import asyncio
import audioop
import io
import math
import os
import wave
from typing import Any, AsyncGenerator, Callable, Generic, List, Optional, Tuple, TypeVar, Union

import aiohttp
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sentry_sdk.tracing import Span as SentrySpan

from vocode.streaming.models.agent import FillerAudioConfig
from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage, BotBackchannel, SilenceMessage
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.pipeline.worker import QueueConsumer
from vocode.streaming.synthesizer.audio_cache import AudioCache
from vocode.streaming.synthesizer.miniaudio_worker import MiniaudioWorker
from vocode.streaming.telephony.constants import MULAW_SILENCE_BYTE, PCM_SILENCE_BYTE
from vocode.streaming.utils import convert_wav, get_chunk_size_per_second
from vocode.streaming.utils.async_requester import AsyncRequestor
from vocode.streaming.utils.create_task import asyncio_create_task_with_done_error_log

FILLER_PHRASES = [
    BaseMessage(text="Um..."),
    BaseMessage(text="Uh..."),
    BaseMessage(text="Uh-huh..."),
    BaseMessage(text="Mm-hmm..."),
    BaseMessage(text="Hmm..."),
    BaseMessage(text="Okay..."),
    BaseMessage(text="Right..."),
    BaseMessage(text="Let me see..."),
]
FILLER_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "filler_audio")
TYPING_NOISE_PATH = "%s/typing-noise.wav" % FILLER_AUDIO_PATH


def encode_as_wav(chunk: bytes, synthesizer_config: SynthesizerConfig) -> bytes:
    output_bytes_io = io.BytesIO()
    in_memory_wav = wave.open(output_bytes_io, "wb")
    in_memory_wav.setnchannels(1)
    assert synthesizer_config.audio_encoding == AudioEncoding.LINEAR16
    in_memory_wav.setsampwidth(2)
    in_memory_wav.setframerate(synthesizer_config.sampling_rate)
    in_memory_wav.writeframes(chunk)
    output_bytes_io.seek(0)
    return output_bytes_io.read()


class SynthesisResult:
    """Holds audio bytes for an utterance and method to know how much of utterance was spoken

    @param chunk_generator - an async generator that that yields ChunkResult objects, which contain chunks of audio and a flag indicating if it is the last chunk
    @param get_message_up_to - takes in the number of seconds spoken and returns the message up to that point
    - *if seconds is None, then it should return the full messages*
    """

    class ChunkResult:
        def __init__(self, chunk: bytes, is_last_chunk: bool):
            self.chunk = chunk
            self.is_last_chunk = is_last_chunk

    def __init__(
        self,
        chunk_generator: AsyncGenerator[ChunkResult, None],
        get_message_up_to: Callable[[Optional[float]], str],
        cached: bool = False,
        is_first: bool = False,
        synthesis_total_span: Optional[SentrySpan] = None,
        ttft_span: Optional[SentrySpan] = None,
    ):
        self.chunk_generator = chunk_generator
        self.get_message_up_to = get_message_up_to
        self.cached = cached
        self.is_first = is_first
        self.synthesis_total_span = synthesis_total_span
        self.ttft_span = ttft_span


class FillerAudio:
    def __init__(
        self,
        message: BaseMessage,
        audio_data: bytes,
        synthesizer_config: SynthesizerConfig,
        is_interruptible: bool = False,
        seconds_per_chunk: int = 1,
    ):
        self.message = message
        self.audio_data = audio_data
        self.synthesizer_config = synthesizer_config
        self.is_interruptible = is_interruptible
        self.seconds_per_chunk = seconds_per_chunk

    def create_synthesis_result(self) -> SynthesisResult:
        chunk_size = (
            get_chunk_size_per_second(
                self.synthesizer_config.audio_encoding,
                self.synthesizer_config.sampling_rate,
            )
            * self.seconds_per_chunk
        )

        async def chunk_generator(chunk_transform=lambda x: x):
            for i in range(0, len(self.audio_data), chunk_size):
                if i + chunk_size > len(self.audio_data):
                    yield SynthesisResult.ChunkResult(chunk_transform(self.audio_data[i:]), True)
                else:
                    yield SynthesisResult.ChunkResult(
                        chunk_transform(self.audio_data[i : i + chunk_size]), False
                    )

        if self.synthesizer_config.should_encode_as_wav:
            output_generator = chunk_generator(
                lambda chunk: encode_as_wav(chunk, self.synthesizer_config)
            )
        else:
            output_generator = chunk_generator()
        return SynthesisResult(output_generator, lambda _: self.message.text)


class CachedAudio:
    def __init__(
        self,
        message: BaseMessage,
        audio_data: bytes,
        synthesizer_config: SynthesizerConfig,
        trailing_silence_seconds: float = 0.0,
    ):
        self.message = message
        self.audio_data = audio_data
        self.synthesizer_config = synthesizer_config
        self.trailing_silence_seconds = trailing_silence_seconds

    def create_synthesis_result(self, chunk_size) -> SynthesisResult:
        async def chunk_generator():
            if isinstance(self.message, BotBackchannel):
                yield SynthesisResult.ChunkResult(
                    self.audio_data, self.trailing_silence_seconds == 0.0
                )
            else:
                for i in range(0, len(self.audio_data), chunk_size):
                    if i + chunk_size > len(self.audio_data):
                        yield SynthesisResult.ChunkResult(
                            self.audio_data[i:], self.trailing_silence_seconds == 0.0
                        )
                    else:
                        yield SynthesisResult.ChunkResult(
                            self.audio_data[i : i + chunk_size], False
                        )
            if self.trailing_silence_seconds > 0:
                silence_synthesis_result = self.create_silence_synthesis_result(chunk_size)
                async for chunk_result in silence_synthesis_result.chunk_generator:
                    yield chunk_result

        if isinstance(self.message, BotBackchannel):

            def get_message_up_to(seconds: Optional[float]):
                return self.message.text

        else:

            def get_message_up_to(seconds: Optional[float]):
                return BaseSynthesizer.get_message_cutoff_from_total_response_length(
                    self.synthesizer_config, self.message, seconds, len(self.audio_data)
                )

        return SynthesisResult(
            chunk_generator=chunk_generator(),
            get_message_up_to=get_message_up_to,
            cached=True,
        )

    def create_silence_synthesis_result(self, chunk_size) -> SynthesisResult:
        async def chunk_generator():
            size_of_silence = int(
                self.trailing_silence_seconds * self.synthesizer_config.sampling_rate
            )
            silence_byte: bytes
            if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
                silence_byte = PCM_SILENCE_BYTE
                size_of_silence *= 2
            elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
                silence_byte = MULAW_SILENCE_BYTE

            for _ in range(
                0,
                size_of_silence,
                chunk_size,
            ):
                yield SynthesisResult.ChunkResult(silence_byte * chunk_size, False)
            yield SynthesisResult.ChunkResult(silence_byte * chunk_size, True)

        def get_message_up_to(seconds):
            return ""

        return SynthesisResult(
            chunk_generator(),
            get_message_up_to,
        )


class SilenceAudio(CachedAudio):
    def __init__(
        self,
        message: SilenceMessage,
        synthesizer_config: SynthesizerConfig,
    ):
        super().__init__(
            message,
            b"",
            synthesizer_config,
            trailing_silence_seconds=message.trailing_silence_seconds,
        )

    def create_synthesis_result(self, chunk_size) -> SynthesisResult:
        return self.create_silence_synthesis_result(chunk_size)


SynthesizerConfigType = TypeVar("SynthesizerConfigType", bound=SynthesizerConfig)


class BaseSynthesizer(Generic[SynthesizerConfigType]):
    def __init__(
        self,
        synthesizer_config: SynthesizerConfigType,
    ):
        self.synthesizer_config = synthesizer_config
        if synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            assert (
                synthesizer_config.sampling_rate == SamplingRate.RATE_8000
            ), "MuLaw encoding only supports 8kHz sampling rate"
        self.filler_audios: List[FillerAudio] = []
        self.async_requestor = AsyncRequestor()
        self.total_chars: int = 0
        self.cost_per_char: Optional[float] = None

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: SynthesizerConfigType) -> str:
        raise NotImplementedError

    async def empty_generator(self):
        yield SynthesisResult.ChunkResult(b"", True)

    def get_synthesizer_config(self) -> SynthesizerConfig:
        return self.synthesizer_config

    def get_typing_noise_filler_audio(self) -> FillerAudio:
        return FillerAudio(
            message=BaseMessage(text="<typing noise>"),
            audio_data=convert_wav(
                TYPING_NOISE_PATH,
                output_sample_rate=self.synthesizer_config.sampling_rate,
                output_encoding=self.synthesizer_config.audio_encoding,
            ),
            synthesizer_config=self.synthesizer_config,
            is_interruptible=True,
            seconds_per_chunk=2,
        )

    def get_cost(self) -> float:
        raise NotImplementedError

    async def set_filler_audios(self, filler_audio_config: FillerAudioConfig):
        if filler_audio_config.use_phrases:
            self.filler_audios = await self.get_phrase_filler_audios()
        elif filler_audio_config.use_typing_noise:
            self.filler_audios = [self.get_typing_noise_filler_audio()]

    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        return []

    def ready_synthesizer(self, chunk_size: int):
        pass

    # given the number of seconds the message was allowed to go until, where did we get in the message?
    @staticmethod
    def get_message_cutoff_from_total_response_length(
        synthesizer_config: SynthesizerConfig,
        message: BaseMessage,
        seconds: Optional[float],
        size_of_output: int,
    ) -> str:
        estimated_output_seconds = size_of_output / synthesizer_config.sampling_rate
        if not message.text:
            return message.text

        if seconds is None:
            return message.text

        estimated_output_seconds_per_char = estimated_output_seconds / len(message.text)
        return message.text[: int(seconds / estimated_output_seconds_per_char)]

    @staticmethod
    def get_message_cutoff_from_voice_speed(
        message: BaseMessage, seconds: Optional[float], words_per_minute: int
    ) -> str:

        if seconds is None:
            return message.text

        words_per_second = words_per_minute / 60
        estimated_words_spoken = math.floor(words_per_second * seconds)
        tokens = word_tokenize(message.text)
        return TreebankWordDetokenizer().detokenize(tokens[:estimated_words_spoken])

    async def get_cached_audio(
        self,
        message: BaseMessage,
    ) -> Optional[CachedAudio]:
        audio_cache = await AudioCache.safe_create()
        cache_phrase = message.cache_phrase or message.text.strip()
        audio_data = await audio_cache.get_audio(
            self.get_voice_identifier(self.synthesizer_config), cache_phrase
        )
        if audio_data is None:
            return None
        logger.info(f"Got cached audio for {cache_phrase}")

        trailing_silence_seconds = 0.0
        if isinstance(message, BotBackchannel):
            trailing_silence_seconds = message.trailing_silence_seconds
        return CachedAudio(message, audio_data, self.synthesizer_config, trailing_silence_seconds)

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        raise NotImplementedError

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        if isinstance(message, SilenceMessage):
            return SilenceAudio(
                message,
                self.synthesizer_config,
            ).create_synthesis_result(chunk_size)

        maybe_cached_audio = await self.get_cached_audio(message)
        if maybe_cached_audio is not None:
            return maybe_cached_audio.create_synthesis_result(chunk_size)
        return await self.create_speech_uncached(
            message,
            chunk_size,
            is_first_text_chunk=is_first_text_chunk,
            is_sole_text_chunk=is_sole_text_chunk,
        )

    async def chunk_result_generator_from_queue(self, chunk_queue: asyncio.Queue[Optional[bytes]]):
        while True:
            try:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break
                yield SynthesisResult.ChunkResult(
                    chunk=chunk,
                    is_last_chunk=False,
                )
            except asyncio.CancelledError:
                break

    # @param file - a file-like object in wav format
    @staticmethod
    def create_synthesis_result_from_wav(
        synthesizer_config: SynthesizerConfig,
        file: Any,
        message: BaseMessage,
        chunk_size: int,
    ) -> SynthesisResult:
        output_bytes = convert_wav(
            file,
            output_sample_rate=synthesizer_config.sampling_rate,
            output_encoding=synthesizer_config.audio_encoding,
        )

        if synthesizer_config.should_encode_as_wav:
            chunk_transform = lambda chunk: encode_as_wav(chunk, synthesizer_config)  # noqa: E731

        else:
            chunk_transform = lambda chunk: chunk  # noqa: E731

        async def chunk_generator(output_bytes):
            for i in range(0, len(output_bytes), chunk_size):
                if i + chunk_size > len(output_bytes):
                    yield SynthesisResult.ChunkResult(chunk_transform(output_bytes[i:]), True)
                else:
                    yield SynthesisResult.ChunkResult(
                        chunk_transform(output_bytes[i : i + chunk_size]), False
                    )

        return SynthesisResult(
            chunk_generator(output_bytes),
            lambda seconds: BaseSynthesizer.get_message_cutoff_from_total_response_length(
                synthesizer_config, message, seconds, len(output_bytes)
            ),
        )

    async def experimental_mp3_streaming_output_generator(
        self,
        response: aiohttp.ClientResponse,
        chunk_size: int,
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        miniaudio_worker_consumer: QueueConsumer = QueueConsumer()
        miniaudio_worker = MiniaudioWorker(
            self.synthesizer_config,
            chunk_size,
        )
        miniaudio_worker.consumer = miniaudio_worker_consumer
        miniaudio_worker.start()
        stream_reader = response.content

        # Create a task to send the mp3 chunks to the MiniaudioWorker's input queue in a separate loop
        async def send_chunks():
            async for chunk in stream_reader.iter_any():
                miniaudio_worker.consume_nonblocking(chunk)
            miniaudio_worker.consume_nonblocking(None)  # sentinel

        try:
            asyncio_create_task_with_done_error_log(send_chunks(), reraise_cancelled=True)

            # Await the output queue of the MiniaudioWorker and yield the wav chunks in another loop
            while True:
                # Get the wav chunk and the flag from the output queue of the MiniaudioWorker
                wav_chunk, is_last = await miniaudio_worker_consumer.input_queue.get()
                if self.synthesizer_config.should_encode_as_wav:
                    wav_chunk = encode_as_wav(wav_chunk, self.synthesizer_config)

                if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
                    wav_chunk = audioop.lin2ulaw(wav_chunk, 2)

                yield SynthesisResult.ChunkResult(wav_chunk, is_last)
                # If this is the last chunk, break the loop
                if is_last:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            await miniaudio_worker.terminate()

    def _resample_chunk(
        self,
        chunk: bytes,
        current_sample_rate: int,
        target_sample_rate: int,
    ) -> bytes:
        resampled_chunk, _ = audioop.ratecv(
            chunk,
            2,
            1,
            current_sample_rate,
            target_sample_rate,
            None,
        )

        return resampled_chunk

    async def tear_down(self):
        pass
