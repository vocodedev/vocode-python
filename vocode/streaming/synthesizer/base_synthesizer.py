import asyncio
import audioop
import math
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import aiohttp
import sentry_sdk
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sentry_sdk.tracing import Span

from vocode.streaming.agent.base_agent import AgentResponse
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.agent import FillerAudioConfig
from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken, SilenceMessage
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.synthesizer.audio_cache import AudioCache
from vocode.streaming.synthesizer.cached_audio import CachedAudio, SilenceAudio
from vocode.streaming.synthesizer.constants import TYPING_NOISE_PATH
from vocode.streaming.synthesizer.filler_audio import FillerAudio
from vocode.streaming.synthesizer.input_streaming_synthesizer import InputStreamingSynthesizer
from vocode.streaming.synthesizer.miniaudio_worker import MiniaudioWorker
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import encode_as_wav
from vocode.streaming.utils import convert_wav
from vocode.streaming.utils.async_requester import AsyncRequestor
from vocode.streaming.utils.create_task import asyncio_create_task_with_done_error_log
from vocode.streaming.utils.worker import (
    AbstractWorker,
    InterruptibleAgentResponseEvent,
    InterruptibleEventFactory,
    InterruptibleWorker,
    QueueConsumer,
)
from vocode.utils.sentry_utils import (
    CustomSentrySpans,
    complete_span_by_op,
    sentry_create_span,
    synthesizer_base_name_if_should_report_to_sentry,
)

if TYPE_CHECKING:
    from vocode.streaming.utils.state_manager import ConversationStateManager


SynthesizerConfigType = TypeVar("SynthesizerConfigType", bound=SynthesizerConfig)


class BaseSynthesizer(
    Generic[SynthesizerConfigType],
    InterruptibleWorker[InterruptibleAgentResponseEvent[AgentResponse]],
):
    conversation_state_manager: "ConversationStateManager"
    interruptible_event_factory: InterruptibleEventFactory

    consumer: AbstractWorker[
        InterruptibleAgentResponseEvent[
            Tuple[Union[BaseMessage, EndOfTurn], Optional[SynthesisResult]]
        ]
    ]

    def __init__(
        self,
        synthesizer_config: SynthesizerConfigType,
    ):
        InterruptibleWorker.__init__(self)
        self.synthesizer_config = synthesizer_config
        if synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            assert (
                synthesizer_config.sampling_rate == SamplingRate.RATE_8000
            ), "MuLaw encoding only supports 8kHz sampling rate"
        self.filler_audios: List[FillerAudio] = []
        self.async_requestor = AsyncRequestor()
        self.total_chars: int = 0
        self.cost_per_char: Optional[float] = None

        self.last_agent_response_tracker: Optional[asyncio.Event] = None
        self.is_first_text_chunk = True

    async def process(
        self, item: InterruptibleAgentResponseEvent[AgentResponse]
    ):  # todo (dow-107): fix typing
        if not self.conversation_state_manager._conversation.synthesis_enabled:
            logger.debug("Synthesis disabled, not synthesizing speech")
            return
        try:
            agent_response = item.payload

            # todo (dow-107): resupport filler audio
            filler_audio_worker = self.conversation_state_manager._conversation.filler_audio_worker
            if filler_audio_worker is not None:
                if filler_audio_worker.interrupt_current_filler_audio():
                    await filler_audio_worker.wait_for_filler_audio_to_finish()

            if isinstance(agent_response.message, EndOfTurn):
                logger.debug("Sending end of turn")
                if isinstance(self, InputStreamingSynthesizer):
                    await self.handle_end_of_turn()
                self.consumer.consume_nonblocking(
                    self.interruptible_event_factory.create_interruptible_agent_response_event(
                        (agent_response.message, None),
                        is_interruptible=item.is_interruptible,
                        agent_response_tracker=item.agent_response_tracker,
                    ),
                )
                self.is_first_text_chunk = True
                return

            synthesizer_base_name: Optional[str] = synthesizer_base_name_if_should_report_to_sentry(
                self
            )
            create_speech_span: Optional[Span] = None
            ttft_span: Optional[Span] = None
            synthesis_span: Optional[Span] = None
            if synthesizer_base_name and agent_response.is_first:
                complete_span_by_op(CustomSentrySpans.LANGUAGE_MODEL_TIME_TO_FIRST_TOKEN)

                sentry_create_span(
                    sentry_callable=sentry_sdk.start_span,
                    op=CustomSentrySpans.SYNTHESIS_TIME_TO_FIRST_TOKEN,
                )

                synthesis_span = sentry_create_span(
                    sentry_callable=sentry_sdk.start_span,
                    op=f"{synthesizer_base_name}{CustomSentrySpans.SYNTHESIZER_SYNTHESIS_TOTAL}",
                )
                if synthesis_span:
                    ttft_span = sentry_create_span(
                        sentry_callable=synthesis_span.start_child,
                        op=f"{synthesizer_base_name}{CustomSentrySpans.SYNTHESIZER_TIME_TO_FIRST_TOKEN}",
                    )
                if ttft_span:
                    create_speech_span = sentry_create_span(
                        sentry_callable=ttft_span.start_child,
                        op=f"{synthesizer_base_name}{CustomSentrySpans.SYNTHESIZER_CREATE_SPEECH}",
                    )
            maybe_synthesis_result: Optional[SynthesisResult] = None
            if isinstance(
                self,
                InputStreamingSynthesizer,
            ) and isinstance(agent_response.message, LLMToken):
                logger.debug("Sending chunk to synthesizer")
                await self.send_token_to_synthesizer(
                    message=agent_response.message,
                    chunk_size=self.chunk_size,
                )
            else:
                logger.debug("Synthesizing speech for message")
                maybe_synthesis_result = await self.create_speech_with_cache(
                    agent_response.message,
                    self.chunk_size,
                    is_first_text_chunk=self.is_first_text_chunk,
                    is_sole_text_chunk=agent_response.is_sole_text_chunk,
                )
            if create_speech_span:
                create_speech_span.finish()
            # For input streaming synthesizers, subsequent chunks are contained in the same SynthesisResult
            if isinstance(self, InputStreamingSynthesizer):
                if not self.is_first_text_chunk:
                    maybe_synthesis_result = None
                elif isinstance(agent_response.message, LLMToken):
                    maybe_synthesis_result = self.get_current_utterance_synthesis_result()
            if maybe_synthesis_result is not None:
                synthesis_result = maybe_synthesis_result
                synthesis_result.is_first = agent_response.is_first
                if not synthesis_result.cached and synthesis_span:
                    synthesis_result.synthesis_total_span = synthesis_span
                    synthesis_result.ttft_span = ttft_span
                self.consumer.consume_nonblocking(
                    self.interruptible_event_factory.create_interruptible_agent_response_event(
                        (agent_response.message, synthesis_result),
                        is_interruptible=item.is_interruptible,
                        agent_response_tracker=item.agent_response_tracker,
                    ),
                )
            self.last_agent_response_tracker = item.agent_response_tracker
            if not isinstance(agent_response.message, SilenceMessage):
                self.is_first_text_chunk = False
        except asyncio.CancelledError:
            pass

    @property
    def chunk_size(self) -> int:
        return self.conversation_state_manager._conversation._get_synthesizer_chunk_size()

    def attach_conversation_state_manager(
        self, conversation_state_manager: "ConversationStateManager"
    ):
        self.conversation_state_manager = conversation_state_manager

    def set_interruptible_event_factory(
        self, interruptible_event_factory: InterruptibleEventFactory
    ):
        self.interruptible_event_factory = interruptible_event_factory

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

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        raise NotImplementedError

    async def create_speech_with_cache(
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
        return await self.create_speech(
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
            miniaudio_worker.terminate()

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
