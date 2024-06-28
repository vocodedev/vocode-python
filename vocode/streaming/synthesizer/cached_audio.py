from typing import Optional

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage, BotBackchannel, SilenceMessage
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import (
    get_message_cutoff_from_total_response_length,
)
from vocode.streaming.telephony.constants import MULAW_SILENCE_BYTE, PCM_SILENCE_BYTE


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
                return get_message_cutoff_from_total_response_length(
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
