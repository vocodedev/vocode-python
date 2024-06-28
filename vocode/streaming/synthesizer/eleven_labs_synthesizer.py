import asyncio
import hashlib
from typing import Optional

from elevenlabs import Voice, VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from loguru import logger

from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import (
    chunk_result_generator_from_queue,
    get_message_cutoff_from_voice_speed,
    resample_chunk,
)
from vocode.streaming.utils.create_task import asyncio_create_task_with_done_error_log

ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"
STREAMED_CHUNK_SIZE = 16000 * 2 // 4  # 1/8 of a second of 16kHz audio with 16-bit samples


class ElevenLabsSynthesizer(AbstractSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        assert synthesizer_config.api_key is not None, "API key must be set"
        assert synthesizer_config.voice_id is not None, "Voice ID must be set"
        self.api_key = synthesizer_config.api_key

        self.elevenlabs_client = AsyncElevenLabs(
            api_key=self.api_key,
        )

        self.voice_id = synthesizer_config.voice_id
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150
        self.upsample = None
        self.sample_rate = self.synthesizer_config.sampling_rate

        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            match self.synthesizer_config.sampling_rate:
                case SamplingRate.RATE_16000:
                    self.output_format = "pcm_16000"
                case SamplingRate.RATE_22050:
                    self.output_format = "pcm_22050"
                case SamplingRate.RATE_24000:
                    self.output_format = "pcm_24000"
                case SamplingRate.RATE_44100:
                    self.output_format = "pcm_44100"
                case SamplingRate.RATE_48000:
                    self.output_format = "pcm_44100"
                    self.upsample = SamplingRate.RATE_48000.value
                    self.sample_rate = SamplingRate.RATE_44100.value
                case _:
                    raise ValueError(
                        f"Unsupported sampling rate: {self.synthesizer_config.sampling_rate}. Elevenlabs only supports 16000, 22050, 24000, and 44100 Hz."
                    )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            self.output_format = "ulaw_8000"
        else:
            raise ValueError(
                f"Unsupported audio encoding: {self.synthesizer_config.audio_encoding}"
            )

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        if self.stability is not None and self.similarity_boost is not None:
            voice = Voice(
                voice_id=self.voice_id,
                settings=VoiceSettings(
                    stability=self.stability, similarity_boost=self.similarity_boost
                ),
            )
        else:
            voice = Voice(voice_id=self.voice_id)
        url = (
            ELEVEN_LABS_BASE_URL
            + f"text-to-speech/{self.voice_id}/stream?output_format={self.output_format}"
        )
        if self.optimize_streaming_latency:
            url += f"&optimize_streaming_latency={self.optimize_streaming_latency}"
        headers = {"xi-api-key": self.api_key}
        body = {
            "text": message.text,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        asyncio_create_task_with_done_error_log(
            self.get_chunks(url, headers, body, chunk_size, chunk_queue),
        )

        return SynthesisResult(
            chunk_result_generator_from_queue(chunk_queue),
            lambda seconds: get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: ElevenLabsSynthesizerConfig):
        hashed_api_key = hashlib.sha256(f"{synthesizer_config.api_key}".encode("utf-8")).hexdigest()
        return ":".join(
            (
                "eleven_labs",
                hashed_api_key,
                str(synthesizer_config.voice_id),
                str(synthesizer_config.model_id),
                str(synthesizer_config.stability),
                str(synthesizer_config.similarity_boost),
                str(synthesizer_config.optimize_streaming_latency),
                synthesizer_config.audio_encoding,
            )
        )

    async def get_chunks(
        self,
        url: str,
        headers: dict,
        body: dict,
        chunk_size: int,
        chunk_queue: asyncio.Queue[Optional[bytes]],
    ):
        try:
            async_client = self.async_requestor.get_client()
            stream = await async_client.send(
                async_client.build_request(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                ),
                stream=True,
            )

            if not stream.is_success:
                error = await stream.aread()
                logger.error(f"ElevenLabs API failed: {stream.status_code} {error.decode('utf-8')}")
                raise Exception(f"ElevenLabs API returned {stream.status_code} status code")
            async for chunk in stream.aiter_bytes(chunk_size):
                if self.upsample:
                    chunk = resample_chunk(
                        chunk,
                        self.sample_rate,
                        self.upsample,
                    )
                chunk_queue.put_nowait(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # treated as sentinel
