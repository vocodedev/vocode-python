import asyncio
from typing import Literal

import aiohttp
from aiohttp import ClientTimeout

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult

TTS_ENDPOINT = "https://play.ht/api/v2/tts/stream"


class PlayHtSynthesizer(BaseSynthesizer[PlayHtSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: PlayHtSynthesizerConfig,
        max_backoff_retries=3,
        backoff_retry_delay=2,
    ):
        super().__init__(
            synthesizer_config,
        )
        self.synthesizer_config = synthesizer_config
        self.api_key = synthesizer_config.api_key
        self.user_id = synthesizer_config.user_id
        if not self.api_key or not self.user_id:
            raise ValueError(
                "You must set the PLAY_HT_API_KEY and PLAY_HT_USER_ID environment variables"
            )
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.max_backoff_retries = max_backoff_retries
        self.backoff_retry_delay = backoff_retry_delay

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        headers = {
            "AUTHORIZATION": f"Bearer {self.api_key}",
            "X-USER-ID": self.user_id,
            "Content-Type": "application/json",
        }
        output_format: Literal["wav", "mulaw"]
        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            output_format = "wav"
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            output_format = "mulaw"
        body = {
            "quality": "draft",
            "voice": self.synthesizer_config.voice_id,
            "text": message.text,
            "sample_rate": self.synthesizer_config.sampling_rate,
            "output_format": output_format,
        }
        if self.synthesizer_config.speed:
            body["speed"] = self.synthesizer_config.speed
        if self.synthesizer_config.seed:
            body["seed"] = self.synthesizer_config.seed
        if self.synthesizer_config.temperature:
            body["temperature"] = self.synthesizer_config.temperature
        if self.synthesizer_config.quality:
            body["quality"] = self.synthesizer_config.quality

        backoff_retry_delay = self.backoff_retry_delay
        max_backoff_retries = self.max_backoff_retries
        for attempt in range(max_backoff_retries):
            response = await self.async_requestor.get_session().post(
                TTS_ENDPOINT,
                headers=headers,
                json=body,
                timeout=ClientTimeout(total=15),
            )
            if not response.ok:
                raise Exception(f"Play.ht API error status code {response.status}")
            if response.status == 429 and attempt < max_backoff_retries - 1:
                await asyncio.sleep(backoff_retry_delay)
                backoff_retry_delay *= 2  # Exponentially increase delay
                continue

            if not response.ok:
                raise Exception(f"Play.ht API error status code {response.status}")

            if self.experimental_streaming:
                return SynthesisResult(
                    self.experimental_mp3_streaming_output_generator(
                        response, chunk_size
                    ),  # should be wav
                    lambda seconds: self.get_message_cutoff_from_voice_speed(
                        message, seconds, self.words_per_minute
                    ),
                )
            else:
                return SynthesisResult(
                    self._streaming_chunk_generator(response, chunk_size, output_format),
                    lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
                )

        raise Exception("Max retries reached for Play.ht API")

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: PlayHtSynthesizerConfig):
        return ":".join(
            (
                "play_ht",
                str(synthesizer_config.user_id),
                synthesizer_config.voice_id,
                str(synthesizer_config.speed),
                str(synthesizer_config.seed),
                str(synthesizer_config.temperature),
                str(synthesizer_config.quality),
                synthesizer_config.audio_encoding,
            )
        )

    @staticmethod
    async def _streaming_chunk_generator(
        response: aiohttp.ClientResponse,
        chunk_size: int,
        output_format: Literal["wav", "mulaw"],
    ):
        if output_format == "wav":
            buffer = b""
            is_first_chunk = True
            async for chunk in response.content.iter_any():
                if is_first_chunk:
                    is_first_chunk = False
                    buffer += chunk[88:]  # size of the wav header
                else:
                    buffer += chunk
                i = 0
                while i < len(buffer) - chunk_size:
                    yield SynthesisResult.ChunkResult(buffer[i : i + chunk_size], False)
                    i += chunk_size
                buffer = buffer[i:]
        elif output_format == "mulaw":
            async for chunk in response.content.iter_chunked(chunk_size):
                yield SynthesisResult.ChunkResult(chunk, False)
        await response.release()
