import asyncio
import audioop
import base64
import io
import json
from typing import Optional

import aiohttp
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    RIME_DEFAULT_REDUCE_LATENCY,
    RIME_DEFAULT_SPEED_ALPHA,
    RimeSynthesizerConfig,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult

# TODO: [OSS] Remove call to internal library with Synthesizers refactor

# https://rime.ai/docs/quickstart

WAV_HEADER_LENGTH = 44


class RimeSynthesizer(BaseSynthesizer[RimeSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: RimeSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.base_url = synthesizer_config.base_url
        self.model_id = synthesizer_config.model_id
        self.speaker = synthesizer_config.speaker
        self.speed_alpha = synthesizer_config.speed_alpha
        self.sampling_rate = synthesizer_config.sampling_rate
        self.reduce_latency = synthesizer_config.reduce_latency
        self.api_key = f"Bearer {getenv('RIME_API_KEY')}"

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: RimeSynthesizerConfig):
        return ":".join(
            (
                "rime",
                synthesizer_config.speaker,
                str(synthesizer_config.speed_alpha),
                synthesizer_config.audio_encoding,
            )
        )

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        body = self.get_request_body(message.text)

        async with self.async_requestor.get_session().post(
            self.base_url,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if not response.ok:
                raise Exception(f"Rime API error: {response.status}, {await response.text()}")
            data = json.loads(await response.text())

            audio_content = data.get("audioContent")
            output_bytes = base64.b64decode(audio_content)[WAV_HEADER_LENGTH:]

            if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
                output_bytes = audioop.lin2ulaw(output_bytes, 2)

            return SynthesisResult(
                self._chunk_generator(output_bytes, chunk_size),
                lambda seconds: self.get_message_cutoff_from_total_response_length(
                    self.synthesizer_config, message, seconds, len(output_bytes)
                ),
            )

    @staticmethod
    async def _chunk_generator(output_bytes, chunk_size):
        for i in range(0, len(output_bytes), chunk_size):
            if i + chunk_size > len(output_bytes):
                yield SynthesisResult.ChunkResult(output_bytes[i:], True)
            else:
                yield SynthesisResult.ChunkResult(output_bytes[i : i + chunk_size], False)

    async def get_chunks(
        self,
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
                    self.base_url,
                    headers=headers,
                    json=body,
                ),
                stream=True,
            )
            if not stream.is_success:
                error = await stream.aread()
                logger.error(f"Rime API failed: {stream.status_code} {error.decode('utf-8')}")
                raise Exception(f"Rime API returned {stream.status_code} status code")
            async for chunk in stream.aiter_bytes(chunk_size):
                chunk_queue.put_nowait(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # treated as sentinel

    def get_request_body(self, text):
        speed_alpha = self.speed_alpha if self.speed_alpha else RIME_DEFAULT_SPEED_ALPHA
        reduce_latency = self.reduce_latency if self.reduce_latency else RIME_DEFAULT_REDUCE_LATENCY

        body = {
            "text": text,
            "speaker": self.speaker,
            "samplingRate": self.sampling_rate,
            "speedAlpha": speed_alpha,
            "reduceLatency": reduce_latency,
        }

        if self.model_id:
            body["modelId"] = self.model_id

        return body
