import io
from typing import Dict, Tuple

import aiohttp

from vocode import getenv
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CoquiSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.utils.async_requester import AsyncRequestor

raise DeprecationWarning("This Synthesizer is deprecated and will be removed in the future.")

COQUI_BASE_URL = "https://app.coqui.ai/api/v2"


class CoquiSynthesizer(AbstractSynthesizer[CoquiSynthesizerConfig]):
    def __init__(self, synthesizer_config: CoquiSynthesizerConfig):
        super().__init__(synthesizer_config)
        self.api_key = synthesizer_config.api_key or getenv("COQUI_API_KEY")
        self.voice_id = synthesizer_config.voice_id
        self.voice_prompt = synthesizer_config.voice_prompt
        self.use_xtts = synthesizer_config.use_xtts

    def get_request(self, text: str) -> Tuple[str, Dict[str, str], Dict[str, object]]:
        url = COQUI_BASE_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "text": text,
            "speed": 1,
        }

        if self.use_xtts:
            # If we have a voice prompt, use that instead of the voice ID
            if self.voice_prompt is not None:
                url = f"{COQUI_BASE_URL}/samples/xtts/render-from-prompt/"
                body["prompt"] = self.voice_prompt
            else:
                url = f"{COQUI_BASE_URL}/samples/xtts/render/"
                body["voice_id"] = self.voice_id
        else:
            if self.voice_prompt is not None:
                url = f"{COQUI_BASE_URL}/samples/from-prompt/"
                body["prompt"] = self.voice_prompt
            else:
                url = f"{COQUI_BASE_URL}/samples"
                body["voice_id"] = self.voice_id
        return url, headers, body

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        url, headers, body = self.get_request(message.text)

        async with AsyncRequestor().get_session().request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            sample = await response.json()
            async with AsyncRequestor().get_session().request(
                "GET",
                sample["audio_url"],
            ) as response:
                read_response = await response.read()

                result = self.create_synthesis_result_from_wav(
                    synthesizer_config=self.synthesizer_config,
                    file=io.BytesIO(read_response),
                    message=message,
                    chunk_size=chunk_size,
                )
                return result
