import asyncio
import audioop
import json
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import websockets
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
    Transcription,
)
from vocode.streaming.models.websocket import AudioMessage
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

ASSEMBLY_AI_URL = "wss://api.assemblyai.com/v2/realtime/ws"


class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("ASSEMBLY_AI_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set ASSEMBLY_AI_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.buffer = bytearray()
        self.audio_cursor = 0

        if isinstance(
            self.transcriber_config.endpointing_config,
            (TimeEndpointingConfig, PunctuationEndpointingConfig),
        ):
            self.transcriber_config.end_utterance_silence_threshold_milliseconds = int(
                self.transcriber_config.endpointing_config.time_cutoff_seconds * 1000
            )
        self.terminate_msg = json.dumps({"terminate_session": True})
        self.end_utterance_silence_threshold_msg = (
            None
            if self.transcriber_config.end_utterance_silence_threshold_milliseconds is None
            else json.dumps(
                {
                    "end_utterance_silence_threshold": self.transcriber_config.end_utterance_silence_threshold_milliseconds
                }
            )
        )

    async def ready(self):
        return True

    async def _run_loop(self):
        await self.process()

    def send_audio(self, chunk):
        if self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            sample_width = 1
            if isinstance(chunk, np.ndarray):
                chunk = chunk.astype(np.int16)
                chunk = chunk.tobytes()
            chunk = audioop.ulaw2lin(chunk, sample_width)

        self.buffer.extend(chunk)

        if (
            len(self.buffer) / (2 * self.transcriber_config.sampling_rate)
        ) >= self.transcriber_config.buffer_size_seconds:
            self.consume_nonblocking(self.buffer)
            self.buffer = bytearray()

    def terminate(self):
        self._ended = True
        super().terminate()

    def get_assembly_ai_url(self):
        url_params = {"sample_rate": self.transcriber_config.sampling_rate}
        if self.transcriber_config.word_boost:
            url_params.update({"word_boost": json.dumps(self.transcriber_config.word_boost)})
        return ASSEMBLY_AI_URL + f"?{urlencode(url_params)}"

    async def process(self):
        self.audio_cursor = 0
        URL = self.get_assembly_ai_url()

        async with websockets.connect(
            URL,
            extra_headers=(("Authorization", self.api_key),),
            ping_interval=5,
            ping_timeout=20,
        ) as ws:
            await asyncio.sleep(0.1)

            if self.end_utterance_silence_threshold_msg:
                await ws.send(self.end_utterance_silence_threshold_msg)

            async def sender(ws):  # sends audio to websocket
                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self.input_queue.get(), 5)
                    except asyncio.exceptions.TimeoutError:
                        break
                    num_channels = 1
                    sample_width = 2
                    self.audio_cursor += len(data) / (
                        self.transcriber_config.sampling_rate * num_channels * sample_width
                    )
                    await ws.send(json.dumps({"audio_data": AudioMessage.from_bytes(data).data}))
                await ws.send(self.terminate_msg)
                logger.debug("Terminating AssemblyAI transcriber sender")

            async def receiver(ws):
                while not self._ended:
                    try:
                        result_str = await ws.recv()
                        data = json.loads(result_str)
                        if "error" in data and data["error"]:
                            raise Exception(data["error"])
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.debug(e)
                        break

                    data = json.loads(result_str)
                    is_final = "message_type" in data and data["message_type"] == "FinalTranscript"

                    if "text" in data and data["text"]:
                        self.produce_nonblocking(
                            Transcription(
                                message=data["text"],
                                confidence=data["confidence"],
                                is_final=is_final,
                            )
                        )

            await asyncio.gather(sender(ws), receiver(ws))
