import asyncio
import audioop
import json
from typing import Optional

import numpy as np
import websockets
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import GladiaTranscriberConfig, Transcription
from vocode.streaming.models.websocket import AudioMessage
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

GLADIA_URL = "wss://api.gladia.io/audio/text/audio-transcription"


class GladiaTranscriber(BaseAsyncTranscriber[GladiaTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: GladiaTranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("GLADIA_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set GLADIA_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        if self.transcriber_config.endpointing_config:
            raise Exception("Gladia endpointing config not supported yet")

        self.buffer = bytearray()

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

    async def process(self):
        async with websockets.connect(GLADIA_URL) as ws:
            await ws.send(
                json.dumps(
                    {
                        "x_gladia_key": self.api_key,
                        "sample_rate": self.transcriber_config.sampling_rate,
                        "encoding": "wav",
                    }
                )
            )

            async def sender(ws):
                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self.input_queue.get(), 5)
                    except asyncio.exceptions.TimeoutError:
                        break

                    await ws.send(
                        json.dumps(
                            {
                                "x_gladia_key": self.api_key,
                                "frames": AudioMessage.from_bytes(data).data,
                            }
                        )
                    )
                logger.debug("Terminating Gladia transcriber sender")

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

                    if data:
                        is_final = data["type"] == "final"

                        if "transcription" in data and data["transcription"]:
                            self.produce_nonblocking(
                                Transcription(
                                    message=data["transcription"],
                                    confidence=data["confidence"],
                                    is_final=is_final,
                                )
                            )

            await asyncio.gather(sender(ws), receiver(ws))
