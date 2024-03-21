import asyncio
import json
import base64
import logging
from typing import Optional
import websockets
import audioop
import numpy as np
from urllib.parse import urlencode
from vocode import getenv

from vocode.streaming.models.transcriber import GladiaTranscriberConfig
from vocode.streaming.models.websocket import AudioMessage
from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from vocode.streaming.models.audio_encoding import AudioEncoding


GLADIA_URL = "wss://api.gladia.io/audio/text/audio-transcription"


avg_latency_hist = meter.create_histogram(
    name="transcriber.gladia.avg_latency",
    unit="seconds",
)
max_latency_hist = meter.create_histogram(
    name="transcriber.gladia.max_latency",
    unit="seconds",
)
min_latency_hist = meter.create_histogram(
    name="transcriber.gladia.min_latency",
    unit="seconds",
)
duration_hist = meter.create_histogram(
    name="transcriber.gladia.duration",
    unit="seconds",
)


class GladiaTranscriber(BaseAsyncTranscriber[GladiaTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: GladiaTranscriberConfig,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("GLADIA_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set GLADIA_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.logger = logger or logging.getLogger(__name__)
        if self.transcriber_config.endpointing_config:
            raise Exception("Gladia endpointing config not supported yet")

        self.buffer = bytearray()
        self.audio_cursor = 0.0

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
            self.input_queue.put_nowait(self.buffer)
            self.buffer = bytearray()

    def terminate(self):
        self._ended = True
        super().terminate()

    async def process(self):
        self.audio_cursor = 0.0
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

                    # TODO: Move this and similar code into a tracing utils file / the superclass.
                    num_channels = 1
                    sample_width = 2
                    self.audio_cursor += len(data) / (
                        self.transcriber_config.sampling_rate
                        * num_channels
                        * sample_width
                    )

                    await ws.send(
                        json.dumps(
                            {
                                "x_gladia_key": self.api_key,
                                "frames": AudioMessage.from_bytes(data).data,
                            }
                        )
                    )
                self.logger.debug("Terminating Gladia transcriber sender")

            async def receiver(ws):
                transcript_cursor = 0.0
                while not self._ended:
                    try:
                        result_str = await ws.recv()
                        data = json.loads(result_str)
                        if "error" in data and data["error"]:
                            raise Exception(data["error"])
                    except websockets.exceptions.ConnectionClosedError as e:
                        self.logger.debug(e)
                        break

                    if data:
                        is_final = data["type"] == "final"

                        # TODO: Move this and similar code into a tracing
                        # utils file / the superclass.
                        if "transcription" in data and data["transcription"]:
                            cur_max_latency = self.audio_cursor - transcript_cursor
                            transcript_cursor = data["time_end"] / 1000
                            cur_min_latency = self.audio_cursor - transcript_cursor
                            duration = (
                                data["time_end"] / 1000 - data["time_begin"] / 1000
                            )

                            avg_latency_hist.record(
                                (cur_min_latency + cur_max_latency) / 2 * duration
                            )
                            duration_hist.record(duration)

                            # Log max and min latencies
                            max_latency_hist.record(cur_max_latency)
                            min_latency_hist.record(max(cur_min_latency, 0))
                            self.output_queue.put_nowait(
                                Transcription(
                                    message=data["transcription"],
                                    confidence=data["confidence"],
                                    is_final=is_final,
                                )
                            )

            await asyncio.gather(sender(ws), receiver(ws))
