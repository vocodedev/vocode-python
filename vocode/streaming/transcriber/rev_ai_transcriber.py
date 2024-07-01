import asyncio
import json
import time
from typing import Optional

import websockets
from loguru import logger
from websockets.client import WebSocketClientProtocol

from vocode import getenv
from vocode.streaming.models.transcriber import (
    RevAITranscriberConfig,
    TimeEndpointingConfig,
    Transcription,
)
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber

NUM_RESTARTS = 5


def getSeconds():
    return time.time()


class RevAITranscriber(BaseAsyncTranscriber[RevAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: RevAITranscriberConfig,
        api_key: Optional[str] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("REV_AI_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set REV_AI_API_KEY environment variable or pass it as a parameter"
            )
        self.closed = False
        self.is_ready = True
        self.last_signal_seconds = 0

    async def ready(self):
        return self.is_ready

    def get_rev_ai_url(self):
        codec = "audio/x-raw"
        layout = "interleaved"
        rate = self.get_transcriber_config().sampling_rate
        audio_format = "S16LE"
        channels = 1

        content_type = (
            f"{codec};layout={layout};rate={rate};format={audio_format};channels={channels}"
        )

        url_params_dict = {
            "access_token": self.api_key,
            "content_type": content_type,
        }

        url_params_arr = [f"{key}={value}" for (key, value) in url_params_dict.items()]
        url = "wss://api.rev.ai/speechtotext/v1/stream?" + "&".join(url_params_arr)
        return url

    async def _run_loop(self):
        restarts = 0
        while not self.closed and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            logger.debug(f"Rev AI connection died, restarting, num_restarts: {restarts}")

    async def process(self):
        async with websockets.connect(self.get_rev_ai_url()) as ws:

            async def sender(ws: WebSocketClientProtocol):
                while not self.closed:
                    try:
                        data = await asyncio.wait_for(self.input_queue.get(), 5)
                    except asyncio.exceptions.TimeoutError:
                        break
                    await ws.send(data)
                await ws.close()
                logger.debug("Terminating Rev.AI transcriber sender")

            async def receiver(ws: WebSocketClientProtocol):
                buffer = ""

                while not self.closed:
                    try:
                        msg = await ws.recv()
                    except Exception as e:
                        logger.debug(f"Got error {e} in Rev.AI receiver")
                        break
                    data = json.loads(msg)

                    if data["type"] == "connected":
                        continue

                    is_done = data["type"] == "final"
                    if (
                        (len(buffer) > 0)
                        and (self.transcriber_config.endpointing_config)
                        and isinstance(
                            self.transcriber_config.endpointing_config,
                            TimeEndpointingConfig,
                        )
                        and (
                            getSeconds()
                            > self.last_signal_seconds
                            + self.transcriber_config.endpointing_config.time_cutoff_seconds
                        )
                    ):
                        is_done = True

                    new_text = "".join([e["value"] for e in data["elements"]])
                    if len(new_text) > len(buffer):
                        self.last_signal_seconds = getSeconds()
                    buffer = new_text

                    confidence = 1.0
                    if is_done:
                        self.produce_nonblocking(
                            Transcription(message=buffer, confidence=confidence, is_final=True)
                        )
                        buffer = ""
                    else:
                        self.produce_nonblocking(
                            Transcription(
                                message=buffer,
                                confidence=confidence,
                                is_final=False,
                            )
                        )

                logger.debug("Terminating Rev.AI transcriber receiver")

            await asyncio.gather(sender(ws), receiver(ws))

    async def terminate(self):
        terminate_msg = json.dumps({"type": "CloseStream"})
        self.consume_nonblocking(terminate_msg)
        self.closed = True
        await super().terminate()
