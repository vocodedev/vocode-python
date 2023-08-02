import asyncio
import json
import logging
from typing import Optional
import websockets
import audioop
import numpy as np
from urllib.parse import urlencode
from vocode import getenv

from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    EndpointingConfig,
    EndpointingType,
)

from vocode.streaming.models.websocket import AudioMessage
from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.endpoint_classifier_model import EndpointClassifier

ASSEMBLY_AI_URL = "wss://api.assemblyai.com/v2/realtime/ws"
PUNCTUATION_TERMINATORS = [".", "!", "?"]


avg_latency_hist = meter.create_histogram(
    name="transcriber.assemblyai.avg_latency",
    unit="seconds",
)
max_latency_hist = meter.create_histogram(
    name="transcriber.assemblyai.max_latency",
    unit="seconds",
)
min_latency_hist = meter.create_histogram(
    name="transcriber.assemblyai.min_latency",
    unit="seconds",
)
duration_hist = meter.create_histogram(
    name="transcriber.assemblyai.duration",
    unit="seconds",
)


class AssemblyAITranscriber(BaseAsyncTranscriber[AssemblyAITranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AssemblyAITranscriberConfig,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("ASSEMBLY_AI_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set ASSEMBLY_AI_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.logger = logger or logging.getLogger(__name__)
        if self.transcriber_config.endpointing_config:
            raise Exception("Assembly AI endpointing config not supported yet")

        self.buffer = bytearray()
        self.audio_cursor = 0
        self.terminate_msg = str.encode(json.dumps({"terminate_session": True}))
        self.classifier = (
            EndpointClassifier()
        )  # classification = classifier.classify_text(text)

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

    def get_assembly_ai_url(self):
        url_params = {"sample_rate": self.transcriber_config.sampling_rate}
        if self.transcriber_config.word_boost:
            url_params.update(
                {"word_boost": json.dumps(self.transcriber_config.word_boost)}
            )
        return ASSEMBLY_AI_URL + f"?{urlencode(url_params)}"

    def is_speech_final(self, assembly_response: dict):
        transcript = assembly_response["channel"]["alternatives"][0]["transcript"]

        # this will be parameter based by default
        if not self.transcriber_config.endpointing_config:
            return transcript and assembly_response["speech_final"]
        elif (
            self.transcriber_config.endpointing_config.type
            == EndpointingType.PUNCTUATION_BASED
        ):
            return (
                transcript
                and transcript.strip()[-1] in PUNCTUATION_TERMINATORS
                and assembly_response["speech_final"]
            ) or (False)
        elif (
            self.transcriber_config.endpointing_config.type
            == EndpointingType.CLASSIFIER_BASED
        ):
            return (
                self.classifier.classify_text(transcript)
                or assembly_response["speech_final"]
            )
        raise Exception("Endpointing config not supported")

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

            async def sender(ws):  # sends audio to websocket
                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self.input_queue.get(), 5)
                    except asyncio.exceptions.TimeoutError:
                        break
                    num_channels = 1
                    sample_width = 2
                    self.audio_cursor += len(data) / (
                        self.transcriber_config.sampling_rate
                        * num_channels
                        * sample_width
                    )
                    await ws.send(
                        json.dumps({"audio_data": AudioMessage.from_bytes(data).data})
                    )
                await ws.send(self.terminate_msg)
                self.logger.debug("Terminating AssemblyAI transcriber sender")

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

                    data = json.loads(result_str)
                    is_final = (
                        "message_type" in data
                        and data["message_type"] == "FinalTranscript"
                    )
                    if self.is_speech_final(data):
                        cur_max_latency = self.audio_cursor - transcript_cursor
                        transcript_cursor = data["audio_end"] / 1000
                        cur_min_latency = self.audio_cursor - transcript_cursor
                        duration = data["audio_end"] / 1000 - data["audio_start"] / 1000

                        avg_latency_hist.record(
                            (cur_min_latency + cur_max_latency) / 2 * duration
                        )
                        duration_hist.record(duration)

                        # Log max and min latencies
                        max_latency_hist.record(cur_max_latency)
                        min_latency_hist.record(max(cur_min_latency, 0))

                        self.output_queue.put_nowait(
                            Transcription(
                                message=data["text"],
                                confidence=data["confidence"],
                                is_final=is_final,
                            )
                        )
                    elif "text" in data and data["text"]:
                        cur_max_latency = self.audio_cursor - transcript_cursor
                        transcript_cursor = data["audio_end"] / 1000
                        cur_min_latency = self.audio_cursor - transcript_cursor
                        duration = data["audio_end"] / 1000 - data["audio_start"] / 1000

                        avg_latency_hist.record(
                            (cur_min_latency + cur_max_latency) / 2 * duration
                        )
                        duration_hist.record(duration)

                        # Log max and min latencies
                        max_latency_hist.record(cur_max_latency)
                        min_latency_hist.record(max(cur_min_latency, 0))

                        self.output_queue.put_nowait(
                            Transcription(
                                message=data["text"],
                                confidence=data["confidence"],
                                is_final=is_final,
                            )
                        )

            await asyncio.gather(sender(ws), receiver(ws))
