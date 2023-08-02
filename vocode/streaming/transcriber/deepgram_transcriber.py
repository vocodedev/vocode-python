import asyncio
import json
import logging
from typing import Optional
import websockets
from websockets.client import WebSocketClientProtocol
import audioop
from urllib.parse import urlencode
from vocode import getenv

from vocode.streaming.models.endpoint_classifier_model import EndpointClassifier


from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    EndpointingConfig,
    EndpointingType,
)
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.endpoint_classifier_model import EndpointClassifier

PUNCTUATION_TERMINATORS = [".", "!", "?"]
NUM_RESTARTS = 5


avg_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.avg_latency",
    unit="seconds",
)
max_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.max_latency",
    unit="seconds",
)
min_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.min_latency",
    unit="seconds",
)
duration_hist = meter.create_histogram(
    name="transcriber.deepgram.duration",
    unit="seconds",
)

classy = EndpointClassifier()


class DeepgramTranscriber(BaseAsyncTranscriber[DeepgramTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: DeepgramTranscriberConfig,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set DEEPGRAM_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.is_ready = False
        self.logger = logger or logging.getLogger(__name__)
        self.audio_cursor = 0.0
        self.classifier = classy

    async def _run_loop(self):
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            self.logger.debug(
                "Deepgram connection died, restarting, num_restarts: %s", restarts
            )

    def send_audio(self, chunk):
        if (
            self.transcriber_config.downsampling
            and self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16
        ):
            chunk, _ = audioop.ratecv(
                chunk,
                2,
                1,
                self.transcriber_config.sampling_rate
                * self.transcriber_config.downsampling,
                self.transcriber_config.sampling_rate,
                None,
            )
        super().send_audio(chunk)

    def terminate(self):
        terminate_msg = json.dumps({"type": "CloseStream"})
        self.input_queue.put_nowait(terminate_msg)
        self._ended = True
        super().terminate()

    def get_deepgram_url(self):
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            encoding = "linear16"
        elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            encoding = "mulaw"
        url_params = {
            "encoding": encoding,
            "sample_rate": self.transcriber_config.sampling_rate,
            "channels": 1,
            "interim_results": "true",
        }
        # trial
        extra_params = {}
        extra_params["interim_results"] = "true"
        extra_params["diarize"] = "true"
        extra_params["tier"] = "phonecall"
        if self.transcriber_config.language:
            extra_params["language"] = self.transcriber_config.language
        if self.transcriber_config.model:
            extra_params["model"] = "nova"
        if self.transcriber_config.tier:
            extra_params["tier"] = self.transcriber_config.tier
        if self.transcriber_config.version:
            extra_params["version"] = self.transcriber_config.version
        if self.transcriber_config.keywords:
            extra_params["keywords"] = self.transcriber_config.keywords
        if (
            self.transcriber_config.endpointing_config
            and self.transcriber_config.endpointing_config.type
            == EndpointingType.PUNCTUATION_BASED
        ):
            extra_params["punctuate"] = "false"
        url_params.update(extra_params)
        return f"wss://api.deepgram.com/v1/listen?{urlencode(url_params)}"

    def calculate_time_silent(self, data: dict):
        end = data["start"] + data["duration"]
        words = data["channel"]["alternatives"][0]["words"]
        if words:
            return end - words[-1]["end"]
        return data["duration"]

    def is_speech_final(self, deep_response: dict, time_silent: float = 0.0):
        transcript = deep_response["channel"]["alternatives"][0]["transcript"]

        # this will be parameter based by default
        if not self.transcriber_config.endpointing_config:
            self.logger.debug("We have selected default based endpointing")

            return transcript and deep_response["speech_final"]
        elif (
            self.transcriber_config.endpointing_config.type
            == EndpointingType.PUNCTUATION_BASED
        ):
            return (
                transcript
                and transcript.strip()[-1] in PUNCTUATION_TERMINATORS
                and deep_response["speech_final"]
            ) or (False)
        elif (
            self.transcriber_config.endpointing_config.type
            == EndpointingType.CLASSIFIER_BASED
        ):
            self.logger.debug("We have selected classifier based endpointing")
            return transcript and self.classifier.classify_text(
                transcript, return_as_int=False
            )  # or deep_response["speech_final"]
        raise Exception("Endpointing config not supported")

    async def process(self):
        self.audio_cursor = 0.0
        extra_headers = {"Authorization": f"Token {self.api_key}"}

        async with websockets.connect(
            self.get_deepgram_url(), extra_headers=extra_headers
        ) as ws:

            async def sender(ws: WebSocketClientProtocol):  # sends audio to websocket
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
                    await ws.send(data)
                self.logger.debug("Terminating Deepgram transcriber sender")

            async def receiver(ws: WebSocketClientProtocol):
                buffer = ""
                time_silent = 0
                transcript_cursor = 0.0
                while not self._ended:
                    try:
                        msg = await ws.recv()
                    except Exception as e:
                        self.logger.debug(f"Got error {e} in Deepgram receiver")
                        break
                    data = json.loads(msg)
                    self.logger.debug(f"Received message from Deepgram: {data['type']}")
                    if (
                        not "is_final" in data
                    ):  # means we've finished receiving transcriptions
                        self.logger.debug("No is_final in Deepgram response, breaking")
                        break
                    cur_max_latency = self.audio_cursor - transcript_cursor
                    transcript_cursor = data["start"] + data["duration"]
                    cur_min_latency = self.audio_cursor - transcript_cursor

                    avg_latency_hist.record(
                        (cur_min_latency + cur_max_latency) / 2 * data["duration"]
                    )
                    duration_hist.record(data["duration"])

                    # Log max and min latencies
                    max_latency_hist.record(cur_max_latency)
                    min_latency_hist.record(max(cur_min_latency, 0))

                    is_final = data["is_final"]
                    speech_final = self.classifier.classify_text(
                        data["channel"]["alternatives"][0]["transcript"],
                        return_as_int=False,
                    )
                    # if speech_final is False:
                    #     self.logger.debug("Classifier returned false")
                    #     break

                    is_final = is_final or speech_final
                    self.logger.debug(
                        f"Probability you stopped speaking: {str(speech_final)}"
                    )
                    top_choice = data["channel"]["alternatives"][0]
                    confidence = top_choice["confidence"]

                    if is_final and top_choice["transcript"]:
                        self.logger.debug(
                            f"Received final transcript: {top_choice['transcript']}"
                        )
                        buffer = f"{buffer} {top_choice['transcript']}"

                        self.output_queue.put_nowait(
                            Transcription(
                                message=buffer, confidence=confidence, is_final=is_final
                            )
                        )
                        buffer = ""
                        time_silent = 0
                    elif top_choice["transcript"] and data["is_final"]:
                        self.logger.debug(
                            "we should not be here, but we are, so we are going to send this"
                        )
                        buffer = f"{buffer} {top_choice['transcript']}"
                    else:
                        time_silent += data["duration"]
                    self.logger.debug(f"Time silent: {time_silent}")
                self.logger.debug("Terminating Deepgram transcriber receiver")

            await asyncio.gather(sender(ws), receiver(ws))
