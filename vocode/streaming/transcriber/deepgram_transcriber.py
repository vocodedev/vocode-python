import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union
from urllib.parse import urlencode

import sentry_sdk
import websockets
from loguru import logger
from pydantic.v1 import BaseModel, Field
from websockets.client import WebSocketClientProtocol

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    LiveResultResponse,
    UtteranceEndResponse,
)
from deepgram.clients.live.v1.response import Word

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    DEEPGRAM_API_WS_URL,
    DeepgramTranscriberConfig,
    EndpointingConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
    Transcription,
)
from vocode.streaming.transcriber.base_transcriber import BaseAsyncTranscriber
from vocode.utils.sentry_utils import (
    CustomSentrySpans,
    sentry_configured,
    sentry_create_span,
)

PUNCTUATION_TERMINATORS = [".", "!", "?"]
NUM_RESTARTS = 5
NUM_AUDIO_CHANNELS = 1


def now():
    return datetime.now(tz=timezone.utc)


class TimeSilentConfig(BaseModel):
    time_cutoff_seconds: float = 1
    post_punctuation_time_seconds: float = 0.5


class InternalPunctuationEndpointingConfig(  # type: ignore
    EndpointingConfig, type="internal_punctuation_based"
):
    time_silent_config: TimeSilentConfig = Field(default_factory=TimeSilentConfig)
    use_single_utterance_endpointing_for_first_utterance: bool = False


class DeepgramEndpointingConfig(EndpointingConfig, type="deepgram"):  # type: ignore
    vad_threshold_ms: int = 500
    utterance_cutoff_ms: int = 1000
    time_silent_config: Optional[TimeSilentConfig] = Field(default_factory=TimeSilentConfig)
    use_single_utterance_endpointing_for_first_utterance: bool = False


class DeepgramTranscriber(BaseAsyncTranscriber[DeepgramTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: DeepgramTranscriberConfig,
        api_key: Optional[str] = "",
    ):
        super().__init__(transcriber_config)

        self._ended = False
        self._deepgram_streaming_config = self.create_deepgram_streaming_config()

        # set up the deepgram client and enable keepalive
        self._client_config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(api_key, self._client_config)
        if not deepgram.api_key:
            raise Exception(
                "Please set DEEPGRAM_API_KEY environment variable or pass it as a parameter"
            )
        self._dgClient = deepgram.listen.asynclive.v("1")

        # subscribe to the transcription event
        self._dgClient.on(LiveTranscriptionEvents.Transcript, self._on_transcribe)
        self._dgClient.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)

        self._ended = False
        self._audio_cursor = 0.0
        self._transcript_cursor = 0.0

        self._min_latency = 0.0
        self._max_latency = 0.0
        self._avg_latency = 0.0

        self._avg_latency_numer = 0.0
        self._avg_latency_denom = 0.0

        self._start_ts: Optional[datetime] = None
        self._connected_ts: Optional[datetime] = None
        self._start_sending_ts: Optional[datetime] = None
        self._start_receiving_ts: Optional[datetime] = None

        self.is_first_transcription = True

        # shared between transcribe and utterance_end
        self._buffer = ""
        self._buffer_avg_confidence = 0.0
        self._num_buffer_utterances = 1
        self._time_silent = 0.0
        self._words_buffer: List[Word] = []
        self._is_final_ts: Optional[datetime] = None

    def create_deepgram_streaming_config(self):
        config: LiveOptions = LiveOptions()

        if self.transcriber_config.model:
            config.model = self.transcriber_config.model
        if self.transcriber_config.language:
            config.language = self.transcriber_config.language
        if self.transcriber_config.tier:
            config.tier = self.transcriber_config.tier
        if self.transcriber_config.version:
            config.version = self.transcriber_config.version
        if self.transcriber_config.keywords:
            config.keywords = self.transcriber_config.keywords

        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            config.encoding = "linear16"
        elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            config.encoding = "mulaw"
        else:
            raise Exception(
                f"Audio encoding not supported {self.transcriber_config.audio_encoding}"
            )

        config.interim_results = True
        config.channels = NUM_AUDIO_CHANNELS
        config.sample_rate = self.transcriber_config.sampling_rate

        if self.transcriber_config.endpointing_config and (
            isinstance(
                self.transcriber_config.endpointing_config,
                PunctuationEndpointingConfig,
            )
            or isinstance(
                self.transcriber_config.endpointing_config,
                InternalPunctuationEndpointingConfig,
            )
            or isinstance(
                self.transcriber_config.endpointing_config,
                DeepgramEndpointingConfig,
            )
        ):
            config.punctuate = True
            config.smart_format = True

        if isinstance(
            self.transcriber_config.endpointing_config,
            DeepgramEndpointingConfig,
        ):
            config.endpointing = int(
                self.transcriber_config.endpointing_config.vad_threshold_ms
                * (1 / self._get_speed_coefficient())
            )
            config.utterance_end_ms = max(
                int(
                    self.transcriber_config.endpointing_config.utterance_cutoff_ms
                    * (1 / self._get_speed_coefficient())
                ),
                1000,
            )  # Deepgram recommends using at least 1000ms since the tick period is ~1s

        return config

    def _get_speed_coefficient(self):
        return self.speed_manager.get_speed_coefficient() if self.speed_manager else 1.0

    async def _run_loop(self):
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            logger.debug(f"Deepgram connection died, restarting, num_restarts: {restarts}")

        logger.error("Deepgram connection died, not restarting")

    def terminate(self):
        self._track_latency_of_transcription_start()
        # Put this in logs until we sentry metrics show up
        # properly on dashboard
        logger.info(
            f"Transcription latency is {self._avg_latency:.4f}s",
            extra={
                "avg_latency": self._avg_latency,
                "min_latency": self._min_latency,
                "max_latency": self._max_latency,
            },
        )
        terminate_msg = json.dumps({"type": "CloseStream"}).encode("utf-8")
        self.input_queue.put_nowait(terminate_msg)
        self._ended = True
        super().terminate()

    async def _on_transcribe(self, parent, result: LiveResultResponse, **kwargs):
        self._track_transcription_latency(
            start=result.start,
            duration=result.duration,
        )
        self._buffer = ""
        self._buffer_avg_confidence = 0.0
        self._num_buffer_utterances = 1
        self._time_silent = 0.0
        self._words_buffer = []
        self._is_final_ts = None

        if (
            result.channel.alternatives[0].transcript
            and result.channel.alternatives[0].confidence > 0.0
            and result.is_final
        ):
            words = result.channel.alternatives[0].words
            if words:
                self._words_buffer.extend(words)
            self._buffer = f"{self._buffer} {result.channel.alternatives[0].transcript}"
            if self._buffer_avg_confidence == 0:
                self._buffer_avg_confidence = result.channel.alternatives[0].confidence
            else:
                self._buffer_avg_confidence = (
                    self._buffer_avg_confidence
                    + result.channel.alternatives[0].confidence / (self._num_buffer_utterances)
                ) * (self._num_buffer_utterances / (self._num_buffer_utterances + 1))
            self._num_buffer_utterances += 1

            self._is_final_ts = now()

        if self._buffer and self._is_endpoint(self._buffer, result, self._time_silent):
            output_ts = now()
            self._track_latency_of_conversation(
                is_final_ts=self._is_final_ts,
                output_ts=output_ts,
            )
            self.output_queue.put_nowait(
                Transcription(
                    message=self._buffer,
                    confidence=self._buffer_avg_confidence,
                    is_final=True,
                    duration_seconds=self.calculate_duration(self._words_buffer),
                )
            )
            self._buffer = ""
            self._buffer_avg_confidence = 0.0
            self._num_buffer_utterances = 1
            self._time_silent = 0.0
            self._words_buffer = []
            self._is_final_ts = None

        if (
            result.channel.alternatives[0].transcript
            and result.channel.alternatives[0].confidence > 0.0
        ):
            if not result.is_final:
                interim_message = f"{self._buffer} {result.channel.alternatives[0].transcript}"
            else:
                interim_message = self._buffer

            self.output_queue.put_nowait(
                Transcription(
                    message=interim_message,
                    confidence=result.channel.alternatives[0].confidence,
                    is_final=False,
                )
            )
            self._time_silent = self.calculate_time_silent(result)
        else:
            self._time_silent += result.duration

        logger.debug("Terminating Deepgram transcriber receiver")

    async def _on_utterance_end(self, parent, utterance_end: UtteranceEndResponse, **kwargs):
        if self._buffer and self._is_endpoint(self._buffer, utterance_end, self._time_silent):
            output_ts = now()
            self._track_latency_of_conversation(
                is_final_ts=self._is_final_ts,
                output_ts=output_ts,
            )
            self.output_queue.put_nowait(
                Transcription(
                    message=self._buffer,
                    confidence=self._buffer_avg_confidence,
                    is_final=True,
                    duration_seconds=self.calculate_duration(self._words_buffer),
                )
            )
            self._buffer = ""
            self._buffer_avg_confidence = 0.0
            self._num_buffer_utterances = 1
            self._time_silent = 0.0
            self._words_buffer = []
            self._is_final_ts = None

        logger.debug("Terminating Deepgram utterance_end receiver")

    def get_input_sample_width(self):
        encoding = self.transcriber_config.audio_encoding
        if encoding == AudioEncoding.LINEAR16:
            return 2
        elif encoding == AudioEncoding.MULAW:
            return 1
        else:
            raise Exception(f"Audio encoding not supported {encoding}")

    def _get_byte_rate(self):
        sample_width = self.get_input_sample_width()
        sample_rate = self.transcriber_config.sampling_rate
        return sample_width * sample_rate * NUM_AUDIO_CHANNELS

    def _is_endpoint(
        self,
        current_buffer: str,
        deepgram_response: Union[UtteranceEndResponse, LiveResultResponse],
        time_silent: float,
    ):
        is_endpoint, log_params = self._compute_is_endpoint_and_log_params(
            current_buffer, deepgram_response, time_silent
        )
        if is_endpoint:
            logger.info("Endpoint detected", extra=log_params)
        return is_endpoint

    @staticmethod
    def _satisfies_time_cutoff(
        *,
        seconds: float,
        deepgram_response: Union[UtteranceEndResponse, LiveResultResponse],
        current_buffer: str,
        time_silent: float,
    ):
        return (
            isinstance(deepgram_response, LiveResultResponse)
            and not deepgram_response.channel.alternatives[0].transcript
            and len(current_buffer) > 0
            and (time_silent + deepgram_response.duration) > seconds
        )

    def _compute_is_endpoint_and_update_log_params_based_on_time_silent(
        self,
        current_buffer: str,
        deepgram_response: LiveResultResponse,
        time_silent: float,
        time_silent_config: TimeSilentConfig,
        existing_log_params: dict,
    ) -> bool:
        if current_buffer.strip():
            if current_buffer.strip()[
                -1
            ] in PUNCTUATION_TERMINATORS and self._satisfies_time_cutoff(
                seconds=time_silent_config.post_punctuation_time_seconds
                * (1.0 / self._get_speed_coefficient()),
                deepgram_response=deepgram_response,
                current_buffer=current_buffer,
                time_silent=time_silent,
            ):
                existing_log_params["source"] = "punctuation"
                return True
            elif self._satisfies_time_cutoff(
                seconds=time_silent_config.time_cutoff_seconds
                * (1.0 / self._get_speed_coefficient()),
                deepgram_response=deepgram_response,
                current_buffer=current_buffer,
                time_silent=time_silent,
            ):
                existing_log_params["source"] = "time_cutoff"
                return True
        return False

    def _compute_is_endpoint_and_log_params(
        self,
        current_buffer: str,
        deepgram_response: Union[UtteranceEndResponse, LiveResultResponse],
        time_silent: float,
    ) -> Tuple[bool, dict]:
        endpointing_config = self.transcriber_config.endpointing_config
        assert endpointing_config is not None

        log_params = {"endpointing_type": endpointing_config.type}

        if isinstance(
            endpointing_config,
            TimeEndpointingConfig,
        ):
            # if it is time based, then return true if there is no transcript
            # and there is some speech to send
            # and the time_silent is greater than the cutoff
            return (
                self._satisfies_time_cutoff(
                    seconds=endpointing_config.time_cutoff_seconds,
                    deepgram_response=deepgram_response,
                    current_buffer=current_buffer,
                    time_silent=time_silent,
                ),
                log_params,
            )
        elif isinstance(
            endpointing_config,
            DeepgramEndpointingConfig,
        ):
            if (
                isinstance(deepgram_response, LiveResultResponse)
                and self.is_first_transcription
                and endpointing_config.use_single_utterance_endpointing_for_first_utterance
            ):
                if (
                    deepgram_response.channel.alternatives[0].transcript
                    and deepgram_response.channel.alternatives[0].transcript.strip()[-1]
                    in PUNCTUATION_TERMINATORS
                ):
                    log_params["source"] = "is_final"
                    return True, log_params
            if isinstance(deepgram_response, UtteranceEndResponse):
                log_params["source"] = "utterance_end"
                return True, log_params
            elif isinstance(deepgram_response, LiveResultResponse):
                if (
                    deepgram_response.channel.alternatives[0].transcript
                    and deepgram_response.speech_final
                ):
                    log_params["source"] = "speech_final"
                    return True, log_params
                elif (
                    endpointing_config.time_silent_config is not None
                    and self._compute_is_endpoint_and_update_log_params_based_on_time_silent(
                        current_buffer,
                        deepgram_response,
                        time_silent,
                        endpointing_config.time_silent_config,
                        log_params,
                    )
                ):
                    return True, log_params
            return False, log_params

        if isinstance(deepgram_response, LiveResultResponse):
            if isinstance(
                endpointing_config,
                PunctuationEndpointingConfig,
            ):
                if (
                    deepgram_response.channel.alternatives[0].transcript
                    and deepgram_response.speech_final
                    and deepgram_response.channel.alternatives[0].transcript.strip()[-1]
                    in PUNCTUATION_TERMINATORS
                ):
                    log_params["source"] = "punctuation"
                    return True, log_params
                elif self._satisfies_time_cutoff(
                    seconds=endpointing_config.time_cutoff_seconds
                    * (1.0 / self._get_speed_coefficient()),
                    deepgram_response=deepgram_response,
                    current_buffer=current_buffer,
                    time_silent=time_silent,
                ):
                    log_params["source"] = "time_cutoff"
                    return True, log_params
            elif isinstance(
                endpointing_config,
                InternalPunctuationEndpointingConfig,
            ):
                if (
                    self.is_first_transcription
                    and endpointing_config.use_single_utterance_endpointing_for_first_utterance
                ):
                    if (
                        deepgram_response.channel.alternatives[0].transcript
                        and deepgram_response.speech_final
                    ):
                        log_params["source"] = "speech_final"
                        return True, log_params
                if self._compute_is_endpoint_and_update_log_params_based_on_time_silent(
                    current_buffer,
                    deepgram_response,
                    time_silent,
                    endpointing_config.time_silent_config,
                    log_params,
                ):
                    return True, log_params
        return False, log_params

    def calculate_time_silent(self, deepgram_transcription_result: LiveResultResponse):
        end = deepgram_transcription_result.start + deepgram_transcription_result.duration
        words = deepgram_transcription_result.channel.alternatives[0].words
        if words:
            return end - words[-1]["end"]
        return deepgram_transcription_result.duration

    def calculate_duration(self, words: List[dict]) -> float:
        if len(words) == 0:
            return 0.0
        return words[-1]["end"] - words[0]["start"]

    async def process(self):
        self._audio_cursor = 0.0
        self._start_ts = now()
        self._connected_ts = now()
        byte_rate = self._get_byte_rate()

        # start client
        await self._dgClient.start(self._deepgram_streaming_config)
        logger.info(f"Connecting to Deepgram")

        try:
            while not self._ended:
                try:
                    data = await asyncio.wait_for(self.input_queue.get(), 5)
                except asyncio.exceptions.TimeoutError:
                    break

                self._audio_cursor += len(data) / byte_rate

                if not self._start_sending_ts:
                    self._start_sending_ts = now()

                await self._dgClient.send(data)

            logger.debug("Terminating Deepgram transcriber sender")
            await self._dgClient.finish()

        except asyncio.exceptions.TimeoutError:
            raise

    @sentry_configured
    def _track_latency_of_transcription_start(
        self,
    ):
        with sentry_create_span(
            sentry_callable=sentry_sdk.start_span,
            op=CustomSentrySpans.LATENCY_OF_TRANSCRIPTION_START,
            start_timestamp=self._start_ts,
        ) as transcription_span:

            with sentry_create_span(
                sentry_callable=transcription_span.start_child,
                op=CustomSentrySpans.START_TO_CONNECTION,
                start_timestamp=self._start_ts,
            ) as span:
                span.finish(end_timestamp=self._connected_ts)

            with sentry_create_span(
                sentry_callable=transcription_span.start_child,
                op=CustomSentrySpans.CONNECTED_TO_FIRST_SEND,
                start_timestamp=self._connected_ts,
            ) as span:
                span.finish(end_timestamp=self._start_sending_ts)

            with sentry_create_span(
                sentry_callable=transcription_span.start_child,
                op=CustomSentrySpans.FIRST_SEND_TO_FIRST_RECEIVE,
                start_timestamp=self._start_sending_ts,
            ) as span:
                span.finish(end_timestamp=self._start_receiving_ts)
            transcription_span.finish(end_timestamp=self._start_receiving_ts)

    def _track_transcription_latency(self, start: float, duration: float):
        cur_max_latency = self._audio_cursor - self._transcript_cursor
        transcript_cursor = start + duration
        cur_min_latency = self._audio_cursor - transcript_cursor

        self._max_latency = max(self._max_latency or 0, cur_max_latency)
        self._min_latency = min(self._min_latency or 10**6, cur_min_latency)

        self._avg_latency_numer += (cur_min_latency + cur_max_latency) / 2 * duration
        self._avg_latency_denom += duration
        self._avg_latency = self._avg_latency_numer / (self._avg_latency_denom or 1)

    def _track_latency_of_conversation(
        self,
        *,
        # is_final_spoken_ts: Optional[datetime],
        is_final_ts: Optional[datetime],
        output_ts: datetime,
    ):
        if not (
            output_ts
            and is_final_ts
            # and is_final_spoken_ts
        ):
            logger.info("Skipping latency measurement when not all timestamps are set")
            return

        if not (output_ts > is_final_ts):  # > is_final_spoken_ts
            logger.error("Skipping latency measurement when timestamps are not in order")
            return
        logger.debug(
            f"Endpoint detected, tracking endpointing_latency={output_ts - is_final_ts}",
        )

        # Ideally, we want to start a latency measurement for the entire conversation delay:
        # from when the user stopped speaking, to the time we start sending audio back.
        # Currently we don't have a way to find from Deepgram what the timetamp is
        # corresponding to their 0.0 in the transcript_cursor and we have observed that
        # the audio_cursor is always less than the transcript cursor, which seems to
        # indicate there is something off in our math for the audio cursor. The audio cursor
        # should represent the total number of seconds of audio that we have submitted to
        # Deepgram and should roughly line up with the transcript cursor.
        #
        # Until we can find a reliable way to find is_final_spoken_ts, we are skipping on
        # sending the transcription_latency span, instead focus on the endpointing_latency
        #
        # There was previous work done using the last word "end" cursor offset from Deepgram
        # in [EPD-926]. However, for similar reasons the logic for adding that to the
        # start_receiving_ts created on our end resulted in times for that audio that were
        # later than "is_final" which does not make sense temporally. We believe that is due
        # to not having an accurate timestamp corresponding to deepgram's 0.0 audio_cursor offset.

        # Span: latency_of_conversation

        # Once we have reliable way to find is_final_spoken_ts we would use that
        # here instead, and uncomment the transcription_latency span
        # start_timestamp=is_final_spoken_ts,
        latency_of_conversation_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span,
            op=CustomSentrySpans.LATENCY_OF_CONVERSATION,
            start_timestamp=is_final_ts,
        )

        # Span: latency_of_conversation.transcription_latency
        # Transcription latency is the time between audio was sent to deepgram for the
        # last word in the buffer when we set `is_final_ts` and then start
        # running
        # Time from `is_final_spoken_ts` to `is_final_ts`
        # with sentry_create_span(
        #     sentry_callable=latency_of_conversation_span.start_child,
        #     op="transcription_latency",
        #     start_timestamp=is_final_spoken_ts,
        # ) as span:
        #     span.finish(end_timestamp=is_final_ts)

        # Span: latency_of_conversation.endpointing_latency
        # Endpointing latency is the time from when we receive the last is_final
        # message from Deepgram and when we send output to the next stage.
        # Time from `is_final_ts` to `output_ts`
        if latency_of_conversation_span:
            with sentry_create_span(
                sentry_callable=latency_of_conversation_span.start_child,
                op=CustomSentrySpans.ENDPOINTING_LATENCY,
                start_timestamp=is_final_ts,
            ) as span:
                span.finish(end_timestamp=output_ts)
