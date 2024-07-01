import queue
from datetime import datetime, timezone

import sentry_sdk
from azure.cognitiveservices.speech.audio import (
    AudioStreamFormat,
    AudioStreamWaveFormat,
    PushAudioInputStream,
)
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import AzureTranscriberConfig, Transcription
from vocode.streaming.transcriber.base_transcriber import BaseThreadAsyncTranscriber
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class AzureTranscriber(BaseThreadAsyncTranscriber[AzureTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: AzureTranscriberConfig,
    ):
        super().__init__(transcriber_config)

        format = None
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            format = AudioStreamFormat(
                samples_per_second=self.transcriber_config.sampling_rate,
                wave_stream_format=AudioStreamWaveFormat.PCM,
            )

        elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            format = AudioStreamFormat(
                samples_per_second=self.transcriber_config.sampling_rate,
                wave_stream_format=AudioStreamWaveFormat.MULAW,
            )

        import azure.cognitiveservices.speech as speechsdk

        self.push_stream = PushAudioInputStream(format)

        config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        speech_config = speechsdk.SpeechConfig(
            subscription=getenv("AZURE_SPEECH_KEY"),
            region=getenv("AZURE_SPEECH_REGION"),
        )

        speech_params = {
            "speech_config": speech_config,
            "audio_config": config,
        }

        if self.transcriber_config.candidate_languages:
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                value="Continuous",
            )
            auto_detect_source_language_config = (
                speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=self.transcriber_config.candidate_languages
                )
            )

            speech_params["auto_detect_source_language_config"] = auto_detect_source_language_config
        else:
            speech_params["language"] = self.transcriber_config.language

        self.speech = speechsdk.SpeechRecognizer(**speech_params)

        self._ended = False
        self.is_ready = False

    def recognized_sentence_final(self, evt):

        sentry_create_span(
            sentry_callable=sentry_sdk.start_span,
            op=CustomSentrySpans.LATENCY_OF_CONVERSATION,
            start_timestamp=datetime.now(tz=timezone.utc),
        )
        self.produce_nonblocking(
            Transcription(message=evt.result.text, confidence=1.0, is_final=True)
        )

    def recognized_sentence_stream(self, evt):
        self.produce_nonblocking(
            Transcription(message=evt.result.text, confidence=1.0, is_final=False)
        )

    def _run_loop(self):
        stream = self.generator()

        def stop_cb(evt):
            logger.debug("CLOSING on {}".format(evt))
            self.speech.stop_continuous_recognition()
            self._ended = True

        self.speech.recognizing.connect(lambda x: self.recognized_sentence_stream(x))
        self.speech.recognized.connect(lambda x: self.recognized_sentence_final(x))
        self.speech.session_started.connect(
            lambda evt: logger.debug("SESSION STARTED: {}".format(evt))
        )
        self.speech.session_stopped.connect(
            lambda evt: logger.debug("SESSION STOPPED {}".format(evt))
        )
        self.speech.canceled.connect(lambda evt: logger.debug("CANCELED {}".format(evt)))

        self.speech.session_stopped.connect(stop_cb)
        self.speech.canceled.connect(stop_cb)
        self.speech.start_continuous_recognition_async()

        for content in stream:
            self.push_stream.write(content)
            if self._ended:
                break

    def generator(self):
        while not self._ended:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            try:
                chunk = self.input_janus_queue.sync_q.get(timeout=5)
            except queue.Empty:
                return

            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.input_janus_queue.sync_q.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

    async def terminate(self):
        self._ended = True
        self.speech.stop_continuous_recognition_async()
        await super().terminate()
