import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import logging
import os
import wave
from typing import Any, Optional
import aiohttp

from vocode import getenv

from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
    tracer,
)
from vocode.streaming.models.synthesizer import GoogleSynthesizerConfig, SynthesizerType
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.utils import convert_wav

from opentelemetry.context.context import Context


class GoogleSynthesizer(BaseSynthesizer[GoogleSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: GoogleSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        from google.cloud import texttospeech_v1beta1 as tts
        import google.auth

        google.auth.default()

        self.tts = tts

        # Instantiates a client
        self.client = tts.TextToSpeechClient()

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        self.voice = tts.VoiceSelectionParams(
            language_code=synthesizer_config.language_code,
            name=synthesizer_config.voice_name,
        )

        # Select the type of audio file you want returned
        self.audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
            speaking_rate=synthesizer_config.speaking_rate,
            pitch=synthesizer_config.pitch,
            effects_profile_id=["telephony-class-application"],
        )
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    def synthesize(self, message: str) -> Any:
        synthesis_input = self.tts.SynthesisInput(text=message)

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        return self.client.synthesize_speech(
            request=self.tts.SynthesizeSpeechRequest(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config,
                enable_time_pointing=[
                    self.tts.SynthesizeSpeechRequest.TimepointType.SSML_MARK
                ],
            )
        )

    # TODO: make this nonblocking, see speech.TextToSpeechAsyncClient
    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.GOOGLE.value.split('_', 1)[-1]}.create_total",
        )
        response: self.tts.SynthesizeSpeechResponse = (  # type: ignore
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor, self.synthesize, message.text
            )
        )
        create_speech_span.end()
        convert_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.GOOGLE.value.split('_', 1)[-1]}.convert",
        )
        output_sample_rate = response.audio_config.sample_rate_hertz

        output_bytes_io = io.BytesIO()
        in_memory_wav = wave.open(output_bytes_io, "wb")
        in_memory_wav.setnchannels(1)
        in_memory_wav.setsampwidth(2)
        in_memory_wav.setframerate(output_sample_rate)
        in_memory_wav.writeframes(response.audio_content)
        output_bytes_io.seek(0)

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )
        convert_span.end()
        return result
