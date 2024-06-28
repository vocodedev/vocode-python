import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import google.auth
from google.cloud import texttospeech as tts  # type: ignore

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import GoogleSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import create_synthesis_result_from_wav


class GoogleSynthesizer(AbstractSynthesizer[GoogleSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: GoogleSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        google.auth.default()

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
        synthesis_input = tts.SynthesisInput(text=message)

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        return self.client.synthesize_speech(
            request=tts.SynthesizeSpeechRequest(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config,
                enable_time_pointing=[tts.SynthesizeSpeechRequest.TimepointType.SSML_MARK],
            )
        )

    # TODO: make this nonblocking, see speech.TextToSpeechAsyncClient
    async def create_speech_with_cache(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        response: tts.SynthesizeSpeechResponse = (  # type: ignore
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor, self.synthesize, message.text
            )
        )
        output_sample_rate = response.audio_config.sample_rate_hertz

        output_bytes_io = io.BytesIO()
        in_memory_wav = wave.open(output_bytes_io, "wb")
        in_memory_wav.setnchannels(1)
        in_memory_wav.setsampwidth(2)
        in_memory_wav.setframerate(output_sample_rate)
        in_memory_wav.writeframes(response.audio_content)
        output_bytes_io.seek(0)

        result = create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )
        return result
