import asyncio
import io
import wave
import audioop
import logging
from typing import Optional, AsyncGenerator, Union, Tuple
from urllib.parse import urljoin, urlencode
import aiohttp
from opentelemetry.trace import Span

from vocode import getenv
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer,
)
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3


ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
MULAW_OUTPUT_FORMAT = "ulaw_8000"


def convert_to_wav(pcm_data: bytes) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(pcm_data)
        return wav_io.getvalue()


async def experimental_streaming_output_generator(
    response: aiohttp.ClientResponse,
    create_speech_span: Optional[Span],
) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
    stream_reader = response.content
    try:
        # Get the wav chunk and the flag from the output queue of the MiniaudioWorker
        buffer = bytearray()
        async for chunk, done in stream_reader.iter_chunks():
            buffer += chunk

            if stream_reader.is_eof():
                yield SynthesisResult.ChunkResult(buffer, stream_reader.is_eof())
                buffer.clear()
            # If this is the last chunk, break the loop
            # if stream_reader.is_eof() and create_speech_span is not None:
            #     create_speech_span.end()
            #     break
    except asyncio.CancelledError:
        pass


class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        import elevenlabs

        self.elevenlabs = elevenlabs

        self.api_key = synthesizer_config.api_key or getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = synthesizer_config.voice_id or ADAM_VOICE_ID
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.output_format = self._get_eleven_labs_format()

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        voice = self.elevenlabs.Voice(voice_id=self.voice_id)
        if self.stability is not None and self.similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )
        # Add additional path if experimental streaming is enabled

        base_url = urljoin(ELEVEN_LABS_BASE_URL, f"text-to-speech/{self.voice_id}")

        if self.experimental_streaming:
            base_url = urljoin(base_url + "/", "stream")

        # Construct query parameters
        query_params = {"output_format": self.output_format}

        if self.optimize_streaming_latency:
            query_params["optimize_streaming_latency"] = self.optimize_streaming_latency

        url = f"{base_url}?{urlencode(query_params)}"

        headers = {"xi-api-key": self.api_key}
        body = {
            "text": message.text,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.create_total",
        )

        session = self.aiohttp_session

        response = await session.request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        if not response.ok:
            response_text = await response.text(encoding="utf-8", errors="ignore")
            raise Exception(
                f"ElevenLabs API returned {response.status} status code with error : {response_text}"
            )

        if self.output_format == DEFAULT_OUTPUT_FORMAT and self.experimental_streaming:
            return SynthesisResult(
                self.experimental_mp3_streaming_output_generator(
                    response, chunk_size, create_speech_span
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
        elif (
            self.output_format == DEFAULT_OUTPUT_FORMAT
            and not self.experimental_streaming
        ):
            audio_data = await response.read()
            create_speech_span.end()
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
            )
            output_bytes_io = decode_mp3(audio_data)

            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )
            convert_span.end()

            return result
        elif (
            self.output_format == MULAW_OUTPUT_FORMAT
            and not self.experimental_streaming
        ):
            audio_data = await response.read()
            # Convert μ-law to linear PCM
            pcm_data = audioop.ulaw2lin(audio_data, 2)
            # Create a WAV file in memory
            wav_data = convert_to_wav(pcm_data)

            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=io.BytesIO(wav_data),
                message=message,
                chunk_size=chunk_size,
            )
            return result
        elif self.output_format == MULAW_OUTPUT_FORMAT and self.experimental_streaming:
            return SynthesisResult(
                experimental_streaming_output_generator(
                    response, create_speech_span
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
        else:
            raise RuntimeError(
                f"Unsupported ElevenLabs configuration: {self.synthesizer_config.sampling_rate}, {self.synthesizer_config.audio_encoding}, {self.output_format}"
            )

    def _get_eleven_labs_format(self):
        sampling_rate = self.synthesizer_config.sampling_rate
        codec = self.synthesizer_config.audio_encoding

        if sampling_rate != 8000 and codec != AudioEncoding.MULAW:
            return DEFAULT_OUTPUT_FORMAT  # default
        else:
            return MULAW_OUTPUT_FORMAT
