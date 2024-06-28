import hashlib
import io
import wave

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CartesiaSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import create_synthesis_result_from_wav
from vocode.streaming.utils.create_task import asyncio_create_task_with_done_error_log


class CartesiaSynthesizer(AbstractSynthesizer[CartesiaSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: CartesiaSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        # Lazy import the cartesia module
        try:
            from cartesia.tts import AsyncCartesiaTTS
        except ImportError as e:
            raise ImportError(f"Missing required dependancies for CartesiaSynthesizer") from e

        self.cartesia_tts = AsyncCartesiaTTS

        self.api_key = synthesizer_config.api_key or getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Cartesia API key")

        if synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            self.channel_width = 2
            match synthesizer_config.sampling_rate:
                case SamplingRate.RATE_44100:
                    self.sampling_rate = 44100
                    self.output_format = "pcm_44100"
                case SamplingRate.RATE_22050:
                    self.sampling_rate = 22050
                    self.output_format = "pcm_22050"
                case SamplingRate.RATE_16000:
                    self.sampling_rate = 16000
                    self.output_format = "pcm_16000"
                case _:
                    raise ValueError(
                        f"Unsupported PCM sampling rate {synthesizer_config.sampling_rate}"
                    )
        elif synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            # Cartesia has issues with MuLaw/8000. Use pcm/16000 and
            # create_synthesis_result_from_wav will handle the conversion to mulaw/8000
            self.channel_width = 2
            self.output_format = "pcm_16000"
            self.sampling_rate = 16000
        else:
            raise ValueError(f"Unsupported audio encoding {synthesizer_config.audio_encoding}")

        self.num_channels = 1
        self.model_id = synthesizer_config.model_id
        self.voice_id = synthesizer_config.voice_id
        self.client = self.cartesia_tts(api_key=self.api_key)
        self.voice_embedding = self.client.get_voice_embedding(voice_id=self.voice_id)

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        generator = await self.client.generate(
            transcript=message.text,
            voice=self.voice_embedding,
            stream=True,
            model_id=self.model_id,
            data_rtype="bytes",
            output_format=self.output_format,
        )

        audio_file = io.BytesIO()
        with wave.open(audio_file, "wb") as wav_file:
            wav_file.setnchannels(self.num_channels)
            wav_file.setsampwidth(self.channel_width)
            wav_file.setframerate(self.sampling_rate)
            async for chunk in generator:
                wav_file.writeframes(chunk["audio"])
        audio_file.seek(0)

        result = create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=audio_file,
            message=message,
            chunk_size=chunk_size,
        )

        return result

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: CartesiaSynthesizerConfig):
        hashed_api_key = hashlib.sha256(f"{synthesizer_config.api_key}".encode("utf-8")).hexdigest()
        return ":".join(
            (
                "cartesia",
                hashed_api_key,
                str(synthesizer_config.voice_id),
                str(synthesizer_config.model_id),
                synthesizer_config.audio_encoding,
            )
        )
