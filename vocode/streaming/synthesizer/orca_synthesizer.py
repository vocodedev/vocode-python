import asyncio
import io
import os
import wave
import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

from vocode import getenv
from vocode.streaming.models.audio import SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import OrcaSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult


class OrcaSynthesizer(BaseSynthesizer[OrcaSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: OrcaSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        # Lazy import the Orca module
        try:
            import pvorca
        except ImportError as e:
            raise ImportError(
                f"Missing required dependancies for Orca"
            ) from e
        
        self.orca_lib = pvorca
        
        self.api_key = synthesizer_config.api_key or getenv("ORCA_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Orca access key")
        
        if synthesizer_config.speech_rate and not (0.7 <= synthesizer_config.speech_rate <= 1.3):
            raise ValueError("Speech rate must be between 0.7 and 1.3 inclusive")

        self.model_path: Optional[str]
        if synthesizer_config.model_file:
            # By default, model files are stored in the same directory as the Orca library
            self.model_path = os.path.join(
                os.path.dirname(self.orca_lib.default_model_path()),
                synthesizer_config.model_file
            )
        else:
            # Will use the default model file
            self.model_path = None

        self.orca = self.orca_lib.create(access_key=self.api_key, model_path=self.model_path)
        self.speech_rate = synthesizer_config.speech_rate
        self.sample_rate = SamplingRate.RATE_22050.value # The only rate supported
        self.output_format = "pcm" # The only format supported
        self.model_file = synthesizer_config.model_file
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)
        
    def synthesize(self, message: str) -> Any:
        pcm, alignments = self.orca.synthesize(
            text=message,
            speech_rate=self.speech_rate
        )
        return pcm
    
    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        raw_bytes: bytes = (
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor, self.synthesize, message.text
            )
        )

        # Convert PCM list of integers to bytes
        pcm_bytes = struct.pack('<' + 'h' * len(raw_bytes), *raw_bytes)

        audio_file = io.BytesIO()
        with wave.open(audio_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)  # 16kHz sample rate
            wav_file.writeframes(pcm_bytes)
        audio_file.seek(0)

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=audio_file,
            message=message,
            chunk_size=chunk_size,
        )
        return result
    
    @classmethod
    def get_voice_identifier(cls, synthesizer_config: OrcaSynthesizerConfig):
        hashed_api_key = hashlib.sha256(f"{synthesizer_config.api_key}".encode("utf-8")).hexdigest()
        return ":".join(
            (
                "orca",
                hashed_api_key,
                str(synthesizer_config.model_file),
                "pcm"
            )
        )
