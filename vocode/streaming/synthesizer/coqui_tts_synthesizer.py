import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydub import AudioSegment
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import create_synthesis_result_from_wav
from TTS.api import TTS

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CoquiTTSSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer


class CoquiTTSSynthesizer(AbstractSynthesizer[CoquiTTSSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: CoquiTTSSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.tts = TTS(**synthesizer_config.tts_kwargs)
        self.speaker = synthesizer_config.speaker
        self.language = synthesizer_config.language
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    async def create_speech_with_cache(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        tts = self.tts
        audio_data = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor,
            tts.tts,
            message.text,
            self.speaker,
            self.language,
        )
        audio_data = np.array(audio_data)

        # Convert the NumPy array to bytes
        audio_data_bytes = (audio_data * 32767).astype(np.int16).tobytes()

        # Create an in-memory file-like object (BytesIO) to store the audio data
        buffer = io.BytesIO(audio_data_bytes)

        audio_segment: AudioSegment = AudioSegment.from_raw(
            buffer, frame_rate=22050, channels=1, sample_width=2  # type: ignore
        )

        output_bytes_io = io.BytesIO()
        audio_segment.export(output_bytes_io, format="wav")  # type: ignore

        result = create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )

        return result
