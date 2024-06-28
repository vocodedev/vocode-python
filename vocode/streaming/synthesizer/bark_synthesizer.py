import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import create_synthesis_result_from_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
from loguru import logger
from scipy.io.wavfile import write as write_wav

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import BarkSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer


class BarkSynthesizer(AbstractSynthesizer[BarkSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: BarkSynthesizerConfig,
    ) -> None:
        super().__init__(synthesizer_config)

        self.SAMPLE_RATE = SAMPLE_RATE
        self.generate_audio = generate_audio
        logger.info("Loading Bark models")
        preload_models(**self.synthesizer_config.preload_kwargs)
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    async def create_speech_with_cache(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        logger.debug("Bark synthesizing audio")
        audio_array = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor,
            self.generate_audio,
            message.text,
            **self.synthesizer_config.generate_kwargs,
        )
        int_audio_arr = (audio_array * np.iinfo(np.int16).max).astype(np.int16)

        output_bytes_io = io.BytesIO()
        write_wav(output_bytes_io, self.SAMPLE_RATE, int_audio_arr)

        result = create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )

        return result
