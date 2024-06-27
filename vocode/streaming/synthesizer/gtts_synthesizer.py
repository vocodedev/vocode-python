import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from gtts import gTTS
from pydub import AudioSegment

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import GTTSSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult


class GTTSSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        synthesizer_config: GTTSSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    async def create_speech_with_cache(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        audio_file = BytesIO()

        def thread():
            tts = gTTS(message.text)
            tts.write_to_fp(audio_file)

        await asyncio.get_event_loop().run_in_executor(self.thread_pool_executor, thread)
        audio_file.seek(0)
        # TODO: probably needs to be in a thread
        audio_segment: AudioSegment = AudioSegment.from_mp3(audio_file)  # type: ignore
        output_bytes_io = BytesIO()
        audio_segment.export(output_bytes_io, format="wav")  # type: ignore

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )
        return result
