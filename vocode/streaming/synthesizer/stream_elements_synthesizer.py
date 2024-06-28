import io

import aiohttp
from pydub import AudioSegment

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import StreamElementsSynthesizerConfig
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.synthesizer_utils import create_synthesis_result_from_wav


class StreamElementsSynthesizer(AbstractSynthesizer[StreamElementsSynthesizerConfig]):
    TTS_ENDPOINT = "https://api.streamelements.com/kappa/v2/speech"

    def __init__(
        self,
        synthesizer_config: StreamElementsSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)
        self.voice = synthesizer_config.voice

    async def create_speech_with_cache(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        url_params = {
            "voice": self.voice,
            "text": message.text,
        }
        async with self.async_requestor.get_session().get(
            self.TTS_ENDPOINT,
            params=url_params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            read_response = await response.read()

            # TODO: probably needs to be in a thread
            audio_segment: AudioSegment = AudioSegment.from_mp3(
                io.BytesIO(read_response)  # type: ignore
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
