import ctypes
import io
import pathlib
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydub import AudioSegment

from vocode.streaming.constants import SENTENCE_ENDINGS
from vocode.streaming.models.transcriber import Transcription, WhisperCPPTranscriberConfig
from vocode.streaming.transcriber.base_transcriber import BaseThreadAsyncTranscriber
from vocode.utils.whisper_cpp.helpers import transcribe
from vocode.utils.whisper_cpp.whisper_params import WhisperFullParams

WHISPER_CPP_SAMPLING_RATE = 16000


class WhisperCPPTranscriber(BaseThreadAsyncTranscriber[WhisperCPPTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: WhisperCPPTranscriberConfig,
    ):
        super().__init__(transcriber_config)
        self._ended = False
        self.buffer_size = round(
            transcriber_config.sampling_rate * transcriber_config.buffer_size_seconds
        )
        self.buffer = np.empty(self.buffer_size, dtype=np.int16)
        self.buffer_index = 0

        # whisper cpp
        # load library and model
        libname = pathlib.Path().absolute() / self.transcriber_config.libname
        self.whisper = ctypes.CDLL(libname)  # type: ignore

        # tell Python what are the return types of the functions
        self.whisper.whisper_init_from_file.restype = ctypes.c_void_p
        self.whisper.whisper_full_default_params.restype = WhisperFullParams
        self.whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p

        # initialize whisper.cpp context
        self.ctx = self.whisper.whisper_init_from_file(
            self.transcriber_config.fname_model.encode("utf-8")
        )

        # get default whisper parameters and adjust as needed
        self.params = self.whisper.whisper_full_default_params()
        self.params.print_realtime = False
        self.params.print_progress = False
        self.params.single_segment = True
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    def create_new_buffer(self):
        buffer = io.BytesIO()
        wav = wave.open(buffer, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(self.transcriber_config.sampling_rate)
        return wav, buffer

    def _run_loop(self):
        in_memory_wav, audio_buffer = self.create_new_buffer()
        message_buffer = ""
        while not self._ended:
            chunk = self.input_janus_queue.sync_q.get()
            in_memory_wav.writeframes(chunk)
            if audio_buffer.tell() >= self.buffer_size * 2:
                audio_buffer.seek(0)
                audio_segment = AudioSegment.from_wav(audio_buffer)
                message, confidence = transcribe(self.whisper, self.params, self.ctx, audio_segment)
                message_buffer += message
                is_final = any(message_buffer.endswith(ending) for ending in SENTENCE_ENDINGS)
                in_memory_wav, audio_buffer = self.create_new_buffer()
                self.produce_nonblocking(
                    Transcription(message=message_buffer, confidence=confidence, is_final=is_final)
                )
                if is_final:
                    message_buffer = ""

    async def terminate(self):
        pass
