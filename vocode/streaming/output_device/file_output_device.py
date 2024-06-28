import asyncio
import wave

import numpy as np

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)


class FileOutputDevice(RateLimitInterruptionsOutputDevice):
    DEFAULT_SAMPLING_RATE = 44100

    def __init__(
        self,
        file_path: str,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.blocksize = self.sampling_rate

        wav = wave.open(file_path, "wb")
        wav.setnchannels(1)  # Mono channel
        wav.setsampwidth(2)  # 16-bit samples
        wav.setframerate(self.sampling_rate)
        self.wav = wav

    async def play(self, chunk: bytes):
        await asyncio.to_thread(lambda: self.wav.writeframes(chunk))

    def terminate(self):
        self.wav.close()
        super().terminate()
