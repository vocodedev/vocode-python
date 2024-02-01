import queue
import audioop
from typing import Optional
import sounddevice as sd
import numpy as np

from .base_output_device import BaseOutputDevice
from vocode.streaming.models.audio_encoding import AudioEncoding


class SpeakerOutput(BaseOutputDevice):
    DEFAULT_SAMPLING_RATE = 44100

    def __init__(
        self,
        device_info: dict,
        sampling_rate: Optional[int] = None,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
        blocksize: Optional[int] = None,
    ):
        self.device_info = device_info
        sampling_rate = sampling_rate or int(
            self.device_info.get("default_samplerate", self.DEFAULT_SAMPLING_RATE)
        )
        super().__init__(sampling_rate, audio_encoding)
        self.blocksize = blocksize or self.sampling_rate
        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sampling_rate,
            dtype=np.int16,
            blocksize=self.blocksize,
            device=int(self.device_info["index"]),
            callback=self.callback,
        )
        self.stream.start()
        self.queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(self, outdata: np.ndarray, frames, time, status):
        if self.queue.empty():
            outdata[:] = 0
            return
        data = self.queue.get()
        outdata[:, 0] = data

    def consume_nonblocking(self, chunk):
        if self.audio_encoding == AudioEncoding.MULAW:
            # Decode MULAW to linear PCM
            chunk = audioop.ulaw2lin(chunk, 2)

        chunk_arr = np.frombuffer(chunk, dtype=np.int16)
        for i in range(0, chunk_arr.shape[0], self.blocksize):
            block = np.zeros(self.blocksize, dtype=np.int16)
            size = min(self.blocksize, chunk_arr.shape[0] - i)
            block[:size] = chunk_arr[i : i + size]
            self.queue.put_nowait(block)

    def terminate(self):
        self.stream.close()

    @classmethod
    def from_default_device(
        cls,
        **kwargs,
    ):
        return cls(sd.query_devices(kind="output"), **kwargs)
