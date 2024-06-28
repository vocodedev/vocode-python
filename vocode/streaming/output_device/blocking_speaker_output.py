import asyncio
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)
from vocode.streaming.utils.worker import ThreadAsyncWorker

DEFAULT_SAMPLING_RATE = 44100


class _PlaybackWorker(ThreadAsyncWorker[bytes]):

    def __init__(self, *, device_info: dict, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.device_info = device_info
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        super().__init__(self.input_queue)
        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sampling_rate,
            dtype=np.int16,
            device=int(self.device_info["index"]),
        )
        self._ended = False
        self.input_queue.put_nowait(self.sampling_rate * b"\x00")
        self.stream.start()

    def _run_loop(self):
        while not self._ended:
            try:
                chunk = self.input_janus_queue.sync_q.get(timeout=1)
                self.stream.write(np.frombuffer(chunk, dtype=np.int16))
            except queue.Empty:
                continue

    def terminate(self):
        self._ended = True
        super().terminate()
        self.stream.close()


class BlockingSpeakerOutput(RateLimitInterruptionsOutputDevice):

    def __init__(
        self,
        device_info: dict,
        sampling_rate: Optional[int] = None,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
    ):
        sampling_rate = sampling_rate or int(
            device_info.get("default_samplerate", DEFAULT_SAMPLING_RATE)
        )
        super().__init__(sampling_rate=sampling_rate, audio_encoding=audio_encoding)
        self.playback_worker = _PlaybackWorker(device_info=device_info, sampling_rate=sampling_rate)

    async def play(self, chunk):
        self.playback_worker.consume_nonblocking(chunk)

    def start(self) -> asyncio.Task:
        self.playback_worker.start()
        return super().start()

    def terminate(self):
        self.playback_worker.terminate()
        super().terminate()

    @classmethod
    def from_default_device(
        cls,
        **kwargs,
    ):
        return cls(sd.query_devices(kind="output"), **kwargs)
