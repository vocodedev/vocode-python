import asyncio
from abc import abstractmethod

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.audio_chunk import AudioChunk
from vocode.streaming.utils.worker import AsyncWorker, InterruptibleEvent


class AbstractOutputDevice(AsyncWorker[InterruptibleEvent[AudioChunk]]):
    """Output devices are workers that are responsible for playing back audio.

    As part of processing:
    - it must call AudioChunk.on_play() when the chunk is played back and set AudioChunk.state = ChunkState.PLAYED
    - it must call AudioChunk.on_interrupt() when the chunk is interrupted and set AudioChunk.state = ChunkState.INTERRUPTED
    - if the interruptible event marker is set, then it must also mark the chunk as interrupted
    """

    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding):
        super().__init__(input_queue=asyncio.Queue())
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding

    @abstractmethod
    def interrupt(self):
        """Must interrupt the currently playing audio"""
        pass
