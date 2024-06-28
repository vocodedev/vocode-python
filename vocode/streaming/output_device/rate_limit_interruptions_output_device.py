import asyncio
import time
from abc import abstractmethod

from vocode.streaming.constants import PER_CHUNK_ALLOWANCE_SECONDS
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState
from vocode.streaming.utils import get_chunk_size_per_second


class RateLimitInterruptionsOutputDevice(AbstractOutputDevice):
    """Output device that works by rate limiting the chunks sent to the output. For interrupts to work properly,
    the next chunk of audio can only be sent after the last chunk is played, so we send
    a chunk of x seconds only after x seconds have passed since the last chunk was sent."""

    def __init__(
        self,
        sampling_rate: int,
        audio_encoding: AudioEncoding,
        per_chunk_allowance_seconds: float = PER_CHUNK_ALLOWANCE_SECONDS,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.per_chunk_allowance_seconds = per_chunk_allowance_seconds

    async def _run_loop(self):
        while True:
            start_time = time.time()
            try:
                item = await self.input_queue.get()
            except asyncio.CancelledError:
                return

            self.interruptible_event = item
            audio_chunk = item.payload

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                continue

            speech_length_seconds = (len(audio_chunk.data)) / get_chunk_size_per_second(
                self.audio_encoding,
                self.sampling_rate,
            )
            await self.play(audio_chunk.data)
            audio_chunk.on_play()
            audio_chunk.state = ChunkState.PLAYED
            end_time = time.time()
            await asyncio.sleep(
                max(
                    speech_length_seconds
                    - (end_time - start_time)
                    - self.per_chunk_allowance_seconds,
                    0,
                ),
            )
            self.interruptible_event.is_interruptible = False

    @abstractmethod
    async def play(self, chunk: bytes):
        """Sends an audio chunk to immediate playback"""
        pass

    def interrupt(self):
        """
        For conversations that use rate-limiting playback as above,
        no custom logic is needed on interrupt, because to end synthesis, all we need to do is stop sending chunks.
        """
        pass
