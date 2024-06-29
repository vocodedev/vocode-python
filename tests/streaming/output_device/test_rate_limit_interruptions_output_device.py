import asyncio

import pytest

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)
from vocode.streaming.pipeline.worker import InterruptibleEvent


class DummyRateLimitInterruptionsOutputDevice(RateLimitInterruptionsOutputDevice):
    async def play(self, chunk: bytes):
        pass


@pytest.mark.asyncio
async def test_calls_callbacks():
    output_device = DummyRateLimitInterruptionsOutputDevice(
        sampling_rate=16000, audio_encoding=AudioEncoding.LINEAR16
    )

    played_event = asyncio.Event()
    interrupted_event = asyncio.Event()
    uninterruptible_played_event = asyncio.Event()

    def on_play():
        played_event.set()

    def on_interrupt():
        interrupted_event.set()

    def uninterruptible_on_play():
        uninterruptible_played_event.set()

    played_audio_chunk = AudioChunk(data=b"")
    played_audio_chunk.on_play = on_play

    interrupted_audio_chunk = AudioChunk(data=b"")
    interrupted_audio_chunk.on_interrupt = on_interrupt

    uninterruptible_audio_chunk = AudioChunk(data=b"")
    uninterruptible_audio_chunk.on_play = uninterruptible_on_play

    interruptible_event = InterruptibleEvent(payload=interrupted_audio_chunk, is_interruptible=True)
    interruptible_event.interruption_event.set()

    uninterruptible_event = InterruptibleEvent(
        payload=uninterruptible_audio_chunk, is_interruptible=False
    )
    uninterruptible_event.interruption_event.set()

    output_device.consume_nonblocking(InterruptibleEvent(payload=played_audio_chunk))
    output_device.consume_nonblocking(interruptible_event)
    output_device.consume_nonblocking(uninterruptible_event)
    output_device.start()

    await played_event.wait()
    assert played_audio_chunk.state == ChunkState.PLAYED

    await interrupted_event.wait()
    assert interrupted_audio_chunk.state == ChunkState.INTERRUPTED

    await uninterruptible_played_event.wait()
    assert uninterruptible_audio_chunk.state == ChunkState.PLAYED

    output_device.terminate()
