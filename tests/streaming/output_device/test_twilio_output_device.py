import asyncio
import base64
import json

import pytest
from pytest_mock import MockerFixture

from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.output_device.twilio_output_device import (
    ChunkFinishedMarkMessage,
    TwilioOutputDevice,
)
from vocode.streaming.pipeline.worker import InterruptibleEvent


@pytest.fixture
def mock_ws(mocker: MockerFixture):
    return mocker.AsyncMock()


@pytest.fixture
def mock_stream_sid():
    return "stream_sid"


@pytest.fixture
def twilio_output_device(mock_ws, mock_stream_sid):
    return TwilioOutputDevice(ws=mock_ws, stream_sid=mock_stream_sid)


@pytest.mark.asyncio
async def test_calls_play_callbacks(twilio_output_device: TwilioOutputDevice):
    played_event = asyncio.Event()

    def on_play():
        played_event.set()

    audio_chunk = AudioChunk(data=b"")
    audio_chunk.on_play = on_play

    twilio_output_device.consume_nonblocking(InterruptibleEvent(payload=audio_chunk))
    twilio_output_device.start()
    twilio_output_device.enqueue_mark_message(
        ChunkFinishedMarkMessage(chunk_id=str(audio_chunk.chunk_id))
    )

    await played_event.wait()
    assert audio_chunk.state == ChunkState.PLAYED

    media_message = json.loads(twilio_output_device.ws.send_text.call_args_list[0][0][0])
    assert media_message["streamSid"] == twilio_output_device.stream_sid
    assert media_message["media"] == {"payload": base64.b64encode(audio_chunk.data).decode("utf-8")}

    mark_message = json.loads(twilio_output_device.ws.send_text.call_args_list[1][0][0])
    assert mark_message["streamSid"] == twilio_output_device.stream_sid
    assert mark_message["mark"]["name"] == str(audio_chunk.chunk_id)

    twilio_output_device.terminate()


@pytest.mark.asyncio
async def test_calls_interrupt_callbacks(twilio_output_device: TwilioOutputDevice):
    interrupted_event = asyncio.Event()

    def on_interrupt():
        interrupted_event.set()

    audio_chunk = AudioChunk(data=b"")
    audio_chunk.on_interrupt = on_interrupt

    interruptible_event = InterruptibleEvent(payload=audio_chunk, is_interruptible=True)

    twilio_output_device.consume_nonblocking(interruptible_event)
    # we start the twilio events task and the mark messages task manually to test this particular case

    # step 1: media is sent into the websocket
    send_twilio_messages_task = asyncio.create_task(twilio_output_device._send_twilio_messages())

    while not twilio_output_device._twilio_events_queue.empty():
        await asyncio.sleep(0.1)

    # step 2: we get an interrupt
    interruptible_event.interrupt()
    twilio_output_device.interrupt()

    # note: this means that the time between the events being interrupted and the clear message being sent, chunks
    # will be marked interrupted - this is OK since the clear message is sent almost instantaneously once queued
    # this is required because it stops queueing new chunks to be sent to the WS immediately

    while not twilio_output_device._twilio_events_queue.empty():
        await asyncio.sleep(0.1)

    # step 3: we get a mark message for the interrupted audio chunk after the clear message
    twilio_output_device.enqueue_mark_message(
        ChunkFinishedMarkMessage(chunk_id=str(audio_chunk.chunk_id))
    )
    process_mark_messages_task = asyncio.create_task(twilio_output_device._process_mark_messages())

    await interrupted_event.wait()
    assert audio_chunk.state == ChunkState.INTERRUPTED

    media_message = json.loads(twilio_output_device.ws.send_text.call_args_list[0][0][0])
    assert media_message["streamSid"] == twilio_output_device.stream_sid
    assert media_message["media"] == {"payload": base64.b64encode(audio_chunk.data).decode("utf-8")}

    mark_message = json.loads(twilio_output_device.ws.send_text.call_args_list[1][0][0])
    assert mark_message["streamSid"] == twilio_output_device.stream_sid
    assert mark_message["mark"]["name"] == str(audio_chunk.chunk_id)

    clear_message = json.loads(twilio_output_device.ws.send_text.call_args_list[2][0][0])
    assert clear_message["streamSid"] == twilio_output_device.stream_sid
    assert clear_message["event"] == "clear"

    send_twilio_messages_task.cancel()
    process_mark_messages_task.cancel()


@pytest.mark.asyncio
async def test_interrupted_audio_chunks_are_not_sent_but_are_marked_interrupted(
    twilio_output_device: TwilioOutputDevice,
):
    interrupted_event = asyncio.Event()

    def on_interrupt():
        interrupted_event.set()

    audio_chunk = AudioChunk(data=b"")
    audio_chunk.on_interrupt = on_interrupt

    interruptible_event = InterruptibleEvent(payload=audio_chunk, is_interruptible=True)
    interruptible_event.interrupt()

    twilio_output_device.consume_nonblocking(interruptible_event)
    twilio_output_device.start()

    await interrupted_event.wait()
    assert audio_chunk.state == ChunkState.INTERRUPTED

    twilio_output_device.ws.send_text.assert_not_called()
