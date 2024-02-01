import asyncio
import pytest
from unittest.mock import Mock, patch
from vocode.streaming.pubsub.base_pubsub import AudioFileWriterSubscriber
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.utils.worker import AsyncWorker
from vocode.streaming.pubsub.base_pubsub import (
    Publisher,
    PubSubManager,
)


@pytest.mark.asyncio
async def test_pubsub():
    pubsub = PubSubManager()

    sub1 = AsyncWorker(input_queue=asyncio.Queue())
    pubsub.subscribe(sub1, "topic1")
    pub = Publisher("pub")
    sub1.start()

    await pub.publish("1", "Hello, topic1!", "String", "topic1", pubsub)

    event = await sub1.input_queue.get()
    assert event.event_id == "1"
    assert event.payload == "Hello, topic1!"
    assert event.payload_type == "String"


@pytest.fixture
def event():
    return Mock(event_id=1, payload_type=AudioEncoding.MULAW, payload=b"\x00\x01")


@pytest.fixture
def audio_writer():
    with patch(
        "vocode.streaming.pubsub.base_pubsub.FileOutputDevice"
    ) as mock_file_output_device:
        yield AudioFileWriterSubscriber("test", save_chunk_in_sec=5, sampling_rate=8000)


@pytest.mark.asyncio
async def test_run_loop(event, audio_writer):
    """
    This test checks that the _run_loop method correctly processes an event from the input_queue,
    creates a FileOutputDevice if necessary, and calls the consume_nonblocking method of the FileOutputDevice
    when the time since the last flush is greater than or equal to save_chunk_in_sec.

    The _run_loop method is run in a separate task using asyncio.create_task, so it can be cancelled later.
    After putting the event in the input_queue, the test waits for a short time to let the loop run,
    then cancels the loop task and waits for it to finish.

    Finally, the test checks that the consume_nonblocking method of the FileOutputDevice was called with the correct argument.
    """

    with patch(
        "vocode.streaming.pubsub.base_pubsub.audioop.ulaw2lin", return_value=b"\x00\x01"
    ), patch("vocode.streaming.pubsub.base_pubsub.time", side_effect=[0, 5, 10]):
        # Put the event in the queue
        await audio_writer.input_queue.put(event)

        # Run the loop in a separate task, so we can stop it later
        loop_task = asyncio.create_task(audio_writer._run_loop())

        # Wait for a short time to let the loop run
        await asyncio.sleep(0.1)

        # Cancel the loop task
        loop_task.cancel()

        # Wait for the loop task to finish
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Check that the FileOutputDevice was created and used correctly
        audio_writer._file_writers[
            event.event_id
        ].consume_nonblocking.assert_called_once_with(b"\x00\x01")


@pytest.mark.asyncio
async def test_terminate(audio_writer):
    with patch(
        "vocode.streaming.pubsub.base_pubsub.FileOutputDevice"
    ) as mock_file_output_device:
        audio_writer._file_writers[1] = mock_file_output_device
        audio_writer.stop()
        mock_file_output_device.terminate.assert_called_once()
