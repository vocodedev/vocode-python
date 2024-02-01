import argparse
import sys

from vocode.streaming.input_device.microphone_input import MicrophoneInput
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber, Transcription
from vocode.streaming.transcriber import *

from vocode.streaming.pubsub.base_pubsub import AudioFileWriterSubscriber, PubSubTopics
from vocode import pubsub


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", action="store_true", help="Log latencies and other statistics"
    )
    args = parser.parse_args()

    if args.trace:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from playground.streaming.tracing_utils import SpecificStatisticsReader

        reader = SpecificStatisticsReader()
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)

    async def print_output(transcriber: BaseTranscriber):
        while True:
            transcription: Transcription = await transcriber.output_queue.get()
            print(transcription)

    async def listen():
        microphone_input = MicrophoneInput.from_default_device()

        transcriber_config = DeepgramTranscriberConfig.from_input_device(
            microphone_input,
            endpointing_config=PunctuationEndpointingConfig(),
            publish_audio=True,
        )

        # replace with the transcriber you want to test
        transcriber = DeepgramTranscriber(transcriber_config)

        transcriber.start()
        asyncio.create_task(print_output(transcriber))

        subscriber = AudioFileWriterSubscriber(
            "AudioFileWriterSubscriber", sampling_rate=transcriber_config.sampling_rate
        )

        pubsub.subscribe(subscriber=subscriber, topic=PubSubTopics.INPUT_AUDIO_STREAMS)
        subscriber.start()

        print("Start speaking...press Ctrl+C to end. ")

        while True:
            chunk = await microphone_input.get_audio()
            transcriber.send_audio(chunk)

    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the listen coroutine in the event loop
        loop.run_until_complete(listen())
    except Exception as exc:
        print("Terminating...")

        # Cancel all running tasks
        for task in asyncio.all_tasks(loop):
            print(f"Cancelling task: {task}")
            task.cancel()

        # Wait for all tasks to finish
        loop.run_until_complete(
            asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True)
        )

        print("Terminated.")
        sys.exit(0)

    finally:
        # Close the event loop
        loop.close()
        import threading

        pubsub.stop()

        for thread in threading.enumerate():
            print(thread)
