import time
import argparse
from typing import Optional
import aiohttp
import sounddevice as sd
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    AzureSynthesizerConfig,
    GoogleSynthesizerConfig,
    PlayHtSynthesizerConfig,
    ElevenLabsSynthesizerConfig,
    RimeSynthesizerConfig,
    PollySynthesizerConfig,
)
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.output_device.speaker_output import SpeakerOutput
from vocode.streaming.synthesizer import *
from vocode.streaming.utils import get_chunk_size_per_second
from playground.streaming.tracing_utils import make_parser_and_maybe_trace


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    make_parser_and_maybe_trace()

    seconds_per_chunk = 1

    async def speak(
        synthesizer: BaseSynthesizer,
        output_device: BaseOutputDevice,
        message: BaseMessage,
        bot_sentiment: Optional[BotSentiment] = None,
    ):
        message_sent = message.text
        cut_off = False
        chunk_size = seconds_per_chunk * get_chunk_size_per_second(
            synthesizer.get_synthesizer_config().audio_encoding,
            synthesizer.get_synthesizer_config().sampling_rate,
        )
        # ClientSession needs to be created within the async task
        synthesizer.aiohttp_session = aiohttp.ClientSession()
        synthesis_result = await synthesizer.create_speech(
            message=message,
            chunk_size=chunk_size,
            bot_sentiment=bot_sentiment,
        )
        chunk_idx = 0
        async for chunk_result in synthesis_result.chunk_generator:
            try:
                start_time = time.time()
                speech_length_seconds = seconds_per_chunk * (
                    len(chunk_result.chunk) / chunk_size
                )
                output_device.consume_nonblocking(chunk_result.chunk)
                end_time = time.time()
                await asyncio.sleep(
                    max(
                        speech_length_seconds - (end_time - start_time),
                        0,
                    )
                )
                print(
                    "Sent chunk {} with size {}".format(
                        chunk_idx, len(chunk_result.chunk)
                    )
                )
                chunk_idx += 1
            except asyncio.CancelledError:
                seconds = chunk_idx * seconds_per_chunk
                print(
                    "Interrupted, stopping text to speech after {} chunks".format(
                        chunk_idx
                    )
                )
                message_sent = f"{synthesis_result.get_message_up_to(seconds)}-"
                cut_off = True
                break

        return message_sent, cut_off

    async def main():
        try:
            while True:
                message_sent, _ = await speak(
                    synthesizer=synthesizer,
                    output_device=speaker_output,
                    message=BaseMessage(text=input("Enter speech to synthesize: ")),
                )
                print("Message sent: ", message_sent)
        except KeyboardInterrupt:
            print("Interrupted, exiting")
        await synthesizer.tear_down()

    # replace with the synthesizer you want to test
    # Note: --trace will not work with AzureSynthesizer

    # speaker_output = SpeakerOutput.from_default_device()
    # synthesizer = AzureSynthesizer(
    #     AzureSynthesizerConfig.from_output_device(speaker_output)
    # )

    # synthesizer = ElevenLabsSynthesizer(
    #     synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device()
    # )
    #
    # speaker_output = SpeakerOutput(
    #     sampling_rate=8000,
    #     audio_encoding=AudioEncoding.MULAW,
    #     device_info=sd.query_devices(kind="output"),
    # )

    # synthesizer = ElevenLabsSynthesizer(
    #     synthesizer_config=ElevenLabsSynthesizerConfig(
    #         experimental_streaming=True,
    #         optimize_streaming_latency=4,
    #         sampling_rate=8000,
    #         audio_encoding=AudioEncoding.MULAW,
    #     )
    # )
    #
    # speaker_output = SpeakerOutput(
    #     sampling_rate=8000,
    #     audio_encoding=AudioEncoding.MULAW,
    #     device_info=sd.query_devices(kind="output"),
    # )

    # synthesizer = ElevenLabsSynthesizer(
    #     synthesizer_config=ElevenLabsSynthesizerConfig(
    #         experimental_streaming=True,
    #         optimize_streaming_latency=0,
    #         sampling_rate=16000,
    #         audio_encoding=AudioEncoding.LINEAR16,
    #     )
    # )
    #
    # speaker_output = SpeakerOutput(
    #     sampling_rate=16000,
    #     audio_encoding=AudioEncoding.LINEAR16,
    #     device_info=sd.query_devices(kind="output"),
    # )

    synthesizer = ElevenLabsSynthesizer(
        synthesizer_config=ElevenLabsSynthesizerConfig(
            experimental_streaming=False,
            stability=0.0,
            similarity_boost=0.0,
            model_id="nova",
            optimize_streaming_latency=0,
            sampling_rate=8000,
            audio_encoding=AudioEncoding.MULAW,
        )
    )

    speaker_output = SpeakerOutput(
        sampling_rate=8000,
        audio_encoding=AudioEncoding.MULAW,
        device_info=sd.query_devices(kind="output"),
    )

    asyncio.run(main())
