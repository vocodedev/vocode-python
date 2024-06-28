import time

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk
from vocode.streaming.output_device.blocking_speaker_output import BlockingSpeakerOutput
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer
from vocode.streaming.utils import get_chunk_size_per_second
from vocode.streaming.utils.worker import InterruptibleEvent

if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    seconds_per_chunk = 1

    async def speak(
        synthesizer: AbstractSynthesizer,
        output_device: AbstractOutputDevice,
        message: BaseMessage,
    ):
        message_sent = message.text
        cut_off = False
        chunk_size = seconds_per_chunk * get_chunk_size_per_second(
            synthesizer.get_synthesizer_config().audio_encoding,
            synthesizer.get_synthesizer_config().sampling_rate,
        )
        # ClientSession needs to be created within the async task
        synthesis_result = await synthesizer.create_speech(
            message=message,
            chunk_size=int(chunk_size),
        )
        chunk_idx = 0
        async for chunk_result in synthesis_result.chunk_generator:
            try:
                start_time = time.time()
                speech_length_seconds = seconds_per_chunk * (len(chunk_result.chunk) / chunk_size)
                output_device.consume_nonblocking(
                    InterruptibleEvent(payload=AudioChunk(data=chunk_result.chunk))
                )
                end_time = time.time()
                await asyncio.sleep(
                    max(
                        speech_length_seconds - (end_time - start_time),
                        0,
                    )
                )
                print("Sent chunk {} with size {}".format(chunk_idx, len(chunk_result.chunk)))
                chunk_idx += 1
            except asyncio.CancelledError:
                seconds = chunk_idx * seconds_per_chunk
                print("Interrupted, stopping text to speech after {} chunks".format(chunk_idx))
                message_sent = f"{synthesis_result.get_message_up_to(seconds)}-"
                cut_off = True
                break

        return message_sent, cut_off

    async def main():
        speaker_output = BlockingSpeakerOutput.from_default_device()
        speaker_output.start()
        synthesizer = AzureSynthesizer(AzureSynthesizerConfig.from_output_device(speaker_output))
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

    # replace with the synthesizer you want to test
    # Note: --trace will not work with AzureSynthesizer
    asyncio.run(main())
