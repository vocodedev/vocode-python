from vocode.streaming.input_device.microphone_input import MicrophoneInput
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, Transcription
from vocode.streaming.pipeline.worker import AsyncWorker
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.transcriber.deepgram_transcriber import (
    DeepgramEndpointingConfig,
    DeepgramTranscriber,
)


class TranscriptionPrinter(AsyncWorker[Transcription]):
    async def _run_loop(self):
        while True:
            transcription: Transcription = await self.input_queue.get()
            print(transcription)


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    async def listen():
        microphone_input = MicrophoneInput.from_default_device()

        # replace with the transcriber you want to test
        transcriber = DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input, endpointing_config=DeepgramEndpointingConfig()
            )
        )
        transcriber.start()
        transcription_printer = TranscriptionPrinter()
        transcriber.consumer = transcription_printer
        transcription_printer.start()
        print("Start speaking...press Ctrl+C to end. ")
        while True:
            chunk = await microphone_input.get_audio()
            transcriber.send_audio(chunk)

    asyncio.run(listen())
