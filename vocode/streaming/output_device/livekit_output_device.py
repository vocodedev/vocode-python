import asyncio
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.utils.worker import AsyncQueueWorker
from livekit import rtc


class LiveKitOutputDevice(AsyncQueueWorker, BaseOutputDevice):
    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding, source: rtc.AudioSource):
        BaseOutputDevice.__init__(self, sampling_rate, audio_encoding)
        AsyncQueueWorker.__init__(self, input_queue=asyncio.Queue())
        self.source = source

    async def process(self, item: bytes):
        audio_frame = rtc.AudioFrame(
            item, self.sampling_rate, num_channels=1, samples_per_channel=len(item) // 2
        )
        return await self.source.capture_frame(audio_frame)
