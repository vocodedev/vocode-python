from openai import audio
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.utils.worker import AsyncQueueWorker
from livekit import rtc


class LiveKitOutputDevice(BaseOutputDevice, AsyncQueueWorker):
    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding, source: rtc.AudioSource):
        super().__init__(sampling_rate, audio_encoding)
        self.source = source

    async def process(self, item: bytes):
        audio_frame = rtc.AudioFrame(
            item, self.sampling_rate, num_channels=1, samples_per_channel=len(item) // 2
        )
        return await self.source.capture_frame(audio_frame)
