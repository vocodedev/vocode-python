import asyncio
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.utils.worker import AsyncQueueWorker
from livekit import rtc

NUM_CHANNELS = 1


class LiveKitOutputDevice(AsyncQueueWorker, BaseOutputDevice):
    source: rtc.AudioSource
    track: rtc.LocalAudioTrack
    room: rtc.Room

    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding):
        BaseOutputDevice.__init__(self, sampling_rate, audio_encoding)
        AsyncQueueWorker.__init__(self, input_queue=asyncio.Queue())

    async def initialize_source(self, room: rtc.Room):
        """Creates the AudioSource that will be used to capture audio frames.

        Can only be called once the room has set up its track callbcks
        """
        self.room = room
        source = rtc.AudioSource(self.sampling_rate, NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("agent-synthesis", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.room.local_participant.publish_track(track, options)
        self.track = track
        self.source = source

    async def uninitialize_source(self):
        await self.room.local_participant.unpublish_track(self.track.sid)

    async def process(self, item: bytes):
        audio_frame = rtc.AudioFrame(
            item, self.sampling_rate, num_channels=1, samples_per_channel=len(item) // 2
        )
        return await self.source.capture_frame(audio_frame)
