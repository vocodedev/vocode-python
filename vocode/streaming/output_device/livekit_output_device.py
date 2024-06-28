import asyncio

from livekit import rtc

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState

NUM_CHANNELS = 1


class LiveKitOutputDevice(AbstractOutputDevice):
    source: rtc.AudioSource
    track: rtc.LocalAudioTrack
    room: rtc.Room

    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding):
        super().__init__(sampling_rate, audio_encoding)

    async def _run_loop(self):
        while True:
            try:
                item = await self.input_queue.get()
            except asyncio.CancelledError:
                return

            self.interruptible_event = item
            audio_chunk = item.payload

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                continue

            await self.play(audio_chunk.data)
            audio_chunk.on_play()
            audio_chunk.state = ChunkState.PLAYED
            self.interruptible_event.is_interruptible = False

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

    async def play(self, item: bytes):
        audio_frame = rtc.AudioFrame(
            item, self.sampling_rate, num_channels=1, samples_per_channel=len(item) // 2
        )
        return await self.source.capture_frame(audio_frame)

    def interrupt(self):
        pass
