import asyncio
from typing import Awaitable, Callable

from loguru import logger
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.output_device.livekit_output_device import LiveKitOutputDevice
from livekit import rtc


class LiveKitConversation(StreamingConversation[LiveKitOutputDevice]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.receive_frames_task: asyncio.Task | None = None

    async def start_room(self, room: rtc.Room):
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)

        await self.output_device.initialize_source(room)

        await super().start()

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            self.receive_frames_task = asyncio.create_task(self._receive_frames(audio_stream))

    async def _receive_frames(
        self,
        audio_stream: rtc.AudioStream,
    ):
        # this is where we will send the frames to transcription
        async for event in audio_stream:
            if self.is_active():
                frame = event.frame
                self.receive_audio(bytes(frame.data))

    def _on_track_unsubscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.mark_terminated()

    async def terminate(self):
        if self.receive_frames_task:
            self.receive_frames_task.cancel()
        await self.output_device.uninitialize_source()
        return await super().terminate()
