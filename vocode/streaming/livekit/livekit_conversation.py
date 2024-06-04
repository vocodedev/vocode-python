import asyncio
from livekit import rtc
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from loguru import logger


class LiveKitOutputDevice(BaseOutputDevice):
    pass


class LiveKitConversation(StreamingConversation):

    def __init__(self, live_kit_room_url: str, live_kit_room_token: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_kit_room_url = live_kit_room_url
        self.live_kit_room_token = live_kit_room_token
        self.room = rtc.Room()
        self.room.on("track_subscribed", self.on_track_subscribed)

    async def start(self):
        await super().start()
        await self.room.connect(self.live_kit_room_url, self.live_kit_room_token)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            video_stream = rtc.AudioStream(track)
            asyncio.ensure_future(self.receive_frames(video_stream))

    async def receive_frames(audio_stream: rtc.AudioStream):
        pass


async def main():
    conversation = LiveKitConversation()
    await conversation.start()


if __name__ == "__main__":
    asyncio.run(main())
