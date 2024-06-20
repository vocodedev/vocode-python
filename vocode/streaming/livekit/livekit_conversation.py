import asyncio
from livekit import rtc
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.output_device.livekit_output_device import LiveKitOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from loguru import logger


class LiveKitConversation(StreamingConversation[LiveKitOutputDevice]):

    def __init__(self, room: rtc.Room, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.room = room
        self.room.on("track_subscribed", self.on_track_subscribed)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            asyncio.ensure_future(self.receive_frames(audio_stream))

    async def receive_frames(self, audio_stream: rtc.AudioStream):
        # this is where we will send the frames to transcription
        async for event in audio_stream:
            if not self.active:
                break

            frame = event.frame
            self.receive_audio(frame.data)


async def main():
    conversation = StreamingConversation()
    await conversation.start()


if __name__ == "__main__":
    asyncio.run(main())
