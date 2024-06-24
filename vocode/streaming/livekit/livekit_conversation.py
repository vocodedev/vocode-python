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


async def main():
    conversation = StreamingConversation()
    await conversation.start()


if __name__ == "__main__":
    asyncio.run(main())
