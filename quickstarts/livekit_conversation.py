import asyncio
import signal
import os
from dotenv import load_dotenv
from livekit import api, rtc
from numpy import source


from vocode.streaming import synthesizer
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.livekit.livekit_conversation import LiveKitConversation
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.output_device.livekit_output_device import LiveKitOutputDevice
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

load_dotenv()


async def create_audio_source(
    room: rtc.Room, sample_rate: int = 48000, num_channels: int = 1
) -> rtc.AudioSource:
    source = rtc.AudioSource(sample_rate, num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-synthesis", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, options)

    return source


async def main(room: rtc.Room):

    # Join the livekit room
    token = (
        api.AccessToken()
        .with_identity("Vocode Agent")
        .with_name("Vocode Agent")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="vocode-room",
            )
        )
        .to_jwt()
    )
    await room.connect(os.getenv("LIVEKIT_URL"), token)

    # Create an audio source for the agent
    source = await create_audio_source(room)

    # Set up output device
    output_device = LiveKitOutputDevice(source=source)

    conversation = LiveKitConversation(
        room=room,
        output_device=output_device,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig(
                audio_encoding=AudioEncoding.LINEAR16,
                sampling_rate=48000,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=os.getenv("DEEPGRAM_API_KEY"),
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""The AI is having a pleasant conversation about life""",
            )
        ),
        synthesizer=ElevenLabsSynthesizer(
            synthesizer_config=ElevenLabsSynthesizerConfig(
                audio_encoding=AudioEncoding.LINEAR16,
                sampling_rate=48000,
                voice_id="ODq5zmih8GrVes37Dizd",
                api_key=os.getenv("ELEVEN_LABS_API_KEY"),
            )
        ),
    )

    await conversation.start()
    print("Conversation started")

    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    while conversation.is_active():
        await asyncio.sleep(1)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room))
    for signal in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
