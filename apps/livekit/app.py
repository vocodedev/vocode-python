import asyncio
from functools import partial
import os
from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
from loguru import logger
from livekit import rtc
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.output_device.livekit_output_device import LiveKitOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber


class Settings(BaseSettings):

    livekit_api_key: str = "ENTER_YOUR_LIVE_KIT_API_KEY"
    livekit_api_secret: str = "ENTER_YOUR_LIVE_KIT_API_SECRET"
    livekit_ws_url: str = "ENTER_YOUR_LIVE_KIT_WS_URL"

    # This means a .env file can be used to overload these settings
    # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


async def create_audio_source(
    room: rtc.Room, sample_rate: int = 48000, num_channels: int = 1
) -> rtc.AudioSource:
    source = rtc.AudioSource(sample_rate, num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-synthesis", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, options)

    return source


async def send_frames_to_conversation(
    conversation: StreamingConversation,
    audio_queue: asyncio.Queue[bytes],
):
    while conversation.is_active():
        chunk = await audio_queue.get()
        conversation.receive_audio(chunk)


def on_track_subscribed(
    audio_queue: asyncio.Queue,
    conversation_ready: asyncio.Event,
    track: rtc.Track,
    publication: rtc.RemoteTrackPublication,
    participant: rtc.RemoteParticipant,
):
    logger.info("track subscribed: %s", publication.sid)
    if track.kind == rtc.TrackKind.KIND_AUDIO:
        audio_stream = rtc.AudioStream(track)
        asyncio.ensure_future(receive_frames(audio_queue, conversation_ready, audio_stream))


async def receive_frames(
    audio_queue: asyncio.Queue,
    conversation_ready: asyncio.Event,
    audio_stream: rtc.AudioStream,
):
    # this is where we will send the frames to transcription
    async for event in audio_stream:
        if conversation_ready.is_set():
            frame = event.frame
            audio_queue.put_nowait(bytes(frame.data))


async def entrypoint(ctx: JobContext):
    conversation_ready = asyncio.Event()
    audio_queue = asyncio.Queue()

    ctx.room.on("track_subscribed", partial(on_track_subscribed, audio_queue, conversation_ready))
    source = await create_audio_source(ctx.room)

    output_device = LiveKitOutputDevice(
        sampling_rate=48000,
        audio_encoding=AudioEncoding.LINEAR16,
        source=source,
    )

    conversation = StreamingConversation(
        output_device=output_device,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig(
                audio_encoding=AudioEncoding.LINEAR16,
                sampling_rate=48000,
                chunk_size=480,
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
    conversation_ready.set()
    asyncio.create_task(send_frames_to_conversation(conversation, audio_queue))


async def request_fnc(req: JobRequest) -> None:
    logger.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    configure_pretty_logging()

    settings = Settings()
    cli.run_app(
        WorkerOptions(
            request_fnc,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_ws_url,
        )
    )
