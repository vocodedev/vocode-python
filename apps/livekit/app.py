import asyncio
import os
from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
from loguru import logger
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
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.livekit.livekit_conversation import LiveKitConversation
from vocode.streaming.action.end_conversation import EndConversationVocodeActionConfig
from vocode.streaming.models.actions import (
    PhraseBasedActionTrigger,
    PhraseBasedActionTriggerConfig,
    PhraseTrigger,
)


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


async def wait_for_termination(conversation: LiveKitConversation, ctx: JobContext):
    await conversation.wait_for_termination()
    await conversation.terminate()
    await ctx.room.disconnect()


async def entrypoint(ctx: JobContext):
    configure_pretty_logging()
    conversation = LiveKitConversation(
        output_device=LiveKitOutputDevice(
            sampling_rate=48000,
            audio_encoding=AudioEncoding.LINEAR16,
        ),
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
                actions=[
                    EndConversationVocodeActionConfig(
                        action_trigger=PhraseBasedActionTrigger(
                            config=PhraseBasedActionTriggerConfig(
                                phrase_triggers=[
                                    PhraseTrigger(
                                        phrase="goodbye",
                                        conditions=["phrase_condition_type_contains"],
                                    )
                                ]
                            )
                        )
                    )
                ],
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
    await conversation.start_room(ctx.room)
    asyncio.create_task(wait_for_termination(conversation, ctx))


async def request_fnc(req: JobRequest) -> None:
    logger.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    settings = Settings()
    cli.run_app(
        WorkerOptions(
            request_fnc,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_ws_url,
        )
    )
