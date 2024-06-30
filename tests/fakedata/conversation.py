import asyncio
import time
from typing import Optional

from pytest_mock import MockerFixture

from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import AgentConfig, ChatGPTAgentConfig
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig, SynthesizerConfig
from vocode.streaming.models.telephony import PhoneCallDirection, TwilioConfig, VonageConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TranscriberConfig
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState
from vocode.streaming.streaming_conversation import (
    StreamingConversation,
    StreamingConversationFactory,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLING_RATE
from vocode.streaming.telephony.conversation.twilio_phone_conversation import (
    TwilioPhoneConversation,
)
from vocode.streaming.telephony.conversation.vonage_phone_conversation import (
    VonagePhoneConversation,
)
from vocode.streaming.transcriber.base_transcriber import AbstractTranscriber, BaseTranscriber
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramEndpointingConfig
from vocode.streaming.utils.events_manager import EventsManager

DEFAULT_DEEPGRAM_TRANSCRIBER_CONFIG = DeepgramTranscriberConfig(
    chunk_size=DEFAULT_CHUNK_SIZE,
    sampling_rate=DEFAULT_SAMPLING_RATE,
    audio_encoding=AudioEncoding.MULAW,
    endpointing_config=DeepgramEndpointingConfig(),
    model="2-phonecall",
    tier="nova",
)

DEFAULT_SYNTHESIZER_CONFIG = PlayHtSynthesizerConfig(
    voice_id="test_voice_id",
    sampling_rate=DEFAULT_SAMPLING_RATE,
    audio_encoding=AudioEncoding.MULAW,
)

DEFAULT_CHAT_GPT_AGENT_CONFIG = ChatGPTAgentConfig(
    prompt_preamble="You are an agent of chaos", initial_message=BaseMessage(text="Hi there!")
)


class DummyOutputDevice(AbstractOutputDevice):

    def __init__(
        self,
        sampling_rate: int,
        audio_encoding: AudioEncoding,
        wait_for_interrupt: bool = False,
        chunks_before_interrupt: int = 1,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.wait_for_interrupt = wait_for_interrupt
        self.chunks_before_interrupt = chunks_before_interrupt
        self.interrupt_event = asyncio.Event()

    async def process(self, item):
        self.interruptible_event = item
        audio_chunk = item.payload

        if item.is_interrupted():
            audio_chunk.on_interrupt()
            audio_chunk.state = ChunkState.INTERRUPTED
        else:
            audio_chunk.on_play()
            audio_chunk.state = ChunkState.PLAYED
            self.interruptible_event.is_interruptible = False

    async def _run_loop(self):
        chunk_counter = 0
        while True:
            try:
                item = await self.input_queue.get()
            except asyncio.CancelledError:
                return
            if self.wait_for_interrupt and chunk_counter == self.chunks_before_interrupt:
                await self.interrupt_event.wait()
            await self.process(item)
            chunk_counter += 1

    def flush(self):
        while True:
            try:
                item = self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.process(item)

    def interrupt(self):
        pass


def create_fake_transcriber(mocker: MockerFixture, transcriber_config: TranscriberConfig):
    transcriber = mocker.MagicMock()
    transcriber.get_transcriber_config = mocker.MagicMock(return_value=transcriber_config)
    return transcriber


def create_fake_transcriber_factory(
    mocker: MockerFixture, transcriber: Optional[AbstractTranscriber] = None
):
    factory = mocker.MagicMock()
    factory.create_transcriber = mocker.MagicMock(
        return_value=transcriber
        or create_fake_transcriber(mocker, DEFAULT_DEEPGRAM_TRANSCRIBER_CONFIG)
    )
    return factory


def create_fake_agent(mocker: MockerFixture, agent_config: AgentConfig):
    agent = mocker.MagicMock()
    agent.get_agent_config = mocker.MagicMock(return_value=agent_config)
    return agent


def create_fake_agent_factory(mocker: MockerFixture, agent: Optional[BaseAgent] = None):
    factory = mocker.MagicMock()
    factory.create_agent = mocker.MagicMock(
        return_value=agent or create_fake_agent(mocker, DEFAULT_CHAT_GPT_AGENT_CONFIG)
    )
    return factory


def create_fake_synthesizer(mocker: MockerFixture, synthesizer_config: SynthesizerConfig):
    synthesizer = mocker.MagicMock()
    synthesizer.get_synthesizer_config = mocker.MagicMock(return_value=synthesizer_config)
    return synthesizer


def create_fake_synthesizer_factory(
    mocker: MockerFixture, synthesizer: Optional[BaseSynthesizer] = None
):
    factory = mocker.MagicMock()
    factory.create_synthesizer = mocker.MagicMock(
        return_value=synthesizer or create_fake_synthesizer(mocker, DEFAULT_SYNTHESIZER_CONFIG)
    )
    return factory


def create_fake_streaming_conversation(
    mocker: MockerFixture,
    transcriber: Optional[BaseTranscriber[TranscriberConfig]] = None,
    agent: Optional[BaseAgent] = None,
    synthesizer: Optional[BaseSynthesizer] = None,
    speed_coefficient: float = 1.0,
    conversation_id: Optional[str] = None,
    events_manager: Optional[EventsManager] = None,
):
    transcriber = transcriber or create_fake_transcriber(
        mocker, DEFAULT_DEEPGRAM_TRANSCRIBER_CONFIG
    )
    agent = agent or create_fake_agent(mocker, DEFAULT_CHAT_GPT_AGENT_CONFIG)
    synthesizer = synthesizer or create_fake_synthesizer(mocker, DEFAULT_SYNTHESIZER_CONFIG)
    return StreamingConversation(
        output_device=DummyOutputDevice(
            sampling_rate=DEFAULT_SAMPLING_RATE, audio_encoding=AudioEncoding.MULAW
        ),
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer,
        speed_coefficient=speed_coefficient,
        conversation_id=conversation_id,
        events_manager=events_manager,
    )


def create_fake_streaming_conversation_factory(
    mocker: MockerFixture,
    transcriber: Optional[BaseTranscriber[TranscriberConfig]] = None,
    agent: Optional[BaseAgent] = None,
    synthesizer: Optional[BaseSynthesizer] = None,
):
    return StreamingConversationFactory(
        transcriber_factory=create_fake_transcriber_factory(mocker, transcriber),
        agent_factory=create_fake_agent_factory(mocker, agent),
        synthesizer_factory=create_fake_synthesizer_factory(mocker, synthesizer),
    )


def create_fake_twilio_phone_conversation_with_streaming_conversation_pipeline(
    mocker: MockerFixture,
    streaming_conversation_factory: StreamingConversationFactory,
    direction: PhoneCallDirection = "outbound",
    from_phone: str = "+1234567890",
    to_phone: str = "+0987654321",
    base_url: str = "http://test.com",
    twilio_sid: str = "test_sid",
    twilio_config: Optional[TwilioConfig] = None,
    config_manager: Optional[BaseConfigManager] = None,
    events_manager: Optional[EventsManager] = None,
):
    return TwilioPhoneConversation(
        direction=direction,
        from_phone=from_phone,
        to_phone=to_phone,
        base_url=base_url,
        config_manager=config_manager,
        pipeline_factory=streaming_conversation_factory,
        pipeline_config=mocker.MagicMock(),
        twilio_sid=twilio_sid,
        twilio_config=twilio_config,
        events_manager=events_manager,
    )


def create_fake_vonage_phone_conversation_with_streaming_conversation_pipeline(
    mocker: MockerFixture,
    streaming_conversation_factory: StreamingConversationFactory,
    direction: PhoneCallDirection = "outbound",
    from_phone: str = "+1234567890",
    to_phone: str = "+0987654321",
    base_url: str = "http://test.com",
    vonage_uuid: str = "test_uuid",
    vonage_config: Optional[VonageConfig] = None,
    config_manager: Optional[BaseConfigManager] = None,
    events_manager: Optional[EventsManager] = None,
):
    return VonagePhoneConversation(
        direction=direction,
        from_phone=from_phone,
        to_phone=to_phone,
        base_url=base_url,
        config_manager=config_manager,
        pipeline_factory=streaming_conversation_factory,
        pipeline_config=mocker.MagicMock(),
        vonage_uuid=vonage_uuid,
        vonage_config=vonage_config or mocker.MagicMock(),
        events_manager=events_manager,
    )
