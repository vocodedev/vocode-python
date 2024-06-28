from abc import abstractmethod
from typing import Literal, Optional, TypeVar, Union

from fastapi import WebSocket
from loguru import logger

from vocode import conversation_id as ctx_conversation_id
from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.events import PhoneCallEndedEvent
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.telephony import PhoneCallDirection
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.output_device.twilio_output_device import TwilioOutputDevice
from vocode.streaming.output_device.vonage_output_device import VonageOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.abstract_factory import AbstractSynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.events_manager import EventsManager

TelephonyOutputDeviceType = TypeVar(
    "TelephonyOutputDeviceType", bound=Union[TwilioOutputDevice, VonageOutputDevice]
)

LOW_INTERRUPT_SENSITIVITY_THRESHOLD = 0.9

TelephonyProvider = Literal["twilio", "vonage"]


class AbstractPhoneConversation(StreamingConversation[TelephonyOutputDeviceType]):
    telephony_provider: TelephonyProvider

    def __init__(
        self,
        direction: PhoneCallDirection,
        from_phone: str,
        to_phone: str,
        base_url: str,
        config_manager: BaseConfigManager,
        output_device: TelephonyOutputDeviceType,
        agent_config: AgentConfig,
        transcriber_config: TranscriberConfig,
        synthesizer_config: SynthesizerConfig,
        agent_factory: AbstractAgentFactory,
        transcriber_factory: AbstractTranscriberFactory,
        synthesizer_factory: AbstractSynthesizerFactory,
        conversation_id: Optional[str] = None,
        events_manager: Optional[EventsManager] = None,
        speed_coefficient: float = 1.0,
    ):
        conversation_id = conversation_id or create_conversation_id()
        ctx_conversation_id.set(conversation_id)

        self.direction = direction
        self.from_phone = from_phone
        self.to_phone = to_phone
        self.base_url = base_url
        super().__init__(
            output_device,
            transcriber_factory.create_transcriber(transcriber_config),
            agent_factory.create_agent(agent_config),
            synthesizer_factory.create_synthesizer(synthesizer_config),
            conversation_id=conversation_id,
            events_manager=events_manager,
            speed_coefficient=speed_coefficient,
        )
        self.transcriptions_worker = self.TranscriptionsWorker(
            input_queue=self.transcriber.output_queue,
            output_queue=self.agent.get_input_queue(),
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.config_manager = config_manager

    def attach_ws(self, ws: WebSocket):
        logger.debug("Trying to attach WS to outbound call")
        self.output_device.ws = ws
        logger.debug("Attached WS to outbound call")

    @abstractmethod
    async def attach_ws_and_start(self, ws: WebSocket):
        pass

    async def terminate(self):
        self.events_manager.publish_event(PhoneCallEndedEvent(conversation_id=self.id))
        await super().terminate()
