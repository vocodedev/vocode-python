from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, Optional, TypeVar, Union

from fastapi import WebSocket
from loguru import logger

from vocode.streaming.models.events import PhoneCallEndedEvent
from vocode.streaming.models.telephony import PhoneCallDirection
from vocode.streaming.output_device.twilio_output_device import TwilioOutputDevice
from vocode.streaming.output_device.vonage_output_device import VonageOutputDevice
from vocode.streaming.pipeline.audio_pipeline import AudioPipeline
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager

TelephonyOutputDeviceType = TypeVar(
    "TelephonyOutputDeviceType", bound=Union[TwilioOutputDevice, VonageOutputDevice]
)

LOW_INTERRUPT_SENSITIVITY_THRESHOLD = 0.9

TelephonyProvider = Literal["twilio", "vonage"]


class AbstractPhoneConversation(Generic[TelephonyOutputDeviceType]):
    telephony_provider: TelephonyProvider

    def __init__(
        self,
        direction: PhoneCallDirection,
        from_phone: str,
        to_phone: str,
        base_url: str,
        config_manager: BaseConfigManager,
        pipeline: AudioPipeline[TelephonyOutputDeviceType],
    ):
        self.direction = direction
        self.from_phone = from_phone
        self.to_phone = to_phone
        self.base_url = base_url

        self.pipeline = pipeline

        self.config_manager = config_manager

    def attach_ws(self, ws: WebSocket):
        logger.debug("Trying to attach WS to outbound call")
        self.pipeline.output_device.ws = ws
        logger.debug("Attached WS to outbound call")

    @abstractmethod
    async def attach_ws_and_start(self, ws: WebSocket):
        pass

    async def terminate(self):
        self.pipeline.events_manager.publish_event(
            PhoneCallEndedEvent(conversation_id=self.pipeline.id)
        )
        await self.pipeline.terminate()