import base64
import json
import os
from enum import Enum
from typing import Optional

from fastapi import WebSocket
from loguru import logger

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.events import PhoneCallConnectedEvent
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.telephony import PhoneCallDirection, TwilioConfig
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.output_device.twilio_output_device import (
    ChunkFinishedMarkMessage,
    TwilioOutputDevice,
)
from vocode.streaming.synthesizer.abstract_factory import AbstractSynthesizerFactory
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.abstract_phone_conversation import (
    AbstractPhoneConversation,
)
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.utils.state_manager import TwilioPhoneConversationStateManager


class TwilioPhoneConversationWebsocketAction(Enum):
    CLOSE_WEBSOCKET = 1


class TwilioPhoneConversation(AbstractPhoneConversation[TwilioOutputDevice]):
    telephony_provider = "twilio"

    def __init__(
        self,
        direction: PhoneCallDirection,
        from_phone: str,
        to_phone: str,
        base_url: str,
        config_manager: BaseConfigManager,
        agent_config: AgentConfig,
        transcriber_config: TranscriberConfig,
        synthesizer_config: SynthesizerConfig,
        twilio_sid: str,
        agent_factory: AbstractAgentFactory,
        transcriber_factory: AbstractTranscriberFactory,
        synthesizer_factory: AbstractSynthesizerFactory,
        twilio_config: Optional[TwilioConfig] = None,
        conversation_id: Optional[str] = None,
        events_manager: Optional[EventsManager] = None,
        record_call: bool = False,
        speed_coefficient: float = 1.0,
        noise_suppression: bool = False,  # is currently a no-op
    ):
        super().__init__(
            direction=direction,
            from_phone=from_phone,
            to_phone=to_phone,
            base_url=base_url,
            config_manager=config_manager,
            output_device=TwilioOutputDevice(),
            agent_config=agent_config,
            transcriber_config=transcriber_config,
            synthesizer_config=synthesizer_config,
            conversation_id=conversation_id,
            events_manager=events_manager,
            transcriber_factory=transcriber_factory,
            agent_factory=agent_factory,
            synthesizer_factory=synthesizer_factory,
            speed_coefficient=speed_coefficient,
        )
        self.config_manager = config_manager
        self.twilio_config = twilio_config or TwilioConfig(
            account_sid=os.environ["TWILIO_ACCOUNT_SID"],
            auth_token=os.environ["TWILIO_AUTH_TOKEN"],
        )
        self.telephony_client = TwilioClient(
            base_url=self.base_url, maybe_twilio_config=self.twilio_config
        )
        self.twilio_sid = twilio_sid
        self.record_call = record_call

    def create_state_manager(self) -> TwilioPhoneConversationStateManager:
        return TwilioPhoneConversationStateManager(self)

    async def attach_ws_and_start(self, ws: WebSocket):
        super().attach_ws(ws)

        await self._wait_for_twilio_start(ws)
        await self.start()
        self.events_manager.publish_event(
            PhoneCallConnectedEvent(
                conversation_id=self.id,
                to_phone_number=self.to_phone,
                from_phone_number=self.from_phone,
            )
        )
        while self.is_active():
            message = await ws.receive_text()
            response = await self._handle_ws_message(message)
            if response == TwilioPhoneConversationWebsocketAction.CLOSE_WEBSOCKET:
                break
        await ws.close(code=1000, reason=None)
        await self.terminate()

    async def _wait_for_twilio_start(self, ws: WebSocket):
        assert isinstance(self.output_device, TwilioOutputDevice)
        while True:
            message = await ws.receive_text()
            if not message:
                continue
            data = json.loads(message)
            if data["event"] == "start":
                logger.debug(f"Media WS: Received event '{data['event']}': {message}")
                self.output_device.stream_sid = data["start"]["streamSid"]
                break

    async def _handle_ws_message(self, message) -> Optional[TwilioPhoneConversationWebsocketAction]:
        if message is None:
            return TwilioPhoneConversationWebsocketAction.CLOSE_WEBSOCKET

        data = json.loads(message)
        if data["event"] == "media":
            media = data["media"]
            chunk = base64.b64decode(media["payload"])
            self.receive_audio(chunk)
        if data["event"] == "mark":
            chunk_id = data["mark"]["name"]
            self.output_device.enqueue_mark_message(ChunkFinishedMarkMessage(chunk_id=chunk_id))
        elif data["event"] == "stop":
            logger.debug(f"Media WS: Received event 'stop': {message}")
            logger.debug("Stopping...")
            return TwilioPhoneConversationWebsocketAction.CLOSE_WEBSOCKET
        return None
