import base64
import json
import os
from enum import Enum
from typing import Optional

from fastapi import WebSocket
from loguru import logger

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.default_factory import DefaultTwilioPhoneConversationActionFactory
from vocode.streaming.action.phone_call_action import TwilioPhoneConversationAction
from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.models.events import PhoneCallConnectedEvent
from vocode.streaming.models.model import BaseModel
from vocode.streaming.models.telephony import PhoneCallDirection, TwilioConfig
from vocode.streaming.output_device.twilio_output_device import (
    ChunkFinishedMarkMessage,
    TwilioOutputDevice,
)
from vocode.streaming.pipeline.abstract_pipeline_factory import AbstractPipelineFactory
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.abstract_phone_conversation import (
    AbstractPhoneConversation,
)
from vocode.streaming.utils.events_manager import EventsManager


class TwilioPhoneConversationWebsocketAction(Enum):
    CLOSE_WEBSOCKET = 1


class TwilioPhoneConversationActionsWorker(ActionsWorker):
    twilio_phone_conversation: "TwilioPhoneConversation"

    def attach_state(self, action: BaseAction):
        super().attach_state(action)
        if isinstance(action, TwilioPhoneConversationAction):
            action.twilio_phone_conversation = self.twilio_phone_conversation


class TwilioPhoneConversation(AbstractPhoneConversation[TwilioOutputDevice]):
    telephony_provider = "twilio"

    def __init__(
        self,
        direction: PhoneCallDirection,
        from_phone: str,
        to_phone: str,
        base_url: str,
        config_manager: BaseConfigManager,
        pipeline_factory: AbstractPipelineFactory[BaseModel, TwilioOutputDevice],
        pipeline_config: BaseModel,
        twilio_sid: str,
        twilio_config: Optional[TwilioConfig] = None,
        id: Optional[str] = None,
        action_factory: Optional[AbstractActionFactory] = None,
        events_manager: Optional[EventsManager] = None,
        record_call: bool = False,
        noise_suppression: bool = False,  # is currently a no-op
    ):
        actions_worker = TwilioPhoneConversationActionsWorker(
            action_factory=action_factory or DefaultTwilioPhoneConversationActionFactory()
        )
        pipeline = pipeline_factory.create_pipeline(
            config=pipeline_config,
            output_device=TwilioOutputDevice(),
            id=id,
            events_manager=events_manager,
            actions_worker=actions_worker,
        )
        actions_worker.twilio_phone_conversation = self

        super().__init__(
            direction=direction,
            from_phone=from_phone,
            to_phone=to_phone,
            base_url=base_url,
            config_manager=config_manager,
            pipeline=pipeline,
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

    async def attach_ws_and_start(self, ws: WebSocket):
        super().attach_ws(ws)

        await self._wait_for_twilio_start(ws)
        await self.pipeline.start()
        self.pipeline.events_manager.publish_event(
            PhoneCallConnectedEvent(
                conversation_id=self.pipeline.id,
                to_phone_number=self.to_phone,
                from_phone_number=self.from_phone,
            )
        )
        while self.pipeline.is_active():
            message = await ws.receive_text()
            response = await self._handle_ws_message(message)
            if response == TwilioPhoneConversationWebsocketAction.CLOSE_WEBSOCKET:
                break
        await ws.close(code=1000, reason=None)
        await self.terminate()

    async def _wait_for_twilio_start(self, ws: WebSocket):
        while True:
            message = await ws.receive_text()
            if not message:
                continue
            data = json.loads(message)
            if data["event"] == "start":
                logger.debug(f"Media WS: Received event '{data['event']}': {message}")
                self.pipeline.output_device.stream_sid = data["start"]["streamSid"]
                break

    async def _handle_ws_message(self, message) -> Optional[TwilioPhoneConversationWebsocketAction]:
        if message is None:
            return TwilioPhoneConversationWebsocketAction.CLOSE_WEBSOCKET

        data = json.loads(message)
        if data["event"] == "media":
            media = data["media"]
            chunk = base64.b64decode(media["payload"])
            self.pipeline.receive_audio(chunk)
        if data["event"] == "mark":
            chunk_id = data["mark"]["name"]
            self.pipeline.output_device.enqueue_mark_message(
                ChunkFinishedMarkMessage(chunk_id=chunk_id)
            )
        elif data["event"] == "stop":
            logger.debug(f"Media WS: Received event 'stop': {message}")
            logger.debug("Stopping...")
            return TwilioPhoneConversationWebsocketAction.CLOSE_WEBSOCKET
        return None

    def create_twilio_client(self):
        return TwilioClient(
            base_url=self.base_url,
            maybe_twilio_config=self.twilio_config,
        )
