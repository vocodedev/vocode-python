from typing import Optional

import sentry_sdk
from fastapi import APIRouter, HTTPException, WebSocket
from loguru import logger

from vocode import sentry_transaction
from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.default_factory import DefaultAgentFactory
from vocode.streaming.models.pipeline import PipelineConfig
from vocode.streaming.models.telephony import BaseCallConfig, TwilioCallConfig, VonageCallConfig
from vocode.streaming.output_device.twilio_output_device import TwilioOutputDevice
from vocode.streaming.output_device.vonage_output_device import VonageOutputDevice
from vocode.streaming.pipeline.abstract_pipeline_factory import AbstractPipelineFactory
from vocode.streaming.streaming_conversation import StreamingConversationFactory
from vocode.streaming.synthesizer.abstract_factory import AbstractSynthesizerFactory
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.abstract_phone_conversation import (
    AbstractPhoneConversation,
)
from vocode.streaming.telephony.conversation.twilio_phone_conversation import (
    TwilioPhoneConversation,
)
from vocode.streaming.telephony.conversation.vonage_phone_conversation import (
    VonagePhoneConversation,
)
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory
from vocode.streaming.utils.base_router import BaseRouter
from vocode.streaming.utils.events_manager import EventsManager


class CallsRouter(BaseRouter):
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        pipeline_factory: AbstractPipelineFactory = StreamingConversationFactory(),
        events_manager: Optional[EventsManager] = None,
    ):
        super().__init__()
        self.base_url = base_url
        self.config_manager = config_manager
        self.pipeline_factory = pipeline_factory
        self.events_manager = events_manager
        self.router = APIRouter()
        self.router.websocket("/connect_call/{id}")(self.connect_call)

    def _from_call_config(
        self,
        base_url: str,
        call_config: BaseCallConfig,
        config_manager: BaseConfigManager,
        conversation_id: str,
        pipeline_factory: AbstractPipelineFactory = StreamingConversationFactory(),
        events_manager: Optional[EventsManager] = None,
    ) -> AbstractPhoneConversation:
        if isinstance(call_config, TwilioCallConfig):
            return TwilioPhoneConversation(
                to_phone=call_config.to_phone,
                from_phone=call_config.from_phone,
                base_url=base_url,
                config_manager=config_manager,
                twilio_config=call_config.twilio_config,
                twilio_sid=call_config.twilio_sid,
                direction=call_config.direction,
                pipeline_factory=pipeline_factory,
                pipeline_config=call_config.pipeline_config,
                id=conversation_id,
                events_manager=events_manager,
            )
        elif isinstance(call_config, VonageCallConfig):
            return VonagePhoneConversation(
                to_phone=call_config.to_phone,
                from_phone=call_config.from_phone,
                base_url=base_url,
                config_manager=config_manager,
                vonage_config=call_config.vonage_config,
                vonage_uuid=call_config.vonage_uuid,
                direction=call_config.direction,
                pipeline_factory=pipeline_factory,
                pipeline_config=call_config.pipeline_config,
                id=conversation_id,
                events_manager=events_manager,
            )
        else:
            raise ValueError(f"Unknown call config type {call_config.type}")

    async def connect_call(self, websocket: WebSocket, id: str):
        with sentry_sdk.start_transaction(op="connect_call") as sentry_txn:
            sentry_transaction.set(sentry_txn)
            await websocket.accept()
            logger.debug("Phone WS connection opened for chat {}".format(id))
            call_config = await self.config_manager.get_config(id)
            if not call_config:
                raise HTTPException(status_code=400, detail="No active phone call")

            phone_conversation = self._from_call_config(
                base_url=self.base_url,
                call_config=call_config,
                config_manager=self.config_manager,
                conversation_id=id,
                pipeline_factory=self.pipeline_factory,
                events_manager=self.events_manager,
            )

            await phone_conversation.attach_ws_and_start(websocket)
            logger.debug("Phone WS connection closed for chat {}".format(id))

    def get_router(self) -> APIRouter:
        return self.router
