from typing import Optional, Dict
import logging

from fastapi import APIRouter, HTTPException, WebSocket
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.synthesizer.factory import SynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import (
    BaseConfigManager,
)

from vocode.streaming.telephony.conversation.call import Call
from vocode.streaming.telephony.templater import Templater
from vocode.streaming.transcriber.factory import TranscriberFactory
from vocode.streaming.utils.base_router import BaseRouter
from vocode.streaming.utils.events_manager import EventsManager


class CallsRouter(BaseRouter):
    def __init__(
        self,
        base_url: str,
        templater: Templater,
        config_manager: BaseConfigManager,
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.base_url = base_url
        self.templater = templater
        self.config_manager = config_manager
        self.transcriber_factory = transcriber_factory
        self.agent_factory = agent_factory
        self.synthesizer_factory = synthesizer_factory
        self.events_manager = events_manager
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.router.websocket("/connect_call/{id}")(self.connect_call)

    async def connect_call(self, websocket: WebSocket, id: str):
        await websocket.accept()
        self.logger.debug("Phone WS connection opened for chat {}".format(id))

        call_config = self.config_manager.get_config(id)
        if not call_config:
            raise HTTPException(status_code=400, detail="No active phone call")

        call = Call.from_call_config(
            base_url=self.base_url,
            call_config=call_config,
            config_manager=self.config_manager,
            conversation_id=id,
            transcriber_factory=self.transcriber_factory,
            agent_factory=self.agent_factory,
            synthesizer_factory=self.synthesizer_factory,
            events_manager=self.events_manager,
            logger=self.logger,
        )

        await call.attach_ws_and_start(websocket)

        self.logger.debug("Phone WS connection closed for chat {}".format(id))

    def get_router(self) -> APIRouter:
        return self.router
