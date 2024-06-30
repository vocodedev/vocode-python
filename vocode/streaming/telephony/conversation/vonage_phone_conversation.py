import os
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.default_factory import DefaultVonagePhoneConversationActionFactory
from vocode.streaming.action.phone_call_action import VonagePhoneConversationAction
from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.models.events import PhoneCallConnectedEvent
from vocode.streaming.models.model import BaseModel
from vocode.streaming.models.telephony import PhoneCallDirection, VonageConfig
from vocode.streaming.output_device.vonage_output_device import VonageOutputDevice
from vocode.streaming.pipeline.abstract_pipeline_factory import AbstractPipelineFactory
from vocode.streaming.telephony.client.vonage_client import VonageClient
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.abstract_phone_conversation import (
    AbstractPhoneConversation,
)
from vocode.streaming.utils.events_manager import EventsManager

KOALA_CHUNK_SIZE = 512  # 16 bit samples, size 256


class VonagePhoneConversationActionsWorker(ActionsWorker):
    vonage_phone_conversation: "VonagePhoneConversation"

    def attach_state(self, action: BaseAction):
        super().attach_state(action)
        if isinstance(action, VonagePhoneConversationAction):
            action.vonage_phone_conversation = self.vonage_phone_conversation


class VonagePhoneConversation(AbstractPhoneConversation[VonageOutputDevice]):
    telephony_provider = "vonage"

    def __init__(
        self,
        direction: PhoneCallDirection,
        from_phone: str,
        to_phone: str,
        base_url: str,
        config_manager: BaseConfigManager,
        pipeline_factory: AbstractPipelineFactory[BaseModel, VonageOutputDevice],
        pipeline_config: BaseModel,
        vonage_uuid: str,
        vonage_config: VonageConfig,
        id: Optional[str] = None,
        action_factory: Optional[AbstractActionFactory] = None,
        events_manager: Optional[EventsManager] = None,
        noise_suppression: bool = False,
    ):
        actions_worker = VonagePhoneConversationActionsWorker(
            action_factory=action_factory or DefaultVonagePhoneConversationActionFactory()
        )
        pipeline = pipeline_factory.create_pipeline(
            config=pipeline_config,
            output_device=VonageOutputDevice(),
            id=id,
            events_manager=events_manager,
            actions_worker=actions_worker,
        )
        actions_worker.vonage_phone_conversation = self

        super().__init__(
            direction=direction,
            from_phone=from_phone,
            to_phone=to_phone,
            base_url=base_url,
            config_manager=config_manager,
            pipeline=pipeline,
        )
        self.vonage_config = vonage_config
        self.telephony_client = VonageClient(
            base_url=self.base_url,
            maybe_vonage_config=self.vonage_config,
        )
        self.vonage_uuid = vonage_uuid
        self.noise_suppression = noise_suppression
        if self.noise_suppression:
            import pvkoala

            logger.info("Using PV koala noise suppression")
            self.buffer = bytearray()
            self.koala = pvkoala.create(
                access_key=os.environ["KOALA_ACCESS_KEY"],
            )

    async def attach_ws_and_start(self, ws: WebSocket):
        # start message
        await ws.receive()
        super().attach_ws(ws)

        await self.pipeline.start()
        self.pipeline.events_manager.publish_event(
            PhoneCallConnectedEvent(
                conversation_id=self.pipeline.id,
                to_phone_number=self.to_phone,
                from_phone_number=self.from_phone,
            )
        )
        disconnected = False
        while self.pipeline.is_active():
            try:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message["code"])
                if "bytes" in message:
                    chunk = message["bytes"]
                    self.receive_audio(chunk)
                else:
                    logger.debug(f"Received non-bytes message: {message}")
            except WebSocketDisconnect as e:
                logger.debug("Websocket disconnected")
                if e.code != 1000:
                    logger.error(f"Websocket disconnected abnormally with code {e.code} {e.reason}")
                disconnected = True
                break
        await self.terminate()
        if not disconnected:
            await ws.close()

    def receive_audio(self, chunk: bytes):
        if self.noise_suppression:
            self.buffer.extend(chunk)

            while len(self.buffer) >= KOALA_CHUNK_SIZE:
                koala_chunk = np.frombuffer(self.buffer[:KOALA_CHUNK_SIZE], dtype=np.int16)
                try:
                    denoised_chunk = np.array(
                        self.koala.process(koala_chunk), dtype=np.int16
                    ).tobytes()
                except Exception:
                    denoised_chunk = koala_chunk.tobytes()
                self.pipeline.receive_audio(denoised_chunk)
                self.buffer = self.buffer[KOALA_CHUNK_SIZE:]
        else:
            self.pipeline.receive_audio(chunk)

    def create_vonage_client(self):
        return VonageClient(
            base_url=self.base_url,
            maybe_vonage_config=self.vonage_config,
        )
