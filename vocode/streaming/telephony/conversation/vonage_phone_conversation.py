import os
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.events import PhoneCallConnectedEvent
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.telephony import PhoneCallDirection, VonageConfig
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.output_device.vonage_output_device import VonageOutputDevice
from vocode.streaming.synthesizer.abstract_factory import AbstractSynthesizerFactory
from vocode.streaming.telephony.client.vonage_client import VonageClient
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.abstract_phone_conversation import (
    AbstractPhoneConversation,
)
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.utils.state_manager import VonagePhoneConversationStateManager

KOALA_CHUNK_SIZE = 512  # 16 bit samples, size 256


class VonagePhoneConversation(AbstractPhoneConversation[VonageOutputDevice]):
    telephony_provider = "vonage"

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
        vonage_uuid: str,
        vonage_config: VonageConfig,
        agent_factory: AbstractAgentFactory,
        transcriber_factory: AbstractTranscriberFactory,
        synthesizer_factory: AbstractSynthesizerFactory,
        conversation_id: Optional[str] = None,
        events_manager: Optional[EventsManager] = None,
        output_to_speaker: bool = False,
        speed_coefficient: float = 1.0,
        noise_suppression: bool = False,
    ):
        self.speed_coefficient = speed_coefficient
        super().__init__(
            direction=direction,
            speed_coefficient=speed_coefficient,
            from_phone=from_phone,
            to_phone=to_phone,
            base_url=base_url,
            config_manager=config_manager,
            output_device=VonageOutputDevice(output_to_speaker=output_to_speaker),
            agent_config=agent_config,
            transcriber_config=transcriber_config,
            synthesizer_config=synthesizer_config,
            conversation_id=conversation_id,
            events_manager=events_manager,
            transcriber_factory=transcriber_factory,
            agent_factory=agent_factory,
            synthesizer_factory=synthesizer_factory,
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

    def create_state_manager(self) -> VonagePhoneConversationStateManager:
        return VonagePhoneConversationStateManager(self)

    async def attach_ws_and_start(self, ws: WebSocket):
        # start message
        await ws.receive()
        super().attach_ws(ws)

        await self.start()
        self.events_manager.publish_event(
            PhoneCallConnectedEvent(
                conversation_id=self.id,
                to_phone_number=self.to_phone,
                from_phone_number=self.from_phone,
            )
        )
        disconnected = False
        while self.is_active():
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
                super().receive_audio(denoised_chunk)
                self.buffer = self.buffer[KOALA_CHUNK_SIZE:]
        else:
            super().receive_audio(chunk)
