import logging
from typing import Callable, Optional
import typing

from fastapi import APIRouter, WebSocket
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.client_backend import InputAudioConfig, OutputAudioConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.models.websocket import (
    AudioConfigStartMessage,
    AudioMessage,
    ReadyMessage,
    WebSocketMessage,
    WebSocketMessageType,
)

from vocode.streaming.output_device.websocket_output_device import WebsocketOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.utils.base_router import BaseRouter

from vocode.streaming.models.events import Event, EventType
from vocode.streaming.models.transcript import TranscriptEvent
from vocode.streaming.utils import events_manager


class ConversationRouter(BaseRouter):
    def __init__(
        self,
        agent: BaseAgent,
        transcriber_thunk: Callable[
            [InputAudioConfig], BaseTranscriber
        ] = lambda input_audio_config: DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_audio_config(
                input_audio_config=input_audio_config,
                endpointing_config=PunctuationEndpointingConfig(),
            )
        ),
        synthesizer_thunk: Callable[
            [OutputAudioConfig], BaseSynthesizer
        ] = lambda output_audio_config: AzureSynthesizer(
            AzureSynthesizerConfig.from_output_audio_config(
                output_audio_config=output_audio_config
            )
        ),
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.sessions: typing.Dict[str, StreamingConversation] = {}
        self.transcriber_thunk = transcriber_thunk
        self.agent = agent
        self.synthesizer_thunk = synthesizer_thunk
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.router.websocket("/conversation")(self.conversation)

    async def get_conversation(
        self,
        output_device: WebsocketOutputDevice,
        start_message: AudioConfigStartMessage,
    ) -> typing.Union[StreamingConversation, None]:
        if start_message.conversation_id is None:
            return None
        conversation = self.sessions.get(start_message.conversation_id)
        if conversation is None:
            return None
        self.logger.debug(f"get_conversation restart...")
        await conversation.restart(output_device, lambda: output_device.ws.send_text(ReadyMessage().json()))
        return conversation
        
    async def new_conversation(
        self,
        output_device: WebsocketOutputDevice,
        start_message: AudioConfigStartMessage,
    ) -> StreamingConversation:
        transcriber = self.transcriber_thunk(start_message.input_audio_config)
        synthesizer = self.synthesizer_thunk(start_message.output_audio_config)
        synthesizer.synthesizer_config.should_encode_as_wav = True
        conversation =  StreamingConversation(
            output_device=output_device,
            transcriber=transcriber,
            agent=self.agent,
            synthesizer=synthesizer,
            conversation_id=start_message.conversation_id,
            events_manager=TranscriptEventManager(output_device, self.logger) if start_message.subscribe_transcript else None,
            logger=self.logger,
        )
        if start_message.conversation_id:
            self.sessions[start_message.conversation_id] = conversation
        await conversation.start(lambda: output_device.ws.send_text(ReadyMessage().json()))
        return conversation
    
    def del_conversation(self, conversation):
        keys = list(self.sessions.keys())
        for key in keys:
            if self.sessions[key] == conversation:
                del self.sessions[key]

    async def conversation(self, websocket: WebSocket):
        await websocket.accept()
        start_message: AudioConfigStartMessage = AudioConfigStartMessage.parse_obj(
            await websocket.receive_json()
        )
        self.logger.debug(f"Conversation started")
        # self.logger.debug(start_message.dict())
        output_device = WebsocketOutputDevice(
            websocket,
            start_message.output_audio_config.sampling_rate,
            start_message.output_audio_config.audio_encoding,
        )
        conversation = await self.get_conversation(output_device, start_message)
        if conversation is None:
            conversation = await self.new_conversation(output_device, start_message)
        while conversation.is_active():
            message: WebSocketMessage = WebSocketMessage.parse_obj(
                await websocket.receive_json()
            )
            if message.type == WebSocketMessageType.STOP:
                break
            audio_message = typing.cast(AudioMessage, message)
            conversation.receive_audio(audio_message.get_bytes())
        output_device.mark_closed()
        self.del_conversation(conversation)
        conversation.terminate()

    def get_router(self) -> APIRouter:
        return self.router

class TranscriptEventManager(events_manager.EventsManager):
    def __init__(self, output_device: WebsocketOutputDevice, logger: Optional[logging.Logger] = None):
        super().__init__(subscriptions=[EventType.TRANSCRIPT])
        self.output_device = output_device
        self.logger = logger or logging.getLogger(__name__)

    def handle_event(self, event: Event):
        if event.type == EventType.TRANSCRIPT:
            transcript_event = typing.cast(TranscriptEvent, event)
            self.output_device.consume_transcript(transcript_event)
            # self.logger.debug(event.dict())

    def restart(self, output_device: WebsocketOutputDevice):
        self.output_device = output_device