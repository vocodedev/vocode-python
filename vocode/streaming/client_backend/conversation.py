import typing
from typing import Callable

from fastapi import APIRouter, WebSocket
from loguru import logger

from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.client_backend import InputAudioConfig, OutputAudioConfig
from vocode.streaming.models.events import Event, EventType
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.models.transcript import TranscriptEvent
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
from vocode.streaming.utils import events_manager
from vocode.streaming.utils.base_router import BaseRouter

BASE_CONVERSATION_ENDPOINT = "/conversation"


class ConversationRouter(BaseRouter):
    def __init__(
        self,
        agent_thunk: Callable[[], BaseAgent],
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
            AzureSynthesizerConfig.from_output_audio_config(output_audio_config=output_audio_config)
        ),
        conversation_endpoint: str = BASE_CONVERSATION_ENDPOINT,
    ):
        super().__init__()
        self.transcriber_thunk = transcriber_thunk
        self.agent_thunk = agent_thunk
        self.synthesizer_thunk = synthesizer_thunk
        self.router = APIRouter()
        self.router.websocket(conversation_endpoint)(self.conversation)

    def get_conversation(
        self,
        output_device: WebsocketOutputDevice,
        start_message: AudioConfigStartMessage,
    ) -> StreamingConversation:
        transcriber = self.transcriber_thunk(start_message.input_audio_config)
        synthesizer = self.synthesizer_thunk(start_message.output_audio_config)
        synthesizer.get_synthesizer_config().should_encode_as_wav = True
        return StreamingConversation(
            output_device=output_device,
            transcriber=transcriber,
            agent=self.agent_thunk(),
            synthesizer=synthesizer,
            conversation_id=start_message.conversation_id,
            events_manager=(
                TranscriptEventManager(output_device)
                if start_message.subscribe_transcript
                else None
            ),
        )

    async def conversation(self, websocket: WebSocket):
        await websocket.accept()
        start_message: AudioConfigStartMessage = AudioConfigStartMessage.parse_obj(
            await websocket.receive_json()
        )
        logger.debug(f"Conversation started")
        output_device = WebsocketOutputDevice(
            websocket,
            start_message.output_audio_config.sampling_rate,
            start_message.output_audio_config.audio_encoding,
        )
        conversation = self.get_conversation(output_device, start_message)
        await conversation.start(lambda: websocket.send_text(ReadyMessage().json()))
        while conversation.is_active():
            message: WebSocketMessage = WebSocketMessage.parse_obj(await websocket.receive_json())
            if message.type == WebSocketMessageType.STOP:
                break
            audio_message = typing.cast(AudioMessage, message)
            conversation.receive_audio(audio_message.get_bytes())
        output_device.mark_closed()
        await conversation.terminate()

    def get_router(self) -> APIRouter:
        return self.router


class TranscriptEventManager(events_manager.EventsManager):
    def __init__(
        self,
        output_device: WebsocketOutputDevice,
    ):
        super().__init__(subscriptions=[EventType.TRANSCRIPT])
        self.output_device = output_device

    async def handle_event(self, event: Event):
        if event.type == EventType.TRANSCRIPT:
            transcript_event = typing.cast(TranscriptEvent, event)
            await self.output_device.send_transcript(transcript_event)
            # logger.debug(event.dict())

    def restart(self, output_device: WebsocketOutputDevice):
        self.output_device = output_device
