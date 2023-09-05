import asyncio
from typing import TYPE_CHECKING, Optional
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import EndpointingConfig
from vocode.streaming.agent.base_agent import AgentResponseMessage
from vocode.streaming.utils.worker import EventTracker

if TYPE_CHECKING:
    from vocode.streaming.streaming_conversation import StreamingConversation
    from vocode.streaming.telephony.conversation.vonage_call import VonageCall
    from vocode.streaming.telephony.conversation.twilio_call import TwilioCall


class ConversationStateManager:
    def __init__(self, conversation: "StreamingConversation"):
        self._conversation = conversation

    def get_transcriber_endpointing_config(self) -> Optional[EndpointingConfig]:
        return (
            self._conversation.transcriber.get_transcriber_config().endpointing_config
        )

    def set_transcriber_endpointing_config(self, endpointing_config: EndpointingConfig):
        assert self.get_transcriber_endpointing_config() is not None
        self._conversation.transcriber.get_transcriber_config().endpointing_config = (
            endpointing_config
        )

    def disable_synthesis(self):
        self._conversation.synthesis_enabled = False

    def enable_synthesis(self):
        self._conversation.synthesis_enabled = True

    def mute_agent(self):
        self._conversation.agent.is_muted = True

    def unmute_agent(self):
        self._conversation.agent.is_muted = False

    async def terminate_conversation(self):
        await self._conversation.terminate()

    def send_bot_message(self, message: BaseMessage) -> asyncio.Event:
        # returns an asyncio.Event that will be set when the agent has finished uttering the message
        event_tracker = EventTracker("action_bot_message")
        self._conversation.agent.produce_interruptible_tracked_event_nonblocking(
            item=AgentResponseMessage(
                message=message,
                is_interruptible=False,
            ),
            is_interruptible=False,
            event_tracker=event_tracker,
        )
        return event_tracker.done_event


class VonageCallStateManager(ConversationStateManager):
    def __init__(self, call: "VonageCall"):
        super().__init__(call)
        self._call = call


class TwilioCallStateManager(ConversationStateManager):
    def __init__(self, call: "TwilioCall"):
        super().__init__(call)
        self._call = call
