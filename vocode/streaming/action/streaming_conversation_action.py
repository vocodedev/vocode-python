from typing import TYPE_CHECKING

from vocode.streaming.action.base_action import ActionConfigType, BaseAction
from vocode.streaming.models.actions import (
    ActionInput,
    ParametersType,
    ResponseType,
    TwilioPhoneConversationActionInput,
    VonagePhoneConversationActionInput,
)

if TYPE_CHECKING:
    from vocode.streaming.streaming_conversation import StreamingConversation


class StreamingConversationAction(BaseAction[ActionConfigType, ParametersType, ResponseType]):
    pipeline: "StreamingConversation"
