import asyncio
from typing import Dict, Any, Optional
from vocode.streaming.action.base_action import ActionConfigType, BaseAction
from vocode.streaming.models.actions import (
    ActionInput,
    ActionOutput,
    ParametersType,
    ResponseType,
    TwilioPhoneCallActionInput,
    VonagePhoneCallActionInput,
)
from vocode.streaming.utils.worker import EventTracker


class VonagePhoneCallAction(BaseAction[ActionConfigType, ParametersType, ResponseType]):
    def create_phone_call_action_input(
        self,
        conversation_id: str,
        params: Dict[str, Any],
        vonage_uuid: str,
        user_message_tracker: Optional[EventTracker] = None,
    ) -> VonagePhoneCallActionInput[ParametersType]:
        if "user_message" in params:
            del params["user_message"]
        return VonagePhoneCallActionInput(
            action_config=self.action_config,
            conversation_id=conversation_id,
            params=self.parameters_type(**params),
            vonage_uuid=vonage_uuid,
            user_message_tracker=user_message_tracker,
        )

    def get_vonage_uuid(self, action_input: ActionInput[ParametersType]) -> str:
        assert isinstance(action_input, VonagePhoneCallActionInput)
        return action_input.vonage_uuid


class TwilioPhoneCallAction(BaseAction[ActionConfigType, ParametersType, ResponseType]):
    def create_phone_call_action_input(
        self,
        conversation_id: str,
        params: Dict[str, Any],
        twilio_sid: str,
        user_message_tracker: Optional[EventTracker] = None,
    ) -> TwilioPhoneCallActionInput[ParametersType]:
        if "user_message" in params:
            del params["user_message"]
        return TwilioPhoneCallActionInput(
            action_config=self.action_config,
            conversation_id=conversation_id,
            params=self.parameters_type(**params),
            twilio_sid=twilio_sid,
            user_message_tracker=user_message_tracker,
        )

    def get_twilio_sid(self, action_input: ActionInput[ParametersType]) -> str:
        assert isinstance(action_input, TwilioPhoneCallActionInput)
        return action_input.twilio_sid
