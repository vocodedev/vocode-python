import asyncio
from enum import Enum
from typing import Annotated, Generic, List, Literal, Optional, TypeVar, Union

from pydantic.v1 import BaseModel, Field

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.model import TypedModel

TriggerType = Literal["action_trigger_function_call", "action_trigger_phrase_based"]


class ActionTriggerConfig(BaseModel):
    pass


class _ActionTrigger(BaseModel):
    type: TriggerType
    config: ActionTriggerConfig


class FunctionCallActionTriggerConfig(ActionTriggerConfig):
    pass


class FunctionCallActionTrigger(_ActionTrigger):
    type: Literal["action_trigger_function_call"] = "action_trigger_function_call"
    config: FunctionCallActionTriggerConfig = Field(default_factory=FunctionCallActionTriggerConfig)


PhraseConditionType = Literal["phrase_condition_type_contains"]


class PhraseTrigger(BaseModel):
    phrase: str
    conditions: List[PhraseConditionType]


class PhraseBasedActionTriggerConfig(ActionTriggerConfig):
    phrase_triggers: List[PhraseTrigger]


class PhraseBasedActionTrigger(_ActionTrigger):
    type: Literal["action_trigger_phrase_based"] = "action_trigger_phrase_based"
    config: PhraseBasedActionTriggerConfig


ActionTrigger = Annotated[
    Union[FunctionCallActionTrigger, PhraseBasedActionTrigger],
    Field(discriminator="type"),
]


class ActionType(str, Enum):
    BASE = "action_base"
    NYLAS_SEND_EMAIL = "action_nylas_send_email"
    WAIT = "action_wait"
    RECORD_EMAIL = "action_record_email"
    END_CONVERSATION = "action_end_conversation"
    EXECUTE_EXTERNAL_ACTION = "action_external"

    TRANSFER_CALL = "action_transfer_call"
    DTMF = "action_dtmf"


ParametersType = TypeVar("ParametersType", bound=BaseModel)

ACTION_STARTED_FORMAT_STRING = "!STARTING ACTION {action_name} WITH PARAMETERS {action_params}!"
ACTION_FINISHED_FORMAT_STRING = "!ACTION {action_name} FINISHED WITH OUTPUT {action_output}!"


class ActionConfig(TypedModel, type=ActionType.BASE):  # type: ignore
    action_trigger: ActionTrigger = FunctionCallActionTrigger(type="action_trigger_function_call")

    def action_attempt_to_string(self, input: "ActionInput") -> str:
        return ACTION_STARTED_FORMAT_STRING.format(
            action_name=self.type,
            action_params=input.params.json(),
        )

    def action_result_to_string(self, input: "ActionInput", output: "ActionOutput") -> str:
        return ACTION_FINISHED_FORMAT_STRING.format(
            action_name=self.type,
            action_output=output.response.json(),
        )


class ActionInput(BaseModel, Generic[ParametersType]):
    action_config: ActionConfig
    conversation_id: str
    params: ParametersType
    user_message_tracker: Optional[asyncio.Event] = None

    class Config:
        arbitrary_types_allowed = True


class FunctionFragment(BaseModel):
    name: str
    arguments: str


class FunctionCall(BaseModel):
    name: str
    arguments: str


class EndOfTurn(BaseModel):
    pass


class VonagePhoneConversationActionInput(ActionInput[ParametersType]):
    vonage_uuid: str


class TwilioPhoneConversationActionInput(ActionInput[ParametersType]):
    twilio_sid: str


ResponseType = TypeVar("ResponseType", bound=BaseModel)


class ActionOutput(BaseModel, Generic[ResponseType]):
    action_type: str
    canned_response: Optional[BaseMessage] = None
    response: ResponseType


class ActionResponse(BaseModel, Generic[ParametersType, ResponseType]):
    conversation_id: str
    action_input: ActionInput
    action_output: ActionOutput
    is_quiet: bool = False


ExternalActionProcessingMode = Literal["muted"]
