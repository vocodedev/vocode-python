from enum import Enum
from typing import Generic, TypeVar
from pydantic import BaseModel


class ActionType(str, Enum):
    BASE = "action_base"
    NYLAS_SEND_EMAIL = "action_nylas_send_email"
    PINECONE_LOOKUP = "action_pinecone_lookup"


ParametersType = TypeVar("ParametersType", bound=BaseModel)


class ActionInput(BaseModel, Generic[ParametersType]):
    action_type: str
    conversation_id: str
    params: ParametersType

class FunctionFragment(BaseModel):
        name: str
        arguments: str

class FunctionCall(BaseModel):
        name: str
        arguments: str

class VonagePhoneCallActionInput(ActionInput[ParametersType]):
    vonage_uuid: str


class TwilioPhoneCallActionInput(ActionInput[ParametersType]):
    twilio_sid: str


ResponseType = TypeVar("ResponseType", bound=BaseModel)


class ActionOutput(BaseModel, Generic[ResponseType]):
    action_type: str
    response: ResponseType
