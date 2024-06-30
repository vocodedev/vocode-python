from typing import Type

from loguru import logger
from pydantic.v1 import BaseModel, Field

from vocode.streaming.action.phone_call_action import (
    TwilioPhoneConversationAction,
    VonagePhoneConversationAction,
)
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput


class DTMFParameters(BaseModel):
    buttons: str = Field(..., description="The buttons to press.")


class DTMFResponse(BaseModel):
    success: bool


class DTMFVocodeActionConfig(VocodeActionConfig, type="action_dtmf"):  # type: ignore
    def action_attempt_to_string(self, input: ActionInput) -> str:
        assert isinstance(input.params, DTMFParameters)
        return "Attempting to press numbers: " f"{list(input.params.buttons)}"

    def action_result_to_string(self, input: ActionInput, output: ActionOutput) -> str:
        assert isinstance(input.params, DTMFParameters)
        assert isinstance(output.response, DTMFResponse)
        return f"Pressed numbers {list(input.params.buttons)} successfully"


FUNCTION_DESCRIPTION = "Presses a string numbers using DTMF tones."


class VonageDTMF(
    VonagePhoneConversationAction[DTMFVocodeActionConfig, DTMFParameters, DTMFResponse]
):
    description: str = FUNCTION_DESCRIPTION
    parameters_type: Type[DTMFParameters] = DTMFParameters
    response_type: Type[DTMFResponse] = DTMFResponse

    def __init__(self, action_config: DTMFVocodeActionConfig):
        super().__init__(action_config, quiet=True)

    async def run(self, action_input: ActionInput[DTMFParameters]) -> ActionOutput[DTMFResponse]:
        buttons = action_input.params.buttons
        vonage_client = self.vonage_phone_conversation.create_vonage_client()
        await vonage_client.send_dtmf(
            vonage_uuid=self.vonage_phone_conversation.vonage_uuid, digits=buttons
        )

        return ActionOutput(
            action_type=action_input.action_config.type,
            response=DTMFResponse(success=True),
        )


class TwilioDTMF(
    TwilioPhoneConversationAction[DTMFVocodeActionConfig, DTMFParameters, DTMFResponse]
):
    description: str = FUNCTION_DESCRIPTION
    parameters_type: Type[DTMFParameters] = DTMFParameters
    response_type: Type[DTMFResponse] = DTMFResponse

    def __init__(self, action_config: DTMFVocodeActionConfig):
        super().__init__(
            action_config,
            quiet=True,
        )

    async def run(self, action_input: ActionInput[DTMFParameters]) -> ActionOutput[DTMFResponse]:
        logger.error("DTMF not yet supported with Twilio")
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=DTMFResponse(success=False),
        )
