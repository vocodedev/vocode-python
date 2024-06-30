import json
from typing import Any, Dict, Optional, Type

from pydantic.v1 import BaseModel

from vocode.streaming.action.external_actions_requester import (
    ExternalActionResponse,
    ExternalActionsRequester,
)
from vocode.streaming.action.streaming_conversation_action import StreamingConversationAction
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput, ExternalActionProcessingMode
from vocode.streaming.models.message import BaseMessage


class ExecuteExternalActionVocodeActionConfig(
    VocodeActionConfig, type="action_external"  # type: ignore
):
    processing_mode: ExternalActionProcessingMode
    name: str
    description: str
    url: str
    input_schema: str
    speak_on_send: bool
    speak_on_receive: bool
    signature_secret: str


class ExecuteExternalActionParameters(BaseModel):
    payload: Dict[str, Any]


class ExecuteExternalActionResponse(BaseModel):
    success: bool
    result: Optional[dict]


class ExecuteExternalAction(
    StreamingConversationAction[
        ExecuteExternalActionVocodeActionConfig,
        ExecuteExternalActionParameters,
        ExecuteExternalActionResponse,
    ]
):
    parameters_type: Type[ExecuteExternalActionParameters] = ExecuteExternalActionParameters
    response_type: Type[ExecuteExternalActionResponse] = ExecuteExternalActionResponse

    def __init__(
        self,
        action_config: ExecuteExternalActionVocodeActionConfig,
    ):
        self.description = action_config.description
        super().__init__(
            action_config,
            quiet=not action_config.speak_on_receive,
            should_respond="always" if action_config.speak_on_send else "never",
            is_interruptible=False,
        )
        self.external_actions_requester = ExternalActionsRequester(url=action_config.url)

    def _user_message_param_info(self):
        return {
            "type": "string",
            "description": (
                "A message to reply to the user with BEFORE we make the function call."
                "\nEssentially a live response informing them that the function is "
                "about to happen."
            ),
        }

    def create_action_params(self, params: Dict[str, Any]) -> ExecuteExternalActionParameters:
        return ExecuteExternalActionParameters(payload=params)

    def get_function_name(self) -> str:
        return self.action_config.name

    def get_parameters_schema(self) -> Dict[str, Any]:
        return json.loads(self.action_config.input_schema)

    async def send_external_action_request(
        self, action_input: ActionInput[ExecuteExternalActionParameters]
    ) -> ExternalActionResponse:
        return await self.external_actions_requester.send_request(
            payload=action_input.params.payload,
            signature_secret=self.action_config.signature_secret,
        )

    async def run(
        self, action_input: ActionInput[ExecuteExternalActionParameters]
    ) -> ActionOutput[ExecuteExternalActionResponse]:
        # TODO: this interruption handling needs to be refactored / DRYd
        if self.should_respond and action_input.user_message_tracker is not None:
            await action_input.user_message_tracker.wait()

        self.pipeline.agent.is_muted = True
        response = await self.send_external_action_request(action_input)
        self.pipeline.agent.is_muted = False

        # TODO (EA): pass specific context based on error
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=ExecuteExternalActionResponse(
                result=response.result if response else None, success=response.success
            ),
            canned_response=(
                BaseMessage(text=response.agent_message)
                if response and response.agent_message
                else None
            ),
        )
