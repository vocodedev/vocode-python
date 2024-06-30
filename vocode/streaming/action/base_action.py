import asyncio
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Literal, Optional, Type, TypeVar

import jsonschema
from jsonschema import Draft202012Validator as SchemaValidator

from vocode.streaming.action.action_utils import exclude_keys_recursive
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ParametersType,
    ResponseType,
)

if TYPE_CHECKING:
    from vocode.streaming.pipeline.audio_pipeline import AudioPipeline

ActionConfigType = TypeVar("ActionConfigType", bound=ActionConfig)


class BaseAction(Generic[ActionConfigType, ParametersType, ResponseType]):  # type: ignore
    description: str = ""
    pipeline: "AudioPipeline"

    def __init__(
        self,
        action_config: ActionConfigType,
        should_respond: Literal["always", "sometimes", "never"] = "never",
        quiet: bool = False,
        is_interruptible: bool = True,
        **kwargs,
    ):
        self.action_config = action_config
        self.should_respond = should_respond
        self.quiet = quiet
        self.is_interruptible = is_interruptible

    async def run(self, action_input: ActionInput[ParametersType]) -> ActionOutput[ResponseType]:
        raise NotImplementedError

    @property
    def parameters_type(self) -> Type[ParametersType]:
        raise NotImplementedError

    @property
    def response_type(self) -> Type[ResponseType]:
        raise NotImplementedError

    def get_parameters_schema(self) -> Dict[str, Any]:
        return self.parameters_type.schema()

    def get_function_name(self) -> str:
        return self.action_config.type

    def get_openai_function(self):
        parameters_schema = self.get_parameters_schema()
        parameters_schema = exclude_keys_recursive(parameters_schema, {"title"})
        if self.should_respond != "never":
            parameters_schema["properties"]["user_message"] = self._user_message_param_info()
            if self.should_respond == "always":
                required: List[str] = parameters_schema.get("required", [])
                required.append("user_message")
                parameters_schema["required"] = required
        try:
            SchemaValidator.check_schema(parameters_schema)
        except jsonschema.exceptions.SchemaError as e:
            raise ValueError(f"Invalid parameters schema for {self.action_config.type}: {e}") from e

        return {
            "name": self.get_function_name(),
            "description": self.description,
            "parameters": parameters_schema,
        }

    def create_action_params(self, params: Dict[str, Any]) -> ParametersType:
        return self.parameters_type(**params)

    def create_action_input(
        self,
        conversation_id: str,
        params: Dict[str, Any],
        user_message_tracker: Optional[asyncio.Event] = None,
    ) -> ActionInput[ParametersType]:
        if "user_message" in params:
            del params["user_message"]
        return ActionInput(
            action_config=self.action_config,
            conversation_id=conversation_id,
            params=self.create_action_params(params),
            user_message_tracker=user_message_tracker,
        )

    def _user_message_param_info(self):
        return {
            "type": "string",
            "description": """A message to reply to the user with BEFORE we make the function call.
                    Essentially a live response informing them that the function is about to happen.
                    Eg Let me check the weather in San Francisco CA for you """,
        }
