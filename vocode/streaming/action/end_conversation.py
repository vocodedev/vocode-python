from typing import Type

from loguru import logger
from pydantic.v1 import BaseModel

from vocode.streaming.action.streaming_conversation_action import StreamingConversationAction
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput

_END_CONVERSATION_ACTION_DESCRIPTION = """
Ends the current conversation; use this when we should hang up. Use this
when: someone has indicated they expect the call to finish, and there are no further objectives
to complete. If the user says Goodbye or a different farewell phrase, this indicates that we should
hang up. Example farewell phrases: goodbye, so long, see you, later, take care, etc. People will
sometimes say a farewell phrase at the end of a longer sentence, so make sure to hang up then
as well. Example: \"This sounds great! Talk to you soon.\""""


class EndConversationParameters(BaseModel):
    pass


class EndConversationResponse(BaseModel):
    success: bool


class EndConversationVocodeActionConfig(
    VocodeActionConfig, type="action_end_conversation"  # type: ignore
):
    def action_attempt_to_string(self, input: ActionInput) -> str:
        assert isinstance(input.params, EndConversationParameters)
        return "Attempting to end conversation"

    def action_result_to_string(self, input: ActionInput, output: ActionOutput) -> str:
        assert isinstance(output.response, EndConversationResponse)
        if output.response.success:
            action_description = "Successfully ended conversation"
        else:
            action_description = "Did not end call because user interrupted"
        return action_description


class EndConversation(
    StreamingConversationAction[
        EndConversationVocodeActionConfig,
        EndConversationParameters,
        EndConversationResponse,
    ]
):
    description: str = _END_CONVERSATION_ACTION_DESCRIPTION
    parameters_type: Type[EndConversationParameters] = EndConversationParameters
    response_type: Type[EndConversationResponse] = EndConversationResponse

    def __init__(
        self,
        action_config: EndConversationVocodeActionConfig,
    ):
        super().__init__(
            action_config,
            quiet=True,
            should_respond="sometimes",
            is_interruptible=False,
        )

    async def _end_of_run_hook(self) -> None:
        """This method is called at the end of the run method. It is optional but intended to be
        overridden if needed."""
        pass

    async def run(
        self, action_input: ActionInput[EndConversationParameters]
    ) -> ActionOutput[EndConversationResponse]:
        if action_input.user_message_tracker is not None:
            await action_input.user_message_tracker.wait()

        if self.pipeline.transcript.was_last_message_interrupted():
            logger.info("Last bot message was interrupted")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=EndConversationResponse(success=False),
            )

        self.pipeline.mark_terminated()

        await self._end_of_run_hook()
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=EndConversationResponse(success=True),
        )
