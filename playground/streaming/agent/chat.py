import asyncio
import typing
from dotenv import load_dotenv
from playground.streaming.tracing_utils import make_parser_and_maybe_trace
from pydantic import BaseModel
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.utils.state_manager import ConversationStateManager
from vocode.streaming.utils.worker import InterruptibleTrackedEvent

load_dotenv()

from vocode.streaming.agent import ChatGPTAgent
from vocode.streaming.agent.base_agent import (
    BaseAgent,
    AgentResponseMessage,
    AgentResponseType,
    TranscriptionAgentInput,
)

from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.utils import create_conversation_id


class ShoutActionConfig(ActionConfig, type="shout"):  # type: ignore
    num_exclamation_marks: int


class ShoutActionParameters(BaseModel):
    name: str


class ShoutActionResponse(BaseModel):
    success: bool


class ShoutAction(
    BaseAction[ShoutActionConfig, ShoutActionParameters, ShoutActionResponse]
):
    description: str = "Shouts someone's name"
    parameters_type: typing.Type[ShoutActionParameters] = ShoutActionParameters
    response_type: typing.Type[ShoutActionResponse] = ShoutActionResponse

    async def run(
        self, action_input: ActionInput[ShoutActionParameters]
    ) -> ActionOutput[ShoutActionResponse]:
        print(
            f"HI THERE {action_input.params.name}{self.action_config.num_exclamation_marks * '!'}"
        )
        return ActionOutput(
            action_type=self.action_config.type,
            response=ShoutActionResponse(success=True),
        )


class ShoutActionFactory(ActionFactory):
    def create_action(self, action_config: ActionConfig) -> BaseAction:
        if isinstance(action_config, ShoutActionConfig):
            return ShoutAction(action_config, should_respond=True)
        else:
            raise Exception("Invalid action type")


class DummyConversationManager(ConversationStateManager):
    pass


async def run_agent(agent: BaseAgent):
    ended = False
    conversation_id = create_conversation_id()

    async def receiver():
        nonlocal ended
        while not ended:
            try:
                event = await agent.get_output_queue().get()
                response = event.payload
                if response.type == AgentResponseType.FILLER_AUDIO:
                    print("Would have sent filler audio")
                elif response.type == AgentResponseType.STOP:
                    print("Agent returned stop")
                    ended = True
                    break
                elif response.type == AgentResponseType.MESSAGE:
                    agent_response = typing.cast(AgentResponseMessage, response)

                    agent.transcript.add_bot_message(
                        agent_response.message.text, conversation_id
                    )
                    print(
                        "AI: "
                        + typing.cast(AgentResponseMessage, response).message.text
                    )
            except asyncio.CancelledError:
                break

    async def sender():
        while not ended:
            try:
                message = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("Human: ")
                )
                agent.consume_nonblocking(
                    agent.interruptible_event_factory.create_interruptible_event(
                        TranscriptionAgentInput(
                            transcription=Transcription(
                                message=message, confidence=1.0, is_final=True
                            ),
                            conversation_id=conversation_id,
                        )
                    )
                )
            except asyncio.CancelledError:
                break

    actions_worker = None
    if isinstance(agent, ChatGPTAgent):
        actions_worker = ActionsWorker(
            input_queue=agent.actions_queue,
            output_queue=agent.get_input_queue(),
            action_factory=agent.action_factory,
        )
        actions_worker.attach_conversation_state_manager(
            agent.conversation_state_manager
        )
        actions_worker.start()

    await asyncio.gather(receiver(), sender())
    if actions_worker is not None:
        actions_worker.terminate()


async def agent_main():
    transcript = Transcript()
    # Replace with your agent!
    agent = ChatGPTAgent(
        ChatGPTAgentConfig(
            prompt_preamble="have a conversation",
            actions=[
                ShoutActionConfig(num_exclamation_marks=3),
            ],
        ),
        action_factory=ShoutActionFactory(),
    )
    agent.attach_conversation_state_manager(DummyConversationManager(conversation=None))
    agent.attach_transcript(transcript)
    if agent.agent_config.initial_message is not None:
        agent.output_queue.put_nowait(
            InterruptibleTrackedEvent(
                payload=AgentResponseMessage(
                    message=agent.agent_config.initial_message
                ),
                event_tracker=asyncio.Event(),
            )
        )
    agent.start()

    try:
        await run_agent(agent)
    except KeyboardInterrupt:
        agent.terminate()


if __name__ == "__main__":
    make_parser_and_maybe_trace()
    asyncio.run(agent_main())
