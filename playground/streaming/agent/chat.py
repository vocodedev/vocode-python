import asyncio
import random
import typing

from dotenv import load_dotenv
from pydantic.v1 import BaseModel

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    EndOfTurn,
    PhraseBasedActionTrigger,
    PhraseBasedActionTriggerConfig,
    PhraseTrigger,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.utils.state_manager import AbstractConversationStateManager
from vocode.streaming.utils.worker import InterruptibleAgentResponseEvent, QueueConsumer

load_dotenv()

from vocode.streaming.agent import ChatGPTAgent
from vocode.streaming.agent.base_agent import (
    AgentResponse,
    AgentResponseMessage,
    AgentResponseType,
    BaseAgent,
    TranscriptionAgentInput,
)
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.utils import create_conversation_id

BACKCHANNELS = ["Got it", "Sure", "Okay", "I understand"]


class ShoutActionConfig(ActionConfig, type="shout"):  # type: ignore
    num_exclamation_marks: int


class ShoutActionParameters(BaseModel):
    pass


class ShoutActionResponse(BaseModel):
    success: bool


class ShoutAction(BaseAction[ShoutActionConfig, ShoutActionParameters, ShoutActionResponse]):
    description: str = "Shouts someone's name"
    parameters_type: typing.Type[ShoutActionParameters] = ShoutActionParameters
    response_type: typing.Type[ShoutActionResponse] = ShoutActionResponse

    async def run(
        self,
        action_input: ActionInput[ShoutActionParameters],
    ) -> ActionOutput[ShoutActionResponse]:
        print(f"HI THERE {self.action_config.num_exclamation_marks * '!'}")
        return ActionOutput(
            action_type=self.action_config.type,
            response=ShoutActionResponse(success=True),
        )


class ShoutActionFactory(AbstractActionFactory):
    def create_action(self, action_config: ActionConfig) -> BaseAction:
        if isinstance(action_config, ShoutActionConfig):
            return ShoutAction(action_config, should_respond="always")
        else:
            raise Exception("Invalid action type")


class DummyConversationManager(AbstractConversationStateManager):
    """For use with Agents operating in a non-call context."""

    def __init__(
        self,
        using_input_streaming_synthesizer: bool = False,
    ):
        self._using_input_streaming_synthesizer = using_input_streaming_synthesizer
        self._conversation_id = create_conversation_id()

    def using_input_streaming_synthesizer(self):
        return self._using_input_streaming_synthesizer

    def get_conversation_id(self):
        return self._conversation_id


async def run_agent(
    agent: BaseAgent, interruption_probability: float, backchannel_probability: float
):
    ended = False
    conversation_id = create_conversation_id()
    agent_response_queue: asyncio.Queue[InterruptibleAgentResponseEvent[AgentResponse]] = (
        asyncio.Queue()
    )
    agent_consumer = QueueConsumer(input_queue=agent_response_queue)
    agent.agent_responses_consumer = agent_consumer

    async def receiver():
        nonlocal ended

        assert agent.transcript is not None

        ignore_until_end_of_turn = False

        while not ended:
            try:
                event = await agent_response_queue.get()
                response = event.payload
                if response.type == AgentResponseType.FILLER_AUDIO:
                    print("Would have sent filler audio")
                elif response.type == AgentResponseType.STOP:
                    print("Agent returned stop")
                    ended = True
                    break
                elif response.type == AgentResponseType.MESSAGE:
                    agent_response = typing.cast(AgentResponseMessage, response)

                    if isinstance(agent_response.message, EndOfTurn):
                        ignore_until_end_of_turn = False
                        if random.random() < backchannel_probability:
                            backchannel = random.choice(BACKCHANNELS)
                            print("Human: " + f"[{backchannel}]")
                            agent.transcript.add_human_message(
                                backchannel,
                                conversation_id,
                                is_backchannel=True,
                            )
                    elif isinstance(agent_response.message, BaseMessage):
                        if ignore_until_end_of_turn:
                            continue

                        message_sent: str
                        is_final: bool
                        # TODO: consider allowing the user to interrupt the agent manually by responding fast
                        if random.random() < interruption_probability:
                            stop_idx = random.randint(0, len(agent_response.message.text))
                            message_sent = agent_response.message.text[:stop_idx]
                            ignore_until_end_of_turn = True
                            is_final = False
                        else:
                            message_sent = agent_response.message.text
                            is_final = True

                        agent.transcript.add_bot_message(
                            message_sent, conversation_id, is_final=is_final
                        )

                        print("AI: " + message_sent + ("-" if not is_final else ""))
            except asyncio.CancelledError:
                break

    async def sender():
        if agent.agent_config.initial_message is not None:
            agent.agent_responses_consumer.consume_nonblocking(
                InterruptibleAgentResponseEvent(
                    payload=AgentResponseMessage(message=agent.agent_config.initial_message),
                    agent_response_tracker=asyncio.Event(),
                )
            )
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
                            twilio_sid="0",
                            vonage_uuid=None,
                        )
                    )
                )
            except asyncio.CancelledError:
                break

    actions_worker = None
    if isinstance(agent, ChatGPTAgent):
        actions_worker = ActionsWorker(
            action_factory=agent.action_factory,
        )
        actions_worker.consumer = agent
        agent.actions_consumer = actions_worker
        actions_worker.attach_conversation_state_manager(agent.conversation_state_manager)
        actions_worker.start()

    await asyncio.gather(receiver(), sender())
    if actions_worker is not None:
        await actions_worker.terminate()


async def agent_main():
    transcript = Transcript()
    # Replace with your agent!
    agent = ChatGPTAgent(
        ChatGPTAgentConfig(
            prompt_preamble="Have a conversation",
            initial_message=BaseMessage(text="Is this Ajay?"),
            actions=[
                ShoutActionConfig(
                    num_exclamation_marks=3,
                    action_trigger=PhraseBasedActionTrigger(
                        type="action_trigger_phrase_based",
                        config=PhraseBasedActionTriggerConfig(
                            phrase_triggers=[
                                PhraseTrigger(
                                    phrase="shout",
                                    conditions=["phrase_condition_type_contains"],
                                ),
                            ],
                        ),
                    ),
                ),
            ],
        ),
        action_factory=ShoutActionFactory(),
    )
    agent.attach_conversation_state_manager(DummyConversationManager())
    agent.attach_transcript(transcript)
    agent.start()

    try:
        await run_agent(agent, interruption_probability=0, backchannel_probability=0)
    except KeyboardInterrupt:
        await agent.terminate()


if __name__ == "__main__":
    asyncio.run(agent_main())
