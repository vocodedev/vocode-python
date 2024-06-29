import asyncio
from typing import List, Optional

import pytest
from pytest_mock import MockerFixture

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.agent.base_agent import (
    AgentResponse,
    AgentResponseMessage,
    BaseAgent,
    GeneratedResponse,
    TranscriptionAgentInput,
)
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.pipeline.worker import (
    InterruptibleAgentResponseEvent,
    InterruptibleEvent,
    QueueConsumer,
)
from vocode.streaming.utils.state_manager import ConversationStateManager


@pytest.fixture(autouse=True)
def mock_env(mocker: MockerFixture):
    mocker.patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "openai_api_key",
        },
    )


def _create_agent(
    mocker: MockerFixture,
    agent_config: ChatGPTAgentConfig,
    transcript: Optional[Transcript] = None,
    action_factory: Optional[AbstractActionFactory] = None,
    conversation_state_manager: Optional[ConversationStateManager] = None,
) -> ChatGPTAgent:
    agent = ChatGPTAgent(agent_config, action_factory=action_factory)
    if transcript:
        agent.attach_transcript(transcript)
    else:
        agent.attach_transcript(Transcript())
    if conversation_state_manager:
        agent.attach_conversation_state_manager(conversation_state_manager)
    else:
        agent.attach_conversation_state_manager(mocker.MagicMock())
    return agent


async def _consume_until_end_of_turn(
    agent_consumer: QueueConsumer[InterruptibleAgentResponseEvent[AgentResponse]],
    timeout: float = 0.1,
) -> List[AgentResponse]:
    agent_responses = []
    try:
        while True:
            agent_response = await asyncio.wait_for(
                agent_consumer.input_queue.get(), timeout=timeout
            )
            agent_responses.append(agent_response.payload)
            if isinstance(agent_response.payload, AgentResponseMessage) and isinstance(
                agent_response.payload.message, EndOfTurn
            ):
                break
    except asyncio.TimeoutError:
        pass
    return agent_responses


def _send_transcription(
    agent: BaseAgent,
    transcription: Transcription,
    agent_response_tracker: Optional[asyncio.Event] = None,
    is_interruptible: bool = False,
):
    agent.consume_nonblocking(
        InterruptibleEvent(
            payload=TranscriptionAgentInput(
                conversation_id="conversation_id", transcription=transcription
            ),
            is_interruptible=is_interruptible,
        )
    )


def _send_action_output(
    agent: BaseAgent,
    action_output: str,
    agent_response_tracker: Optional[asyncio.Event] = None,
    is_interruptible: bool = False,
):
    agent.consume_nonblocking(
        InterruptibleEvent(
            payload=action_output,
            is_interruptible=is_interruptible,
            agent_response_tracker=agent_response_tracker,
        )
    )


def _mock_generate_response(
    mocker: MockerFixture, agent: BaseAgent, generated_responses: List[GeneratedResponse]
):
    async def mock_generate_response(*args, **kwargs):
        for response in generated_responses:
            yield response

    mocker.patch.object(agent, "generate_response", mock_generate_response)


@pytest.mark.asyncio
async def test_generate_responses(mocker: MockerFixture):
    agent_config = ChatGPTAgentConfig(
        prompt_preamble="Have a pleasant conversation about life",
        generate_responses=True,
    )
    agent = _create_agent(mocker, agent_config)
    _mock_generate_response(
        mocker,
        agent,
        [
            GeneratedResponse(
                message=BaseMessage(text="Hi, how are you doing today?"), is_interruptible=True
            )
        ],
    )
    _send_transcription(
        agent,
        Transcription(message="Hello?", confidence=1.0, is_final=True),
    )
    agent_consumer = QueueConsumer()
    agent.agent_responses_consumer = agent_consumer
    agent.start()
    agent_responses = await _consume_until_end_of_turn(agent_consumer)
    agent.terminate()

    messages = [response.message for response in agent_responses]

    assert messages == [BaseMessage(text="Hi, how are you doing today?"), EndOfTurn()]


@pytest.mark.asyncio
async def test_function_call(mocker: MockerFixture):
    # TODO: assert that when we return a function call with a user message, it sends out a message alongside
    # an end of turn with the same agent response tracker
    pass


@pytest.mark.asyncio
async def test_action_response_agent_input(mocker: MockerFixture):
    # TODO: assert that the canned response is optionally sent if the action is not quiet
    # and that it goes through the normal flow when the action is not quiet
    pass
