import asyncio
from typing import Generator, Type
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic.v1 import BaseModel
from pytest_mock import MockerFixture

from tests.fakedata.conversation import create_fake_agent, create_fake_streaming_conversation
from tests.fakedata.id import generate_uuid
from vocode.streaming.action.end_conversation import (
    EndConversation,
    EndConversationParameters,
    EndConversationVocodeActionConfig,
)
from vocode.streaming.models.actions import (
    ActionInput,
    TwilioPhoneConversationActionInput,
    VonagePhoneConversationActionInput,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.utils import create_conversation_id


class EndConversationActionTestCase(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    action: EndConversation
    conversation_id: str


@pytest.fixture
def end_conversation_action_test_case(mocker: MockerFixture) -> EndConversationActionTestCase:
    action = EndConversation(action_config=EndConversationVocodeActionConfig())
    return EndConversationActionTestCase(
        action=action,
        conversation_id=create_conversation_id(),
    )


@pytest.fixture
def user_message_tracker() -> asyncio.Event:
    tracker = asyncio.Event()
    tracker.set()
    return tracker


@pytest.fixture
def mock_streaming_conversation_with_end_conversation_action(
    mocker, end_conversation_action_test_case: EndConversationActionTestCase
):
    mock_streaming_conversation = create_fake_streaming_conversation(
        mocker,
        agent=create_fake_agent(
            mocker,
            agent_config=ChatGPTAgentConfig(
                prompt_preamble="", actions=[end_conversation_action_test_case.action.action_config]
            ),
        ),
    )
    mock_streaming_conversation.actions_worker.attach_state(
        end_conversation_action_test_case.action
    )
    mock_streaming_conversation.active = True
    return mock_streaming_conversation


@pytest.mark.asyncio
async def test_end_conversation_success(
    mocker: MockerFixture,
    mock_env: Generator,
    mock_streaming_conversation_with_end_conversation_action: StreamingConversation,
    end_conversation_action_test_case: EndConversationActionTestCase,
    user_message_tracker: asyncio.Event,
):

    action_input = ActionInput(
        action_config=EndConversationVocodeActionConfig(),
        conversation_id=end_conversation_action_test_case.conversation_id,
        params=EndConversationParameters(),
        user_message_tracker=user_message_tracker,
    )

    response = await end_conversation_action_test_case.action.run(action_input=action_input)

    assert response.response.success
    assert not mock_streaming_conversation_with_end_conversation_action.is_active()


@pytest.mark.asyncio
async def test_end_conversation_fails_if_interrupted(
    mocker: MockerFixture,
    mock_env: Generator,
    mock_streaming_conversation_with_end_conversation_action: StreamingConversation,
    end_conversation_action_test_case: EndConversationActionTestCase,
    user_message_tracker: asyncio.Event,
):
    mock_streaming_conversation_with_end_conversation_action.transcript.add_bot_message(
        "Unfinished", conversation_id=end_conversation_action_test_case.conversation_id
    )

    action_input = ActionInput(
        action_config=EndConversationVocodeActionConfig(),
        conversation_id=end_conversation_action_test_case.conversation_id,
        params=EndConversationParameters(),
        user_message_tracker=user_message_tracker,
    )

    response = await end_conversation_action_test_case.action.run(action_input=action_input)

    assert not response.response.success
    assert mock_streaming_conversation_with_end_conversation_action.is_active()
