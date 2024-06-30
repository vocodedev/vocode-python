import base64
import json
import os

import pytest
from pytest_mock import MockerFixture

from tests.fakedata.conversation import (
    create_fake_agent,
    create_fake_streaming_conversation,
    create_fake_streaming_conversation_factory,
    create_fake_twilio_phone_conversation_with_streaming_conversation_pipeline,
    create_fake_vonage_phone_conversation_with_streaming_conversation_pipeline,
)
from tests.fakedata.id import generate_uuid
from vocode.streaming.action.execute_external_action import (
    ExecuteExternalAction,
    ExecuteExternalActionParameters,
    ExecuteExternalActionVocodeActionConfig,
)
from vocode.streaming.action.external_actions_requester import ExternalActionResponse
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.actions import (
    ActionInput,
    TwilioPhoneConversationActionInput,
    VonagePhoneConversationActionInput,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.telephony.conversation.twilio_phone_conversation import (
    TwilioPhoneConversation,
)
from vocode.streaming.telephony.conversation.vonage_phone_conversation import (
    VonagePhoneConversation,
)
from vocode.streaming.utils import create_conversation_id

ACTION_INPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "length": {"type": "string", "enum": ["30m", "1hr"]},
        "time": {"type": "string", "pattern": r"^\d{2}:\d0[ap]m$"},
    },
}


@pytest.fixture
def action_config() -> dict:
    """Provides a common action configuration for tests."""
    return {
        "processing_mode": "muted",
        "name": "name",
        "description": "A description",
        "url": "https://example.com",
        "input_schema": json.dumps(ACTION_INPUT_SCHEMA),
        "speak_on_send": True,
        "speak_on_receive": True,
        "signature_secret": base64.b64encode(os.urandom(32)).decode(),
    }


@pytest.fixture
def execute_action_setup(mocker, action_config) -> ExecuteExternalAction:
    """Common setup for creating an ExecuteExternalAction instance."""
    action = ExecuteExternalAction(
        action_config=ExecuteExternalActionVocodeActionConfig(**action_config),
    )
    mocked_requester = mocker.AsyncMock()
    mocked_requester.send_request.return_value = ExternalActionResponse(
        result={"test": "test"},
        agent_message="message!",
        success=True,
    )
    action.external_actions_requester = mocked_requester
    return action


@pytest.fixture
def mock_agent_with_execute_external_action(mocker: MockerFixture, action_config) -> BaseAgent:
    return create_fake_agent(
        mocker,
        agent_config=ChatGPTAgentConfig(prompt_preamble="", actions=[action_config]),
    )


@pytest.fixture
def mock_streaming_conversation(
    mocker: MockerFixture, mock_agent_with_execute_external_action: BaseAgent
) -> StreamingConversation:
    return create_fake_streaming_conversation(mocker, agent=mock_agent_with_execute_external_action)


@pytest.mark.asyncio
async def test_execute_external_action_success(
    mocker,
    mock_streaming_conversation: StreamingConversation,
    execute_action_setup: ExecuteExternalAction,
):
    mock_streaming_conversation.actions_worker.attach_state(execute_action_setup)
    response = await execute_action_setup.run(
        action_input=ActionInput(
            action_config=execute_action_setup.action_config,
            conversation_id=create_conversation_id(),
            params=ExecuteExternalActionParameters(payload={}),
        ),
    )

    assert response.response.success
    assert response.response.result == {"test": "test"}
