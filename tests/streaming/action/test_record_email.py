import pytest

from tests.fakedata.id import generate_uuid
from vocode.streaming.action.record_email import (
    RecordEmail,
    RecordEmailParameters,
    RecordEmailVocodeActionConfig,
)
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.utils import create_conversation_id

# id is just a description of the parameterized test case's input
EMAIL_TEST_CASES = [
    pytest.param("kian@vocode.dev", True, id="valid_email_with_dev"),
    pytest.param("kian+tester@vocode.net", True, id="valid_email_with_plus_in_local_part"),
    pytest.param("kian-tester@vocode.org", True, id="valid_email_with_dash_in_local_part"),
    pytest.param("kian.vocode@dev.com", True, id="valid_email_with_dot_in_local_part"),
    pytest.param("kian@vocode", False, id="missing_tld"),
    pytest.param("kian", False, id="missing_at_and_tld"),
    pytest.param("kian@", False, id="missing_domain"),
    pytest.param("@vocode.dev", False, id="missing_local_part"),
    pytest.param(".kian@vocode.dev", False, id="leading_dot_in_local_part"),
    pytest.param("kian.@vocode.dev", False, id="trailing_dot_in_local_part"),
    pytest.param("kian..tester@vocode.com", False, id="consecutive_dots_in_local_part"),
    pytest.param("kian@vo..code.com", False, id="consecutive_dots_in_domain"),
    pytest.param("kian@vocode.com.", False, id="trailing_dot_in_domain"),
    pytest.param("ki'an@vocode.net", False, id="apostrophe_in_local_part"),
    pytest.param("kian@vocode..org", False, id="consecutive_dots_in_tld"),
    pytest.param("kian@.vocode.net", False, id="leading_dot_in_domain"),
]


@pytest.fixture
def record_email_action() -> RecordEmail:
    return RecordEmail(action_config=RecordEmailVocodeActionConfig())


@pytest.mark.asyncio
@pytest.mark.parametrize("email_input,expected_success", EMAIL_TEST_CASES)
async def test_email_validation(
    record_email_action: RecordEmail, email_input: str, expected_success: bool
):
    res = await record_email_action.run(
        action_input=ActionInput(
            action_config=RecordEmailVocodeActionConfig(),
            conversation_id=create_conversation_id(),
            params=RecordEmailParameters(
                raw_value="",
                formatted_value=email_input,
            ),
        ),
    )
    assert res.response.success == expected_success
