import asyncio
from typing import List
from unittest.mock import MagicMock

import pytest
from pydantic.v1 import BaseModel
from pytest_mock import MockerFixture

from tests.fakedata.conversation import (
    DEFAULT_CHAT_GPT_AGENT_CONFIG,
    create_fake_agent,
    create_fake_streaming_conversation,
)
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.models.agent import InterruptSensitivity
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.models.transcript import ActionStart, Message, Transcript
from vocode.streaming.utils.worker import AsyncWorker


class ShouldIgnoreUtteranceTestCase(BaseModel):
    transcript: Transcript
    human_transcription: str
    expected: bool


async def _consume_worker_output(worker: AsyncWorker, timeout: float = 0.1):
    try:
        return await asyncio.wait_for(worker.output_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def test_interrupt_sensitivity(mocker: MockerFixture):
    from vocode.streaming.streaming_conversation import StreamingConversation

    mock_instance = mocker.MagicMock(spec=StreamingConversation.TranscriptionsWorker)
    mocker.patch.object(
        mock_instance,
        "is_transcription_backchannel",
        wraps=StreamingConversation.TranscriptionsWorker.is_transcription_backchannel,
    )
    mock_instance.conversation = mocker.MagicMock()
    mock_instance.conversation.agent = mocker.MagicMock()

    mock_instance.conversation.agent.get_agent_config = mocker.MagicMock(
        return_value=mocker.MagicMock(interrupt_sensitivity="high")
    )

    test_transcription = Transcription(message="test", confidence=1.0, is_final=True)
    assert mock_instance.is_transcription_backchannel(mock_instance, test_transcription) is False

    mock_instance.conversation.agent.get_agent_config = mocker.MagicMock(
        return_value=mocker.MagicMock(interrupt_sensitivity="low")
    )
    assert mock_instance.is_transcription_backchannel(mock_instance, test_transcription) is True


@pytest.mark.parametrize(
    "test_case",
    [
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[],
            ),
            human_transcription="hi",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=True,
                        is_end_of_turn=True,
                    ),
                ],
            ),
            human_transcription="hi",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=False,
                        is_end_of_turn=False,
                    ),
                ],
            ),
            human_transcription="one two three four",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=False,
                        is_end_of_turn=False,
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=True,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=True,
                        is_end_of_turn=True,
                    ),
                    Message(
                        sender=Sender.HUMAN,
                        text="hi there",
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=False,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=False,
                    ),
                    ActionStart(
                        action_type="action",
                        action_input=MagicMock(spec=ActionInput),
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=True,
        ),
        ShouldIgnoreUtteranceTestCase(
            transcript=Transcript(
                event_logs=[
                    Message(
                        sender=Sender.BOT,
                        text="hi there",
                        is_final=True,
                        is_end_of_turn=True,
                    ),
                    ActionStart(
                        action_type="action",
                        action_input=MagicMock(spec=ActionInput),
                    ),
                ],
            ),
            human_transcription="hello?",
            expected=False,
        ),
    ],
)
def test_should_ignore_utterance(
    mocker: MockerFixture,
    test_case: ShouldIgnoreUtteranceTestCase,
):
    from vocode.streaming.streaming_conversation import StreamingConversation

    conversation = mocker.MagicMock()
    transcriptions_worker = StreamingConversation.TranscriptionsWorker(
        input_queue=mocker.MagicMock(),
        output_queue=mocker.MagicMock(),
        conversation=conversation,
        interruptible_event_factory=mocker.MagicMock(),
    )
    conversation = mocker.MagicMock()
    conversation.interrupt_sensitivity = "low"
    transcriptions_worker.has_associated_ignored_utterance = False

    transcriptions_worker.conversation.transcript = test_case.transcript
    assert (
        transcriptions_worker.should_ignore_utterance(
            Transcription(message=test_case.human_transcription, confidence=1.0, is_final=True),
        )
        == test_case.expected
    )


class TranscriptionsWorkerTestCase(BaseModel):
    transcriptions: List[Transcription]
    transcript: Transcript
    should_broadcast_interrupt: bool
    initial_message_ongoing: bool
    should_kick_off_pipeline: bool
    interrupt_sensitivity: InterruptSensitivity


async def _mock_streaming_conversation_constructor(
    mocker: MockerFixture,
    interrupt_sensitivity: InterruptSensitivity = "low",
):

    streaming_conversation = create_fake_streaming_conversation(
        mocker,
        agent=create_fake_agent(
            mocker,
            DEFAULT_CHAT_GPT_AGENT_CONFIG.copy(
                update={"interrupt_sensitivity": interrupt_sensitivity}
            ),
        ),
    )
    streaming_conversation.broadcast_interrupt = mocker.MagicMock(
        wraps=streaming_conversation.broadcast_interrupt
    )
    return streaming_conversation


@pytest.mark.asyncio
async def test_transcriptions_worker_ignores_utterances_before_initial_message(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.transcriptions_worker.input_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.output_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.start()
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="sup",
            confidence=1.0,
            is_final=True,
        ),
    )
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="hi, who is",
            confidence=1.0,
            is_final=False,
        ),
    )
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="hi, who is calling?",
            confidence=1.0,
            is_final=True,
        ),
    )
    assert await _consume_worker_output(streaming_conversation.transcriptions_worker) is None
    assert not streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcript.add_bot_message(
        text="hi there", is_final=True, conversation_id="test"
    )
    streaming_conversation.initial_message_tracker.set()

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="hi, who is this?",
            confidence=1.0,
            is_final=True,
        ),
    )

    transcription_agent_input = await _consume_worker_output(
        streaming_conversation.transcriptions_worker
    )
    assert transcription_agent_input.payload.transcription.message == "hi, who is this?"
    assert streaming_conversation.broadcast_interrupt.called

    assert streaming_conversation.transcript.event_logs[0].sender == Sender.BOT

    human_backchannels = streaming_conversation.transcript.event_logs[1:]
    assert all(
        backchannel.sender == Sender.HUMAN and backchannel.is_backchannel
        for backchannel in human_backchannels
    )
    streaming_conversation.transcriptions_worker.terminate()


@pytest.mark.asyncio
async def test_transcriptions_worker_ignores_associated_ignored_utterance(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.transcriptions_worker.input_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.output_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.start()
    streaming_conversation.initial_message_tracker.set()
    streaming_conversation.transcript.add_bot_message(
        text="Hi, I was wondering",
        is_final=False,
        conversation_id="test",
    )

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="i'm listening",
            confidence=1.0,
            is_final=False,
        ),
    )
    assert await _consume_worker_output(streaming_conversation.transcriptions_worker) is None
    assert not streaming_conversation.broadcast_interrupt.called  # ignored for length of response

    streaming_conversation.transcript.event_logs[-1].text = (
        "Hi, I was wondering if you had a chance to look at my email?"
    )
    streaming_conversation.transcript.event_logs[-1].is_final = True
    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="I'm listening.",
            confidence=1.0,
            is_final=True,
        ),
    )

    assert await _consume_worker_output(streaming_conversation.transcriptions_worker) is None
    assert not streaming_conversation.broadcast_interrupt.called  # ignored for length of response

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="I have not yet gotten a chance.",
            confidence=1.0,
            is_final=True,
        ),
    )

    transcription_agent_input = await _consume_worker_output(
        streaming_conversation.transcriptions_worker
    )
    assert (
        transcription_agent_input.payload.transcription.message == "I have not yet gotten a chance."
    )
    assert streaming_conversation.broadcast_interrupt.called
    assert [message.text for message in streaming_conversation.transcript.event_logs] == [
        "Hi, I was wondering if you had a chance to look at my email?",
        "I'm listening.",
    ]
    assert streaming_conversation.transcript.event_logs[-1].is_backchannel
    streaming_conversation.transcriptions_worker.terminate()


@pytest.mark.asyncio
async def test_transcriptions_worker_interrupts_on_interim_transcripts(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.transcriptions_worker.input_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.output_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.start()
    streaming_conversation.initial_message_tracker.set()
    streaming_conversation.transcript.add_bot_message(
        text="Hi, I was wondering",
        is_final=False,
        conversation_id="test",
    )

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry, could you stop",
            confidence=1.0,
            is_final=False,
        ),
    )

    assert await _consume_worker_output(streaming_conversation.transcriptions_worker) is None
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry, could you stop talking please?",
            confidence=1.0,
            is_final=True,
        ),
    )

    transcription_agent_input = await _consume_worker_output(
        streaming_conversation.transcriptions_worker
    )
    assert (
        transcription_agent_input.payload.transcription.message
        == "Sorry, could you stop talking please?"
    )

    assert streaming_conversation.transcript.event_logs[-1].sender == Sender.BOT
    assert streaming_conversation.transcript.event_logs[-1].text == "Hi, I was wondering"
    streaming_conversation.transcriptions_worker.terminate()


@pytest.mark.asyncio
async def test_transcriptions_worker_interrupts_immediately_before_bot_has_begun_turn(
    mocker: MockerFixture,
):
    streaming_conversation = await _mock_streaming_conversation_constructor(
        mocker,
    )

    streaming_conversation.transcriptions_worker.input_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.output_queue = asyncio.Queue()
    streaming_conversation.transcriptions_worker.start()
    streaming_conversation.initial_message_tracker.set()

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry,",
            confidence=1.0,
            is_final=False,
        ),
    )
    assert await _consume_worker_output(streaming_conversation.transcriptions_worker) is None
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Sorry, what?",
            confidence=1.0,
            is_final=True,
        ),
    )
    transcription_agent_input = await _consume_worker_output(
        streaming_conversation.transcriptions_worker
    )
    assert transcription_agent_input.payload.transcription.message == "Sorry, what?"
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcript.add_bot_message(
        text="", is_final=False, conversation_id="test"
    )

    streaming_conversation.transcriptions_worker.consume_nonblocking(
        Transcription(
            message="Couldn't",
            confidence=1.0,
            is_final=False,
        ),
    )

    assert await _consume_worker_output(streaming_conversation.transcriptions_worker) is None
    assert streaming_conversation.broadcast_interrupt.called

    streaming_conversation.transcriptions_worker.terminate()
