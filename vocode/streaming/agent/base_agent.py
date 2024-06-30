from __future__ import annotations

import asyncio
import json
import random
import typing
from enum import Enum
from typing import TYPE_CHECKING, AsyncGenerator, Dict, Generic, Optional, Tuple, TypeVar, Union

import sentry_sdk
from loguru import logger
from pydantic.v1 import BaseModel

from vocode import sentry_span_tags
from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.action.execute_external_action import ExecuteExternalActionVocodeActionConfig
from vocode.streaming.action.phone_call_action import (
    TwilioPhoneConversationAction,
    VonagePhoneConversationAction,
)
from vocode.streaming.agent.goodbye import is_goodbye_simple
from vocode.streaming.agent.phrase_trigger import matches_phrase_trigger
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    EndOfTurn,
    FunctionCall,
)
from vocode.streaming.models.agent import AgentConfig, ChatGPTAgentConfig, LLMAgentConfig
from vocode.streaming.models.events import Sender
from vocode.streaming.models.message import BaseMessage, BotBackchannel, SilenceMessage
from vocode.streaming.models.model import TypedModel
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.models.transcript import Message, Transcript
from vocode.streaming.pipeline.worker import (
    AbstractWorker,
    InterruptibleAgentResponseEvent,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleWorker,
)
from vocode.streaming.utils import unrepeating_randomizer
from vocode.streaming.utils.speed_manager import SpeedManager
from vocode.streaming.utils.state_manager import (
    TwilioPhoneConversationStateManager,
    VonagePhoneConversationStateManager,
)
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span

if TYPE_CHECKING:
    from vocode.streaming.utils.state_manager import AbstractConversationStateManager

AGENT_TRACE_NAME = "agent"
POST_QUESTION_BACKCHANNELS = [
    "Oh okay, got it.",
    "Oh, okay, got it.",
    "Oh okay got it.",
    "Ohhh, okay, got it.",
    "Oh got it, okay.",
    "Ohhh got it, okay.",
    "Ohhh, got it, okay.",
    "Oh, got it, okay.",
]


class AgentInputType(str, Enum):
    BASE = "agent_input_base"
    TRANSCRIPTION = "agent_input_transcription"
    ACTION_RESULT = "agent_input_action_result"


class AgentInput(TypedModel, type=AgentInputType.BASE.value):  # type: ignore
    conversation_id: str
    agent_response_tracker: Optional[asyncio.Event] = None

    class Config:
        arbitrary_types_allowed = True


class TranscriptionAgentInput(AgentInput, type=AgentInputType.TRANSCRIPTION.value):  # type: ignore
    transcription: Transcription


class ActionResultAgentInput(AgentInput, type=AgentInputType.ACTION_RESULT.value):  # type: ignore
    action_input: ActionInput
    action_output: ActionOutput
    is_quiet: bool = False


class AgentResponseType(str, Enum):
    BASE = "agent_response_base"
    MESSAGE = "agent_response_message"
    STOP = "agent_response_stop"
    FILLER_AUDIO = "agent_response_filler_audio"


class AgentResponse(TypedModel, type=AgentResponseType.BASE.value):  # type: ignore
    pass


class AgentResponseMessage(AgentResponse, type=AgentResponseType.MESSAGE.value):  # type: ignore
    message: Union[BaseMessage, EndOfTurn]
    is_interruptible: bool = True
    # Whether the message is the first message in the response; has metrics implications
    is_first: bool = False
    # If the response is not being chunked up into multiple sentences, this is set to True
    is_sole_text_chunk: bool = False


class AgentResponseStop(AgentResponse, type=AgentResponseType.STOP.value):  # type: ignore
    pass


class AgentResponseFillerAudio(
    AgentResponse,
    type=AgentResponseType.FILLER_AUDIO.value,  # type: ignore
):
    pass


class GeneratedResponse(BaseModel):
    message: Union[BaseMessage, FunctionCall, EndOfTurn]
    is_interruptible: bool
    streamed: bool = False


class StreamedResponse(GeneratedResponse):
    streamed: bool = True


AgentConfigType = TypeVar("AgentConfigType", bound=AgentConfig)


class AbstractAgent(Generic[AgentConfigType]):
    def __init__(self, agent_config: AgentConfigType):
        self.agent_config = agent_config

    def get_agent_config(self) -> AgentConfig:
        return self.agent_config

    def update_last_bot_message_on_cut_off(self, message: str):
        """Updates the last bot message in the conversation history when the human cuts off the bot's response."""
        pass

    def get_cut_off_response(self) -> str:
        assert isinstance(self.agent_config, LLMAgentConfig) or isinstance(
            self.agent_config,
            ChatGPTAgentConfig,
        ), "Set cutoff response is only implemented in LLMAgent and ChatGPTAgent"
        assert self.agent_config.cut_off_response is not None
        on_cut_off_messages = self.agent_config.cut_off_response.messages
        assert len(on_cut_off_messages) > 0
        return random.choice(on_cut_off_messages).text


class BaseAgent(AbstractAgent[AgentConfigType], InterruptibleWorker):
    agent_responses_consumer: AbstractWorker[InterruptibleAgentResponseEvent[AgentResponse]]
    actions_consumer: Optional[AbstractWorker[InterruptibleEvent[ActionInput]]]

    def __init__(
        self,
        agent_config: AgentConfigType,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        interruptible_event_factory: InterruptibleEventFactory = InterruptibleEventFactory(),
    ):
        self.input_queue: asyncio.Queue[InterruptibleEvent[AgentInput]] = asyncio.Queue()
        AbstractAgent.__init__(self, agent_config=agent_config)
        InterruptibleWorker.__init__(
            self,
            interruptible_event_factory=interruptible_event_factory,
        )
        self.action_factory = action_factory
        self.transcript: Optional[Transcript] = None

        self.functions = self.get_functions() if self.agent_config.actions else None
        self.is_muted = False

        self.post_question_bot_backchannel_randomizer = unrepeating_randomizer(
            POST_QUESTION_BACKCHANNELS,
        )

    def get_functions(self):
        raise NotImplementedError

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    def attach_conversation_state_manager(
        self,
        conversation_state_manager: AbstractConversationStateManager,
    ):
        self.conversation_state_manager = conversation_state_manager

    def attach_speed_manager(self, speed_manager: SpeedManager):
        self.speed_manager = speed_manager

    def _get_speed_adjusted_silence_seconds(self, seconds: float) -> float:
        speed_coefficient = (
            self.speed_manager.get_speed_coefficient() if self.speed_manager is not None else 1.0
        )
        return seconds / speed_coefficient

    def set_interruptible_event_factory(self, factory: InterruptibleEventFactory):
        self.interruptible_event_factory = factory

    def get_input_queue(
        self,
    ) -> asyncio.Queue[InterruptibleEvent[AgentInput]]:
        return self.input_queue

    def is_first_response(self):
        assert self.transcript is not None

        num_bot_messages = sum(
            1
            for event_log in self.transcript.event_logs
            if isinstance(event_log, Message) and event_log.sender == Sender.BOT
        )

        return num_bot_messages <= (1 if self.agent_config.initial_message is not None else 0)


class RespondAgent(BaseAgent[AgentConfigType]):
    async def _maybe_prepend_interrupt_responses(
        self,
        transcription: Transcription,
        responses_stream: AsyncGenerator[GeneratedResponse, None],
    ) -> AsyncGenerator[GeneratedResponse, None]:
        if transcription.is_interrupt:
            if self.agent_config.cut_off_response:
                cut_off_response = self.get_cut_off_response()
                yield GeneratedResponse(
                    message=BaseMessage(text=cut_off_response),
                    is_interruptible=False,
                )
                return
            if transcription.bot_was_in_medias_res:
                silence_message = SilenceMessage()
                silence_message.trailing_silence_seconds = self._get_speed_adjusted_silence_seconds(
                    silence_message.trailing_silence_seconds
                )
                yield GeneratedResponse(message=silence_message, is_interruptible=True)
        async for response in responses_stream:
            yield response

    async def handle_generate_response(
        self,
        transcription: Transcription,
        agent_input: AgentInput,
    ) -> bool:
        conversation_id = agent_input.conversation_id
        responses = self._maybe_prepend_interrupt_responses(
            transcription=transcription,
            responses_stream=self.generate_response(
                transcription.message,
                is_interrupt=transcription.is_interrupt,
                conversation_id=conversation_id,
                bot_was_in_medias_res=transcription.bot_was_in_medias_res,
            ),
        )
        is_first_response_of_turn = True
        function_call = None

        responses_buffer = ""
        end_of_turn_agent_response_tracker = None

        async for generated_response in responses:
            if is_first_response_of_turn:
                message_type = "UNKNOWN"
                match generated_response.message:
                    case SilenceMessage():  # type: ignore[misc]
                        message_type = "silence"
                    case BotBackchannel():  # type: ignore[misc]
                        message_type = "backchannel"
                    case BaseMessage():  # type: ignore[misc]
                        message_type = "message"
                    case FunctionCall():  # type: ignore[misc]
                        message_type = "function_call"
                    case _:
                        logger.warning(
                            "Unknown message type received for Sentry metrics "
                            f"reporting: {type(generated_response.message)}",
                        )
                span_tags = sentry_span_tags.value
                if span_tags:
                    span_tags["message_type"] = message_type
                    sentry_span_tags.set(span_tags)

            if isinstance(generated_response.message, FunctionCall):
                function_call = generated_response.message
                continue

            agent_response_tracker = agent_input.agent_response_tracker or asyncio.Event()
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(
                        message=generated_response.message,
                        is_first=is_first_response_of_turn,
                    ),
                    is_interruptible=self.agent_config.allow_agent_to_be_cut_off
                    and generated_response.is_interruptible,
                    agent_response_tracker=agent_response_tracker,
                ),
            )
            if isinstance(generated_response.message, BaseMessage):
                responses_buffer = f"{responses_buffer} {generated_response.message.text}"
            elif isinstance(generated_response.message, EndOfTurn):
                end_of_turn_agent_response_tracker = agent_response_tracker

            if self.agent_config.end_conversation_on_goodbye and isinstance(
                generated_response.message,
                BaseMessage,
            ):
                if is_goodbye_simple(
                    message=generated_response.message.text,
                    phrases=self.agent_config.goodbye_phrases,
                ):
                    logger.debug("Simple goodbye detected, ending conversation")
                    return True
            is_first_response_of_turn = False

        # if the client (the implemented agent) doesn't create an EndOfTurn, then we need to create one
        if not end_of_turn_agent_response_tracker:
            end_of_turn_agent_response_tracker = (
                agent_input.agent_response_tracker or asyncio.Event()
            )
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(
                        message=EndOfTurn(),
                        is_first=is_first_response_of_turn,
                    ),
                    is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
                    agent_response_tracker=end_of_turn_agent_response_tracker,
                ),
            )

        phrase_trigger_match = (
            matches_phrase_trigger(responses_buffer, self.agent_config.actions)
            if self.agent_config.actions
            else None
        )
        if phrase_trigger_match:
            action_config = self._get_action_config(phrase_trigger_match)
            assert action_config is not None
            action = self.action_factory.create_action(action_config)
            action_input = self.create_action_input(
                action,
                agent_input,
                {},
                end_of_turn_agent_response_tracker,
            )
            self.enqueue_action_input(action, action_input, agent_input.conversation_id)

        # TODO: implement should_stop for generate_responses
        if function_call and self.agent_config.actions is not None:
            await self.call_function(function_call, agent_input)
        return False

    async def handle_respond(self, transcription: Transcription, conversation_id: str) -> bool:
        try:
            response, should_stop = await self.respond(
                transcription.message,
                is_interrupt=transcription.is_interrupt,
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.error(f"Error while generating response: {e}", exc_info=True)
            response = None
            return True
        if response:
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(message=BaseMessage(text=response)),
                    is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
                )
            )
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(message=EndOfTurn()),
                    is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
                )
            )
            return should_stop
        else:
            logger.debug("No response generated")
        return False

    async def process(self, item: InterruptibleEvent[AgentInput]):
        assert self.transcript is not None
        try:
            agent_input = item.payload
            if isinstance(agent_input, TranscriptionAgentInput):
                transcription = typing.cast(TranscriptionAgentInput, agent_input).transcription
                self.transcript.add_human_message(
                    text=transcription.message,
                    conversation_id=agent_input.conversation_id,
                )
            elif isinstance(agent_input, ActionResultAgentInput):
                self.transcript.add_action_finish_log(
                    action_input=agent_input.action_input,
                    action_output=agent_input.action_output,
                    conversation_id=agent_input.conversation_id,
                )
                if agent_input.is_quiet:
                    # Do not generate a response to quiet actions
                    logger.debug("Action is quiet, skipping response generation")
                    return
                if agent_input.action_output.canned_response is not None:
                    self.agent_responses_consumer.consume_nonblocking(
                        self.interruptible_event_factory.create_interruptible_agent_response_event(
                            AgentResponseMessage(
                                message=agent_input.action_output.canned_response,
                                is_sole_text_chunk=True,
                            ),
                            is_interruptible=True,
                        )
                    )
                    self.agent_responses_consumer.consume_nonblocking(
                        self.interruptible_event_factory.create_interruptible_agent_response_event(
                            AgentResponseMessage(message=EndOfTurn()),
                        )
                    )
                    return
                transcription = Transcription(
                    message=agent_input.action_output.response.json(),
                    confidence=1.0,
                    is_final=True,
                )
            else:
                raise ValueError("Invalid AgentInput type")

            if self.is_muted:
                logger.debug("Agent is muted, skipping processing")
                return

            if self.agent_config.send_filler_audio:
                self.agent_responses_consumer.consume_nonblocking(
                    self.interruptible_event_factory.create_interruptible_agent_response_event(
                        AgentResponseFillerAudio(),
                    )
                )

            logger.debug("Responding to transcription")
            should_stop = False
            if self.agent_config.generate_responses:
                # TODO (EA): this is quite ugly but necessary to have the agent act properly after an action completes
                if not isinstance(agent_input, ActionResultAgentInput):
                    sentry_create_span(
                        sentry_callable=sentry_sdk.start_span,
                        op=CustomSentrySpans.LANGUAGE_MODEL_TIME_TO_FIRST_TOKEN,
                    )
                should_stop = await self.handle_generate_response(transcription, agent_input)
            else:
                should_stop = await self.handle_respond(transcription, agent_input.conversation_id)

            if should_stop:
                logger.debug("Agent requested to stop")
                self.agent_responses_consumer.consume_nonblocking(
                    self.interruptible_event_factory.create_interruptible_agent_response_event(
                        AgentResponseStop(),
                    )
                )
                return
        except asyncio.CancelledError:
            pass

    def _get_action_config(self, function_name: str) -> Optional[ActionConfig]:
        if self.agent_config.actions is None:
            return None
        for action_config in self.agent_config.actions:
            if action_config.type == function_name or (
                isinstance(action_config, ExecuteExternalActionVocodeActionConfig)
                and action_config.name == function_name
            ):
                return action_config
        return None

    async def call_function(self, function_call: FunctionCall, agent_input: AgentInput):
        action_config = self._get_action_config(function_call.name)
        if action_config is None:
            logger.error(f"Function {function_call.name} not found in agent config, skipping")
            return
        action = self.action_factory.create_action(action_config)
        params = json.loads(function_call.arguments)
        user_message_tracker = None
        if "user_message" in params:
            user_message = params["user_message"]
            user_message_tracker = asyncio.Event()
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(
                        message=BaseMessage(text=user_message),
                        is_sole_text_chunk=True,
                    ),
                    is_interruptible=action.is_interruptible,
                )
            )
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(message=EndOfTurn()),
                    agent_response_tracker=user_message_tracker,
                )
            )
        action_input = self.create_action_input(action, agent_input, params, user_message_tracker)
        self.enqueue_action_input(action, action_input, agent_input.conversation_id)

    def create_action_input(
        self,
        action: BaseAction,
        agent_input: AgentInput,
        params: Dict,
        user_message_tracker: Optional[asyncio.Event] = None,
    ) -> ActionInput:
        action_input: ActionInput
        if isinstance(action, VonagePhoneConversationAction):
            assert isinstance(
                self.conversation_state_manager,
                VonagePhoneConversationStateManager,
            ), "Cannot use a VonagePhoneConversationAction unless the attached conversation is a VonagePhoneConversation"
            action_input = action.create_phone_conversation_action_input(
                conversation_id=agent_input.conversation_id,
                params=params,
                vonage_uuid=self.conversation_state_manager.get_vonage_uuid(),
                user_message_tracker=user_message_tracker,
            )
        elif isinstance(action, TwilioPhoneConversationAction):
            assert isinstance(
                self.conversation_state_manager, TwilioPhoneConversationStateManager
            ), "Cannot use a TwilioPhoneConversationAction unless the attached conversation is a TwilioPhoneConversation"
            action_input = action.create_phone_conversation_action_input(
                conversation_id=agent_input.conversation_id,
                params=params,
                twilio_sid=self.conversation_state_manager.get_twilio_sid(),
                user_message_tracker=user_message_tracker,
            )
        else:
            action_input = action.create_action_input(
                agent_input.conversation_id,
                params,
                user_message_tracker,
            )
        return action_input

    def enqueue_action_input(
        self,
        action: BaseAction,
        action_input: ActionInput,
        conversation_id: str,
    ):
        if self.actions_consumer is None:
            logger.warning("No actions consumer attached, skipping action")
            return
        event = self.interruptible_event_factory.create_interruptible_event(
            action_input,
            is_interruptible=action.is_interruptible,
        )
        assert self.transcript is not None
        self.transcript.add_action_start_log(
            action_input=action_input,
            conversation_id=conversation_id,
        )
        self.actions_consumer.consume_nonblocking(event)

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        raise NotImplementedError

    def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[
        GeneratedResponse,
        None,
    ]:  # tuple of the content and whether it is interruptible
        raise NotImplementedError
