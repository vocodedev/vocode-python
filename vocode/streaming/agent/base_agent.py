from __future__ import annotations

import asyncio
from enum import Enum
import json
import logging
import random
from typing import (
    AsyncGenerator,
    Generator,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
)
import typing
from opentelemetry import trace
from opentelemetry.trace import Span
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.action.phone_call_action import (
    TwilioPhoneCallAction,
    VonagePhoneCallAction,
)
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    FunctionCall,
    FunctionFragment,
)

from vocode.streaming.models.agent import (
    AgentConfig,
    ChatGPTAgentConfig,
    LLMAgentConfig,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.model import BaseModel, TypedModel
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.utils import remove_non_letters_digits
from vocode.streaming.utils.goodbye_model import GoodbyeModel
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.utils.worker import (
    EventTracker,
    InterruptibleTrackedEvent,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleWorker,
)

if TYPE_CHECKING:
    from vocode.streaming.utils.state_manager import ConversationStateManager

tracer = trace.get_tracer(__name__)
AGENT_TRACE_NAME = "agent"


class AgentInputType(str, Enum):
    BASE = "agent_input_base"
    TRANSCRIPTION = "agent_input_transcription"
    ACTION_RESULT = "agent_input_action_result"


class AgentInput(TypedModel, type=AgentInputType.BASE.value):
    conversation_id: str
    vonage_uuid: Optional[str]
    twilio_sid: Optional[str]


class TranscriptionAgentInput(AgentInput, type=AgentInputType.TRANSCRIPTION.value):
    transcription: Transcription


class ActionResultAgentInput(AgentInput, type=AgentInputType.ACTION_RESULT.value):
    action_input: ActionInput
    action_output: ActionOutput
    is_quiet: bool = False


class AgentResponseType(str, Enum):
    BASE = "agent_response_base"
    MESSAGE = "agent_response_message"
    STOP = "agent_response_stop"
    FILLER_AUDIO = "agent_response_filler_audio"


class AgentResponse(TypedModel, type=AgentResponseType.BASE.value):
    pass


class AgentResponseMessage(AgentResponse, type=AgentResponseType.MESSAGE.value):
    message: BaseMessage
    is_interruptible: bool = True


class AgentResponseStop(AgentResponse, type=AgentResponseType.STOP.value):
    pass


class AgentResponseFillerAudio(
    AgentResponse, type=AgentResponseType.FILLER_AUDIO.value
):
    pass


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
            self.agent_config, ChatGPTAgentConfig
        ), "Set cutoff response is only implemented in LLMAgent and ChatGPTAgent"
        assert self.agent_config.cut_off_response is not None
        on_cut_off_messages = self.agent_config.cut_off_response.messages
        assert len(on_cut_off_messages) > 0
        return random.choice(on_cut_off_messages).text


class BaseAgent(AbstractAgent[AgentConfigType], InterruptibleWorker):
    def __init__(
        self,
        agent_config: AgentConfigType,
        action_factory: ActionFactory = ActionFactory(),
        interruptible_event_factory: InterruptibleEventFactory = InterruptibleEventFactory(),
        logger: Optional[logging.Logger] = None,
    ):
        self.input_queue: asyncio.Queue[
            InterruptibleTrackedEvent[AgentInput]
        ] = asyncio.Queue()
        self.output_queue: asyncio.Queue[
            InterruptibleTrackedEvent[AgentResponse]
        ] = asyncio.Queue()
        AbstractAgent.__init__(self, agent_config=agent_config)
        InterruptibleWorker.__init__(
            self,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            interruptible_event_factory=interruptible_event_factory,
        )
        self.action_factory = action_factory
        self.actions_queue: asyncio.Queue[
            InterruptibleEvent[ActionInput]
        ] = asyncio.Queue()
        self.logger = logger or logging.getLogger(__name__)
        self.goodbye_model = None
        if self.agent_config.end_conversation_on_goodbye:
            self.goodbye_model = GoodbyeModel()
            self.goodbye_model_initialize_task = asyncio.create_task(
                self.goodbye_model.initialize_embeddings()
            )
        self.transcript: Optional[Transcript] = None

        self.functions = self.get_functions() if self.agent_config.actions else None
        self.is_muted = False

    def get_functions(self):
        raise NotImplementedError

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    def attach_conversation_state_manager(
        self, conversation_state_manager: ConversationStateManager
    ):
        self.conversation_state_manager = conversation_state_manager

    def set_interruptible_event_factory(self, factory: InterruptibleEventFactory):
        self.interruptible_event_factory = factory

    def get_input_queue(
        self,
    ) -> asyncio.Queue[InterruptibleTrackedEvent[AgentInput]]:
        return self.input_queue

    def get_output_queue(
        self,
    ) -> asyncio.Queue[InterruptibleTrackedEvent[AgentResponse]]:
        return self.output_queue

    def create_goodbye_detection_task(self, message: str) -> asyncio.Task:
        assert self.goodbye_model is not None
        return asyncio.create_task(self.goodbye_model.is_goodbye(message))


class RespondAgent(BaseAgent[AgentConfigType]):
    async def handle_generate_response(
        self, transcription: Transcription, agent_input: AgentInput
    ) -> bool:
        conversation_id = agent_input.conversation_id
        tracer_name_start = await self.get_tracer_name_start()
        agent_span = tracer.start_span(
            f"{tracer_name_start}.generate_total"  # type: ignore
        )
        agent_span_first = tracer.start_span(
            f"{tracer_name_start}.generate_first"  # type: ignore
        )
        responses = self.generate_response(
            transcription.message,
            is_interrupt=transcription.is_interrupt,
            conversation_id=conversation_id,
        )
        is_first_response = True
        function_call = None
        async for response in responses:
            if isinstance(response, FunctionCall):
                function_call = response
                continue
            if is_first_response:
                agent_span_first.end()
                is_first_response = False
            self.produce_interruptible_tracked_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=response)),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
                span_name="agent_response_message",
            )
        # TODO: implement should_stop for generate_responses
        agent_span.end()
        if function_call and self.agent_config.actions is not None:
            await self.call_function(function_call, agent_input)
        return False

    async def handle_respond(
        self, transcription: Transcription, conversation_id: str
    ) -> bool:
        try:
            tracer_name_start = await self.get_tracer_name_start()
            with tracer.start_as_current_span(f"{tracer_name_start}.respond_total"):
                response, should_stop = await self.respond(
                    transcription.message,
                    is_interrupt=transcription.is_interrupt,
                    conversation_id=conversation_id,
                )
        except Exception as e:
            self.logger.error(f"Error while generating response: {e}", exc_info=True)
            response = None
            return True
        if response:
            self.produce_interruptible_tracked_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=response)),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
                span_name="agent_response_message",
            )
            return should_stop
        else:
            self.logger.debug("No response generated")
        return False

    async def process(self, item: InterruptibleTrackedEvent[AgentInput]):
        if self.is_muted:
            self.logger.debug("Agent is muted, skipping processing")
            return
        assert self.transcript is not None
        try:
            agent_input = item.payload
            if isinstance(agent_input, TranscriptionAgentInput):
                transcription = typing.cast(
                    TranscriptionAgentInput, agent_input
                ).transcription
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
                    self.logger.debug("Action is quiet, skipping response generation")
                    return
                transcription = Transcription(
                    message=agent_input.action_output.response.json(),
                    confidence=1.0,
                    is_final=True,
                )
            else:
                raise ValueError("Invalid AgentInput type")

            goodbye_detected_task = None
            if self.agent_config.end_conversation_on_goodbye:
                goodbye_detected_task = self.create_goodbye_detection_task(
                    transcription.message
                )
            if self.agent_config.send_filler_audio:
                self.produce_interruptible_tracked_event_nonblocking(
                    AgentResponseFillerAudio(), span_name="agent_response_filler_audio"
                )
            self.logger.debug("Responding to transcription")
            item.event_tracker.set()
            should_stop = False
            if self.agent_config.generate_responses:
                should_stop = await self.handle_generate_response(
                    transcription, agent_input
                )
            else:
                should_stop = await self.handle_respond(
                    transcription, agent_input.conversation_id
                )

            if should_stop:
                self.logger.debug("Agent requested to stop")
                self.produce_interruptible_tracked_event_nonblocking(
                    AgentResponseStop(),
                    span_name="agent_response_stop",
                )
                return
            if goodbye_detected_task:
                try:
                    goodbye_detected = await asyncio.wait_for(
                        goodbye_detected_task, 0.1
                    )
                    if goodbye_detected:
                        self.logger.debug("Goodbye detected, ending conversation")
                        self.produce_interruptible_tracked_event_nonblocking(
                            AgentResponseStop(),
                            span_name="agent_response_stop",
                        )
                        return
                except asyncio.TimeoutError:
                    self.logger.debug("Goodbye detection timed out")
        except asyncio.CancelledError:
            pass

    def _get_action_config(self, function_name: str) -> Optional[ActionConfig]:
        if self.agent_config.actions is None:
            return None
        for action_config in self.agent_config.actions:
            if action_config.type == function_name:
                return action_config
        return None

    async def call_function(self, function_call: FunctionCall, agent_input: AgentInput):
        action_config = self._get_action_config(function_call.name)
        if action_config is None:
            self.logger.error(
                f"Function {function_call.name} not found in agent config, skipping"
            )
            return
        action = self.action_factory.create_action(action_config)
        params = json.loads(function_call.arguments)
        user_message_tracker = None
        if "user_message" in params:
            user_message = params["user_message"]
            user_message_tracker = EventTracker("user_message")
            self.produce_interruptible_tracked_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=user_message)),
                event_tracker=user_message_tracker,
            )
        action_input: ActionInput
        if isinstance(action, VonagePhoneCallAction):
            assert (
                agent_input.vonage_uuid is not None
            ), "Cannot use VonagePhoneCallActionFactory unless the attached conversation is a VonageCall"
            action_input = action.create_phone_call_action_input(
                agent_input.conversation_id,
                params,
                agent_input.vonage_uuid,
                user_message_tracker,
            )
        elif isinstance(action, TwilioPhoneCallAction):
            assert (
                agent_input.twilio_sid is not None
            ), "Cannot use TwilioPhoneCallActionFactory unless the attached conversation is a TwilioCall"
            action_input = action.create_phone_call_action_input(
                agent_input.conversation_id,
                params,
                agent_input.twilio_sid,
                user_message_tracker,
            )
        else:
            action_input = action.create_action_input(
                agent_input.conversation_id,
                params,
                user_message_tracker,
            )
        event = self.interruptible_event_factory.create_interruptible_event(
            action_input, is_interruptible=action.is_interruptible
        )
        assert self.transcript is not None
        self.transcript.add_action_start_log(
            action_input=action_input,
            conversation_id=agent_input.conversation_id,
        )
        self.actions_queue.put_nowait(event)

    async def get_tracer_name_start(self) -> str:
        if hasattr(self, "tracer_name_start"):
            return self.tracer_name_start
        if (
            hasattr(self.agent_config, "azure_params")
            and self.agent_config.azure_params is not None
        ):
            beginning_agent_name = self.agent_config.type.rsplit("_", 1)[0]
            engine = self.agent_config.azure_params.engine
            tracer_name_start = (
                f"{AGENT_TRACE_NAME}.{beginning_agent_name}_azuregpt-{engine}"
            )
        else:
            optional_model_name = (
                f"-{self.agent_config.model_name}"
                if hasattr(self.agent_config, "model_name")
                else ""
            )
            tracer_name_start = remove_non_letters_digits(
                f"{AGENT_TRACE_NAME}.{self.agent_config.type}{optional_model_name}"
            )
        self.tracer_name_start: str = tracer_name_start
        return tracer_name_start

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
    ) -> AsyncGenerator[Union[str, FunctionCall], None]:
        raise NotImplementedError
