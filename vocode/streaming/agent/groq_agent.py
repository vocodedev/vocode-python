import os
import random
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeVar, Union

import sentry_sdk
from groq import AsyncGroq
from loguru import logger

from vocode import sentry_span_tags
from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.openai_utils import (
    get_openai_chat_messages_from_transcript,
    merge_event_logs,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
)
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.actions import FunctionCallActionTrigger
from vocode.streaming.models.agent import GroqAgentConfig
from vocode.streaming.models.events import Sender
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken
from vocode.streaming.models.transcript import EventLog, Message, Transcript
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class GroqAgent(RespondAgent[GroqAgentConfig]):
    groq_client: AsyncGroq

    def __init__(
        self,
        agent_config: GroqAgentConfig,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

        if not self.groq_client.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed in")

        if self.agent_config.vector_db_config:
            self.vector_db = vector_db_factory.create_vector_db(self.agent_config.vector_db_config)

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
            if isinstance(action_config.action_trigger, FunctionCallActionTrigger)
        ]

    def format_groq_chat_messages_from_transcript(
        self,
        transcript: Transcript,
        prompt_preamble: str,
    ) -> List[dict]:
        # merge consecutive bot messages
        merged_event_logs: List[EventLog] = merge_event_logs(event_logs=transcript.event_logs)

        chat_messages: List[Dict[str, Optional[Any]]]
        chat_messages = get_openai_chat_messages_from_transcript(
            merged_event_logs=merged_event_logs,
            prompt_preamble=prompt_preamble,
        )

        return chat_messages

    def get_chat_parameters(self, messages: Optional[List] = None, use_functions: bool = True):
        assert self.transcript is not None

        messages = messages or self.format_groq_chat_messages_from_transcript(
            self.transcript,
            self.agent_config.prompt_preamble,
        )

        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
            "model": self.agent_config.model_name,
        }

        if use_functions and self.functions:
            parameters["functions"] = self.functions

        return parameters

    async def _create_groq_stream(self, chat_parameters: Dict[str, Any]) -> AsyncGenerator:
        try:
            stream = await self.groq_client.chat.completions.create(**chat_parameters)
        except Exception as e:
            logger.error(
                f"Error while hitting Groq with chat_parameters: {chat_parameters}",
                exc_info=True,
            )
            raise e
        return stream

    def should_backchannel(self, human_input: str) -> bool:
        return (
            not self.is_first_response()
            and not human_input.strip().endswith("?")
            and random.random() < self.agent_config.backchannel_probability
        )

    def choose_backchannel(self) -> Optional[BotBackchannel]:
        backchannel = None
        if self.transcript is not None:
            last_bot_message: Optional[Message] = None
            for event_log in self.transcript.event_logs[::-1]:
                if isinstance(event_log, Message) and event_log.sender == Sender.BOT:
                    last_bot_message = event_log
                    break
            if last_bot_message and last_bot_message.text.strip().endswith("?"):
                return BotBackchannel(text=self.post_question_bot_backchannel_randomizer())
        return backchannel

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        assert self.transcript is not None

        chat_parameters = {}
        if self.agent_config.vector_db_config:
            try:
                docs_with_scores = await self.vector_db.similarity_search_with_score(
                    self.transcript.get_last_user_message()[1]
                )
                docs_with_scores_str = "\n\n".join(
                    [
                        "Document: "
                        + doc[0].metadata["source"]
                        + f" (Confidence: {doc[1]})\n"
                        + doc[0].lc_kwargs["page_content"].replace(r"\n", "\n")
                        for doc in docs_with_scores
                    ]
                )
                vector_db_result = (
                    f"Found {len(docs_with_scores)} similar documents:\n{docs_with_scores_str}"
                )
                messages = self.format_groq_chat_messages_from_transcript(
                    self.transcript,
                    self.agent_config.prompt_preamble,
                )
                messages.insert(-1, vector_db_result_to_openai_chat_message(vector_db_result))
                chat_parameters = self.get_chat_parameters(messages)
            except Exception as e:
                logger.error(f"Error while hitting vector db: {e}", exc_info=True)
                chat_parameters = self.get_chat_parameters()
        else:
            chat_parameters = self.get_chat_parameters()
        chat_parameters["stream"] = True

        groq_chat_messages: List = chat_parameters.get("messages", [])

        backchannelled = "false"
        backchannel: Optional[BotBackchannel] = None
        if (
            self.agent_config.use_backchannels
            and not bot_was_in_medias_res
            and self.should_backchannel(human_input)
        ):
            backchannel = self.choose_backchannel()
        elif self.agent_config.first_response_filler_message and self.is_first_response():
            backchannel = BotBackchannel(text=self.agent_config.first_response_filler_message)

        if backchannel is not None:
            # The LLM needs the backchannel context manually - otherwise we're in a race condition
            # between sending the response and generating Groq's response
            groq_chat_messages.append({"role": "assistant", "content": backchannel.text})
            yield GeneratedResponse(
                message=backchannel,
                is_interruptible=True,
            )
            backchannelled = "true"

        span_tags = sentry_span_tags.value
        if span_tags:
            span_tags["prior_backchannel"] = backchannelled
            sentry_span_tags.set(span_tags)

        first_sentence_total_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.LLM_FIRST_SENTENCE_TOTAL
        )

        ttft_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.TIME_TO_FIRST_TOKEN
        )

        stream = await self._create_groq_stream(chat_parameters)

        response_generator = collate_response_async
        using_input_streaming_synthesizer = (
            self.streaming_conversation.using_input_streaming_synthesizer()
        )
        if using_input_streaming_synthesizer:
            response_generator = stream_response_async
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=openai_get_tokens(
                stream,
            ),
            get_functions=True,
            sentry_span=ttft_span,
        ):
            if first_sentence_total_span:
                first_sentence_total_span.finish()

            ResponseClass = (
                StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
            )
            MessageType = LLMToken if using_input_streaming_synthesizer else BaseMessage
            if isinstance(message, str):
                yield ResponseClass(
                    message=MessageType(text=message),
                    is_interruptible=True,
                )
            else:
                yield ResponseClass(
                    message=message,
                    is_interruptible=True,
                )
