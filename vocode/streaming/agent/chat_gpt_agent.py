import os
import random
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeVar, Union

import sentry_sdk
from loguru import logger
from openai import DEFAULT_MAX_RETRIES as OPENAI_DEFAULT_MAX_RETRIES
from openai import AsyncAzureOpenAI, AsyncOpenAI, NotFoundError, RateLimitError

from vocode import sentry_span_tags
from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.openai_utils import (
    format_openai_chat_messages_from_transcript,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
)
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.actions import FunctionCallActionTrigger
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.events import Sender
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken
from vocode.streaming.models.transcript import Message
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span

ChatGPTAgentConfigType = TypeVar("ChatGPTAgentConfigType", bound=ChatGPTAgentConfig)


def instantiate_openai_client(agent_config: ChatGPTAgentConfig, model_fallback: bool = False):
    if agent_config.azure_params:
        return AsyncAzureOpenAI(
            azure_endpoint=agent_config.azure_params.base_url,
            api_key=agent_config.azure_params.api_key,
            api_version=agent_config.azure_params.api_version,
            max_retries=0 if model_fallback else OPENAI_DEFAULT_MAX_RETRIES,
        )
    else:
        if agent_config.openai_api_key is not None:
            logger.info("Using OpenAI API key override")
        if agent_config.base_url_override is not None:
            logger.info(f"Using OpenAI base URL override: {agent_config.base_url}")
        return AsyncOpenAI(
            api_key=agent_config.openai_api_key or os.environ["OPENAI_API_KEY"],
            base_url=agent_config.base_url_override or "https://api.openai.com/v1",
            max_retries=0 if model_fallback else OPENAI_DEFAULT_MAX_RETRIES,
        )


class ChatGPTAgent(RespondAgent[ChatGPTAgentConfigType]):
    openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI]

    def __init__(
        self,
        agent_config: ChatGPTAgentConfigType,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.openai_client = instantiate_openai_client(
            agent_config, model_fallback=agent_config.llm_fallback is not None
        )

        if not self.openai_client.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")

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

    def get_chat_parameters(self, messages: Optional[List] = None, use_functions: bool = True):
        assert self.transcript is not None
        is_azure = self._is_azure_model()

        messages = messages or format_openai_chat_messages_from_transcript(
            self.transcript,
            self.get_model_name_for_tokenizer(),
            self.functions,
            self.agent_config.prompt_preamble,
        )

        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
        }

        if is_azure:
            assert self.agent_config.azure_params is not None
            parameters["model"] = self.agent_config.azure_params.deployment_name
        else:
            parameters["model"] = self.agent_config.model_name

        if use_functions and self.functions:
            parameters["functions"] = self.functions

        return parameters

    def _is_azure_model(self) -> bool:
        return self.agent_config.azure_params is not None

    def get_model_name_for_tokenizer(self):
        if not self.agent_config.azure_params:
            return self.agent_config.model_name
        else:
            return self.agent_config.azure_params.openai_model_name

    def apply_model_fallback(self, chat_parameters: Dict[str, Any]):
        if self.agent_config.llm_fallback is None:
            return
        if self.agent_config.llm_fallback.provider == "openai":
            self.agent_config.model_name = self.agent_config.llm_fallback.model_name
            if isinstance(self.openai_client, AsyncAzureOpenAI):
                self.agent_config.azure_params = None
        else:
            if self.agent_config.azure_params:
                self.agent_config.azure_params.deployment_name = (
                    self.agent_config.llm_fallback.model_name
                )
                if isinstance(self.openai_client, AsyncOpenAI):
                    # TODO: handle OpenAI fallback to Azure
                    pass

        self.openai_client = instantiate_openai_client(self.agent_config, model_fallback=False)
        chat_parameters["model"] = self.agent_config.llm_fallback.model_name

    async def _create_openai_stream_with_fallback(
        self, chat_parameters: Dict[str, Any]
    ) -> AsyncGenerator:
        try:
            stream = await self.openai_client.chat.completions.create(**chat_parameters)
        except (NotFoundError, RateLimitError) as e:
            logger.error(
                f"{'Model not found' if isinstance(e, NotFoundError) else 'Rate limit error'} for model_name: {chat_parameters.get('model')}. Applying fallback.",
                exc_info=True,
            )
            self.apply_model_fallback(chat_parameters)
            stream = await self.openai_client.chat.completions.create(**chat_parameters)
        except Exception as e:
            logger.error(
                f"Error while hitting OpenAI with chat_parameters: {chat_parameters}",
                exc_info=True,
            )
            raise e
        return stream

    async def _create_openai_stream(self, chat_parameters: Dict[str, Any]) -> AsyncGenerator:
        if self.agent_config.llm_fallback is not None and self.openai_client.max_retries == 0:
            stream = await self._create_openai_stream_with_fallback(chat_parameters)
        else:
            try:
                stream = await self.openai_client.chat.completions.create(**chat_parameters)
            except Exception as e:
                logger.error(
                    f"Error while hitting OpenAI with chat_parameters: {chat_parameters}",
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
                messages = format_openai_chat_messages_from_transcript(
                    self.transcript,
                    self.agent_config.model_name,
                    self.functions,
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

        openai_chat_messages: List = chat_parameters.get("messages", [])

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
            # between sending the response and generating ChatGPT's response
            openai_chat_messages.append({"role": "assistant", "content": backchannel.text})
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

        stream = await self._create_openai_stream(chat_parameters)

        response_generator = collate_response_async
        using_input_streaming_synthesizer = (
            self.conversation_state_manager.using_input_streaming_synthesizer()
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
