import os
from typing import Any, AsyncGenerator, Dict

import sentry_sdk
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import MessageStreamEvent
from loguru import logger

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.anthropic_utils import format_anthropic_chat_messages_from_transcript
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.actions import FunctionFragment
from vocode.streaming.models.agent import AnthropicAgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class AnthropicAgent(RespondAgent[AnthropicAgentConfig]):
    anthropic_client: AsyncAnthropic

    def __init__(
        self,
        agent_config: AnthropicAgentConfig,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def get_chat_parameters(self, messages: list = [], use_functions: bool = True):
        assert self.transcript is not None

        parameters: dict[str, Any] = {
            "messages": messages,
            "system": self.agent_config.prompt_preamble,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
            "stream": True,
        }

        parameters["model"] = self.agent_config.model_name

        return parameters

    async def token_generator(
        self,
        gen: AsyncStream[MessageStreamEvent],
    ) -> AsyncGenerator[str | FunctionFragment, None]:
        async for chunk in gen:
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text

    async def _get_anthropic_stream(self, chat_parameters: Dict[str, Any]):
        return await self.anthropic_client.messages.create(**chat_parameters)

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        if not self.transcript:
            raise ValueError("A transcript is not attached to the agent")
        messages = format_anthropic_chat_messages_from_transcript(transcript=self.transcript)
        chat_parameters = self.get_chat_parameters(messages)
        try:
            first_sentence_total_span = sentry_create_span(
                sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.LLM_FIRST_SENTENCE_TOTAL
            )

            ttft_span = sentry_create_span(
                sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.TIME_TO_FIRST_TOKEN
            )
            stream = await self._get_anthropic_stream(chat_parameters)
        except Exception as e:
            logger.error(
                f"Error while hitting Anthropic with chat_parameters: {chat_parameters}",
                exc_info=True,
            )
            raise e

        response_generator = collate_response_async

        using_input_streaming_synthesizer = (
            self.streaming_conversation.using_input_streaming_synthesizer()
        )
        if using_input_streaming_synthesizer:
            response_generator = stream_response_async
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=self.token_generator(
                stream,
            ),
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
