from typing import AsyncGenerator, AsyncIterator, Optional

import sentry_sdk
from langchain.chat_models import init_chat_model
from langchain_core.messages.base import BaseMessage as LangchainBaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from loguru import logger

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.anthropic_utils import merge_bot_messages_for_langchain
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.agent import LangchainAgentConfig
from vocode.streaming.models.events import Sender
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.models.transcript import Message
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class LangchainAgent(RespondAgent[LangchainAgentConfig]):

    def __init__(
        self,
        agent_config: LangchainAgentConfig,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        chain: Optional[Runnable] = None,
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.chain = chain if chain else self.create_chain()

    def create_chain(self):
        model = init_chat_model(
            model=self.agent_config.model_name,
            model_provider=self.agent_config.provider,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        messages_for_prompt_template = [("placeholder", "{chat_history}")]
        if self.agent_config.prompt_preamble:
            messages_for_prompt_template.insert(0, ("system", self.agent_config.prompt_preamble))
        prompt_template = ChatPromptTemplate.from_messages(messages_for_prompt_template)
        chain = prompt_template | model
        return chain

    async def token_generator(
        self,
        gen: AsyncIterator[LangchainBaseMessage],
    ) -> AsyncGenerator[str, None]:
        async for chunk in gen:
            if isinstance(chunk.content, str):
                yield chunk.content
            else:
                raise ValueError(
                    f"Received unexpected message type {type(chunk)} from Langchain. Expected str."
                )

    def format_langchain_messages_from_transcript(self) -> list[tuple]:
        if not self.transcript:
            raise ValueError("A transcript is not attached to the agent")
        messages = []
        for event_log in self.transcript.event_logs:
            if isinstance(event_log, Message):
                messages.append(
                    (
                        "ai" if event_log.sender == Sender.BOT else "human",
                        event_log.to_string(include_sender=False),
                    )
                )
            else:
                raise ValueError(
                    f"Invalid event log type {type(event_log)}. Langchain currently only supports human and bot messages"
                )

        if self.agent_config.provider == "anthropic":
            messages = merge_bot_messages_for_langchain(messages)

        return messages

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        if not self.transcript:
            raise ValueError("A transcript is not attached to the agent")
        try:
            first_sentence_total_span = sentry_create_span(
                sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.LLM_FIRST_SENTENCE_TOTAL
            )

            ttft_span = sentry_create_span(
                sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.TIME_TO_FIRST_TOKEN
            )
            stream = self.chain.astream(
                {"chat_history": self.format_langchain_messages_from_transcript()}
            )
        except Exception as e:
            logger.error(
                f"Error while hitting Langchain",
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
