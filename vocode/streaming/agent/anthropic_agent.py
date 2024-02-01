from typing import AsyncGenerator, Optional, Tuple
from langchain import ConversationChain
import logging

from typing import Optional, Tuple
from vocode.streaming.agent.base_agent import RespondAgent

from vocode.streaming.agent.utils import get_sentence_from_buffer

from langchain import ConversationChain
from langchain.schema import ChatMessage, AIMessage, HumanMessage
from langchain.chat_models import ChatAnthropic
import logging
from vocode import getenv

from vocode.streaming.models.agent import ChatAnthropicAgentConfig


from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

from vocode import getenv
from vocode.streaming.models.agent import ChatAnthropicAgentConfig
from langchain.memory import ConversationBufferMemory

SENTENCE_ENDINGS = [".", "!", "?"]


class ChatAnthropicAgent(RespondAgent[ChatAnthropicAgentConfig]):
    def __init__(
        self,
        agent_config: ChatAnthropicAgentConfig,
        logger: Optional[logging.Logger] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        import anthropic

        anthropic_api_key = anthropic_api_key or getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in environment or passed in"
            )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.llm = ChatAnthropic(
            model=agent_config.model_name,  # type: ignore
            anthropic_api_key=anthropic_api_key,
        )

        self.anthropic_client = (
            anthropic.AsyncAnthropic(api_key=anthropic_api_key)
            if agent_config.generate_responses
            else None
        )

        self.memory = ConversationBufferMemory(return_messages=True)
        self.memory.chat_memory.messages.append(
            HumanMessage(content=self.agent_config.prompt_preamble)
        )
        if agent_config.initial_message:
            self.memory.chat_memory.messages.append(
                AIMessage(content=agent_config.initial_message.text)
            )

        self.conversation = ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm
        )

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        text = await self.conversation.apredict(input=human_input)
        self.logger.debug(f"LLM response: {text}")
        return text, False

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        self.memory.chat_memory.messages.append(HumanMessage(content=human_input))

        bot_memory_message = AIMessage(content="")
        self.memory.chat_memory.messages.append(bot_memory_message)
        prompt = self.llm._convert_messages_to_prompt(self.memory.chat_memory.messages)

        if self.anthropic_client:
            streamed_response = await self.anthropic_client.completions.create(
                prompt=prompt,
                max_tokens_to_sample=self.agent_config.max_tokens_to_sample,
                model=self.agent_config.model_name,
                stream=True,
            )

            buffer = ""
            async for completion in streamed_response:
                buffer += completion.completion
                sentence, remainder = get_sentence_from_buffer(buffer)
                if sentence:
                    bot_memory_message.content = bot_memory_message.content + sentence
                    buffer = remainder
                    yield sentence, True
                continue

    def update_last_bot_message_on_cut_off(self, message: str):
        for memory_message in self.memory.chat_memory.messages[::-1]:
            if (
                isinstance(memory_message, ChatMessage)
                and memory_message.role == "assistant"
            ) or isinstance(memory_message, AIMessage):
                memory_message.content = message
                return
