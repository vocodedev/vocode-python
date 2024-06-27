import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_community.chat_models import ChatVertexAI
from loguru import logger

from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import ChatVertexAIAgentConfig


class ChatVertexAIAgent(RespondAgent[ChatVertexAIAgentConfig]):
    def __init__(
        self,
        agent_config: ChatVertexAIAgentConfig,
    ):
        super().__init__(agent_config=agent_config)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.llm = ChatVertexAI()

        self.memory = ConversationBufferMemory(return_messages=True)
        self.memory.chat_memory.messages.append(
            SystemMessage(content=self.agent_config.prompt_preamble)
        )

        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)
        if agent_config.initial_message:
            raise NotImplementedError("initial_message not supported for Vertex AI")
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Optional[str]:
        # Vertex AI doesn't allow async, so we run in a separate thread
        text = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor,
            lambda input: self.conversation.predict(input=input),
            human_input,
        )

        logger.debug(f"LLM response: {text}")
        return text
