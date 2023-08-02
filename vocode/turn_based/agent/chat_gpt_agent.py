from typing import List, Optional
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from vocode import getenv

from vocode.turn_based.agent.base_agent import BaseAgent


class ChatGPTAgent(BaseAgent):
    def __init__(
        self,
        system_prompt: str,
        api_key: Optional[str] = None,
        initial_message: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 100,
        memory: Optional[ConversationBufferMemory] = None,
    ):
        super().__init__(system_prompt=system_prompt, initial_message=initial_message)
        openai.api_key = getenv("OPENAI_API_KEY", api_key)
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = (
            memory if memory else ConversationBufferMemory(return_messages=True)
        )
        if initial_message:
            self.memory.chat_memory.add_ai_message(initial_message)
        self.llm = ChatOpenAI(  # type: ignore
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.conversation = ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm
        )

    def respond(self, human_input: str):
        return self.conversation.predict(input=human_input)

    def respond_to_message_history(self, messages: List = []):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
