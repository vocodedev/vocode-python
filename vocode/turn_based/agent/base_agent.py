from typing import List, Optional


class BaseAgent:
    def __init__(self, system_prompt: str, initial_message: Optional[str] = None):
        self.system_prompt = system_prompt
        self.initial_message = initial_message

    def respond(self, human_input: str):
        raise NotImplementedError

    def respond_to_message_history(self, messages: List = []):
        raise NotImplementedError
