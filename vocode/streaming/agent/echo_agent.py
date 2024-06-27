from typing import AsyncGenerator, Optional, Tuple

from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent
from vocode.streaming.models.agent import EchoAgentConfig
from vocode.streaming.models.message import BaseMessage


class EchoAgent(RespondAgent[EchoAgentConfig]):
    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Optional[str]:
        return human_input

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        yield GeneratedResponse(message=BaseMessage(text=human_input), is_interruptible=True)

    def update_last_bot_message_on_cut_off(self, message: str):
        pass
