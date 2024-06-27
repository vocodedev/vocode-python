from typing import Optional, Tuple

from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import GPT4AllAgentConfig
from vocode.turn_based.agent.gpt4all_agent import GPT4AllAgent as TurnBasedGPT4AllAgent

raise DeprecationWarning("This Agent is deprecated and will be removed in the future.")


class GPT4AllAgent(RespondAgent[GPT4AllAgentConfig]):
    def __init__(self, agent_config: GPT4AllAgentConfig):
        super().__init__(agent_config=agent_config)
        self.turn_based_agent = TurnBasedGPT4AllAgent(
            model_path=agent_config.model_path,
            system_prompt=agent_config.prompt_preamble,
            initial_message=(
                agent_config.initial_message.text if agent_config.initial_message else None
            ),
        )

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Optional[str]:
        return await self.turn_based_agent.respond_async(human_input)
