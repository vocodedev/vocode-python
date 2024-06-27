from typing import Optional, Tuple, cast

import aiohttp
from loguru import logger

from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import (
    RESTfulAgentInput,
    RESTfulAgentOutput,
    RESTfulAgentOutputType,
    RESTfulAgentText,
    RESTfulUserImplementedAgentConfig,
)


class RESTfulUserImplementedAgent(RespondAgent[RESTfulUserImplementedAgentConfig]):
    def __init__(
        self,
        agent_config: RESTfulUserImplementedAgentConfig,
    ):
        super().__init__(agent_config)
        if self.agent_config.generate_responses:
            raise NotImplementedError(
                "Use the WebSocket user implemented agent to stream responses"
            )

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Optional[str]:
        config = self.agent_config.respond
        try:
            # TODO: cache session
            async with aiohttp.ClientSession() as session:
                payload = RESTfulAgentInput(
                    human_input=human_input, conversation_id=conversation_id
                ).dict()
                async with session.request(
                    config.method,
                    config.url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    assert response.status == 200
                    output: RESTfulAgentOutput = RESTfulAgentOutput.parse_obj(await response.json())
                    output_response = None
                    if output.type == RESTfulAgentOutputType.TEXT:
                        output_response = cast(RESTfulAgentText, output).response
                    elif output.type == RESTfulAgentOutputType.END:
                        await self.conversation_state_manager.terminate_conversation()
                    return output_response
        except Exception as e:
            logger.error(f"Error in response from RESTful agent: {e}")
            return None
