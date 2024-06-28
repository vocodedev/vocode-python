from abc import abstractmethod
from loguru import logger
from vocode.streaming.agent.agent_response import AgentResponse
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.synthesizer.abstract_synthesizer import (
    AbstractSynthesizer,
    SynthesizerConfigType,
)
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult


class AbstractInputStreamingSynthesizer(AbstractSynthesizer[SynthesizerConfigType]):

    @abstractmethod
    def get_current_utterance_synthesis_result(self):
        raise NotImplementedError

    @abstractmethod
    async def send_token_to_synthesizer(
        self,
        message: LLMToken,
        chunk_size: int,
    ):
        raise NotImplementedError

    async def _synthesize_agent_response(
        self, agent_response: AgentResponse
    ) -> SynthesisResult | None:
        if isinstance(agent_response.message, LLMToken):
            logger.debug("Sending chunk to synthesizer")
            await self.send_token_to_synthesizer(
                message=agent_response.message,
                chunk_size=self._chunk_size,
            )
            if self.is_first_text_chunk:
                return self.get_current_utterance_synthesis_result
        elif isinstance(agent_response.message, BaseMessage):
            logger.debug("Synthesizing speech for message")
            return await self.create_speech_with_cache(
                agent_response.message,
                self._chunk_size,
                is_first_text_chunk=self.is_first_text_chunk,
                is_sole_text_chunk=agent_response.is_sole_text_chunk,
            )
