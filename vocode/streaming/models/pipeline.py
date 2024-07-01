from enum import Enum

from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.model import TypedModel
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.transcriber import TranscriberConfig


class PipelineType(str, Enum):
    BASE = "pipeline_base"
    STREAMING_CONVERSATION = "pipeline_streaming_conversation"


class PipelineConfig(TypedModel, type=PipelineType.BASE):
    pass


class StreamingConversationConfig(TypedModel, type=PipelineType.STREAMING_CONVERSATION):
    transcriber_config: TranscriberConfig
    agent_config: AgentConfig
    synthesizer_config: SynthesizerConfig
    speed_coefficient: float = 1.0
