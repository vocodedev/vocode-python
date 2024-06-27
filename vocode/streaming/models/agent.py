from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic.v1 import validator

from vocode.streaming.models.actions import ActionConfig
from vocode.streaming.models.message import BaseMessage

from .model import BaseModel, TypedModel
from .vector_db import VectorDBConfig

FILLER_AUDIO_DEFAULT_SILENCE_THRESHOLD_SECONDS = 0.5
LLM_AGENT_DEFAULT_TEMPERATURE = 1.0
LLM_AGENT_DEFAULT_MAX_TOKENS = 256
LLM_AGENT_DEFAULT_MODEL_NAME = "text-curie-001"
CHAT_GPT_AGENT_DEFAULT_MODEL_NAME = "gpt-3.5-turbo-1106"
CHAT_GPT_AGENT_16K_MODEL_NAME = "gpt-3.5-turbo-0613-16k"
ACTION_AGENT_DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"
CHAT_ANTHROPIC_DEFAULT_MODEL_NAME = "claude-3-haiku-20240307"
CHAT_VERTEX_AI_DEFAULT_MODEL_NAME = "chat-bison@001"
AZURE_OPENAI_DEFAULT_API_TYPE = "azure"
AZURE_OPENAI_DEFAULT_API_VERSION = "2023-07-01-preview"
AZURE_OPENAI_DEFAULT_ENGINE = "gpt-35-turbo"
AZURE_OPENAI_GPT_35_16K_ENGINE = "gpt-35-turbo-16k"
AZURE_OPENAI_GPT_4_ENGINE = "vocode-api-gpt4"
OPENAI_GPT_4_MODEL_NAME = "gpt-4"
OPENAI_GPT_4_32K_MODEL_NAME = "gpt-4-32k"
OPENAI_GPT_4_O_MODEL_NAME = "gpt-4o"
OPENAI_GPT_35_TURBO_1106_MODEL_NAME = "gpt-3.5-turbo-1106"
OPENAI_GPT_4_1106_PREVIEW_MODEL_NAME = "gpt-4-1106-preview"
ANTHROPIC_CLAUDE_3_HAIKU_MODEL_NAME = "claude-3-haiku-20240307"
ANTHROPIC_CLAUDE_3_SONNET_MODEL_NAME = "claude-3-sonnet-20240229"
ANTHROPIC_CLAUDE_3_OPUS_MODEL_NAME = "claude-3-opus-20240229"
GROQ_DEFAULT_MODEL_NAME = "llama3-70b-8192"
GROQ_LLAMA3_8B_MODEL_NAME = "llama3-8b-8192"
GROQ_LLAMA3_70B_MODEL_NAME = "llama3-70b-8192"
GROQ_MIXTRAL_8X7B_MODEL_NAME = "mixtral-8x7b-32768"
GROQ_GEMMA_7B_MODEL_NAME = "gemma-7b-it"

InterruptSensitivity = Literal["low", "high"]


class AgentType(str, Enum):
    BASE = "agent_base"
    LLM = "agent_llm"
    CHAT_GPT_ALPHA = "agent_chat_gpt_alpha"
    CHAT_GPT = "agent_chat_gpt"
    ANTHROPIC = "agent_anthropic"
    CHAT_VERTEX_AI = "agent_chat_vertex_ai"
    ECHO = "agent_echo"
    GPT4ALL = "agent_gpt4all"
    LLAMACPP = "agent_llamacpp"
    GROQ = "agent_groq"
    INFORMATION_RETRIEVAL = "agent_information_retrieval"
    RESTFUL_USER_IMPLEMENTED = "agent_restful_user_implemented"
    WEBSOCKET_USER_IMPLEMENTED = "agent_websocket_user_implemented"
    ACTION = "agent_action"
    LANGCHAIN = "agent_langchain"


class FillerAudioConfig(BaseModel):
    silence_threshold_seconds: float = FILLER_AUDIO_DEFAULT_SILENCE_THRESHOLD_SECONDS
    use_phrases: bool = True
    use_typing_noise: bool = False

    @validator("use_typing_noise")
    def typing_noise_excludes_phrases(cls, v, values):
        if v and values.get("use_phrases"):
            values["use_phrases"] = False
        if not v and not values.get("use_phrases"):
            raise ValueError("must use either typing noise or phrases for filler audio")
        return v


class WebhookConfig(BaseModel):
    url: str


class AzureOpenAIConfig(BaseModel):
    base_url: str
    api_key: str
    region: str
    deployment_name: str
    openai_model_name: str
    api_type: str = AZURE_OPENAI_DEFAULT_API_TYPE
    api_version: Optional[str] = AZURE_OPENAI_DEFAULT_API_VERSION


class CutOffResponse(BaseModel):
    messages: List[BaseMessage] = [BaseMessage(text="Sorry?")]


class AgentConfig(TypedModel, type=AgentType.BASE.value):  # type: ignore
    initial_message: Optional[BaseMessage] = None
    generate_responses: bool = True
    allowed_idle_time_seconds: Optional[float] = None
    num_check_human_present_times: int = 0
    allow_agent_to_be_cut_off: bool = True
    end_conversation_on_goodbye: bool = False
    send_filler_audio: Union[bool, FillerAudioConfig] = False
    webhook_config: Optional[WebhookConfig] = None
    actions: Optional[List[ActionConfig]] = None
    initial_message_delay: float = 0.0
    goodbye_phrases: Optional[List[str]] = None
    interrupt_sensitivity: InterruptSensitivity = "low"
    cut_off_response: Optional[CutOffResponse] = None


class LLMAgentConfig(AgentConfig, type=AgentType.LLM.value):  # type: ignore
    prompt_preamble: str
    model_name: str = LLM_AGENT_DEFAULT_MODEL_NAME
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS


class LLMFallback(BaseModel):
    provider: Literal["openai", "azure"]
    model_name: str


class ChatGPTAgentConfig(AgentConfig, type=AgentType.CHAT_GPT.value):  # type: ignore
    openai_api_key: Optional[str] = None
    prompt_preamble: str
    model_name: str = CHAT_GPT_AGENT_DEFAULT_MODEL_NAME
    base_url_override: Optional[str] = None
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    azure_params: Optional[AzureOpenAIConfig] = None
    vector_db_config: Optional[VectorDBConfig] = None
    # TODO: the below fields should moved up to AgentConfig, and their logic should live in BaseAgent
    use_backchannels: bool = False
    backchannel_probability: float = 0.7
    first_response_filler_message: Optional[str] = None
    llm_fallback: Optional[LLMFallback] = None


class AnthropicAgentConfig(AgentConfig, type=AgentType.ANTHROPIC.value):  # type: ignore
    prompt_preamble: str
    model_name: str = CHAT_ANTHROPIC_DEFAULT_MODEL_NAME
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE


class LangchainAgentConfig(AgentConfig, type=AgentType.LANGCHAIN.value):  # type: ignore
    prompt_preamble: str
    model_name: str
    provider: str
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS


class ChatVertexAIAgentConfig(AgentConfig, type=AgentType.CHAT_VERTEX_AI.value):  # type: ignore
    prompt_preamble: str
    model_name: str = CHAT_VERTEX_AI_DEFAULT_MODEL_NAME
    generate_responses: bool = False  # Google Vertex AI doesn't support streaming


class GroqAgentConfig(AgentConfig, type=AgentType.GROQ.value):  # type: ignore
    groq_api_key: Optional[str] = None
    prompt_preamble: str
    model_name: str = GROQ_DEFAULT_MODEL_NAME
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    vector_db_config: Optional[VectorDBConfig] = None
    # TODO: the below fields should moved up to AgentConfig, and their logic should live in BaseAgent
    use_backchannels: bool = False
    backchannel_probability: float = 0.7
    first_response_filler_message: Optional[str] = None


class InformationRetrievalAgentConfig(
    AgentConfig, type=AgentType.INFORMATION_RETRIEVAL.value  # type: ignore
):
    recipient_descriptor: str
    caller_descriptor: str
    goal_description: str
    fields: List[str]
    # TODO: add fields for IVR, voicemail


class EchoAgentConfig(AgentConfig, type=AgentType.ECHO.value):  # type: ignore
    pass


class GPT4AllAgentConfig(AgentConfig, type=AgentType.GPT4ALL.value):  # type: ignore
    prompt_preamble: str
    model_path: str
    generate_responses: bool = False


class RESTfulUserImplementedAgentConfig(
    AgentConfig, type=AgentType.RESTFUL_USER_IMPLEMENTED.value  # type: ignore
):
    class EndpointConfig(BaseModel):
        url: str
        method: str = "POST"

    respond: EndpointConfig
    generate_responses: bool = False
    # generate_response: Optional[EndpointConfig]


class RESTfulAgentInput(BaseModel):
    conversation_id: str
    human_input: str


class RESTfulAgentOutputType(str, Enum):
    BASE = "restful_agent_base"
    TEXT = "restful_agent_text"
    END = "restful_agent_end"


class RESTfulAgentOutput(TypedModel, type=RESTfulAgentOutputType.BASE):  # type: ignore
    pass


class RESTfulAgentText(RESTfulAgentOutput, type=RESTfulAgentOutputType.TEXT):  # type: ignore
    response: str


class RESTfulAgentEnd(RESTfulAgentOutput, type=RESTfulAgentOutputType.END):  # type: ignore
    pass
