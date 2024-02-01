from enum import Enum
from typing import List, Optional

from pydantic import validator

from vocode.streaming.input_device.base_input_device import BaseInputDevice
from vocode.streaming.models.client_backend import InputAudioConfig
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLING_RATE,
)
from .audio_encoding import AudioEncoding
from .model import TypedModel

AZURE_DEFAULT_LANGUAGE = "en-US"


class TranscriberType(str, Enum):
    BASE = "transcriber_base"
    DEEPGRAM = "transcriber_deepgram"
    GOOGLE = "transcriber_google"
    ASSEMBLY_AI = "transcriber_assembly_ai"
    WHISPER_CPP = "transcriber_whisper_cpp"
    REV_AI = "transcriber_rev_ai"
    AZURE = "transcriber_azure"
    GLADIA = "transcriber_gladia"


class EndpointingType(str, Enum):
    BASE = "endpointing_base"
    TIME_BASED = "endpointing_time_based"
    PUNCTUATION_BASED = "endpointing_punctuation_based"


class EndpointingConfig(TypedModel, type=EndpointingType.BASE):
    pass


class TimeEndpointingConfig(EndpointingConfig, type=EndpointingType.TIME_BASED):
    time_cutoff_seconds: float = 0.4


class PunctuationEndpointingConfig(
    EndpointingConfig, type=EndpointingType.PUNCTUATION_BASED
):
    time_cutoff_seconds: float = 0.4


class TranscriberConfig(TypedModel, type=TranscriberType.BASE.value):
    sampling_rate: int
    audio_encoding: AudioEncoding
    chunk_size: int
    endpointing_config: Optional[EndpointingConfig] = None
    downsampling: Optional[int] = None
    min_interrupt_confidence: Optional[float] = None
    mute_during_speech: bool = False
    publish_audio: bool = False

    @validator("min_interrupt_confidence")
    def min_interrupt_confidence_must_be_between_0_and_1(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("must be between 0 and 1")
        return v

    @classmethod
    def from_input_device(
        cls,
        input_device: BaseInputDevice,
        endpointing_config: Optional[EndpointingConfig] = None,
        **kwargs,
    ):
        return cls(
            sampling_rate=input_device.sampling_rate,
            audio_encoding=input_device.audio_encoding,
            chunk_size=input_device.chunk_size,
            endpointing_config=endpointing_config,
            **kwargs,
        )

    # TODO(EPD-186): switch to from_twilio_input_device and from_vonage_input_device
    @classmethod
    def from_telephone_input_device(
        cls,
        endpointing_config: Optional[EndpointingConfig] = None,
        **kwargs,
    ):
        return cls(
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=DEFAULT_AUDIO_ENCODING,
            chunk_size=DEFAULT_CHUNK_SIZE,
            endpointing_config=endpointing_config,
            **kwargs,
        )

    @classmethod
    def from_input_audio_config(cls, input_audio_config: InputAudioConfig, **kwargs):
        return cls(
            sampling_rate=input_audio_config.sampling_rate,
            audio_encoding=input_audio_config.audio_encoding,
            chunk_size=input_audio_config.chunk_size,
            downsampling=input_audio_config.downsampling,
            **kwargs,
        )


class DeepgramTranscriberConfig(TranscriberConfig, type=TranscriberType.DEEPGRAM.value):
    language: Optional[str] = None
    model: Optional[str] = "nova"
    tier: Optional[str] = None
    version: Optional[str] = None
    keywords: Optional[list] = None


class GladiaTranscriberConfig(TranscriberConfig, type=TranscriberType.GLADIA.value):
    buffer_size_seconds: float = 0.1


class GoogleTranscriberConfig(TranscriberConfig, type=TranscriberType.GOOGLE.value):
    model: Optional[str] = None
    language_code: str = "en-US"


class AzureTranscriberConfig(TranscriberConfig, type=TranscriberType.AZURE.value):
    language: str = AZURE_DEFAULT_LANGUAGE
    candidate_languages: Optional[List[str]] = None


class AssemblyAITranscriberConfig(
    TranscriberConfig, type=TranscriberType.ASSEMBLY_AI.value
):
    buffer_size_seconds: float = 0.1
    word_boost: Optional[List[str]] = None


class WhisperCPPTranscriberConfig(
    TranscriberConfig, type=TranscriberType.WHISPER_CPP.value
):
    buffer_size_seconds: float = 1
    libname: str
    fname_model: str


class RevAITranscriberConfig(TranscriberConfig, type=TranscriberType.REV_AI.value):
    pass
