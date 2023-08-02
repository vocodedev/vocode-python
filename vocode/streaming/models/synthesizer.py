from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import validator
from vocode.streaming.models.client_backend import OutputAudioConfig

from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_SAMPLING_RATE,
)
from .model import BaseModel, TypedModel
from .audio_encoding import AudioEncoding


class SynthesizerType(str, Enum):
    BASE = "synthesizer_base"
    AZURE = "synthesizer_azure"
    GOOGLE = "synthesizer_google"
    ELEVEN_LABS = "synthesizer_eleven_labs"
    RIME = "synthesizer_rime"
    PLAY_HT = "synthesizer_play_ht"
    GTTS = "synthesizer_gtts"
    STREAM_ELEMENTS = "synthesizer_stream_elements"
    COQUI_TTS = "synthesizer_coqui_tts"
    COQUI = "synthesizer_coqui"
    BARK = "synthesizer_bark"


class SentimentConfig(BaseModel):
    emotions: List[str] = ["angry", "friendly", "sad", "whispering"]

    @validator("emotions")
    def emotions_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("must have at least one emotion")
        return v


class SynthesizerConfig(TypedModel, type=SynthesizerType.BASE.value):
    sampling_rate: int
    audio_encoding: AudioEncoding
    should_encode_as_wav: bool = False
    sentiment_config: Optional[SentimentConfig] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_output_device(cls, output_device: BaseOutputDevice, **kwargs):
        return cls(
            sampling_rate=output_device.sampling_rate,
            audio_encoding=output_device.audio_encoding,
            **kwargs
        )

    # TODO(EPD-186): switch to from_twilio_output_device and from_vonage_output_device
    @classmethod
    def from_telephone_output_device(cls, **kwargs):
        return cls(
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=DEFAULT_AUDIO_ENCODING,
            **kwargs
        )

    @classmethod
    def from_output_audio_config(cls, output_audio_config: OutputAudioConfig, **kwargs):
        return cls(
            sampling_rate=output_audio_config.sampling_rate,
            audio_encoding=output_audio_config.audio_encoding,
            **kwargs
        )


AZURE_SYNTHESIZER_DEFAULT_VOICE_NAME = "en-US-SteffanNeural"
AZURE_SYNTHESIZER_DEFAULT_PITCH = 0
AZURE_SYNTHESIZER_DEFAULT_RATE = 15


class AzureSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.AZURE.value):
    voice_name: str = AZURE_SYNTHESIZER_DEFAULT_VOICE_NAME
    pitch: int = AZURE_SYNTHESIZER_DEFAULT_PITCH
    rate: int = AZURE_SYNTHESIZER_DEFAULT_RATE


DEFAULT_GOOGLE_LANGUAGE_CODE = "en-US"
DEFAULT_GOOGLE_VOICE_NAME = "en-US-Neural2-I"
DEFAULT_GOOGLE_PITCH = 0
DEFAULT_GOOGLE_SPEAKING_RATE = 1.2


class GoogleSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.GOOGLE.value):
    language_code: str = DEFAULT_GOOGLE_LANGUAGE_CODE
    voice_name: str = DEFAULT_GOOGLE_VOICE_NAME
    pitch: float = DEFAULT_GOOGLE_PITCH
    speaking_rate: float = DEFAULT_GOOGLE_SPEAKING_RATE


ELEVEN_LABS_ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"


class ElevenLabsSynthesizerConfig(
    SynthesizerConfig, type=SynthesizerType.ELEVEN_LABS.value
):
    api_key: Optional[str] = None
    voice_id: Optional[str] = ELEVEN_LABS_ADAM_VOICE_ID
    stability: Optional[float]
    similarity_boost: Optional[float]
    optimize_streaming_latency: Optional[int]
    model_id: Optional[str]

    @validator("voice_id")
    def set_name(cls, voice_id):
        return voice_id or ELEVEN_LABS_ADAM_VOICE_ID

    @validator("similarity_boost", always=True)
    def stability_and_similarity_boost_check(cls, similarity_boost, values):
        stability = values.get("stability")
        if (stability is None) != (similarity_boost is None):
            raise ValueError(
                "Both stability and similarity_boost must be set or not set."
            )
        return similarity_boost

    @validator("optimize_streaming_latency")
    def optimize_streaming_latency_check(cls, optimize_streaming_latency):
        if optimize_streaming_latency is not None and not (
            0 <= optimize_streaming_latency <= 4
        ):
            raise ValueError("optimize_streaming_latency must be between 0 and 4.")
        return optimize_streaming_latency


RIME_DEFAULT_SPEAKER = "young_male_unmarked-1"
RIME_DEFAULT_SAMPLE_RATE = 22050
RIME_DEFAULT_BASE_URL = "https://rjmopratfrdjgmfmaios.functions.supabase.co/rime-tts"
RIME_DEFAULT_USE_NEW_FORMAT = False


class RimeSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.RIME.value):
    speaker: str = RIME_DEFAULT_SPEAKER
    sampling_rate: int = RIME_DEFAULT_SAMPLE_RATE
    base_url: str = RIME_DEFAULT_BASE_URL
    use_new_format: bool = RIME_DEFAULT_USE_NEW_FORMAT


COQUI_DEFAULT_SPEAKER_ID = "ebe2db86-62a6-49a1-907a-9a1360d4416e"


class CoquiSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.COQUI.value):
    api_key: Optional[str] = None
    voice_id: Optional[str] = COQUI_DEFAULT_SPEAKER_ID
    voice_prompt: Optional[str] = None
    use_xtts: Optional[bool] = True

    @validator("voice_id", always=True)
    def override_voice_id_with_prompt(cls, voice_id, values):
        if values.get("voice_prompt"):
            return None
        return voice_id or COQUI_DEFAULT_SPEAKER_ID


PLAYHT_DEFAULT_VOICE_ID = "larry"


class PlayHtSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.PLAY_HT.value):
    voice_id: str = PLAYHT_DEFAULT_VOICE_ID
    speed: Optional[str] = None
    preset: Optional[str] = None


class CoquiTTSSynthesizerConfig(
    SynthesizerConfig, type=SynthesizerType.COQUI_TTS.value
):
    tts_kwargs: dict = {}
    speaker: Optional[str] = None
    language: Optional[str] = None


class GTTSSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.GTTS.value):
    pass


STREAM_ELEMENTS_SYNTHESIZER_DEFAULT_VOICE = "Brian"


class StreamElementsSynthesizerConfig(
    SynthesizerConfig, type=SynthesizerType.STREAM_ELEMENTS.value
):
    voice: str = STREAM_ELEMENTS_SYNTHESIZER_DEFAULT_VOICE


class BarkSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.BARK.value):
    preload_kwargs: Dict[str, Any] = {}
    generate_kwargs: Dict[str, Any] = {}
