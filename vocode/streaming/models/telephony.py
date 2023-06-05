from typing import Optional
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.model import BaseModel
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.models.transcript import Transcript

class TwilioConfig(BaseModel):
    account_sid: str
    auth_token: str
    record: bool = False


class CallEntity(BaseModel):
    phone_number: str


class CreateInboundCall(BaseModel):
    transcriber_config: Optional[TranscriberConfig] = None
    agent_config: AgentConfig
    synthesizer_config: Optional[SynthesizerConfig] = None
    twilio_sid: str
    conversation_id: Optional[str] = None
    twilio_config: Optional[TwilioConfig] = None


class EndOutboundCall(BaseModel):
    call_id: str
    twilio_config: Optional[TwilioConfig] = None


class CreateOutboundCall(BaseModel):
    recipient: CallEntity
    caller: CallEntity
    transcriber_config: Optional[TranscriberConfig] = None
    agent_config: AgentConfig
    synthesizer_config: Optional[SynthesizerConfig] = None
    conversation_id: Optional[str] = None
    twilio_config: Optional[TwilioConfig] = None
    # TODO add IVR/etc.


class DialIntoZoomCall(BaseModel):
    recipient: CallEntity
    caller: CallEntity
    zoom_meeting_id: str
    zoom_meeting_password: Optional[str]
    transcriber_config: Optional[TranscriberConfig] = None
    agent_config: AgentConfig
    synthesizer_config: Optional[SynthesizerConfig] = None
    conversation_id: Optional[str] = None
    twilio_config: Optional[TwilioConfig] = None


class CallConfig(BaseModel):
    transcriber_config: TranscriberConfig
    agent_config: AgentConfig
    synthesizer_config: SynthesizerConfig
    twilio_config: TwilioConfig
    twilio_sid: str
    twilio_from: Optional[str]
    twilio_to: Optional[str]
    transcript: Optional[Transcript]
