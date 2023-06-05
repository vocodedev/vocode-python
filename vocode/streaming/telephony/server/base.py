import logging
from typing import List, Optional
from fastapi import APIRouter, Form, Response
from pydantic import BaseModel
from vocode import getenv
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.synthesizer import (
    AzureSynthesizerConfig,
    SynthesizerConfig,
)
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TranscriberConfig,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.synthesizer.factory import SynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import (
    BaseConfigManager,
)
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLING_RATE,
)

from vocode.streaming.telephony.server.router.calls import CallsRouter
from vocode.streaming.models.telephony import (
    CallConfig,
    TwilioConfig,
)

from vocode.streaming.telephony.conversation.call import Call
from vocode.streaming.telephony.templater import Templater
from vocode.streaming.telephony.utils import create_twilio_client, end_twilio_call
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.transcriber.factory import TranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.events_manager import EventsManager


class InboundCallConfig(BaseModel):
    url: str
    agent_config: AgentConfig
    twilio_config: Optional[TwilioConfig] = None
    transcriber_config: Optional[TranscriberConfig] = None
    synthesizer_config: Optional[SynthesizerConfig] = None


class TelephonyServer:
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        inbound_call_configs: List[InboundCallConfig] = [],
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.config_manager = config_manager
        self.templater = Templater()
        self.events_manager = events_manager
        self.router.include_router(
            CallsRouter(
                base_url=base_url,
                templater=self.templater,
                config_manager=self.config_manager,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=self.events_manager,
                logger=self.logger,
            ).get_router()
        )
        for config in inbound_call_configs:
            self.router.add_api_route(
                config.url,
                self.create_inbound_route(
                    agent_config=config.agent_config,
                    twilio_config=config.twilio_config,
                    transcriber_config=config.transcriber_config,
                    synthesizer_config=config.synthesizer_config,
                ),
                methods=["POST"],
            )
            self.logger.info(
                f"Set up inbound call TwiML at https://{base_url}{config.url}"
            )

    def create_inbound_route(
        self,
        agent_config: AgentConfig,
        twilio_config: Optional[TwilioConfig] = None,
        transcriber_config: Optional[TranscriberConfig] = None,
        synthesizer_config: Optional[SynthesizerConfig] = None,
    ):
        def route(
            twilio_sid: str = Form(alias="CallSid"),
            twilio_from: str = Form(alias="From"),
            twilio_to: str = Form(alias="To"),
        ) -> Response:
            call_config = CallConfig(
                transcriber_config=transcriber_config
                or DeepgramTranscriberConfig(
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    audio_encoding=DEFAULT_AUDIO_ENCODING,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    model="phonecall",
                    tier="nova",
                    endpointing_config=PunctuationEndpointingConfig(),
                ),
                agent_config=agent_config,
                synthesizer_config=synthesizer_config
                or AzureSynthesizerConfig(
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    audio_encoding=DEFAULT_AUDIO_ENCODING,
                ),
                twilio_config=twilio_config
                or TwilioConfig(
                    account_sid=getenv("TWILIO_ACCOUNT_SID"),
                    auth_token=getenv("TWILIO_AUTH_TOKEN"),
                ),
                twilio_sid=twilio_sid,
                twilio_from=twilio_from,
                twilio_to=twilio_to,
                transcript=None,
            )

            conversation_id = create_conversation_id()
            self.config_manager.save_config(conversation_id, call_config)
            return self.templater.get_connection_twiml(
                base_url=self.base_url, call_id=conversation_id
            )

        return route

    async def end_outbound_call(self, conversation_id: str):
        # TODO validation via twilio_client
        call_config = self.config_manager.get_config(conversation_id)
        if not call_config:
            raise ValueError(f"Could not find call config for {conversation_id}")
        end_twilio_call(
            create_twilio_client(call_config.twilio_config),
            call_config.twilio_sid,
        )
        return {"id": conversation_id}

    def get_router(self) -> APIRouter:
        return self.router
