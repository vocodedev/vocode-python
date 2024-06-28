from typing import TYPE_CHECKING, Optional

from vocode.streaming.models.transcriber import EndpointingConfig
from vocode.streaming.synthesizer.input_streaming_synthesizer import InputStreamingSynthesizer
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.telephony.client.vonage_client import VonageClient
from vocode.streaming.utils.redis_conversation_message_queue import RedisConversationMessageQueue

if TYPE_CHECKING:
    from vocode.streaming.streaming_conversation import StreamingConversation
    from vocode.streaming.telephony.conversation.abstract_phone_conversation import (
        AbstractPhoneConversation,
    )
    from vocode.streaming.telephony.conversation.twilio_phone_conversation import (
        TwilioPhoneConversation,
    )
    from vocode.streaming.telephony.conversation.vonage_phone_conversation import (
        VonagePhoneConversation,
    )


# TODO: make this a proper ABC
class AbstractConversationStateManager:
    @property
    def logger(self):
        raise NotImplementedError

    @property
    def transcript(self):
        raise NotImplementedError

    def get_transcriber_endpointing_config(self) -> Optional[EndpointingConfig]:
        raise NotImplementedError

    def set_transcriber_endpointing_config(self, endpointing_config: EndpointingConfig):
        raise NotImplementedError

    def disable_synthesis(self):
        raise NotImplementedError

    def enable_synthesis(self):
        raise NotImplementedError

    def mute_agent(self):
        raise NotImplementedError

    def unmute_agent(self):
        raise NotImplementedError

    def using_input_streaming_synthesizer(self):
        raise NotImplementedError

    async def terminate_conversation(self):
        raise NotImplementedError

    def get_conversation_id(self):
        raise NotImplementedError


class AbstractPhoneConversationStateManager(AbstractConversationStateManager):
    def get_config_manager(self):
        raise NotImplementedError

    def get_to_phone(self):
        raise NotImplementedError

    def get_from_phone(self):
        raise NotImplementedError


class ConversationStateManager(AbstractConversationStateManager):
    def __init__(self, conversation: "StreamingConversation"):
        self._conversation = conversation
        if not hasattr(self, "redis_message_queue"):
            self.redis_message_queue = RedisConversationMessageQueue()

    @property
    def transcript(self):
        return self._conversation.transcript

    def get_transcriber_endpointing_config(self) -> Optional[EndpointingConfig]:
        return self._conversation.transcriber.get_transcriber_config().endpointing_config

    def set_transcriber_endpointing_config(self, endpointing_config: EndpointingConfig):
        assert self.get_transcriber_endpointing_config() is not None
        self._conversation.transcriber.get_transcriber_config().endpointing_config = (
            endpointing_config
        )

    def disable_synthesis(self):
        self._conversation.synthesis_enabled = False

    def enable_synthesis(self):
        self._conversation.synthesis_enabled = True

    def mute_agent(self):
        self._conversation.agent.is_muted = True

    def unmute_agent(self):
        self._conversation.agent.is_muted = False

    def using_input_streaming_synthesizer(self):
        return isinstance(
            self._conversation.synthesizer,
            InputStreamingSynthesizer,
        )

    async def terminate_conversation(self):
        self._conversation.mark_terminated()

    def set_call_check_for_idle_paused(self, value: bool):
        if not self._conversation:
            return
        self._conversation.set_check_for_idle_paused(value)

    def get_conversation_id(self):
        return self._conversation.id


class PhoneConversationStateManager(
    AbstractPhoneConversationStateManager, ConversationStateManager
):
    def __init__(self, conversation: "AbstractPhoneConversation"):
        ConversationStateManager.__init__(self, conversation.pipeline)
        self._phone_conversation = conversation

    def get_config_manager(self):
        return self._phone_conversation.config_manager

    def get_to_phone(self):
        return self._phone_conversation.to_phone

    def get_from_phone(self):
        return self._phone_conversation.from_phone

    def get_direction(self):
        return self._phone_conversation.direction


class VonagePhoneConversationStateManager(PhoneConversationStateManager):
    def __init__(self, conversation: "VonagePhoneConversation"):
        super().__init__(conversation=conversation)
        self._vonage_phone_conversation = conversation

    def create_vonage_client(self):
        return VonageClient(
            base_url=self._vonage_phone_conversation.base_url,
            maybe_vonage_config=self._vonage_phone_conversation.vonage_config,
        )


class TwilioPhoneConversationStateManager(PhoneConversationStateManager):
    def __init__(self, conversation: "TwilioPhoneConversation"):
        super().__init__(conversation=conversation)
        self._twilio_phone_conversation = conversation

    def get_twilio_config(self):
        return self._twilio_phone_conversation.twilio_config

    def create_twilio_client(self):
        return TwilioClient(
            base_url=self._twilio_phone_conversation.base_url,
            maybe_twilio_config=self.get_twilio_config(),
        )
