import logging
import time

from typing import Optional
from twilio.rest import Client
from xml.etree import ElementTree as ET

from vocode.streaming.models.telephony import BaseCallConfig, TwilioConfig
from vocode.streaming.telephony.client.base_telephony_client import BaseTelephonyClient
from vocode.streaming.telephony.templater import Templater


class TwilioClient(BaseTelephonyClient):
    def __init__(self, base_url: str, twilio_config: TwilioConfig):
        super().__init__(base_url)
        self.twilio_config = twilio_config
        # TODO: this is blocking
        self.twilio_client = Client(twilio_config.account_sid, twilio_config.auth_token)
        try:
            # Test credentials
            self.twilio_client.api.accounts(twilio_config.account_sid).fetch()
        except Exception as e:
            raise RuntimeError(
                "Could not create Twilio client. Invalid credentials"
            ) from e
        self.templater = Templater()

    def get_telephony_config(self):
        return self.twilio_config

    async def create_call(
        self,
        conversation_id: str,
        to_phone: str,
        from_phone: str,
        record: bool = False,
        digits: Optional[str] = None,
    ) -> str:
        # TODO: Make this async. This is blocking.
        twiml = self.get_connection_twiml(conversation_id=conversation_id)
        twilio_call = self.twilio_client.calls.create(
            twiml=twiml.body.decode("utf-8"),
            to=to_phone,
            from_=from_phone,
            send_digits=digits,
            record=record,
            **self.get_telephony_config().extra_params,
        )
        return twilio_call.sid

    def get_connection_twiml(self, conversation_id: str):
        return self.templater.get_connection_twiml(
            base_url=self.base_url, call_id=conversation_id
        )

    def end_call(self, twilio_sid):
        logging.info("I am ending the call now within the twilio client code")
        current_call = self.twilio_client.calls(twilio_sid).fetch()

        logging.info(f"The current parent call SID is {current_call.parent_call_sid}")
        # if the call is part of a conference, we should just let it keep going
        if current_call.parent_call_sid is not None:
            return False

        # fail-safe in case something is really wrong with the call and it still hasn't hung up
        call = self.twilio_client.calls(twilio_sid).fetch()

        # Check the call's duration
        if (
            call.duration is not None and int(call.duration) > 5 * 60
        ):  # duration is in seconds
            # The call has been going for more than 5 minutes - terminate it
            response = self.twilio_client.calls(twilio_sid).update(status="completed")
            return response.status == "completed"

        time.sleep(
            10
        )  # for testing purposes only, if for some reason it just keeps going even when it's a conference
        response = self.twilio_client.calls(twilio_sid).update(status="completed")
        return response.status == "completed"

    def validate_outbound_call(
        self,
        to_phone: str,
        from_phone: str,
        mobile_only: bool = False,  # originally to conform with California law; we leave as False for testing purposes
    ):
        if len(to_phone) < 8:
            raise ValueError("Invalid 'to' phone")

        if not mobile_only:
            return
        line_type_intelligence = (
            self.twilio_client.lookups.v2.phone_numbers(to_phone)
            .fetch(fields="line_type_intelligence")
            .line_type_intelligence
        )
        if not line_type_intelligence or (
            line_type_intelligence and line_type_intelligence["type"] != "mobile"
        ):
            raise ValueError("Can only call mobile phones")
