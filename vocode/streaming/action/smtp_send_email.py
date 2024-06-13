from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
import os
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)


class SmtpSendEmailActionConfig(ActionConfig, type=ActionType.SMTP_SEND_EMAIL):
    username: str = Field(..., description="The username for the SMTP server.")
    password: str = Field(..., description="The password for the SMTP server.")
    smtp_host: str = Field(..., description="The host of the SMTP server.")
    smtp_port: int = Field(..., description="The port of the SMTP server.")
    from_name: str = Field(..., description="The name of the sender.")


class SmtpSendEmailParameters(BaseModel):
    recipient_email: str = Field(..., description="The email address of the recipient.")
    body: str = Field(..., description="The body of the email.")
    subject: Optional[str] = Field(None, description="The subject of the email.")


class SmtpSendEmailResponse(BaseModel):
    success: bool


class SmtpSendEmail(
    BaseAction[
        SmtpSendEmailActionConfig, SmtpSendEmailParameters, SmtpSendEmailResponse
    ]
):
    description: str = "Sends an email using Smtp server API."
    parameters_type: Type[SmtpSendEmailParameters] = SmtpSendEmailParameters
    response_type: Type[SmtpSendEmailResponse] = SmtpSendEmailResponse

    async def run(
        self, action_input: ActionInput[SmtpSendEmailParameters]
    ) -> ActionOutput[SmtpSendEmailResponse]:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        # Set up the email
        msg = MIMEMultipart()
        msg["From"] = f"{self.action_config.from_name} <{self.action_config.username}>"
        msg["To"] = action_input.params.recipient_email
        msg["Subject"] = action_input.params.subject

        # Add the body to the email
        msg.attach(MIMEText(action_input.params.body, "plain"))

        # Connect to the SMTP server
        with smtplib.SMTP(
            self.action_config.smtp_host, self.action_config.smtp_port
        ) as server:
            # Authenticate to the SMTP server
            server.starttls()
            server.login(self.action_config.username, self.action_config.password)

            # Send the email
            server.send_message(msg)

        return ActionOutput(
            action_type=self.action_config.type,
            response=SmtpSendEmailResponse(success=True),
        )
