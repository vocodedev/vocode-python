from typing import Union
from pydantic.v1 import BaseModel
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.message import BaseMessage


class AgentResponse(BaseModel):
    message: Union[BaseMessage, EndOfTurn]
    is_interruptible: bool = True
    # Whether the message is the first message in the response; has metrics implications
    is_first: bool = False
    # If the response is not being chunked up into multiple sentences, this is set to True
    is_sole_text_chunk: bool = False
