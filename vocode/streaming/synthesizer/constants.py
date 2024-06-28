import os
from vocode.streaming.models.message import BaseMessage


FILLER_PHRASES = [
    BaseMessage(text="Um..."),
    BaseMessage(text="Uh..."),
    BaseMessage(text="Uh-huh..."),
    BaseMessage(text="Mm-hmm..."),
    BaseMessage(text="Hmm..."),
    BaseMessage(text="Okay..."),
    BaseMessage(text="Right..."),
    BaseMessage(text="Let me see..."),
]
FILLER_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "filler_audio_data")
TYPING_NOISE_PATH = "%s/typing-noise.wav" % FILLER_AUDIO_PATH
