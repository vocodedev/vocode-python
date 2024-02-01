import os
from vocode.streaming.pubsub.base_pubsub import PubSubManager

environment = {}


def setenv(**kwargs):
    for key, value in kwargs.items():
        environment[key] = value


def getenv(key, default=None):
    return environment.get(key) or os.getenv(key, default)


api_key = getenv("VOCODE_API_KEY")
base_url = getenv("VOCODE_BASE_URL", "api.vocode.dev")


pubsub = PubSubManager()
