import pytest
from aioresponses import aioresponses, CallbackResult
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    PlayHtSynthesizerConfig,
)
import re
from vocode.streaming.synthesizer.eleven_labs_synthesizer import (
    ElevenLabsSynthesizer,
    ELEVEN_LABS_BASE_URL,
)
from vocode.streaming.synthesizer.play_ht_synthesizer import (
    PlayHtSynthesizer,
    TTS_ENDPOINT,
)

import re
from tests.streaming.data.loader import get_audio_path

import asyncio
import pytest

DEFAULT_PARAMS = {"sampling_rate": 16000, "audio_encoding": AudioEncoding.LINEAR16}

MOCK_API_KEY = "my_api_key"
MOCK_USER_ID = "my_user_id"


def create_eleven_labs_request_handler(optimize_streaming_latency=False):
    def request_handler(url, headers, **kwargs):
        if optimize_streaming_latency and not re.search(
            r"optimize_streaming_latency=\d", url
        ):
            raise Exception("optimize_streaming_latency not found in url")
        if headers["xi-api-key"] != MOCK_API_KEY:
            return CallbackResult(status=401)
        with open(get_audio_path("fake_audio.mp3"), "rb") as audio_file:
            return CallbackResult(content_type="audio/mpeg", body=audio_file.read())

    return request_handler


@pytest.fixture
def mock_eleven_labs_api():
    with aioresponses() as m:
        pattern = re.compile(rf"{re.escape(ELEVEN_LABS_BASE_URL)}text-to-speech/\w+")
        m.post(pattern, callback=create_eleven_labs_request_handler())
        yield m


@pytest.fixture(scope="module")
async def fixture_eleven_labs_synthesizer_with_api_key():
    params = DEFAULT_PARAMS.copy()
    params["api_key"] = MOCK_API_KEY
    return ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig(**params))


@pytest.fixture(scope="module")
async def fixture_eleven_labs_synthesizer_wrong_api_key():
    params = DEFAULT_PARAMS.copy()
    params["api_key"] = "wrong_api_key"
    return ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig(**params))


@pytest.fixture(scope="module")
async def fixture_eleven_labs_synthesizer_env_api_key():
    params = DEFAULT_PARAMS.copy()
    import os

    os.environ["ELEVEN_LABS_API_KEY"] = MOCK_API_KEY
    return ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig(**params))


# PlayHT Setup


def create_play_ht_request_handler():
    def request_handler(url, headers, **kwargs):
        if headers["AUTHORIZATION"] != MOCK_API_KEY:
            return CallbackResult(status=401)
        if headers["X-USER-ID"] != MOCK_USER_ID:
            return CallbackResult(status=401)
        with open(get_audio_path("fake_audio.mp3"), "rb") as audio_file:
            return CallbackResult(content_type="audio/mpeg", body=audio_file.read())

    return request_handler


@pytest.fixture
def mock_play_ht_api():
    with aioresponses() as m:
        m.post(TTS_ENDPOINT, callback=create_play_ht_request_handler())
        yield m


@pytest.fixture(scope="module")
async def fixture_play_ht_synthesizer_with_api_key():
    params = DEFAULT_PARAMS.copy()
    params["api_key"] = MOCK_API_KEY
    params["user_id"] = MOCK_USER_ID
    return PlayHtSynthesizer(PlayHtSynthesizerConfig(**params))


@pytest.fixture(scope="module")
async def fixture_play_ht_synthesizer_wrong_api_key():
    params = DEFAULT_PARAMS.copy()
    params["api_key"] = "wrong_api_key"
    params["user_id"] = MOCK_USER_ID
    return PlayHtSynthesizer(PlayHtSynthesizerConfig(**params))


@pytest.fixture(scope="module")
async def fixture_play_ht_synthesizer_wrong_user_id():
    params = DEFAULT_PARAMS.copy()
    params["api_key"] = MOCK_API_KEY
    params["user_id"] = "wrong_api_key"
    return PlayHtSynthesizer(PlayHtSynthesizerConfig(**params))


@pytest.fixture(scope="module")
async def fixture_play_ht_synthesizer_env_api_key():
    params = DEFAULT_PARAMS.copy()
    import os

    os.environ["PLAY_HT_API_KEY"] = MOCK_API_KEY
    os.environ["PLAY_HT_USER_ID"] = MOCK_USER_ID
    return PlayHtSynthesizer(PlayHtSynthesizerConfig(**params))
