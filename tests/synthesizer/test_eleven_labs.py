import asyncio
from pydantic import ValidationError
import pytest
from vocode.streaming.synthesizer.base_synthesizer import SynthesisResult
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from aioresponses import aioresponses
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.audio_encoding import AudioEncoding
from pydub import AudioSegment


async def assert_synthesis_result_valid(synthesizer: ElevenLabsSynthesizer):
    response = await synthesizer.create_speech(BaseMessage(text="Hello, world!"), 1024)
    assert isinstance(response, SynthesisResult)
    assert response.chunk_generator is not None
    audio = AudioSegment.empty()
    async for chunk in response.chunk_generator:
        audio += AudioSegment(
            chunk.chunk,
            frame_rate=synthesizer.synthesizer_config.sampling_rate,
            sample_width=2,
            channels=1,
        )


@pytest.mark.asyncio
async def test_with_api_key(
    fixture_eleven_labs_synthesizer_with_api_key: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    await assert_synthesis_result_valid(
        await fixture_eleven_labs_synthesizer_with_api_key
    )


@pytest.mark.asyncio
async def test_with_wrong_api_key(
    fixture_eleven_labs_synthesizer_wrong_api_key: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    with pytest.raises(Exception, match="ElevenLabs API returned 401 status code"):
        await (await fixture_eleven_labs_synthesizer_wrong_api_key).create_speech(
            BaseMessage(text="Hello, world!"), 1024
        )


@pytest.mark.asyncio
async def test_with_env_api_key(
    fixture_eleven_labs_synthesizer_env_api_key: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    await assert_synthesis_result_valid(
        await fixture_eleven_labs_synthesizer_env_api_key
    )


import pytest
from unittest.mock import AsyncMock, patch
import aiohttp
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.synthesizer.eleven_labs_synthesizer import (
    ElevenLabsSynthesizer,
    ElevenLabsSynthesizerConfig,
)


# Mock the API response
@pytest.fixture
def mock_api_response():
    import io

    async def response(*args, **kwargs):
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.ok = True
        mock_response.status = 200
        # Here, simulate a WAV file or the expected audio format
        mock_response.read = AsyncMock(
            return_value=bytearray(
                b"UklGRoRdAABXQVZFZm10IBIAAAAHAAEAQB8AAEAfAAABAAgAAABmYWN0BAAAAABdAABkYXRhAF0AAPv9A="
            )
        )
        return mock_response

    return response


@pytest.fixture
def mock_streaming_api_response():
    import aiohttp
    from unittest.mock import AsyncMock, MagicMock

    async def response(*args, **kwargs):
        # Create a mock for aiohttp.ClientResponse
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.ok = True
        mock_response.status = 200

        # Simulate the streaming of data
        # Here we define the chunks of data that will be streamed
        chunks = [
            bytearray(b"UklGRoRdAABXQV"),
            bytearray(b"ZFZm10IBIAAAAH"),
            bytearray(b"AAEAQB8AAEAfAA"),
            bytearray(b"ABAAgAAABmYWN0"),
            bytearray(b"BAAAAABdAABkYXRhAF0AAPv9A="),
            bytearray(),  # An empty byte array to simulate the end of the stream
        ]

        # Create a mock for aiohttp.StreamReader
        mock_stream_reader = MagicMock(spec=aiohttp.StreamReader)
        mock_stream_reader.iter_any = AsyncMock(side_effect=chunks)

        # Assign the mock stream reader to the content attribute
        mock_response.content = mock_stream_reader

        return mock_response

    return response


# Test the create_speech method
@pytest.mark.asyncio
async def test_create_speech_mu_law_no_streaming(mock_api_response):
    synthesizer_config = ElevenLabsSynthesizerConfig(
        api_key="fake_api_key",
        voice_id="fake_voice_id",
        sampling_rate=8000,
        audio_encoding=AudioEncoding.MULAW,
    )
    synthesizer = ElevenLabsSynthesizer(synthesizer_config)

    # Mock aiohttp.ClientSession.request
    with patch("aiohttp.ClientSession.request", new=mock_api_response):
        result = await synthesizer.create_speech(BaseMessage(text="Hello"), 1024)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.chunk_generator is not None
        audio = AudioSegment.empty()
        async for chunk in result.chunk_generator:
            audio += AudioSegment(
                chunk.chunk,
                frame_rate=synthesizer.synthesizer_config.sampling_rate,
                sample_width=2,
                channels=1,
            )


@pytest.mark.asyncio
async def test_create_speech_mu_law_streaming(mock_streaming_api_response):
    synthesizer_config = ElevenLabsSynthesizerConfig(
        api_key="fake_api_key",
        voice_id="fake_voice_id",
        sampling_rate=8000,
        experimental_streaming=True,
        audio_encoding=AudioEncoding.MULAW,
    )
    synthesizer = ElevenLabsSynthesizer(synthesizer_config)

    # Mock aiohttp.ClientSession.request
    with patch("aiohttp.ClientSession.request", new=mock_streaming_api_response):
        result = await synthesizer.create_speech(BaseMessage(text="Hello"), 1024)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.chunk_generator is not None
        audio = AudioSegment.empty()
        async for chunk in result.chunk_generator:
            audio += AudioSegment(
                chunk.chunk,
                frame_rate=synthesizer.synthesizer_config.sampling_rate,
                sample_width=2,
                channels=1,
            )
