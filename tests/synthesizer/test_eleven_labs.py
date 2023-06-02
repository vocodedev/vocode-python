from pydantic import ValidationError
import pytest
from vocode.streaming.synthesizer.base_synthesizer import SynthesisResult
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from aioresponses import aioresponses
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.audio_encoding import AudioEncoding
from pydub import AudioSegment


async def asset_synthesis_result_valid(synthesizer: ElevenLabsSynthesizer):
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
    eleven_labs_synthesizer_with_api_key: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    await asset_synthesis_result_valid(eleven_labs_synthesizer_with_api_key)


@pytest.mark.asyncio
async def test_with_wrong_api_key(
    eleven_labs_synthesizer_wrong_api_key: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    with pytest.raises(Exception, match="ElevenLabs API returned 401 status code"):
        await eleven_labs_synthesizer_wrong_api_key.create_speech(
            BaseMessage(text="Hello, world!"), 1024
        )


@pytest.mark.asyncio
async def test_with_env_api_key(
    eleven_labs_synthesizer_env_api_key: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    await asset_synthesis_result_valid(eleven_labs_synthesizer_env_api_key)


@pytest.mark.asyncio
async def test_with_stability_similarity(
    eleven_labs_synthesizer_stability_similarity: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    await asset_synthesis_result_valid(eleven_labs_synthesizer_stability_similarity)


@pytest.mark.asyncio
async def test_optimize_streaming_latency(
    eleven_labs_synthesizer_optimize_streaming_latency: ElevenLabsSynthesizer,
    mock_eleven_labs_api_optimize_streaming_latency: aioresponses,
):
    await asset_synthesis_result_valid(
        eleven_labs_synthesizer_optimize_streaming_latency
    )


@pytest.mark.asyncio
async def test_voice_id(
    eleven_labs_synthesizer_voice_id: ElevenLabsSynthesizer,
    mock_eleven_labs_api: aioresponses,
):
    await asset_synthesis_result_valid(eleven_labs_synthesizer_voice_id)


@pytest.mark.asyncio
async def test_create_synthesizer_with_only_stability():
    params = {
        "sampling_rate": 16000,
        "audio_encoding": AudioEncoding.LINEAR16,
        "stability": 1,
    }
    with pytest.raises(
        ValidationError,
        match="Both stability and similarity_boost must be set or not set.",
    ):
        ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig(**params))
