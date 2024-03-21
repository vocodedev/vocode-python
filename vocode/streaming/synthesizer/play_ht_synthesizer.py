import asyncio
import logging
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from vocode import getenv
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig, SynthesizerType
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer,
)
from vocode.streaming.utils.mp3_helper import decode_mp3

TTS_ENDPOINT = "https://api.play.ht/api/v2/tts/stream"


class PlayHtSynthesizer(BaseSynthesizer[PlayHtSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: PlayHtSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[ClientSession] = None,
        max_backoff_retries=3,
        backoff_retry_delay=2,
    ):
        super().__init__(synthesizer_config, aiohttp_session)
        self.synthesizer_config = synthesizer_config
        self.api_key = synthesizer_config.api_key or getenv("PLAY_HT_API_KEY")
        self.user_id = synthesizer_config.user_id or getenv("PLAY_HT_USER_ID")
        if not self.api_key or not self.user_id:
            raise ValueError(
                "You must set the PLAY_HT_API_KEY and PLAY_HT_USER_ID environment variables"
            )
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.max_backoff_retries = max_backoff_retries
        self.backoff_retry_delay = backoff_retry_delay

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        headers = {
            "AUTHORIZATION": self.api_key,
            "X-USER-ID": self.user_id,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        body = {
            "quality": "draft",
            "voice": self.synthesizer_config.voice,
            "text": message.text,
            "sample_rate": self.synthesizer_config.sampling_rate,
            "voice_engine": self.synthesizer_config.voice_engine,
        }
        if self.synthesizer_config.speed:
            body["speed"] = self.synthesizer_config.speed
        if self.synthesizer_config.seed:
            body["seed"] = self.synthesizer_config.seed
        if self.synthesizer_config.temperature:
            body["temperature"] = self.synthesizer_config.temperature

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.PLAY_HT.value.split('_', 1)[-1]}.create_total",
        )

        backoff_retry_delay = self.backoff_retry_delay
        max_backoff_retries = self.max_backoff_retries

        for attempt in range(max_backoff_retries):
            response = await self.aiohttp_session.post(
                TTS_ENDPOINT,
                headers=headers,
                json=body,
                timeout=ClientTimeout(total=15),
            )

            if response.status == 429 and attempt < max_backoff_retries - 1:
                await asyncio.sleep(backoff_retry_delay)
                backoff_retry_delay *= 2  # Exponentially increase delay
                continue

            if not response.ok:
                raise Exception(f"Play.ht API error status code {response.status}")

            if self.experimental_streaming:
                return SynthesisResult(
                    self.experimental_mp3_streaming_output_generator(
                        response, chunk_size, create_speech_span
                    ),
                    lambda seconds: self.get_message_cutoff_from_voice_speed(
                        message, seconds, self.words_per_minute
                    ),
                )
            else:
                read_response = await response.read()
                create_speech_span.end()
                convert_span = tracer.start_span(
                    f"synthesizer.{SynthesizerType.PLAY_HT.value.split('_', 1)[-1]}.convert",
                )
                output_bytes_io = decode_mp3(read_response)

                result = self.create_synthesis_result_from_wav(
                    synthesizer_config=self.synthesizer_config,
                    file=output_bytes_io,
                    message=message,
                    chunk_size=chunk_size,
                )
                convert_span.end()
                return result

        raise Exception("Max retries reached for Play.ht API")
