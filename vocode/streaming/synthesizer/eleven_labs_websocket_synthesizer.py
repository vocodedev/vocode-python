import asyncio
import audioop
import base64
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import websockets
from loguru import logger
from pydantic import BaseModel, conint

from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.synthesizer.input_streaming_synthesizer import InputStreamingSynthesizer

NONCE = "071b5f21-3b24-4427-817e-62508007ae60"
ELEVEN_LABS_BASE_URL = "wss://api.elevenlabs.io/v1/"


# Based on https://github.com/elevenlabs/elevenlabs-python/blob/main/src/elevenlabs/tts.py
async def string_chunker(string: str) -> AsyncGenerator[str, None]:
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    for text in string:
        if buffer.endswith(splitters):
            yield buffer if buffer.endswith(" ") else buffer + " "
            buffer = text
        elif text.startswith(splitters):
            output = buffer + text[0]
            yield output if output.endswith(" ") else output + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer != "":
        yield buffer + " "


class ElevenLabsWebsocketVoiceSettings(BaseModel):
    stability: float
    similarity_boost: float


class ElevenLabsWebsocketGenerationConfig(BaseModel):
    chunk_length_schedule: list[conint(ge=50, le=500)]  # type: ignore
    # TODO: Replace above with below for pydantic 2.X
    # chunk_length_schedule: list[Annotated[int, Field(strict=True, ge=50, le=500)]]


class ElevenLabsWebsocketMessage(BaseModel):
    text: str
    try_trigger_generation: bool = True
    flush: bool = False


class ElevenLabsWebsocketFirstMessage(ElevenLabsWebsocketMessage, BaseModel):
    voice_settings: ElevenLabsWebsocketVoiceSettings | None = None
    generation_config: ElevenLabsWebsocketGenerationConfig | None = None
    xi_api_key: str


class ElevenLabsWebsocketResponseAlignment(BaseModel):
    chars: list[str]
    charStartTimesMs: list[int]
    charDurationsMs: list[int]


class ElevenLabsWebsocketResponse(BaseModel):
    audio: Optional[str] = None
    isFinal: Optional[bool] = None
    normalizedAlignment: Optional[ElevenLabsWebsocketResponseAlignment] = None
    alignment: Optional[ElevenLabsWebsocketResponseAlignment] = None

    def __str__(self):
        return f"ElevenLabsWebsocketResponse(has_audio={self.audio is not None}, isFinal={self.isFinal}, alignment={self.alignment})"


class ElevenLabsWSSynthesizer(
    BaseSynthesizer[ElevenLabsSynthesizerConfig], InputStreamingSynthesizer
):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
    ):
        super().__init__(
            synthesizer_config,
        )

        assert synthesizer_config.api_key is not None, "API key must be set"
        assert synthesizer_config.voice_id is not None, "Voice ID must be set"
        self.api_key = synthesizer_config.api_key
        self.voice_id = synthesizer_config.voice_id
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150

        self.text_chunk_queue: asyncio.Queue[Optional[BotBackchannel | LLMToken]] = asyncio.Queue()
        self.voice_packet_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.current_turn_utterances_by_chunk: List[Tuple[str, float]] = []
        self.sample_width = 2 if synthesizer_config.audio_encoding == AudioEncoding.LINEAR16 else 1

        self.websocket_listener: asyncio.Task | None = None
        self.websocket_tasks: dict[str, asyncio.Task | None] = {
            "listener": None,
            "writer": None,
        }
        self.end_of_turn = False

        self.upsample = None
        self.sample_rate = self.synthesizer_config.sampling_rate

        # While this looks useless, we need to assign the response of `asyncio.gather`
        # to *something* or we risk garbage collection of the running coroutines spawned
        # by `asyncio.gather`.
        self.websocket_functions: list[None] | None = None

        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            match self.synthesizer_config.sampling_rate:
                case SamplingRate.RATE_16000:
                    self.output_format = "pcm_16000"
                case SamplingRate.RATE_22050:
                    self.output_format = "pcm_22050"
                case SamplingRate.RATE_24000:
                    self.output_format = "pcm_24000"
                case SamplingRate.RATE_44100:
                    self.output_format = "pcm_44100"
                case SamplingRate.RATE_48000:
                    self.output_format = "pcm_44100"
                    self.upsample = SamplingRate.RATE_48000.value
                    self.sample_rate = SamplingRate.RATE_44100.value
                case _:
                    raise ValueError(
                        f"Unsupported sampling rate: {self.synthesizer_config.sampling_rate}. Elevenlabs only supports 16000, 22050, 24000, and 44100 Hz."
                    )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            self.output_format = "ulaw_8000"
        else:
            raise ValueError(
                f"Unsupported audio encoding: {self.synthesizer_config.audio_encoding}"
            )

    def get_eleven_labs_websocket_voice_settings(self):
        if self.stability is None or self.similarity_boost is None:
            return None
        return ElevenLabsWebsocketVoiceSettings(
            stability=self.stability,
            similarity_boost=self.similarity_boost,
        )

    def reduce_chunk_amplitude(self, chunk: bytes, factor: float) -> bytes:
        if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            chunk = audioop.ulaw2lin(chunk, 2)
        pcm = np.frombuffer(chunk, dtype=np.int16)
        pcm = (pcm * factor).astype(np.int16)
        pcm_bytes = pcm.tobytes()
        if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            return audioop.lin2ulaw(pcm_bytes, 2)
        else:
            return pcm_bytes

    async def establish_websocket_listeners(self, chunk_size):
        url = (
            ELEVEN_LABS_BASE_URL
            + f"text-to-speech/{self.voice_id}/stream-input?output_format={self.output_format}"
        )
        if self.optimize_streaming_latency:
            url += f"&optimize_streaming_latency={self.optimize_streaming_latency}"
        if self.model_id:
            url += f"&model_id={self.model_id}"
        headers = {"xi-api-key": self.api_key}

        backchannelled = False

        async with websockets.connect(
            url,
            extra_headers=headers,
        ) as ws:

            async def write() -> None:
                nonlocal backchannelled
                try:
                    first_message = True
                    while True:
                        message = await self.text_chunk_queue.get()
                        if not message:
                            break
                        if first_message and isinstance(message, BotBackchannel):
                            backchannelled = True
                        eleven_labs_ws_message = (
                            ElevenLabsWebsocketMessage(
                                text=message.text,
                                flush=not isinstance(message, LLMToken),
                            ).json()
                            if not first_message
                            else ElevenLabsWebsocketFirstMessage(
                                text=message.text,
                                voice_settings=self.get_eleven_labs_websocket_voice_settings(),
                                generation_config=ElevenLabsWebsocketGenerationConfig(
                                    chunk_length_schedule=[50],
                                ),
                                flush=not isinstance(message, LLMToken),
                                xi_api_key=self.api_key,
                            ).json(exclude_none=True)
                        )
                        await ws.send(eleven_labs_ws_message)
                        first_message = False
                finally:
                    await ws.send(ElevenLabsWebsocketMessage(text="").json())

            async def listen() -> None:
                """Listen to the websocket for audio data and stream it."""

                first_message = True
                buffer = bytearray()
                while True:
                    message = await ws.recv()
                    if "audio" not in message:
                        continue
                    response = ElevenLabsWebsocketResponse.model_validate_json(message)
                    if response.audio:
                        decoded = base64.b64decode(response.audio)
                        seconds = len(decoded) / (
                            self.sample_width * self.synthesizer_config.sampling_rate
                        )

                        if self.upsample:
                            decoded = self._resample_chunk(
                                decoded,
                                self.sample_rate,
                                self.upsample,
                            )
                            seconds = len(decoded) / (self.sample_width * self.sample_rate)

                        if response.alignment:
                            utterance_chunk = "".join(response.alignment.chars) + " "
                            self.current_turn_utterances_by_chunk.append((utterance_chunk, seconds))
                        # For backchannels, send them all as one chunk (so it can't be interrupted) and reduce the volume
                        # so that in the case of a false endpoint, the backchannel is not too loud.
                        if first_message and backchannelled:
                            buffer.extend(decoded)
                            logger.info("First message was a backchannel, reducing volume.")
                            reduced_amplitude_buffer = self.reduce_chunk_amplitude(
                                buffer, factor=self.synthesizer_config.backchannel_amplitude_factor
                            )
                            await self.voice_packet_queue.put(reduced_amplitude_buffer)
                            buffer = bytearray()
                            first_message = False
                        else:
                            buffer.extend(decoded)
                            for chunk_idx in range(0, len(buffer) - chunk_size, chunk_size):
                                await self.voice_packet_queue.put(
                                    buffer[chunk_idx : chunk_idx + chunk_size]
                                )
                            buffer = buffer[len(buffer) - (len(buffer) % chunk_size) :]

                    if response.isFinal:
                        await self.voice_packet_queue.put(None)
                        break

            self.websocket_tasks["listener"] = asyncio.create_task(listen())
            self.websocket_tasks["writer"] = asyncio.create_task(write())
            self.websocket_functions = await asyncio.gather(*self.websocket_tasks.values())

    def get_current_utterance_synthesis_result(self):
        return SynthesisResult(
            self.chunk_result_generator_from_queue(self.voice_packet_queue),
            lambda seconds: self.get_current_message_so_far(seconds),
        )

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ):
        """
        Ran when doing utterance parsing.
        ie: "Hello, my name is foo."
        """
        if not self.websocket_listener:
            self.websocket_listener = asyncio.create_task(
                self.establish_websocket_listeners(chunk_size)
            )

        if isinstance(message, BotBackchannel):
            if not message.text.endswith(" "):
                message.text += " "
            await self.text_chunk_queue.put(message)
            self.total_chars += len(message.text)
        else:
            async for text in string_chunker(message.text):
                await self.text_chunk_queue.put(LLMToken(text=text))
                self.total_chars += len(text)

        return self.get_current_utterance_synthesis_result()

    async def send_token_to_synthesizer(self, message: LLMToken, chunk_size: int):
        """
        Ran when parsing a single chunk of text.

        ie: "Hello,"
        """
        self.total_chars += len(message.text)

        if not self.websocket_listener:
            self.websocket_listener = asyncio.create_task(
                self.establish_websocket_listeners(chunk_size)
            )

        await self.text_chunk_queue.put(message)
        return None

    def _cleanup_websocket_tasks(self):
        for task in self.websocket_tasks.values():
            if task is not None:
                task.cancel()
        self.text_chunk_queue = asyncio.Queue()
        self.voice_packet_queue = asyncio.Queue()
        if self.websocket_listener is not None:
            self.websocket_listener.cancel()

    def ready_synthesizer(self, chunk_size: int):
        self._cleanup_websocket_tasks()
        self.websocket_listener = asyncio.create_task(
            self.establish_websocket_listeners(chunk_size)
        )

    def get_current_message_so_far(self, seconds: Optional[float]) -> str:
        seconds_idx = 0.0
        buffer = ""
        for utterance, duration in self.current_turn_utterances_by_chunk:
            if seconds is not None and seconds_idx > seconds:
                return buffer
            buffer += utterance
            seconds_idx += duration
        return buffer

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: ElevenLabsSynthesizerConfig):
        return ElevenLabsSynthesizer.get_voice_identifier(synthesizer_config)

    async def handle_end_of_turn(self):
        self.end_of_turn = True
        await self.text_chunk_queue.put(None)
        self.current_turn_utterances_by_chunk = []

    async def cancel_websocket_tasks(self):
        self._cleanup_websocket_tasks()

    async def tear_down(self):
        await self.cancel_websocket_tasks()
        await super().tear_down()
