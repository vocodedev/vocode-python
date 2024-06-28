import asyncio
import audioop
import os
from typing import AsyncGenerator, AsyncIterator, Optional

import numpy as np
from loguru import logger
from pyht import AsyncClient
from pyht.client import CongestionCtrl, TTSOptions
from pyht.protos import api_pb2

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.synthesizer.play_ht_synthesizer import (
    PlayHtSynthesizer as VocodePlayHtSynthesizer,
)
from vocode.streaming.synthesizer.synthesizer_utils import (
    chunk_result_generator_from_queue,
    get_message_cutoff_from_voice_speed,
    split_text,
)
from vocode.streaming.utils import generate_from_async_iter_with_lookahead, generate_with_is_last
from vocode.streaming.utils.create_task import asyncio_create_task_with_done_error_log

PLAY_HT_ON_PREM_ADDR = os.environ.get("VOCODE_PLAYHT_ON_PREM_ADDR", None)
PLAY_HT_V2_MAX_CHARS = 200
EXPERIMENTAL_VOICE_AMPLITUDE_THRESHOLD = 200


class PlayHtSynthesizerV2(VocodePlayHtSynthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.playht_client_saas = AsyncClient(
            user_id=self.user_id,
            api_key=self.api_key,
        )
        self.playht_client_on_prem = None

        if self.synthesizer_config.on_prem:
            logger.info(f"Creating on-prem PlayHT with gRPC address {PLAY_HT_ON_PREM_ADDR}")
            advanced_options = AsyncClient.AdvancedOptions(
                grpc_addr=PLAY_HT_ON_PREM_ADDR,
                fallback_enabled=True,
                congestion_ctrl=CongestionCtrl.STATIC_MAR_2023,
            )

            self.playht_client_on_prem = AsyncClient(
                user_id=self.user_id,
                api_key=self.api_key,
                advanced=advanced_options,
            )

        if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            audio_format = api_pb2.FORMAT_MULAW
        elif self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            audio_format = api_pb2.FORMAT_WAV

        sample_rate = self.synthesizer_config.sampling_rate
        self.playht_options = TTSOptions(
            voice=self.synthesizer_config.voice_id,
            # PlayHT runs significantly slower when sampling rate is not 24KHz
            # 24KHz is the default for PlayHT
            sample_rate=sample_rate if sample_rate > 24000 else 24000,
            speed=self.synthesizer_config.speed if self.synthesizer_config.speed else 1,
            format=audio_format,
            text_guidance=self.synthesizer_config.text_guidance,
            voice_guidance=self.synthesizer_config.voice_guidance,
            temperature=self.synthesizer_config.temperature,
            top_p=self.synthesizer_config.top_p,
        )
        if self.synthesizer_config.quality:
            self.playht_options.quality = self.synthesizer_config.quality

    @property
    def playht_client(self) -> AsyncClient:
        if self.playht_client_on_prem is not None:
            return self.playht_client_on_prem
        return self.playht_client_saas

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:

        self.total_chars += len(message.text)
        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        asyncio_create_task_with_done_error_log(
            self.get_chunks(
                message,
                chunk_size,
                chunk_queue,
                cut_leading_silence=not is_first_text_chunk
                and self.synthesizer_config.experimental_remove_silence,
                cut_trailing_silence=not is_sole_text_chunk
                and self.synthesizer_config.experimental_remove_silence,
            ),
        )

        return SynthesisResult(
            chunk_result_generator_from_queue(chunk_queue),
            lambda seconds: get_message_cutoff_from_voice_speed(
                message,
                seconds,
                self.words_per_minute,
            ),
        )

    def _contains_voice_experimental(self, chunk: bytes):
        pcm = np.frombuffer(
            (
                audioop.ulaw2lin(chunk, 2)
                if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW
                else chunk
            ),
            dtype=np.int16,
        )
        return np.max(np.abs(pcm)) > EXPERIMENTAL_VOICE_AMPLITUDE_THRESHOLD

    @staticmethod
    def _enumerate_by_chunk_size(buffer: bytes, chunk_size: int):
        for buffer_idx in range(0, len(buffer) - chunk_size, chunk_size):
            yield buffer_idx, buffer[buffer_idx : buffer_idx + chunk_size]

    async def _downsample_from_24khz(self, chunk: bytes) -> bytes:
        if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            return await self._downsample_mulaw(chunk)
        elif self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            return await self._downsample_pcm(chunk)
        else:
            raise Exception(f"Unsupported audio format: {self.synthesizer_config.audio_encoding}")

    async def _downsample_pcm(self, chunk: bytes) -> bytes:
        downsampled_chunk, _ = audioop.ratecv(
            chunk,
            2,
            1,
            24000,
            self.synthesizer_config.sampling_rate,
            None,
        )

        return downsampled_chunk

    async def _downsample_mulaw(self, chunk: bytes) -> bytes:
        pcm_data = audioop.ulaw2lin(chunk, 2)
        downsampled_pcm_data = await self._downsample_pcm(pcm_data)
        downsampled_chunk = audioop.lin2ulaw(downsampled_pcm_data, 2)
        return downsampled_chunk

    async def downsample_async_generator(self, async_gen: AsyncGenerator[bytes, None]):
        async for play_ht_chunk in async_gen:
            if self.synthesizer_config.sampling_rate >= 24000:
                yield play_ht_chunk
            else:
                downsampled_chunk = await self._downsample_from_24khz(play_ht_chunk)
                yield downsampled_chunk

    async def _cut_leading_trailing_silence(
        self,
        async_iter: AsyncIterator[bytes],
        chunk_size: int,
        cut_leading_silence: bool = True,
        cut_trailing_silence: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        buffer: bytearray = bytearray()

        async def generate_chunks(
            play_ht_chunk: bytes,
            cut_leading_silence=False,
        ) -> AsyncGenerator[bytes, None]:
            """Yields chunks of size chunk_size from play_ht_chunk and leaves the remainder in buffer.

            If cut_leading_silence is True, does not yield chunks until it detects voice.
            """
            nonlocal buffer

            buffer.extend(play_ht_chunk)
            detected_voice = False
            for buffer_idx, chunk in self._enumerate_by_chunk_size(buffer, chunk_size):
                if cut_leading_silence and not detected_voice:
                    if self._contains_voice_experimental(chunk):
                        detected_voice = True
                        yield chunk
                    if detected_voice:
                        logger.debug(f"Cut off {buffer_idx} bytes of leading silence")
                else:
                    yield chunk
            buffer = buffer[len(buffer) - (len(buffer) % chunk_size) :]

        async def _cut_out_trailing_silence(
            trailing_chunk: bytes,
        ) -> AsyncGenerator[bytes, None]:
            """Yields chunks of size chunk_size from trailing_chunk until it detects silence."""
            for buffer_idx, chunk in self._enumerate_by_chunk_size(trailing_chunk, chunk_size):
                if not self._contains_voice_experimental(chunk):
                    logger.debug(
                        f"Cutting off {len(trailing_chunk) - buffer_idx} bytes of trailing silence",
                    )
                    break
                yield chunk

        # Yield from the first audio chunk, no matter what, for latency
        try:
            play_ht_chunk = await async_iter.__anext__()
        except StopAsyncIteration:
            return

        async for chunk in generate_chunks(play_ht_chunk, cut_leading_silence=cut_leading_silence):
            yield chunk

        async for lookahead_buffer, is_last in generate_with_is_last(
            generate_from_async_iter_with_lookahead(async_iter, 2),
        ):
            if not is_last:
                async for chunk in generate_chunks(lookahead_buffer[0]):
                    yield chunk
            else:
                trailing_chunk = b"".join(lookahead_buffer)
                async for chunk in (
                    _cut_out_trailing_silence(trailing_chunk)
                    if cut_trailing_silence
                    else generate_chunks(trailing_chunk)
                ):
                    yield chunk

    async def get_chunks(
        self,
        message: BaseMessage,
        chunk_size: int,
        chunk_queue: asyncio.Queue[Optional[bytes]],
        cut_leading_silence: bool,
        cut_trailing_silence: bool,
    ):
        buffer = bytearray()
        try:
            playht_bytes_generators = [
                self.playht_client.tts(
                    text,
                    self.playht_options,
                )
                for text in split_text(
                    string_to_split=message.text,
                    max_text_length=PLAY_HT_V2_MAX_CHARS,
                )
            ]
            downsampled_generators = [
                self.downsample_async_generator(gen) for gen in playht_bytes_generators
            ]

            for async_gen in downsampled_generators:
                async_iter = async_gen.__aiter__()
                if (
                    self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16
                ):  # skip the first chunk, which contains wav header
                    await async_iter.__anext__()
                if not cut_trailing_silence and not cut_leading_silence:
                    while True:
                        try:
                            play_ht_chunk = await async_iter.__anext__()
                        except StopAsyncIteration:
                            break

                        buffer.extend(play_ht_chunk)
                        for _, chunk in self._enumerate_by_chunk_size(buffer, chunk_size):
                            chunk_queue.put_nowait(chunk)
                        buffer = buffer[len(buffer) - (len(buffer) % chunk_size) :]
                    if len(buffer) > 0:
                        chunk_queue.put_nowait(buffer)
                else:
                    async for chunk in self._cut_leading_trailing_silence(
                        async_iter,
                        chunk_size,
                        cut_leading_silence=cut_leading_silence,
                        cut_trailing_silence=cut_trailing_silence,
                    ):
                        chunk_queue.put_nowait(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # treated as sentinel

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: PlayHtSynthesizerConfig):
        return ":".join(
            (
                "play_ht_v2",
                synthesizer_config.voice_id,
                str(synthesizer_config.user_id),
                str(synthesizer_config.speed),
                str(synthesizer_config.seed),
                str(synthesizer_config.temperature),
                synthesizer_config.audio_encoding,
            ),
        )

    async def tear_down(self):
        await self.playht_client.close()
        await super().tear_down()
