from __future__ import annotations

import asyncio
import queue
from typing import Tuple, Union

import miniaudio
from loguru import logger

from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.utils import convert_wav
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.utils.worker import AbstractWorker, ThreadAsyncWorker


class MiniaudioWorker(ThreadAsyncWorker[Union[bytes, None]]):
    consumer: AbstractWorker[Tuple[bytes, bool]]

    def __init__(
        self,
        synthesizer_config: SynthesizerConfig,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.synthesizer_config = synthesizer_config
        self.chunk_size = chunk_size
        self._ended = False

    async def run_thread_forwarding(self):
        try:
            await asyncio.gather(
                self._forward_to_thread(),
                self._forward_from_thread(),
            )
        except asyncio.CancelledError:
            return

    async def _forward_from_thread(self):
        while True:
            try:
                chunk, done = await self.output_janus_queue.async_q.get()
                self.consumer.consume_nonblocking((chunk, done))
            except asyncio.CancelledError:
                break

    def _run_loop(self):
        # tracks the mp3 so far
        current_mp3_buffer = bytearray()
        # tracks the wav so far
        current_wav_buffer = bytearray()
        # the leftover chunks of the wav that haven't been sent to the output queue yet
        current_wav_output_buffer = bytearray()
        while not self._ended:
            # Get a tuple of (mp3_chunk, is_last) from the input queue
            try:
                mp3_chunk = self.input_janus_queue.sync_q.get(timeout=1)
            except queue.Empty:
                continue
            if mp3_chunk is None:
                current_mp3_buffer.clear()
                current_wav_buffer.clear()
                self.output_janus_queue.sync_q.put((bytes(current_wav_output_buffer), True))
                current_wav_output_buffer.clear()
                continue
            try:
                current_mp3_buffer.extend(mp3_chunk)
                output_bytes = decode_mp3(bytes(current_mp3_buffer))
            except miniaudio.DecodeError as e:
                # TODO: better logging
                logger.exception("MiniaudioWorker error: " + str(e), exc_info=True)
                self.output_janus_queue.sync_q.put(
                    (bytes(current_wav_output_buffer), True)
                )  # sentinel
                continue
            converted_output_bytes = convert_wav(
                output_bytes,
                output_sample_rate=self.synthesizer_config.sampling_rate,
                output_encoding=self.synthesizer_config.audio_encoding,
            )
            # take the difference between the current_wav_buffer and the converted_output_bytes
            # and put the difference in the output buffer
            new_bytes = converted_output_bytes[len(current_wav_buffer) :]
            current_wav_output_buffer.extend(new_bytes)

            # chunk up new_bytes in chunks of chunk_size bytes, but keep the last chunk (less than chunk size) in the wav output buffer
            output_buffer_idx = 0
            while output_buffer_idx < len(current_wav_output_buffer) - self.chunk_size:
                chunk = current_wav_output_buffer[
                    output_buffer_idx : output_buffer_idx + self.chunk_size
                ]
                self.output_janus_queue.sync_q.put(
                    (chunk, False)
                )  # don't need to use bytes() since we already sliced it (which is a copy)
                output_buffer_idx += self.chunk_size

            current_wav_output_buffer = current_wav_output_buffer[output_buffer_idx:]
            current_wav_buffer.extend(new_bytes)

    async def terminate(self):
        self._ended = True
        await super().terminate()
