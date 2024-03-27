from __future__ import annotations

import io
import queue

from typing import Optional, Tuple, Union
import asyncio
import miniaudio

from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.utils import convert_wav
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.utils.worker import ThreadAsyncWorker, logger


class ID3TagProcessor:
    def __init__(self):
        self.buffer = bytearray()
        self.id3_tag_size = 0
        self.id3_tag_processed = False

    def process_chunk(self, chunk):
        self.buffer += chunk

        if not self.id3_tag_processed:
            if self.buffer.startswith(b"ID3"):
                if len(self.buffer) >= 10:
                    self.id3_tag_size = self.calculate_id3_size(self.buffer[:10])
                    if len(self.buffer) >= self.id3_tag_size:
                        # Skip the ID3 tag
                        self.buffer = self.buffer[self.id3_tag_size :]
                        self.id3_tag_processed = True
            else:
                self.id3_tag_processed = True

        if self.id3_tag_processed:
            # Return the audio data and clear the buffer
            audio_data = self.buffer
            self.buffer = bytearray()
            return audio_data

        return bytearray()  # Return an empty bytearray if still processing the tag

    def calculate_id3_size(self, header):
        # Extract the four bytes that represent the size
        size_bytes = header[6:10]
        # Calculate the size (each byte is only 7 bits)
        tag_size = 0
        for byte in size_bytes:
            tag_size = (tag_size << 7) | (byte & 0x7F)

        # The size does not include the 10-byte header
        return tag_size + 10


class MiniaudioWorker(ThreadAsyncWorker[Union[bytes, None]]):
    def __init__(
        self,
        synthesizer_config: SynthesizerConfig,
        chunk_size: int,
        input_queue: asyncio.Queue[Union[bytes, None]],
        output_queue: asyncio.Queue[Tuple[bytes, bool]],
    ) -> None:
        super().__init__(input_queue, output_queue)
        self.output_queue = output_queue  # for typing
        self.synthesizer_config = synthesizer_config
        self.chunk_size = chunk_size
        self._ended = False

    def _run_loop(self):
        # tracks the mp3 so far
        id3_processor = ID3TagProcessor()
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
                self.output_janus_queue.sync_q.put(
                    (bytes(current_wav_output_buffer), True)
                )
                current_wav_output_buffer.clear()
                continue
            try:
                processed_chunk = id3_processor.process_chunk(mp3_chunk)
                if processed_chunk:
                    current_mp3_buffer.extend(processed_chunk)
                    output_bytes_io = io.BytesIO(bytes(current_mp3_buffer))
                    # Ensure the stream is at the start
                    output_bytes_io.seek(0)
                    # Check if there is data in the stream
                    if output_bytes_io.getbuffer().nbytes > 0:
                        output_bytes = decode_mp3(output_bytes_io.read())
                        # Further processing...
                    else:
                        # Handle the case where there is no data
                        continue
                else:
                    # Handle empty processed_chunk
                    continue
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

    def terminate(self):
        self._ended = True
        super().terminate()
