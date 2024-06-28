from __future__ import annotations

import asyncio
import audioop
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import TranscriberConfig, Transcription
from vocode.streaming.utils.speed_manager import SpeedManager
from vocode.streaming.utils.worker import AsyncWorker, ThreadAsyncWorker

TranscriberConfigType = TypeVar("TranscriberConfigType", bound=TranscriberConfig)


class AbstractTranscriber(Generic[TranscriberConfigType], ABC):
    def __init__(self, transcriber_config: TranscriberConfigType):
        self.transcriber_config = transcriber_config
        self.is_muted = False
        self.speed_manager: Optional[SpeedManager] = None
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Transcription] = asyncio.Queue()

    def attach_speed_manager(self, speed_manager: SpeedManager):
        self.speed_manager = speed_manager

    def mute(self):
        self.is_muted = True

    def unmute(self):
        self.is_muted = False

    def get_transcriber_config(self) -> TranscriberConfigType:
        return self.transcriber_config

    async def ready(self):
        return True

    def create_silent_chunk(self, chunk_size, sample_width=2):
        linear_audio = b"\0" * chunk_size
        if self.get_transcriber_config().audio_encoding == AudioEncoding.LINEAR16:
            return linear_audio
        elif self.get_transcriber_config().audio_encoding == AudioEncoding.MULAW:
            return audioop.lin2ulaw(linear_audio, sample_width)

    @abstractmethod
    async def _run_loop(self):
        pass

    def send_audio(self, chunk):
        if not self.is_muted:
            self.consume_nonblocking(chunk)
        else:
            self.consume_nonblocking(self.create_silent_chunk(len(chunk)))

    @abstractmethod
    def terminate(self):
        pass


class BaseAsyncTranscriber(AbstractTranscriber[TranscriberConfigType], AsyncWorker[bytes]):  # type: ignore
    def __init__(self, transcriber_config: TranscriberConfigType):
        AbstractTranscriber.__init__(self, transcriber_config)
        AsyncWorker.__init__(self, self.input_queue, self.output_queue)

    def terminate(self):
        AsyncWorker.terminate(self)


class BaseThreadAsyncTranscriber(  # type: ignore
    AbstractTranscriber[TranscriberConfigType], ThreadAsyncWorker[bytes]
):
    def __init__(self, transcriber_config: TranscriberConfigType):
        AbstractTranscriber.__init__(self, transcriber_config)
        ThreadAsyncWorker.__init__(self, self.input_queue, self.output_queue)

    def _run_loop(self):
        raise NotImplementedError

    def terminate(self):
        ThreadAsyncWorker.terminate(self)


BaseTranscriber = Union[
    BaseAsyncTranscriber[TranscriberConfigType],
    BaseThreadAsyncTranscriber[TranscriberConfigType],
]
