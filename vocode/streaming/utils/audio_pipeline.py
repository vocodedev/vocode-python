from abc import abstractmethod
from typing import Generic, TypeVar

from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.utils.worker import AbstractWorker

OutputDeviceType = TypeVar("OutputDeviceType", bound=AbstractOutputDevice)


class AudioPipeline(AbstractWorker[bytes], Generic[OutputDeviceType]):
    output_device: OutputDeviceType
    events_manager: EventsManager
    id: str

    def receive_audio(self, chunk: bytes):
        self.consume_nonblocking(chunk)

    @abstractmethod
    def is_active(self):
        raise NotImplementedError
