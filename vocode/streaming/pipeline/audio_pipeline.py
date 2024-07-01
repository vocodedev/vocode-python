from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.pipeline.worker import AbstractWorker
from vocode.streaming.utils.events_manager import EventsManager

OutputDeviceType = TypeVar("OutputDeviceType", bound=AbstractOutputDevice)


class AudioPipeline(AbstractWorker[bytes], Generic[OutputDeviceType]):
    output_device: OutputDeviceType
    events_manager: EventsManager
    actions_worker: Optional[ActionsWorker]
    id: str

    def receive_audio(self, chunk: bytes):
        self.consume_nonblocking(chunk)

    @abstractmethod
    def is_active(self):
        raise NotImplementedError
