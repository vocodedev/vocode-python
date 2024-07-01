from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from vocode.streaming.action.worker import ActionsWorker
from vocode.streaming.models.model import BaseModel
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.pipeline.audio_pipeline import AudioPipeline, OutputDeviceType
from vocode.streaming.utils.events_manager import EventsManager

PipelineConfigType = TypeVar("PipelineConfigType", bound=BaseModel)


class AbstractPipelineFactory(Generic[PipelineConfigType, OutputDeviceType], ABC):

    @abstractmethod
    def create_pipeline(
        self,
        config: PipelineConfigType,
        output_device: OutputDeviceType,
        id: Optional[str] = None,
        events_manager: Optional[EventsManager] = None,
        actions_worker: Optional[ActionsWorker] = None,
    ) -> AudioPipeline[OutputDeviceType]:
        raise NotImplementedError
