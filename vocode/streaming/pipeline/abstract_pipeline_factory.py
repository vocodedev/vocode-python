from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from vocode.streaming.models.model import BaseModel
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.pipeline.audio_pipeline import AudioPipeline
from vocode.streaming.utils.events_manager import EventsManager

PipelineConfigType = TypeVar("PipelineConfigType", bound=BaseModel)


class AbstractPipelineFactory(Generic[PipelineConfigType], ABC):

    @abstractmethod
    def create_pipeline(
        self,
        config: PipelineConfigType,
        output_device: AbstractOutputDevice,
        id: Optional[str] = None,
        events_manager: Optional[EventsManager] = None,
    ) -> AudioPipeline:
        raise NotImplementedError
