from abc import ABC, abstractmethod

from pydub import AudioSegment


class AbstractOutputDevice(ABC):

    @abstractmethod
    def send_audio(self, audio: AudioSegment) -> None:
        pass

    def terminate(self):
        pass
