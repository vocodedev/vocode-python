from typing import Optional

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from vocode.turn_based.output_device.abstract_output_device import AbstractOutputDevice


class SpeakerOutput(AbstractOutputDevice):
    DEFAULT_SAMPLING_RATE = 44100

    def __init__(
        self,
        device_info: dict,
        sampling_rate: Optional[int] = None,
    ):
        self.device_info = device_info
        self.sampling_rate = sampling_rate or int(
            self.device_info.get("default_samplerate", self.DEFAULT_SAMPLING_RATE)
        )
        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sampling_rate,
            dtype=np.int16,
            device=int(self.device_info["index"]),
        )
        self.stream.start()

    @classmethod
    def from_default_device(cls, sampling_rate: Optional[int] = None):
        return cls(sd.query_devices(kind="output"), sampling_rate)

    def send_audio(self, audio_segment: AudioSegment):
        raw_data = audio_segment.raw_data
        if audio_segment.frame_rate != self.sampling_rate:
            raw_data = audio_segment.set_frame_rate(self.sampling_rate).raw_data
        self.stream.write(np.frombuffer(raw_data, dtype=np.int16))

    def terminate(self):
        self.stream.close()
