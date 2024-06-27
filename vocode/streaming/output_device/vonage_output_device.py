from typing import Optional

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from vocode.streaming.output_device.blocking_speaker_output import BlockingSpeakerOutput
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)
from vocode.streaming.telephony.constants import (
    PCM_SILENCE_BYTE,
    VONAGE_AUDIO_ENCODING,
    VONAGE_CHUNK_SIZE,
    VONAGE_SAMPLING_RATE,
)


class VonageOutputDevice(RateLimitInterruptionsOutputDevice):
    def __init__(
        self,
        ws: Optional[WebSocket] = None,
        output_to_speaker: bool = False,
    ):
        super().__init__(sampling_rate=VONAGE_SAMPLING_RATE, audio_encoding=VONAGE_AUDIO_ENCODING)
        self.ws = ws
        self.output_to_speaker = output_to_speaker
        if output_to_speaker:
            self.output_speaker = BlockingSpeakerOutput.from_default_device(
                sampling_rate=VONAGE_SAMPLING_RATE, blocksize=VONAGE_CHUNK_SIZE // 2
            )

    async def play(self, chunk: bytes):
        if self.output_to_speaker:
            self.output_speaker.consume_nonblocking(chunk)
        for i in range(0, len(chunk), VONAGE_CHUNK_SIZE):
            subchunk = chunk[i : i + VONAGE_CHUNK_SIZE]
            if len(subchunk) % 2 == 1:
                subchunk += PCM_SILENCE_BYTE  # pad with silence, Vonage goes crazy otherwise
            if self.ws and self.ws.application_state == WebSocketState.DISCONNECTED:
                await self.ws.send_bytes(subchunk)
