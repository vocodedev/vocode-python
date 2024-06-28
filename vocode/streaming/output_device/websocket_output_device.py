from __future__ import annotations

import asyncio

from fastapi import WebSocket

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcript import TranscriptEvent
from vocode.streaming.models.websocket import AudioMessage, TranscriptMessage
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)


class WebsocketOutputDevice(RateLimitInterruptionsOutputDevice):
    def __init__(self, ws: WebSocket, sampling_rate: int, audio_encoding: AudioEncoding):
        super().__init__(sampling_rate, audio_encoding)
        self.ws = ws
        self.active = False
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    def start(self):
        self.active = True
        return super().start()

    def mark_closed(self):
        self.active = False

    async def play(self, chunk: bytes):
        await self.ws.send_text(AudioMessage.from_bytes(chunk).json())

    async def send_transcript(self, event: TranscriptEvent):
        if self.active:
            transcript_message = TranscriptMessage.from_event(event)
            await self.ws.send_text(transcript_message.json())
