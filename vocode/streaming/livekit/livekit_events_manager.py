import uuid
from typing import TYPE_CHECKING, List

from livekit import rtc

from vocode.streaming.models.events import Event, EventType, Sender
from vocode.streaming.models.transcript import TranscriptEvent
from vocode.streaming.utils.events_manager import EventsManager

if TYPE_CHECKING:
    from vocode.streaming.livekit.livekit_conversation import LiveKitConversation


class LiveKitEventsManager(EventsManager):
    _conversation: "LiveKitConversation"

    def __init__(
        self,
        subscriptions: List[EventType] = [],
    ):
        if EventType.TRANSCRIPT not in subscriptions:
            subscriptions.append(EventType.TRANSCRIPT)
        super().__init__(subscriptions)

    def attach_conversation(self, conversation: "LiveKitConversation"):
        self._conversation = conversation

    async def handle_event(self, event: Event):
        if isinstance(event, TranscriptEvent):
            participant = (
                self._conversation.room.local_participant
                if event.sender == Sender.BOT
                else self._conversation.user_participant
            )
            track = (
                self._conversation.user_track
                if event.sender == Sender.HUMAN
                else self._conversation.output_device.track
            )

            transcription = rtc.Transcription(
                participant_identity=participant.identity,
                track_id=track.sid,
                segments=[
                    rtc.TranscriptionSegment(
                        id=str(uuid.uuid4()),
                        text=event.text,
                        start_time=int(event.timestamp),
                        end_time=int(event.timestamp),
                        final=True,
                    )
                ],
                language="",
            )
            await self._conversation.room.local_participant.publish_transcription(transcription)
