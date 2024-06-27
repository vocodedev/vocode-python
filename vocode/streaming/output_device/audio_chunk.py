import uuid
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class ChunkState(int, Enum):
    UNPLAYED = 0
    PLAYED = 1
    INTERRUPTED = 2


@dataclass
class AudioChunk:
    data: bytes
    state: ChunkState = ChunkState.UNPLAYED
    chunk_id: UUID = field(default_factory=uuid.uuid4)

    @staticmethod
    def on_play():
        pass

    @staticmethod
    def on_interrupt():
        pass

    def __hash__(self) -> int:
        return hash(self.chunk_id)
