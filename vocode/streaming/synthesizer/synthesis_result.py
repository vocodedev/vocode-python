from sentry_sdk.tracing import Span as SentrySpan


from typing import AsyncGenerator, Callable, Optional


class SynthesisResult:
    """Holds audio bytes for an utterance and method to know how much of utterance was spoken

    @param chunk_generator - an async generator that that yields ChunkResult objects, which contain chunks of audio and a flag indicating if it is the last chunk
    @param get_message_up_to - takes in the number of seconds spoken and returns the message up to that point
    - *if seconds is None, then it should return the full messages*
    """

    class ChunkResult:
        def __init__(self, chunk: bytes, is_last_chunk: bool):
            self.chunk = chunk
            self.is_last_chunk = is_last_chunk

    def __init__(
        self,
        chunk_generator: AsyncGenerator[ChunkResult, None],
        get_message_up_to: Callable[[Optional[float]], str],
        cached: bool = False,
        is_first: bool = False,
        synthesis_total_span: Optional[SentrySpan] = None,
        ttft_span: Optional[SentrySpan] = None,
    ):
        self.chunk_generator = chunk_generator
        self.get_message_up_to = get_message_up_to
        self.cached = cached
        self.is_first = is_first
        self.synthesis_total_span = synthesis_total_span
        self.ttft_span = ttft_span
