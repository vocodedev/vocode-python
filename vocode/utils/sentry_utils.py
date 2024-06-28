import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import sentry_sdk
from loguru import logger
from sentry_sdk.tracing import Span, Transaction, _SpanRecorder

from vocode import get_serialized_ctx_wrappers, sentry_transaction

if TYPE_CHECKING:
    from vocode.streaming.synthesizer.abstract_synthesizer import AbstractSynthesizer

_SYNTHESIZER_NAMES = {
    "AzureSynthesizer": "azure",
    "ElevenLabsSynthesizer": "eleven_labs",
    "ElevenLabsWSSynthesizer": "eleven_labs_ws",
    "PlayHtSynthesizer": "play_ht",
    "PlayHtSynthesizerV2": "play_ht_v2",
    "RimeSynthesizer": "rime",
}

_FILTERED_SPANS = {"middleware.starlette.receive", "middleware.starlette.send", "Queue.get"}


class CustomSentrySpans:
    CONNECTED_TO_FIRST_SEND = "connected_to_first_send"
    ENDPOINTING_LATENCY = "endpointing_latency"
    FIRST_SEND_TO_FIRST_RECEIVE = "first_send_to_first_receive"
    LANGUAGE_MODEL_TIME_TO_FIRST_TOKEN = "language_model_time_to_first_token"
    LATENCY_OF_CONVERSATION = "latency_of_conversation"
    LATENCY_OF_TRANSCRIPTION_START = "latency_of_transcription_start"
    LLM_FIRST_SENTENCE_TOTAL = "llm_first_sentence_total"
    START_TO_CONNECTION = "start_to_connection"
    SYNTHESIS_GENERATE_FIRST_CHUNK = "synthesis_generate_first_chunk"
    SYNTHESIS_TIME_TO_FIRST_TOKEN = "synthesis_time_to_first_token"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"

    SYNTHESIZER_SYNTHESIS_TOTAL = ".synthesis_total"
    SYNTHESIZER_TIME_TO_FIRST_TOKEN = ".time_to_first_token"
    SYNTHESIZER_CREATE_SPEECH = ".create_speech"

    @classmethod
    def is_present(cls, value):
        for attr in dir(cls):
            if not attr.startswith("__"):
                attr_value = getattr(cls, attr)
                if isinstance(attr_value, str) and (
                    attr_value == value or value.endswith(attr_value)
                ):
                    return True
        return False


class TransactionNotSampled(Exception):
    pass


class NoMatchingSpan(Exception):
    pass


class MissingTransaction(Exception):
    pass


class SentryConfiguredContextManager:
    """
    A context manager that only executes a function if Sentry is configured.

    Attributes:
        func (Callable): The function to be executed.
        args (Tuple): The positional arguments to pass to the function.
        kwargs (Dict): The keyword arguments to pass to the function.
        result (Any): The result of the function execution.
    """

    def __init__(self, func: Callable, *args: Tuple, **kwargs: Dict) -> None:
        """
        Constructs all the necessary attributes for the SentryConfiguredContextManager object.

        Args:
            func (Callable): The function to be executed.
            *args (Tuple): The positional arguments to pass to the function.
            **kwargs (Dict): The keyword arguments to pass to the function.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result: Optional[Any] = None

    @property
    def is_configured(self) -> bool:
        """
        Checks if Sentry is configured.

        Returns:
            bool: True if Sentry is configured, False otherwise.
        """
        client = sentry_sdk.Hub.current.client
        if client is not None and client.options is not None and "dsn" in client.options:
            return True
        return False

    def __enter__(self) -> Optional[Any]:
        """
        Executes the function if Sentry is configured.

        Returns:
            Any: The result of the function execution, or None if Sentry is not configured.
        """
        if self.is_configured:
            self.result = self.func(*self.args, **self.kwargs)
            return self.result
        else:
            return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Defines the behavior when exiting the context.

        Args:
            exc_type (Any): The type of the exception.
            exc_val (Any): The value of the exception.
            exc_tb (Any): The traceback of the exception.
        """
        pass

    def __call__(self) -> Optional[Any]:
        """
        Executes the function if Sentry is configured, and prints a message if it's not.

        Returns:
            Any: The result of the function execution, or None if Sentry is not configured.
        """
        if self.is_configured:
            return self.func(*self.args, **self.kwargs)
        else:
            logger.debug("Sentry is not configured, skipping function execution.")
            return None

    def execute(self) -> Optional[Any]:
        """Executes the wrapped function immediately and returns its result."""
        return self()


def sentry_configured(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        context_manager = SentryConfiguredContextManager(func, *args, **kwargs)
        if "with" not in func.__code__.co_names:
            return context_manager.execute()
        return context_manager

    return wrapper


def synthesizer_base_name_if_should_report_to_sentry(
    synthesizer: "AbstractSynthesizer",
) -> Optional[str]:
    """Returns a synthesizer name if we should report metrics to Sentry for this
    kind of synthesizer; else returns None.
    """
    return f"synthesizer.{_SYNTHESIZER_NAMES.get(synthesizer.__class__.__qualname__)}"


@sentry_configured
def set_tags(span: Span) -> Span:
    serialized_ctx_wrappers = get_serialized_ctx_wrappers()
    for name, value in serialized_ctx_wrappers.items():
        span.set_tag(name, value)
    if not span.description:
        span.description = serialized_ctx_wrappers.get("conversation_id") or span.op
    return span


@sentry_configured
def get_span_by_op(op_value) -> Span:
    transaction: Transaction = sentry_sdk.Hub.current.scope.transaction or sentry_transaction.value
    if transaction is None:
        raise MissingTransaction("No transaction found.")
    elif not transaction.sampled:
        raise TransactionNotSampled("Transaction is not sampled.")
    elif not transaction._span_recorder:
        raise MissingTransaction("No span recorder found.")
    # Probably not great accessing an internal variable but transaction spans aren't
    # exposed publicly so it is what it is.
    span_matches = [
        span
        for span in transaction._span_recorder.spans
        if span.op == op_value and span.timestamp is None
    ]
    if span_matches:
        most_recent_span = max(span_matches, key=lambda span: span.start_timestamp, default=None)
        if most_recent_span is not None:
            return set_tags(most_recent_span)
    raise NoMatchingSpan(f"No span found with op '{op_value}'.")


@sentry_configured
def complete_span_by_op(op_value):
    try:
        span = get_span_by_op(op_value)
    except TransactionNotSampled as e:
        logger.debug(f"Transaction not sampled")
        return None
    except NoMatchingSpan:
        # TODO: Fix sentry running out of span depth
        logger.warning(f"No matching span found for op '{op_value}'")
        return None
    except MissingTransaction as e:
        logger.error(f"Missing top level transaction: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting span by op '{op_value}': {e}")
        return None
    span.finish()


@sentry_configured
def sentry_create_span(*args, sentry_callable: Callable, **kwargs) -> Span:
    span = sentry_callable(*args, **kwargs)

    return set_tags(span)


class SpanRecorder(_SpanRecorder):

    def __init__(self, maxlen):
        self.maxlen = 900
        self._auto_spans = []
        self._custom_spans = []
        self._low_prio_spans = []

    def add(self, span: Span):
        if span.op in _FILTERED_SPANS and span.description in _FILTERED_SPANS:
            self._low_prio_spans.append(span)
        else:
            if CustomSentrySpans.is_present(span.op):
                self._custom_spans.append(span)
            else:
                self._auto_spans.append(span)

    @property
    def spans(self):
        return (self._custom_spans + self._auto_spans + self._low_prio_spans)[: self.maxlen]


def init_span_recorder(self, maxlen: int):
    if self._span_recorder is None:
        self._span_recorder = SpanRecorder(maxlen)


Span.init_span_recorder = init_span_recorder  # type: ignore
