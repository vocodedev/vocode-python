import asyncio
import audioop
import random
import secrets
import wave
from string import ascii_letters, digits
from typing import Any, AsyncGenerator, AsyncIterator, Callable, List, Tuple, TypeVar

from vocode.streaming.models.audio import AudioEncoding

custom_alphabet = ascii_letters + digits + ".-_"

ChoiceType = TypeVar("ChoiceType")


def create_loop_in_thread(loop: asyncio.AbstractEventLoop, long_running_task=None):
    asyncio.set_event_loop(loop)
    if long_running_task:
        loop.run_until_complete(long_running_task)
    else:
        loop.run_forever()


def convert_linear_audio(
    raw_wav: bytes,
    input_sample_rate=24000,
    output_sample_rate=8000,
    output_encoding=AudioEncoding.LINEAR16,
    output_sample_width=2,
):
    # downsample
    if input_sample_rate != output_sample_rate:
        raw_wav, _ = audioop.ratecv(raw_wav, 2, 1, input_sample_rate, output_sample_rate, None)

    if output_encoding == AudioEncoding.LINEAR16:
        return raw_wav
    elif output_encoding == AudioEncoding.MULAW:
        return audioop.lin2ulaw(raw_wav, output_sample_width)


def convert_wav(
    file: Any,
    output_sample_rate=8000,
    output_encoding=AudioEncoding.LINEAR16,
):
    with wave.open(file, "rb") as wav:
        raw_wav = wav.readframes(wav.getnframes())
        return convert_linear_audio(
            raw_wav,
            input_sample_rate=wav.getframerate(),
            output_sample_rate=output_sample_rate,
            output_encoding=output_encoding,
            output_sample_width=wav.getsampwidth(),
        )


def get_chunk_size_per_second(audio_encoding: AudioEncoding, sampling_rate: int) -> int:
    if audio_encoding == AudioEncoding.LINEAR16:
        return sampling_rate * 2
    elif audio_encoding == AudioEncoding.MULAW:
        return sampling_rate
    else:
        raise Exception("Unsupported audio encoding")


def create_conversation_id() -> str:
    return secrets.token_urlsafe(16)


def create_utterance_id() -> str:
    return secrets.token_hex(16)


def remove_non_letters_digits(text):
    return "".join(i for i in text if i in custom_alphabet)


def unrepeating_randomizer(l: List[ChoiceType]) -> Callable[[], ChoiceType]:
    last_choice = None

    def next() -> ChoiceType:
        nonlocal last_choice
        choice = random.choice(l)
        while choice == last_choice:
            choice = random.choice(l)
        last_choice = choice
        return choice

    return next


AsyncIteratorGenericType = TypeVar("AsyncIteratorGenericType")


async def generate_with_is_last(
    async_gen: AsyncGenerator[AsyncIteratorGenericType, None],
) -> AsyncGenerator[Tuple[AsyncIteratorGenericType, bool], None]:
    async_iter = async_gen.__aiter__()
    try:
        next_item = await async_iter.__anext__()
    except StopAsyncIteration:
        assert False, "Cannot generate with is_last from an empty async generator"
    while True:
        try:
            item = await async_iter.__anext__()
            yield next_item, False
            next_item = item
        except StopAsyncIteration:
            yield next_item, True
            break


async def generate_from_async_iter_with_lookahead(
    async_iter: AsyncIterator[AsyncIteratorGenericType],
    lookahead: int,
) -> AsyncGenerator[List[AsyncIteratorGenericType], None]:
    """Yield sliding window lists of length `lookahead + 1` from an async iterator.

    If the length of async iterator < lookahead + 1, then it should just yield the whole
        async iterator as a list.
    """
    assert lookahead > 0
    buffer = []

    stream_length = 0
    while True:
        try:
            next_item = await async_iter.__anext__()
            stream_length += 1
            buffer.append(next_item)
            if len(buffer) == lookahead + 1:
                yield buffer
                buffer = buffer[1:]
        except StopAsyncIteration:
            if buffer and stream_length <= lookahead:
                yield buffer
            return


async def enumerate_async_iter(
    async_iter: AsyncIterator[AsyncIteratorGenericType],
) -> AsyncGenerator[Tuple[int, AsyncIteratorGenericType], None]:
    i = 0
    async for item in async_iter:
        yield i, item
        i += 1
