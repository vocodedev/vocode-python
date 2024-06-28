import asyncio
import audioop
import io
import math
from typing import Any, List, Optional
import wave

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.synthesizer.synthesis_result import SynthesisResult
from vocode.streaming.utils import convert_wav
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def create_synthesis_result_from_wav(
    synthesizer_config: SynthesizerConfig,
    file: Any,
    message: BaseMessage,
    chunk_size: int,
) -> SynthesisResult:
    output_bytes = convert_wav(
        file,
        output_sample_rate=synthesizer_config.sampling_rate,
        output_encoding=synthesizer_config.audio_encoding,
    )

    if synthesizer_config.should_encode_as_wav:
        chunk_transform = lambda chunk: encode_as_wav(chunk, synthesizer_config)  # noqa: E731

    else:
        chunk_transform = lambda chunk: chunk  # noqa: E731

    async def chunk_generator(output_bytes):
        for i in range(0, len(output_bytes), chunk_size):
            if i + chunk_size > len(output_bytes):
                yield SynthesisResult.ChunkResult(chunk_transform(output_bytes[i:]), True)
            else:
                yield SynthesisResult.ChunkResult(
                    chunk_transform(output_bytes[i : i + chunk_size]), False
                )

    return SynthesisResult(
        chunk_generator(output_bytes),
        lambda seconds: get_message_cutoff_from_total_response_length(
            synthesizer_config, message, seconds, len(output_bytes)
        ),
    )


async def chunk_result_generator_from_queue(self, chunk_queue: asyncio.Queue[Optional[bytes]]):
    while True:
        try:
            chunk = await chunk_queue.get()
            if chunk is None:
                break
            yield SynthesisResult.ChunkResult(
                chunk=chunk,
                is_last_chunk=False,
            )
        except asyncio.CancelledError:
            break


def resample_chunk(
    chunk: bytes,
    current_sample_rate: int,
    target_sample_rate: int,
) -> bytes:
    resampled_chunk, _ = audioop.ratecv(
        chunk,
        2,
        1,
        current_sample_rate,
        target_sample_rate,
        None,
    )

    return resampled_chunk


def get_message_cutoff_from_total_response_length(
    synthesizer_config: SynthesizerConfig,
    message: BaseMessage,
    seconds: Optional[float],
    size_of_output: int,
) -> str:
    estimated_output_seconds = size_of_output / synthesizer_config.sampling_rate
    if not message.text:
        return message.text

    if seconds is None:
        return message.text

    estimated_output_seconds_per_char = estimated_output_seconds / len(message.text)
    return message.text[: int(seconds / estimated_output_seconds_per_char)]


def get_message_cutoff_from_voice_speed(
    message: BaseMessage, seconds: Optional[float], words_per_minute: int
) -> str:

    if seconds is None:
        return message.text

    words_per_second = words_per_minute / 60
    estimated_words_spoken = math.floor(words_per_second * seconds)
    tokens = word_tokenize(message.text)
    return TreebankWordDetokenizer().detokenize(tokens[:estimated_words_spoken])


def split_text(string_to_split: str, max_text_length: int) -> List[str]:
    # Base case: if the string_to_split is less than or equal to max_text_length characters, return it as a single element array
    if len(string_to_split) <= max_text_length:
        return [string_to_split.strip()]

    # Recursive case: find the index of the last sentence ender in the first max_text_length characters of the string_to_split
    sentence_enders = [".", "!", "?"]
    index = -1
    for ender in sentence_enders:
        i = string_to_split[:max_text_length].rfind(ender)
        if i > index:
            index = i

    # If there is a sentence ender, split the string_to_split at that index plus one and strip any spaces from both parts
    if index != -1:
        first_part = string_to_split[: index + 1].strip()
        second_part = string_to_split[index + 1 :].strip()

    # If there is no sentence ender, find the index of the last comma in the first max_text_length characters of the string_to_split
    else:
        index = string_to_split[:max_text_length].rfind(",")
        # If there is a comma, split the string_to_split at that index plus one and strip any spaces from both parts
        if index != -1:
            first_part = string_to_split[: index + 1].strip()
            second_part = string_to_split[index + 1 :].strip()
        # If there is no comma, find the index of the last space in the first max_text_length characters of the string_to_split
        else:
            index = string_to_split[:max_text_length].rfind(" ")
            # If there is a space, split the string_to_split at that index and strip any spaces from both parts
            if index != -1:
                first_part = string_to_split[:index].strip()
                second_part = string_to_split[index:].strip()

            # If there is no space, split the string_to_split at max_text_length characters and strip any spaces from both parts
            else:
                first_part = string_to_split[:max_text_length].strip()
                second_part = string_to_split[max_text_length:].strip()

    # Append the first part to the result array
    result = [first_part]

    # Call the function recursively on the remaining part of the string_to_split and extend the result array with it, unless it is empty
    if second_part != "":
        result.extend(split_text(string_to_split=second_part, max_text_length=max_text_length))

    # Return the result array
    return result


def encode_as_wav(chunk: bytes, synthesizer_config: SynthesizerConfig) -> bytes:
    output_bytes_io = io.BytesIO()
    in_memory_wav = wave.open(output_bytes_io, "wb")
    in_memory_wav.setnchannels(1)
    assert synthesizer_config.audio_encoding == AudioEncoding.LINEAR16
    in_memory_wav.setsampwidth(2)
    in_memory_wav.setframerate(synthesizer_config.sampling_rate)
    in_memory_wav.writeframes(chunk)
    output_bytes_io.seek(0)
    return output_bytes_io.read()


async def empty_generator(self):
    yield SynthesisResult.ChunkResult(b"", True)
