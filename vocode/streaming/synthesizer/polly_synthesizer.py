import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import boto3

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PollySynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
)


class PollySynthesizer(BaseSynthesizer[PollySynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: PollySynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        client = boto3.client("polly")

        # AWS Polly supports sampling rate of 8k and 16k for pcm output
        if synthesizer_config.sampling_rate not in [8000, 16000]:
            raise Exception(
                "Sampling rate not supported by AWS Polly",
                synthesizer_config.sampling_rate,
            )

        self.sampling_rate = synthesizer_config.sampling_rate
        self.client = client
        self.language_code = synthesizer_config.language_code
        self.voice_id = synthesizer_config.voice_id
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    def synthesize(self, message: str) -> Any:
        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        return self.client.synthesize_speech(
            Text=message,
            LanguageCode=self.language_code,
            TextType="text",
            OutputFormat="pcm",
            VoiceId=self.voice_id,
            SampleRate=str(self.sampling_rate),
        )

    def get_speech_marks(self, message: str) -> Any:
        return self.client.synthesize_speech(
            Text=message,
            LanguageCode=self.language_code,
            TextType="text",
            OutputFormat="json",
            VoiceId=self.voice_id,
            SampleRate=str(self.sampling_rate),
            SpeechMarkTypes=["word"],
        )

    # given the number of seconds the message was allowed to go until, where did we get in the message?
    def get_message_up_to(
        self,
        message: str,
        seconds: Optional[float],
        word_events,
    ) -> str:
        if seconds is None:
            return message
        for event in word_events:
            # time field is in ms
            if event["time"] > seconds * 1000:
                return message[: event["start"]]
        return message

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        audio_response = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor, self.synthesize, message.text
        )
        audio_stream = audio_response.get("AudioStream")

        speech_marks_response = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor, self.get_speech_marks, message.text
        )
        word_events = [
            json.loads(v)
            for v in speech_marks_response.get("AudioStream").read().decode().split()
            if v
        ]

        async def chunk_generator(audio_data_stream, chunk_transform=lambda x: x):
            audio_buffer = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor,
                lambda: audio_stream.read(chunk_size),
            )
            if len(audio_buffer) != chunk_size:
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer), True)
                return
            else:
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer), False)
            while True:
                audio_buffer = audio_stream.read(chunk_size)
                if len(audio_buffer) != chunk_size:
                    yield SynthesisResult.ChunkResult(
                        chunk_transform(audio_buffer[: len(audio_buffer)]), True
                    )
                    break
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer), False)

        if self.synthesizer_config.should_encode_as_wav:
            output_generator = chunk_generator(
                audio_stream,
                lambda chunk: encode_as_wav(chunk, self.synthesizer_config),
            )
        else:
            output_generator = chunk_generator(audio_stream)

        return SynthesisResult(
            output_generator,
            lambda seconds: self.get_message_up_to(
                message.text,
                seconds,
                word_events,
            ),
        )
