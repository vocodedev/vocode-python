import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from xml.etree import ElementTree

import azure.cognitiveservices.speech as speechsdk
from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage, SSMLMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import (
    FILLER_AUDIO_PATH,
    FILLER_PHRASES,
    BaseSynthesizer,
    FillerAudio,
    SynthesisResult,
    encode_as_wav,
)

NAMESPACES = {
    "mstts": "https://www.w3.org/2001/mstts",
    "": "https://www.w3.org/2001/10/synthesis",
}

ElementTree.register_namespace("", NAMESPACES[""])
ElementTree.register_namespace("mstts", NAMESPACES["mstts"])

_AZURE_INSIDE_VOICE_REGEX = r"<voice[^>]*>(.*?)<\/voice>"


class WordBoundaryEventPool:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(
            {
                "text": event.text,
                "text_offset": event.text_offset,
                "audio_offset": (event.audio_offset + 5000) / (10000 * 1000),
                "boudary_type": event.boundary_type,
            }
        )

    def get_events_sorted(self):
        return sorted(self.events, key=lambda event: event["audio_offset"])


class AzureSynthesizer(BaseSynthesizer[AzureSynthesizerConfig]):
    OFFSET_MS = 100

    def __init__(
        self,
        synthesizer_config: AzureSynthesizerConfig,
        azure_speech_key: Optional[str] = None,
        azure_speech_region: Optional[str] = None,
    ):
        super().__init__(synthesizer_config)
        # Instantiates a client
        azure_speech_key = azure_speech_key or getenv("AZURE_SPEECH_KEY")
        azure_speech_region = azure_speech_region or getenv("AZURE_SPEECH_REGION")
        if not azure_speech_key:
            raise ValueError(
                "Please set AZURE_SPEECH_KEY environment variable or pass it as a parameter"
            )
        if not azure_speech_region:
            raise ValueError(
                "Please set AZURE_SPEECH_REGION environment variable or pass it as a parameter"
            )
        speech_config = speechsdk.SpeechConfig(
            subscription=azure_speech_key, region=azure_speech_region
        )
        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            if self.synthesizer_config.sampling_rate == SamplingRate.RATE_44100:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm
                )
            if self.synthesizer_config.sampling_rate == SamplingRate.RATE_48000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
                )
            if self.synthesizer_config.sampling_rate == SamplingRate.RATE_24000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
                )
            elif self.synthesizer_config.sampling_rate == SamplingRate.RATE_16000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
                )
            elif self.synthesizer_config.sampling_rate == SamplingRate.RATE_8000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm
                )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
            )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )

        self.voice_name = self.synthesizer_config.voice_name
        self.pitch = self.synthesizer_config.pitch
        self.rate = self.synthesizer_config.rate
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: AzureSynthesizerConfig) -> str:
        return ":".join(
            (
                "azure",
                synthesizer_config.voice_name,
                str(synthesizer_config.pitch),
                str(synthesizer_config.rate),
                synthesizer_config.language_code,
                synthesizer_config.audio_encoding,
            )
        )

    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        filler_phrase_audios = []
        for filler_phrase in FILLER_PHRASES:
            cache_key = "-".join(
                (
                    str(filler_phrase.text),
                    str(self.synthesizer_config.type),
                    str(self.synthesizer_config.audio_encoding),
                    str(self.synthesizer_config.sampling_rate),
                    str(self.voice_name),
                    str(self.pitch),
                    str(self.rate),
                )
            )
            filler_audio_path = os.path.join(FILLER_AUDIO_PATH, f"{cache_key}.bytes")
            if os.path.exists(filler_audio_path):
                audio_data = open(filler_audio_path, "rb").read()
            else:
                logger.debug(f"Generating filler audio for {filler_phrase.text}")
                ssml = self.create_ssml(filler_phrase.text)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor, self.synthesizer.speak_ssml, ssml
                )
                offset = self.synthesizer_config.sampling_rate * self.OFFSET_MS // 1000
                audio_data = result.audio_data[offset:]
                with open(filler_audio_path, "wb") as f:
                    f.write(audio_data)
            filler_phrase_audios.append(
                FillerAudio(
                    filler_phrase,
                    audio_data,
                    self.synthesizer_config,
                )
            )
        return filler_phrase_audios

    def add_marks(self, message: str, index=0) -> str:
        search_result = re.search(r"([\.\,\:\;\-\—]+)", message)
        if search_result is None:
            return message
        start, end = search_result.span()
        with_mark = message[:start] + f'<mark name="{index}" />' + message[start:end]
        rest = message[end:]
        rest_stripped = re.sub(r"^(.+)([\.\,\:\;\-\—]+)$", r"\1", rest)
        if len(rest_stripped) == 0:
            return with_mark
        return with_mark + self.add_marks(rest_stripped, index + 1)

    def word_boundary_cb(self, evt, pool):
        pool.add(evt)

    def create_ssml(self, message: str) -> str:
        voice_language_code = self.synthesizer_config.voice_name[:5]
        ssml_root = ElementTree.fromstring(
            f'<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="{voice_language_code}"></speak>'
        )
        voice = ElementTree.SubElement(ssml_root, "voice")
        voice.set("name", self.voice_name)
        if self.synthesizer_config.language_code != "en-US":
            lang = ElementTree.SubElement(voice, "{%s}lang" % NAMESPACES.get(""))
            lang.set("xml:lang", self.synthesizer_config.language_code)
            voice_root = lang
        else:
            voice_root = voice
        # this ugly hack is necessary so we can limit the gap between sentences
        # for normal sentences, it seems like the gap is > 500ms, so we're able to reduce it to 500ms
        # for very tiny sentences, the API hangs - so we heuristically only update the silence gap
        # if there is more than one word in the sentence
        if " " in message:
            silence = ElementTree.SubElement(voice_root, "{%s}silence" % NAMESPACES.get("mstts"))
            silence.set("value", "500ms")
            silence.set("type", "Tailing-exact")
        prosody = ElementTree.SubElement(voice_root, "prosody")
        prosody.set("pitch", f"{self.pitch}%")
        prosody.set("rate", f"{self.rate}%")
        prosody.text = message.strip()
        ssml = ElementTree.tostring(ssml_root, encoding="unicode")
        regmatch = re.search(_AZURE_INSIDE_VOICE_REGEX, ssml, re.DOTALL)
        if regmatch:
            self.total_chars += len(regmatch.group(1))
        return ssml

    def synthesize_ssml(self, ssml: str) -> speechsdk.AudioDataStream:
        result = self.synthesizer.start_speaking_ssml_async(ssml).get()
        return speechsdk.AudioDataStream(result)

    def ready_synthesizer(self, chunk_size: int):
        # TODO: remove warming up the synthesizer for now
        # connection = speechsdk.Connection.from_speech_synthesizer(self.synthesizer)
        # connection.open(True)
        pass

    # given the number of seconds the message was allowed to go until, where did we get in the message?
    def get_message_up_to(
        self,
        message: str,
        ssml: str,
        seconds: Optional[float],
        word_boundary_event_pool: WordBoundaryEventPool,
    ) -> str:
        if seconds is None:
            return message
        events = word_boundary_event_pool.get_events_sorted()
        for event in events:
            if event["audio_offset"] > seconds:
                ssml_fragment = ssml[: event["text_offset"]]
                # TODO: this is a little hacky, but it works for now
                return ssml_fragment.split(">")[-1]
        return message

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        # offset = int(self.OFFSET_MS * (self.synthesizer_config.sampling_rate / 1000))
        offset = 0
        logger.debug(f"Synthesizing message: {message}")

        # Azure will return no audio for certain strings like "-", "[-", and "!"
        # which causes the `chunk_generator` below to hang. Return an empty
        # generator for these cases.
        if not re.search(r"\w", message.text):
            return SynthesisResult(
                self.empty_generator(),
                lambda _: message.text,
            )

        async def chunk_generator(
            audio_data_stream: speechsdk.AudioDataStream, chunk_transform=lambda x: x
        ):
            audio_buffer = bytes(chunk_size)
            filled_size = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor,
                lambda: audio_data_stream.read_data(audio_buffer),
            )
            if filled_size != chunk_size:
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer[offset:]), True)
                return
            else:
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer[offset:]), False)
            while True:
                audio_buffer = bytes(chunk_size)
                filled_size = audio_data_stream.read_data(audio_buffer)
                if filled_size != chunk_size:
                    yield SynthesisResult.ChunkResult(
                        chunk_transform(audio_buffer[: filled_size - offset]), True
                    )
                    break
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer), False)

        word_boundary_event_pool = WordBoundaryEventPool()
        self.synthesizer.synthesis_word_boundary.connect(
            lambda event: self.word_boundary_cb(event, word_boundary_event_pool)
        )
        ssml = message.ssml if isinstance(message, SSMLMessage) else self.create_ssml(message.text)
        audio_data_stream = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor, self.synthesize_ssml, ssml
        )
        if self.synthesizer_config.should_encode_as_wav:
            output_generator = chunk_generator(
                audio_data_stream,
                lambda chunk: encode_as_wav(chunk, self.synthesizer_config),
            )
        else:
            output_generator = chunk_generator(audio_data_stream)

        return SynthesisResult(
            output_generator,
            lambda seconds: self.get_message_up_to(
                message.text, ssml, seconds, word_boundary_event_pool
            ),
        )
