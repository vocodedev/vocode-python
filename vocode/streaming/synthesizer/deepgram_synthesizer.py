import asyncio
from typing import Optional

from deepgram import DeepgramClient, DeepgramClientOptions, SpeakOptions

from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import DeepgramSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.utils.create_task import asyncio_create_task_with_done_error_log
from vocode.streaming.utils.mp3_helper import decode_mp3


class DeepgramSynthesizer(BaseSynthesizer[DeepgramSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: DeepgramSynthesizerConfig,
        api_key: Optional[str] = "",
    ):
        super().__init__(synthesizer_config)

        # save config
        self.synthesizer_config = synthesizer_config

        self._config: SpeakOptions = SpeakOptions()
        self._config.encoding = self.synthesizer_config.audio_encoding

        if not self._config.model:
            self._config.model = self.synthesizer_config.model
        if not self._config.bit_rate:
            self._config.bit_rate = self.synthesizer_config.bit_rate
        if not self._config.container:
            self._config.container = self.synthesizer_config.container
        if not self._config.sample_rate:
            self._config.sample_rate = self.synthesizer_config.sampling_rate

        # deepgram client
        config: DeepgramClientOptions = DeepgramClientOptions()
        deepgram: DeepgramClient = DeepgramClient(api_key, config)
        if not deepgram.api_key:
            raise Exception(
                "Please set DEEPGRAM_API_KEY environment variable or pass it as a parameter"
            )
        self._dgClient = deepgram.asyncspeak.v("1")

        # off script options
        self.experimental_streaming = synthesizer_config.experimental_streaming

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        # if self.experimental_streaming:
        #     # TODO: streaming
        #     return SynthesisResult(
        #         self.experimental_mp3_streaming_output_generator(
        #             response, chunk_size, create_speech_span
        #         ),  # should be wav
        #         lambda seconds: self.get_message_cutoff_from_voice_speed(
        #             message, seconds, self.words_per_minute
        #         ),
        #     )
        # else:
        # TODO: indent when implementing streaming
        try:
            response = await self._dgClient.stream({"text": message.text}, options=self._config)

            output_bytes_io = decode_mp3(bytes(response.stream.getbuffer()))
            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )

            return result

        except Exception as e:
            raise
