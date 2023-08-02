import logging
from typing import Optional
import typing
from vocode.streaming.models.transcriber import (
    AssemblyAITranscriberConfig,
    AzureTranscriberConfig,
    DeepgramTranscriberConfig,
    WhisperCPPTranscriberConfig,
    GoogleTranscriberConfig,
    RevAITranscriberConfig,
    TranscriberConfig,
    TranscriberType,
)
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriber
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.google_transcriber import GoogleTranscriber
from vocode.streaming.transcriber.rev_ai_transcriber import RevAITranscriber
from vocode.streaming.transcriber.azure_transcriber import AzureTranscriber
from vocode.streaming.transcriber.whisper_cpp_transcriber import WhisperCPPTranscriber


class TranscriberFactory:
    def create_transcriber(
        self,
        transcriber_config: TranscriberConfig,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(transcriber_config, WhisperCPPTranscriberConfig):
            return WhisperCPPTranscriber(transcriber_config, logger=logger)
        elif isinstance(transcriber_config, DeepgramTranscriberConfig):
            return DeepgramTranscriber(transcriber_config, logger=logger)
        elif isinstance(transcriber_config, GoogleTranscriberConfig):
            return GoogleTranscriber(transcriber_config, logger=logger)
        elif isinstance(transcriber_config, AssemblyAITranscriberConfig):
            return AssemblyAITranscriber(transcriber_config, logger=logger)
        elif isinstance(transcriber_config, RevAITranscriberConfig):
            return RevAITranscriber(transcriber_config, logger=logger)
        elif isinstance(transcriber_config, AzureTranscriberConfig):
            return AzureTranscriber(transcriber_config, logger=logger)
        else:
            raise Exception("Invalid transcriber config")
