import asyncio
import logging
import signal
from dotenv import load_dotenv


load_dotenv()

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import *
from vocode.streaming.agent import *
from vocode.streaming.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.pubsub.base_pubsub import *
from vocode import pubsub

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False,
        logger=logger,
    )

    transcriber_config = DeepgramTranscriberConfig.from_input_device(
        microphone_input,
        endpointing_config=PunctuationEndpointingConfig(),
        publish_audio=True,
    )

    audio_recorder = AudioFileWriterSubscriber(
        "streaming_conversation",
        save_chunk_in_sec=1,
        sampling_rate=transcriber_config.sampling_rate,
    )

    pubsub.subscribe(audio_recorder, PubSubTopics.INPUT_AUDIO_STREAMS)
    audio_recorder.start()

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(transcriber_config),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""The AI is having a pleasant conversation about life. Keep your responses short. 
                Use simple and accessible English.""",
            )
        ),
        synthesizer=RimeSynthesizer(
            RimeSynthesizerConfig.from_output_device(
                speaker_output, speaker="middle_female_latina-5"
            )
        ),
        logger=logger,
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")

    def sigint_handler(signum, frame):
        asyncio.create_task(conversation.terminate())
        pubsub.stop()

    signal.signal(signal.SIGINT, sigint_handler)

    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
