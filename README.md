<div align="center">

![Hero](https://user-images.githubusercontent.com/6234599/228337850-e32bb01d-3701-47ef-a433-3221c9e0e56e.png)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/vocodehq.svg?style=social&label=Follow%20%40vocodehq)](https://twitter.com/vocodehq) [![GitHub Repo stars](https://img.shields.io/github/stars/vocodedev/vocode-python?style=social)](https://github.com/vocodedev/vocode-python)
[![Downloads](https://static.pepy.tech/badge/vocode/month)](https://pepy.tech/project/vocode)

[Community](https://discord.gg/NaU4mMgcnC) | [Docs](https://docs.vocode.dev) | [Dashboard](https://app.vocode.dev)

</div>

# <span><img style='vertical-align:middle; display:inline;' src="https://user-images.githubusercontent.com/6234599/228339858-95a0873a-2d40-4542-963a-6358d19086f5.svg"  width="5%" height="5%">&nbsp; vocode</span>

### **Build voice-based LLM apps in minutes**

Vocode is an open source library that makes it easy to build voice-based LLM apps. Using Vocode, you can build real-time streaming conversations with LLMs and deploy them to phone calls, Zoom meetings, and more. You can also build personal assistants or apps like voice-based chess. Vocode provides easy abstractions and integrations so that everything you need is in a single library.

We're actively looking for community maintainers, so please reach out if interested!

# ⭐️ Features

- 🗣 [Spin up a conversation with your system audio](https://docs.vocode.dev/python-quickstart)
- ➡️ 📞 [Set up a phone number that responds with a LLM-based agent](https://docs.vocode.dev/telephony#inbound-calls)
- 📞 ➡️ [Send out phone calls from your phone number managed by an LLM-based agent](https://docs.vocode.dev/telephony#outbound-calls)
- 🧑‍💻 [Dial into a Zoom call](https://github.com/vocodedev/vocode-python/blob/main/vocode/streaming/telephony/hosted/zoom_dial_in.py)
- 🤖 [Use an outbound call to a real phone number in a Langchain agent](https://docs.vocode.dev/langchain-agent)
- Out of the box integrations with:
  - Transcription services, including:
    - [AssemblyAI](https://www.assemblyai.com/)
    - [Deepgram](https://deepgram.com/)
    - [Gladia](https://gladia.io)
    - [Google Cloud](https://cloud.google.com/speech-to-text)
    - [Microsoft Azure](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text)
    - [RevAI](https://www.rev.ai/)
    - [Whisper](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)
    - [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
    
  - LLMs, including:
    - [ChatGPT](https://openai.com/blog/chatgpt)
    - [GPT-4](https://platform.openai.com/docs/models/gpt-4)
    - [Anthropic](https://www.anthropic.com/)
    - [GPT4All](https://github.com/nomic-ai/gpt4all)
  - Synthesis services, including:
    - [Rime.ai](https://rime.ai)
    - [Microsoft Azure](https://azure.microsoft.com/en-us/products/cognitive-services/text-to-speech/)
    - [Google Cloud](https://cloud.google.com/text-to-speech)
    - [Play.ht](https://play.ht)
    - [Eleven Labs](https://elevenlabs.io/)
    - [Coqui](https://coqui.ai/)
    - [Coqui (OSS)](https://github.com/coqui-ai/TTS)
    - [gTTS](https://gtts.readthedocs.io/)
    - [StreamElements](https://streamelements.com/)
    - [Bark](https://github.com/suno-ai/bark)
    - [AWS Polly](https://aws.amazon.com/polly/)

Check out our React SDK [here](https://github.com/vocodedev/vocode-react-sdk)!

# 🫂 Contribution and Roadmap

We're an open source project and are extremely open to contributors adding new features, integrations, and documentation! Please don't hesitate to reach out and get started building with us.

For more information on contributing, see our [Contribution Guide](https://github.com/vocodedev/vocode-python/blob/main/contributing.md).

And check out our [Roadmap](https://github.com/vocodedev/vocode-python/blob/main/roadmap.md).

We'd love to talk to you on [Discord](https://discord.gg/NaU4mMgcnC) about new ideas and contributing!

# 🚀 Quickstart

```bash
pip install 'vocode'
```

```python
import asyncio
import logging
import signal
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import *
from vocode.streaming.agent import *
from vocode.streaming.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.message import BaseMessage
import vocode

# these can also be set as environment variables
vocode.setenv(
    OPENAI_API_KEY="<your OpenAI key>",
    DEEPGRAM_API_KEY="<your Deepgram key>",
    AZURE_SPEECH_KEY="<your Azure key>",
    AZURE_SPEECH_REGION="<your Azure region>",
)


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
        use_blocking_speaker_output=True
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""The AI is having a pleasant conversation about life""",
            )
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(speaker_output)
        ),
        logger=logger,
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
```

# 📞 Phone call quickstarts

- [Telephony Server - Self-hosted](https://docs.vocode.dev/telephony)

# 🌱 Documentation

[docs.vocode.dev](https://docs.vocode.dev/)


# Recording audio input (human speech):

To record input audio stream to a wav file to /recordings folder, you can see examples in: `quickstarts/streaming_conversation_audio_recorded.py` and
`playground/streaming/transcribe/transcribe.py`