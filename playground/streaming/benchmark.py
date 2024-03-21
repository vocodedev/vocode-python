"""Basic benchmarking script.

Example usage:
DEEPGRAM_API_KEY=[KEY] ASSEMBLY_AI_KEY=[KEY] poetry run python playground/streaming/benchmark.py --transcribers all

Powershell:
 $env:DEEPGRAM_API_KEY = 'KEY';
 $env:ASSEMBLY_AI_API_KEY = 'KEY';
 poetry run python .\playground\streaming\benchmark.py --transcribers all;
 Remove-Item Env:\DEEPGRAM_API_KEY; Remove-Item Env:\ASSEMBLY_AI_API_KEY
"""

from collections import defaultdict
import os
import re
import json
import argparse
import asyncio
import logging
from tqdm import tqdm
import sounddevice as sd
from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from vocode.streaming.agent.base_agent import TranscriptionAgentInput
from vocode.streaming.agent.vertex_ai_agent import ChatVertexAIAgent
from vocode.streaming.input_device.file_input_device import FileInputDevice
from vocode.streaming.agent import ChatGPTAgent, ChatAnthropicAgent
from vocode.streaming.input_device.microphone_input import MicrophoneInput
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.agent import (
    AzureOpenAIConfig,
    ChatGPTAgentConfig,
    ChatAnthropicAgentConfig,
    ChatVertexAIAgentConfig,
)
from vocode.streaming.models.synthesizer import (
    AzureSynthesizerConfig,
    BarkSynthesizerConfig,
    CoquiSynthesizerConfig,
    CoquiTTSSynthesizerConfig,
    ElevenLabsSynthesizerConfig,
    GTTSSynthesizerConfig,
    GoogleSynthesizerConfig,
    PlayHtSynthesizerConfig,
    RimeSynthesizerConfig,
    StreamElementsSynthesizerConfig,
)
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    AssemblyAITranscriberConfig,
    GladiaTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.output_device.file_output_device import FileOutputDevice
from vocode.streaming.synthesizer import (
    AzureSynthesizer,
    BarkSynthesizer,
    CoquiSynthesizer,
    CoquiTTSSynthesizer,
    ElevenLabsSynthesizer,
    GTTSSynthesizer,
    GoogleSynthesizer,
    PlayHtSynthesizer,
    RimeSynthesizer,
    StreamElementsSynthesizer,
)
from vocode.streaming.transcriber import (
    DeepgramTranscriber,
    AssemblyAITranscriber,
    GladiaTranscriber,
)
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.utils import get_chunk_size_per_second, remove_non_letters_digits
from vocode.streaming.utils.worker import InterruptibleEvent
from playground.streaming.tracing_utils import get_final_metrics

logger = logging.getLogger(__name__)
logging.basicConfig()

logger.setLevel(logging.DEBUG)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


# Create the parser
parser = argparse.ArgumentParser(
    description="Benchmark Vocode's transcribers, agents, and synthesizers.\n"
    + "Example usage: python playground/streaming/benchmark.py --all --all_num_cycles 3 --create_graphs"
)

synthesizer_classes = {
    "elevenlabs": (ElevenLabsSynthesizer, ElevenLabsSynthesizerConfig),
    "azure": (AzureSynthesizer, AzureSynthesizerConfig),
    "bark": (BarkSynthesizer, BarkSynthesizerConfig),
    # "coqui": (CoquiSynthesizer, CoquiSynthesizerConfig),
    # "coquitts": (CoquiTTSSynthesizer, CoquiTTSSynthesizerConfig),
    "google": (GoogleSynthesizer, GoogleSynthesizerConfig),
    "gtts": (GTTSSynthesizer, GTTSSynthesizerConfig),
    "playht": (PlayHtSynthesizer, PlayHtSynthesizerConfig),
    "rime": (RimeSynthesizer, RimeSynthesizerConfig),
    "streamelements": (StreamElementsSynthesizer, StreamElementsSynthesizerConfig),
}


# These synthesizers stream output so they need to be traced within this file.
STREAMING_SYNTHESIZERS = ["azure", "elevenlabs"]


TRANSCRIBER_CHOICES = ["deepgram", "assemblyai", "gladia"]
AGENT_CHOICES = [
    "gpt_gpt-3.5-turbo",
    "gpt_gpt-4",
    "azuregpt_gpt-35-turbo",
    "anthropic_claude-v1",
    "anthropic_claude-instant-v1",
    "vertex_ai_chat-bison@001",
]
SYNTHESIZER_CHOICES = list(synthesizer_classes)

parser.add_argument(
    "--transcribers",
    type=str,
    nargs="*",
    default=[],
    choices=TRANSCRIBER_CHOICES + ["all"],
    help="The list of transcribers to benchmark",
)
parser.add_argument(
    "--agents",
    type=str,
    nargs="*",
    default=[],
    choices=AGENT_CHOICES + ["all"],
    help="The list of agents to benchmark. Each agent should be of the form <company>_<model_name>.",
)
parser.add_argument(
    "--synthesizers",
    type=str,
    nargs="*",
    default=[],
    choices=SYNTHESIZER_CHOICES + ["all"],
    help="The list of synthesizers to benchmark",
)
parser.add_argument(
    "--transcriber_audio",
    type=str,
    default="playground/streaming/test.wav",
    help="Path to the audio file to transcribe",
)
parser.add_argument(
    "--transcriber_use_mic",
    action="store_true",
    help="Use the microphone as the input device for the transcriber. "
    + "Overrides --transcriber_audio. Be silent for ≈5 seconds to end transcription.",
)
parser.add_argument(
    "--synthesizer_text",
    type=str,
    default="Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
    help="The text for synthesizers to synthesize",
)
parser.add_argument(
    "--agent_prompt_preamble",
    type=str,
    default="The AI is having a very short and pleasant conversation about life",
    help="The prompt preamble to use for the agent",
)
parser.add_argument(
    "--agent_first_input",
    type=str,
    default="What is the meaning of life?",
    help="The initial message sent to the agent (this is a transcribed sentence that the agent should respond to).",
)
parser.add_argument(
    "--no_generate_responses",
    action="store_true",
    help="Disable streaming generated responses for agents",
)
parser.add_argument(
    "--transcriber_num_cycles",
    type=int,
    default=1,
    help="The number of transcriber runs to perform. Results are averaged over the runs.",
)
parser.add_argument(
    "--synthesizer_num_cycles",
    type=int,
    default=1,
    help="The number of synthesizer runs to perform. Results are averaged over the runs.",
)
parser.add_argument(
    "--all_num_cycles",
    type=int,
    default=None,
    help="The number of transcriber, agent, and synthesizer runs to perform. Overrides all other num_cycle arguments.",
)
parser.add_argument(
    "--agent_num_cycles",
    type=int,
    default=1,
    help="The number of agent runs to perform. Results are averaged over the runs.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Run all supported transcribers, agents, and synthesizers. Ignores other arguments.",
)
parser.add_argument(
    "--create_graphs",
    action="store_true",
    help="Create graphs from the benchmark results. Requires matplotlib.",
)
parser.add_argument(
    "--just_graphs",
    action="store_true",
    help="Skips computing statistics. Loads the last saved benchmark result "
    + "JSON file and creates graphs from it.",
)
parser.add_argument(
    "--results_file",
    type=str,
    default="benchmark_results.json",
    help="The file to save the benchmark JSON results to",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="benchmark_results",
    help="The directory to save the text-to-speech output and JSON results to",
)
args = parser.parse_args()
if args.all:
    print("--all is set! Running all supported transcribers, agents, and synthesizers.")
    args.transcribers = TRANSCRIBER_CHOICES
    args.agents = AGENT_CHOICES
    args.synthesizers = SYNTHESIZER_CHOICES

if "all" in args.transcribers:
    args.transcribers = TRANSCRIBER_CHOICES
if "all" in args.agents:
    args.agents = AGENT_CHOICES
if "all" in args.synthesizers:
    args.synthesizers = SYNTHESIZER_CHOICES

if args.all_num_cycles is not None:
    args.transcriber_num_cycles = args.all_num_cycles
    args.agent_num_cycles = args.all_num_cycles
    args.synthesizer_num_cycles = args.all_num_cycles

if args.create_graphs or args.just_graphs:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "ERROR: The --create_graphs flag requires matplotlib. Please "
            + "install matplotlib and try again."
        )
        exit(1)

if args.just_graphs:
    print(
        "--just_graphs is set! Skipping computing statistics and instead "
        + "generating graphs from the last saved benchmark result JSON file."
    )

should_generate_responses = not args.no_generate_responses

os.makedirs(args.results_dir, exist_ok=True)


def get_transcriber(transcriber_name, file_input):
    if transcriber_name == "deepgram":
        transcriber = DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                file_input,
                endpointing_config=PunctuationEndpointingConfig(),
            )
        )
    elif transcriber_name == "assemblyai":
        transcriber = AssemblyAITranscriber(
            AssemblyAITranscriberConfig.from_input_device(
                file_input,
            )
        )
    elif transcriber_name == "gladia":
        transcriber = GladiaTranscriber(
            GladiaTranscriberConfig.from_input_device(
                file_input,
            )
        )
    return transcriber


trace.set_tracer_provider(TracerProvider(resource=Resource.create({})))
span_exporter = InMemorySpanExporter()
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(span_exporter))  # type: ignore

reader = InMemoryMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)


async def run_agents():
    for agent_name in tqdm(args.agents, desc="Agents"):
        company, model_name = agent_name.rsplit("_", 1)
        length_meter = meter.create_counter(
            remove_non_letters_digits(
                f"agent.agent_chat_{company}-{model_name}.total_characters"
            ),
        )
        for _ in tqdm(range(args.agent_num_cycles), desc="Agent Cycles"):
            if company == "gpt":
                agent = ChatGPTAgent(
                    ChatGPTAgentConfig(
                        initial_message=None,
                        prompt_preamble=args.agent_prompt_preamble,
                        allow_agent_to_be_cut_off=False,
                        model_name=model_name,
                        generate_responses=should_generate_responses,
                    )
                )
            elif company == "azuregpt":
                agent = ChatGPTAgent(
                    ChatGPTAgentConfig(
                        initial_message=None,
                        prompt_preamble=args.agent_prompt_preamble,
                        allow_agent_to_be_cut_off=False,
                        azure_params=AzureOpenAIConfig(engine=model_name),
                        generate_responses=should_generate_responses,
                    )
                )
            elif company == "anthropic":
                agent = ChatAnthropicAgent(
                    ChatAnthropicAgentConfig(
                        initial_message=None,
                        allow_agent_to_be_cut_off=False,
                        model_name=model_name,
                        generate_responses=should_generate_responses,
                    )
                )
            elif company == "vertex_ai":
                agent = ChatVertexAIAgent(
                    ChatVertexAIAgentConfig(
                        initial_message=None,
                        prompt_preamble=args.agent_prompt_preamble,
                        allow_agent_to_be_cut_off=False,
                        model_name=model_name,
                        generate_responses=False,
                    )
                )
            agent.attach_transcript(Transcript())
            agent_task = agent.start()
            message = TranscriptionAgentInput(
                transcription=Transcription(
                    message=args.agent_first_input, confidence=1.0, is_final=True
                ),
                conversation_id=0,
            )
            agent.consume_nonblocking(
                agent.interruptible_event_factory.create_interruptible_event(message)
            )

            while True:
                try:
                    message = await asyncio.wait_for(
                        agent.output_queue.get(), timeout=15
                    )
                    length_meter.add(len(message.payload.message.text))
                    logger.debug(
                        f"[Agent: {agent_name}] Response from API: {message.payload.message.text}"
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        f"[Agent: {agent_name}] Agent queue is empty, stopping..."
                    )
                    break


async def run_synthesizers():
    def create_file_output_device(synthesizer_name, extra_info=""):
        return FileOutputDevice(
            os.path.join(args.results_dir, f"{synthesizer_name}{extra_info}.wav"),
        )

    for synthesizer_cycle_idx in tqdm(
        range(args.synthesizer_num_cycles), desc="Synthesizer Cycles"
    ):
        for synthesizer_name in args.synthesizers:
            file_output = create_file_output_device(
                synthesizer_name, f"-run={synthesizer_cycle_idx}"
            )
            synthesizer_class, synthesizer_config_class = synthesizer_classes[
                synthesizer_name
            ]
            extra_config = {}
            if synthesizer_name == "playht":
                extra_config["voice_id"] = "larry"
            elif synthesizer_name == "rime":
                extra_config["speaker"] = "young_male_unmarked-1"
            synthesizer = synthesizer_class(
                synthesizer_config_class.from_output_device(file_output, **extra_config)
            )

            chunk_size = get_chunk_size_per_second(
                synthesizer.get_synthesizer_config().audio_encoding,
                synthesizer.get_synthesizer_config().sampling_rate,
            )

            current_synthesizer_is_streaming = (
                synthesizer_name in STREAMING_SYNTHESIZERS
            )
            if current_synthesizer_is_streaming:
                total_synthesis_span = tracer.start_span(
                    f"synthesizer.{synthesizer_name}.create_total"
                )
                first_synthesis_span = tracer.start_span(
                    f"synthesizer.{synthesizer_name}.create_first"
                )

            try:
                synthesis_result = await synthesizer.create_speech(
                    message=BaseMessage(text=args.synthesizer_text),
                    chunk_size=chunk_size,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[Synthesizer: {synthesizer_name}] Timed out while synthesizing. Skipping {synthesizer_name}..."
                )
                continue
            except Exception as e:
                logger.error(
                    f"[Synthesizer: {synthesizer_name}] Exception while synthesizing: {e}. Skipping {synthesizer_name}..."
                )
                continue
            chunk_generator = synthesis_result.chunk_generator

            with tqdm(desc=f"{synthesizer_name.title()} Synthesizing") as pbar:
                first_chunk = True
                while True:
                    pbar.update(1)
                    chunk_result = await chunk_generator.__anext__()
                    if current_synthesizer_is_streaming and first_chunk:
                        first_chunk = False
                        first_synthesis_span.end()
                    file_output.consume_nonblocking(chunk_result.chunk)
                    if chunk_result.is_last_chunk:
                        break

            if current_synthesizer_is_streaming:
                total_synthesis_span.end()


async def run_transcribers():
    sample_rate = 44100
    chunk_size = 2048
    sleep_time = chunk_size / sample_rate
    if args.transcriber_use_mic:
        input_device_info = sd.query_devices(kind="input")
        input_device = MicrophoneInput(input_device_info)
    else:
        input_device = FileInputDevice(
            args.transcriber_audio,
            chunk_size=chunk_size,
            silent_duration=0.01,
            skip_initial_load=True,
        )

    for transcriber_cycle_idx in tqdm(
        range(args.transcriber_num_cycles), desc="Transcriber Cycles"
    ):
        for transcriber_name in tqdm(args.transcribers, desc="Transcribers"):
            transcriber = get_transcriber(transcriber_name, input_device)
            if not args.transcriber_use_mic:
                input_device.load()
            transcriber_task = transcriber.start()

            if args.transcriber_use_mic:

                async def record_audio_task():
                    while True:
                        chunk = await input_device.get_audio()
                        transcriber.send_audio(chunk)

                send_audio = asyncio.create_task(record_audio_task())
            else:

                async def send_audio_task():
                    while not input_device.is_done():
                        chunk = await input_device.get_audio()
                        transcriber.send_audio(chunk)
                        await asyncio.sleep(sleep_time)

                send_audio = asyncio.create_task(send_audio_task())

            # `get` from `transcriber.output_queue` until it's empty for 5 seconds
            pbar = tqdm(
                desc=f"{transcriber_name.title()} Transcribing",
                total=input_device.duration if not args.transcriber_use_mic else None,
                unit="chunk",
            )
            while True:
                try:
                    transcription = await asyncio.wait_for(
                        transcriber.output_queue.get(), timeout=5
                    )
                    # update the progress bar status
                    pbar.update(round(transcriber.audio_cursor - pbar.n, 2))
                except asyncio.TimeoutError:
                    logger.debug(
                        f"[Transcriber: {transcriber_name}] Transcriber queue is empty, stopping transcription..."
                    )
                    send_audio.cancel()
                    break
            if not args.transcriber_use_mic:
                pbar.update(pbar.total - pbar.n)
            transcriber.terminate()


def create_graphs(final_results):
    logger.info("Creating graphs from benchmark results...")
    results_split = []
    for name, value in final_results.items():
        first_name = name.split(".", 1)
        second_name = first_name[1].rsplit(".", 1)
        results_split.append((first_name[0], *second_name, value))

    graph_data = defaultdict(lambda: defaultdict(list))
    for category, name, metric, value in results_split:
        graph_data[f"{category} - {metric}"]["labels"].append(name)
        graph_data[f"{category} - {metric}"]["values"].append(value)

    graph_dir = os.path.join(args.results_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    for graph_title, data in graph_data.items():
        plt.title(graph_title)
        plt.bar(data["labels"], data["values"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, f"{graph_title}.png"))
        plt.clf()


async def main():
    result_file_path = os.path.join(args.results_dir, args.results_file)
    if not args.just_graphs:
        if args.agents:
            await run_agents()
        if args.transcribers:
            await run_transcribers()
        if args.synthesizers:
            await run_synthesizers()

        trace_results = span_exporter.get_finished_spans()
        final_spans = defaultdict(list)
        for span in trace_results:
            duration_ns = span.end_time - span.start_time
            duration_s = duration_ns / 1e9
            final_spans[span.name].append(duration_s)

        scope_metrics = reader.get_metrics_data().resource_metrics[0].scope_metrics
        final_metrics = get_final_metrics(scope_metrics, final_spans=final_spans)

        final_spans = {k: sum(v) / len(v) for k, v in final_spans.items() if len(v) > 0}
        if len(scope_metrics) > 0:
            final_results = {**final_spans, **final_metrics}
        else:
            final_results = final_spans
        print(json.dumps(final_results, indent=4))
        if args.results_file:
            with open(result_file_path, "w") as f:
                json.dump(final_results, f, indent=4)
    else:
        with open(result_file_path, "r") as f:
            final_results = json.load(f)

    if args.create_graphs or args.just_graphs:
        create_graphs(final_results)

    print("Benchmarking complete!")


if __name__ == "__main__":
    asyncio.run(main())
