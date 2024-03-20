import os
import io
import logging
import asyncio
from functools import partial
import threading

import discord
from discord.sinks import Sink, Filters, default_filters, AudioData
from discord.ext import tasks

from vocode.streaming.transcriber import DeepgramTranscriber, GoogleTranscriber
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, GoogleTranscriberConfig
from pydub import AudioSegment
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.agent import ChatGPTAgent
from vocode.streaming.synthesizer import AzureSynthesizer, StreamElementsSynthesizer
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig, StreamElementsSynthesizerConfig
from vocode.streaming.models.transcriber import PunctuationEndpointingConfig
from discord_output_device import DiscordOutputDevice
from discord_streaming_conversation import DiscordStreamingConversation

bot = discord.Bot(debug_guilds=[1113610683737722913])
connections = {}

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

conversation = None
discord_output = None
IDLE_TIME_SECONDS = 10


import wave

# f = wave.open("receive.wav", "wb")
# f.setnchannels(1)
# f.setsampwidth(2)
# f.setframerate(44100)

import time
class VocodeSink(Sink):
    @Filters.container
    def write(self, data, user):
        # if user not in self.audio_data:
        #     file = io.BytesIO()
        #     self.audio_data.update({user: AudioData(file)})

        # file = self.audio_data[user]
        # file.write(data)
        data = AudioSegment.from_raw(
            io.BytesIO(data), sample_width=2, frame_rate=48000, channels=2
        )
        # convert data audio segment to wav with 1 channel and frame rate 44100
        data = data.set_channels(1)
        data = data.set_frame_rate(44100)
        data = data.set_sample_width(2)
        raw = io.BytesIO()
        data.export(raw, format="raw")
        # f.writeframes(raw.getvalue())
        raw.seek(0)
        # print(time.time())
        conversation.receive_audio(raw.read())
        # transcriber.send_audio(raw)
        # write the data to a wave file

    def format_audio(self, *args, **kwargs):
        pass


async def finished_callback(
    sink, channel: discord.TextChannel, conversation=None, *args
):
    recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
    await sink.vc.disconnect()
    # files = [
    #     discord.File(audio.file, f"{user_id}.{sink.encoding}")
    #     for user_id, audio in sink.audio_data.items()
    # ]
    if conversation.active:
        conversation.terminate()
    else:
        await channel.send("Leaving voice channel due to lack of voice activity.")
    await channel.send(
        f"Finished! Recorded audio for {', '.join(recorded_users)}.",  # files=files
    )

def create_loop_in_thread(loop: asyncio.AbstractEventLoop, long_running_task=None):
    asyncio.set_event_loop(loop)
    if long_running_task:
        loop.run_until_complete(long_running_task)
    else:
        loop.run_forever()

async def start_conversation():
    global conversation, discord_output
    discord_output = DiscordOutputDevice(None)
    # transcriber = GoogleTranscriber(
    #     GoogleTranscriberConfig(
    #         sampling_rate=44100,
    #         audio_encoding=AudioEncoding.LINEAR16,
    #         chunk_size=2048,
    #     )
    # )
    transcriber = DeepgramTranscriber(
        DeepgramTranscriberConfig(
            sampling_rate=44100,
            audio_encoding=AudioEncoding.LINEAR16,
            chunk_size=2048,
            endpointing_config=PunctuationEndpointingConfig(),
        )
    )
    agent = ChatGPTAgent(
        ChatGPTAgentConfig(
            initial_message=None,
            prompt_preamble="""The AI is having a pleasant conversation about life""",
            allowed_idle_time_seconds=IDLE_TIME_SECONDS,
            generate_responses=False,
        )
    )
    # synthesizer = AzureSynthesizer(
    #     AzureSynthesizerConfig.from_output_device(discord_output)
    # )
    synthesizer = StreamElementsSynthesizer(
        StreamElementsSynthesizerConfig.from_output_device(discord_output)
    )

    conversation = DiscordStreamingConversation(
        output_device=discord_output,
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer,
        logger=logger,
    )
    print("Conversation created")

    await conversation.start()

create_loop_in_thread(asyncio.new_event_loop(), start_conversation())

@bot.command()
async def start(ctx: discord.ApplicationContext):
    """Record your voice!"""
    global conversation, discord_output

    voice = ctx.author.voice

    if not voice:
        return await ctx.respond("You're not in a vc right now")

    vc = await voice.channel.connect()
    discord_output.init(vc)
    conversation.provide_discord_connection(vc, connections)
    connections.update({ctx.guild.id: vc})

    vc.start_recording(
        VocodeSink(),
        partial(finished_callback, conversation=conversation),
        ctx.channel,
    )
    # asyncio.create_task(print_transcription_task())

    await ctx.respond("The recording has started!")


@bot.command()
async def stop(ctx: discord.ApplicationContext):
    """Stop recording."""
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.delete()
    else:
        await ctx.respond("Not recording in this guild.")


@bot.event
async def on_ready():
    pass


# async def main():
# transcriber.start()
# async def print_transcription_task():
#     while True:
#         transcription = await transcriber.output_queue.get()
#         print(transcription)

# asyncio.create_task(print_transcription_task())

bot.run(os.environ["DISCORD_BOT_TOKEN"])

# if __name__ == "__main__":
#     main()


# @client.tree.command()
# async def join(interaction: discord.Interaction):
#     """Joins the voice channel you are in."""
#     # get the name of the voice channel that the user is in
#     voice_channel = interaction.user.voice
#     if voice_channel is None:
#         await interaction.response.send_message('You are not in a voice channel.')
#     else:
#         channel = interaction.user.voice.channel
#         await interaction.response.send_message(f'You are in {channel.name}')
#         voice_client = await channel.connect()
#         source = discord.PCMVolumeTransformer(discord.FFmpegPCMAudio("test.wav"))
#         voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)
#         while voice_client.is_playing():
#             await asyncio.sleep(1)
#         await voice_client.disconnect()
