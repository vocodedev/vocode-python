import io
from pydub import AudioSegment
import discord
from janus import Queue
import asyncio
import numpy as np

from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.utils.worker import ThreadAsyncWorker


class PlayFuncWriterWorker(ThreadAsyncWorker):
    def __init__(self, input_queue: Queue, vc) -> None:
        super().__init__(input_queue)
        self.ready_to_send = True
        self.vc = vc

    def play_audio(self, buffer):
        data = AudioSegment.from_raw(
            io.BytesIO(buffer), sample_width=2, frame_rate=44100, channels=1
        )
        data = data.set_channels(2)
        data = data.set_frame_rate(48000)
        data = data.set_sample_width(2)
        raw = io.BytesIO()
        data.export(raw, format="raw")
        audio = discord.PCMAudio(raw)

        def after(error):
            if error:
                print(f"Player error: {error}")
            self.ready_to_send = True

        self.vc.play(discord.PCMVolumeTransformer(audio), after=after)

    def _run_loop(self):
        while True:
            try:
                if self.ready_to_send:
                    self.ready_to_send = False
                    block = self.input_janus_queue.sync_q.get()
                    self.play_audio(block)
            except asyncio.CancelledError:
                return


class DiscordOutputDevice(BaseOutputDevice):
    DEFAULT_SAMPLING_RATE = 44100

    def __init__(
        self,
        play_func,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.blocksize = self.sampling_rate
        self.queue: Queue[np.ndarray] = Queue()
        # self.thread_worker = None
        self.process_task = asyncio.create_task(self.process())
        self.ready_to_send = True
    
    async def process(self):
        while True:
            if self.ready_to_send:
                self.ready_to_send = False
                block = await self.queue.async_q.get()
                self.play_audio(block)
        
    def play_audio(self, buffer):
        data = AudioSegment.from_raw(
            io.BytesIO(buffer), sample_width=2, frame_rate=44100, channels=1
        )
        data = data.set_channels(2)
        data = data.set_frame_rate(48000)
        data = data.set_sample_width(2)
        raw = io.BytesIO()
        data.export(raw, format="raw")
        audio = discord.PCMAudio(raw)

        def after(error):
            if error:
                print(f"Player error: {error}")
            self.ready_to_send = True
        
        self.vc.play(discord.PCMVolumeTransformer(audio), after=after)

    def consume_nonblocking(self, chunk):
        # if self.thread_worker is None:
        #     raise RuntimeError(
        #         "Output device not inited. Call discord_output.init(voice_channel) first."
        #     )

        chunk_arr = np.frombuffer(chunk, dtype=np.int16)
        for i in range(0, chunk_arr.shape[0], self.blocksize):
            block = np.zeros(self.blocksize, dtype=np.int16)
            size = min(self.blocksize, chunk_arr.shape[0] - i)
            block[:size] = chunk_arr[i : i + size]
            self.queue.sync_q.put_nowait(block)

    def init(self, vc):
        self.vc = vc
        # self.thread_worker = PlayFuncWriterWorker(self.queue, vc)
        # self.thread_worker.start()
        pass

    def terminate(self):
        self.thread_worker.terminate()
