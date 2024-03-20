import asyncio
from vocode.streaming.streaming_conversation import StreamingConversation


class DiscordStreamingConversation(StreamingConversation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vc = None
        self.connections = None

    def provide_discord_connection(self, vc, connections):
        self.vc = vc
        self.connections = connections

    def terminate(self):
        if self.vc is not None:
            self.vc.stop_recording()
            del self.connections[self.vc.guild.id]
        return super().terminate()
