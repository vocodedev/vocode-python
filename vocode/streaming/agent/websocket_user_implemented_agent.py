import asyncio
import json
import logging
from typing import Dict
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.utils.worker import (
    InterruptibleTrackedEvent,
    InterruptibleEvent,
)
import websockets
from websockets.client import (
    connect,
    WebSocketClientProtocol,
)

from typing import Awaitable, Callable, Optional, cast
from vocode.streaming.agent.base_agent import (
    AgentInput,
    AgentResponse,
    AgentResponseMessage,
    AgentResponseStop,
    AgentResponseFillerAudio,
    BaseAgent,
    TranscriptionAgentInput,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.websocket_agent import (
    WebSocketAgentMessage,
    WebSocketAgentStopMessage,
    WebSocketAgentTextMessage,
    WebSocketUserImplementedAgentConfig,
)

NUM_RESTARTS = 5


class WebSocketUserImplementedAgent(BaseAgent[WebSocketUserImplementedAgentConfig]):
    input_queue: asyncio.Queue[InterruptibleTrackedEvent[AgentInput]]
    output_queue: asyncio.Queue[InterruptibleTrackedEvent[AgentResponse]]

    def __init__(
        self,
        agent_config: WebSocketUserImplementedAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_config = agent_config
        self.logger = logger or logging.getLogger(__name__)

        self.has_ended = False
        super().__init__(agent_config=agent_config, logger=logger)

    def get_agent_config(self) -> WebSocketUserImplementedAgentConfig:
        return self.agent_config

    async def _run_loop(self) -> None:
        restarts = 0
        self.logger.info("Starting Socket Agent")
        while not self.has_ended and restarts < NUM_RESTARTS:
            await self._process()
            restarts += 1
            self.logger.debug(
                "Socket Agent connection died, restarting, num_restarts: %s", restarts
            )

    def _handle_incoming_socket_message(self, message: WebSocketAgentMessage) -> None:
        self.logger.info("Handling incoming message from Socket Agent: %s", message)

        agent_response: AgentResponse

        if isinstance(message, WebSocketAgentTextMessage):
            agent_response = AgentResponseMessage(
                message=BaseMessage(text=message.data.text)
            )
        elif isinstance(message, WebSocketAgentStopMessage):
            agent_response = AgentResponseStop()
            self.has_ended = True
        else:
            raise Exception("Unknown Socket message type")

        self.logger.info("Putting interruptible agent response event in output queue")
        self.produce_interruptible_tracked_event_nonblocking(
            agent_response,
            self.get_agent_config().allow_agent_to_be_cut_off,
            span_name="websocket_agent_response",
        )

    async def _process(self) -> None:
        socket_url = self.get_agent_config().respond.url
        self.logger.info("Connecting to web socket agent %s", socket_url)

        async with connect(socket_url) as ws:

            async def sender(
                ws: WebSocketClientProtocol,
            ) -> None:  # sends audio to websocket
                while not self.has_ended:
                    self.logger.info("Waiting for data from agent request queue")
                    try:
                        input: InterruptibleTrackedEvent = await self.input_queue.get()
                        input.event_tracker.set()
                        payload = input.payload
                        if isinstance(payload, TranscriptionAgentInput):
                            transcription = payload.transcription
                            self.logger.info(
                                "Transcription message: %s", transcription.message
                            )
                            agent_request = WebSocketAgentTextMessage.from_text(
                                transcription.message,
                                conversation_id=payload.conversation_id,
                            )
                            agent_request_json = agent_request.json()
                            self.logger.info(
                                f"Sending data to web socket agent: {agent_request_json}"
                            )
                            if isinstance(agent_request, AgentResponseStop):
                                # In practice, it doesn't make sense for the client to send a text and stop message to the agent service
                                self.has_ended = True

                            await ws.send(agent_request_json)

                    except asyncio.exceptions.TimeoutError:
                        break

                    except Exception as e:
                        self.logger.error(
                            f'WebSocket Agent Send Error: "{e}" in Web Socket User Implemented Agent sender'
                        )
                        break

                self.logger.debug("Terminating web socket agent sender")

            async def receiver(ws: WebSocketClientProtocol) -> None:
                while not self.has_ended:
                    try:
                        msg = await ws.recv()
                        self.logger.info("Received data from web socket agent")
                        data = json.loads(msg)
                        message = WebSocketAgentMessage.parse_obj(data)
                        self._handle_incoming_socket_message(message)

                    except websockets.exceptions.ConnectionClosed as e:
                        self.logger.error(
                            f'WebSocket Agent Receive Error: Connection Closed - "{e}"'
                        )
                        break

                    except websockets.exceptions.ConnectionClosedOK as e:
                        self.logger.error(
                            f'WebSocket Agent Receive Error: Connection Closed OK - "{e}"'
                        )
                        break

                    except websockets.exceptions.InvalidStatus as e:
                        self.logger.error(
                            f'WebSocket Agent Receive Error: Invalid Status - "{e}"'
                        )
                        break

                    except Exception as e:
                        self.logger.error(f'WebSocket Agent Receive Error: "{e}"')
                        break

                self.logger.debug(
                    "Terminating Web Socket User Implemented Agent receiver"
                )

            await asyncio.gather(sender(ws), receiver(ws))

    def terminate(self):
        self.produce_interruptible_tracked_event_nonblocking(
            AgentResponseStop(), span_name="websocket_agent_response_stop"
        )
        super().terminate()
