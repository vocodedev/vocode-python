import asyncio
import json

import websockets
from loguru import logger
from websockets.client import WebSocketClientProtocol, connect

from vocode.streaming.agent.base_agent import (
    AgentInput,
    AgentResponse,
    AgentResponseMessage,
    AgentResponseStop,
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
from vocode.streaming.utils.worker import InterruptibleAgentResponseEvent, InterruptibleEvent

NUM_RESTARTS = 5


class WebSocketUserImplementedAgent(BaseAgent[WebSocketUserImplementedAgentConfig]):
    input_queue: asyncio.Queue[InterruptibleEvent[AgentInput]]

    def __init__(
        self,
        agent_config: WebSocketUserImplementedAgentConfig,
    ):
        self.agent_config = agent_config

        self.has_ended = False
        super().__init__(agent_config=agent_config)

    def get_agent_config(self) -> WebSocketUserImplementedAgentConfig:
        return self.agent_config

    async def _run_loop(self) -> None:
        restarts = 0
        logger.info("Starting Socket Agent")
        while not self.has_ended and restarts < NUM_RESTARTS:
            await self._process()
            restarts += 1
            logger.debug("Socket Agent connection died, restarting, num_restarts: %s", restarts)

    def _handle_incoming_socket_message(self, message: WebSocketAgentMessage) -> None:
        logger.info("Handling incoming message from Socket Agent: %s", message)

        agent_response: AgentResponse

        if isinstance(message, WebSocketAgentTextMessage):
            agent_response = AgentResponseMessage(message=BaseMessage(text=message.data.text))
        elif isinstance(message, WebSocketAgentStopMessage):
            agent_response = AgentResponseStop()
            self.has_ended = True
        else:
            raise Exception("Unknown Socket message type")

        logger.info("Putting interruptible agent response event in output queue")
        self.agent_responses_consumer.consume_nonblocking(
            self.interruptible_event_factory.create_interruptible_agent_response_event(
                agent_response,
                is_interruptible=self.get_agent_config().allow_agent_to_be_cut_off,
            )
        )

    async def _process(self) -> None:
        socket_url = self.get_agent_config().respond.url
        logger.info("Connecting to web socket agent %s", socket_url)

        async with connect(socket_url) as ws:

            async def sender(
                ws: WebSocketClientProtocol,
            ) -> None:  # sends audio to websocket
                while not self.has_ended:
                    logger.info("Waiting for data from agent request queue")
                    try:
                        input = await self.input_queue.get()
                        payload = input.payload
                        if isinstance(payload, TranscriptionAgentInput):
                            transcription = payload.transcription
                            logger.info("Transcription message: %s", transcription.message)
                            agent_request = WebSocketAgentTextMessage.from_text(
                                transcription.message,
                                conversation_id=payload.conversation_id,
                            )
                            agent_request_json = agent_request.json()
                            logger.info(f"Sending data to web socket agent: {agent_request_json}")
                            if isinstance(agent_request, AgentResponseStop):
                                # In practice, it doesn't make sense for the client to send a text and stop message to the agent service
                                self.has_ended = True

                            await ws.send(agent_request_json)

                    except asyncio.exceptions.TimeoutError:
                        break

                    except Exception as e:
                        logger.error(
                            f'WebSocket Agent Send Error: "{e}" in Web Socket User Implemented Agent sender'
                        )
                        break

                logger.debug("Terminating web socket agent sender")

            async def receiver(ws: WebSocketClientProtocol) -> None:
                while not self.has_ended:
                    try:
                        msg = await ws.recv()
                        logger.info("Received data from web socket agent")
                        data = json.loads(msg)
                        message = WebSocketAgentMessage.parse_obj(data)
                        self._handle_incoming_socket_message(message)

                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f'WebSocket Agent Receive Error: Connection Closed - "{e}"')
                        break

                    except websockets.exceptions.ConnectionClosedOK as e:
                        logger.error(f'WebSocket Agent Receive Error: Connection Closed OK - "{e}"')
                        break

                    except websockets.exceptions.InvalidStatus as e:
                        logger.error(f'WebSocket Agent Receive Error: Invalid Status - "{e}"')
                        break

                    except Exception as e:
                        logger.error(f'WebSocket Agent Receive Error: "{e}"')
                        break

                logger.debug("Terminating Web Socket User Implemented Agent receiver")

            await asyncio.gather(sender(ws), receiver(ws))

    async def terminate(self):
        self.agent_responses_consumer.consume_nonblocking(
            self.interruptible_event_factory.create_interruptible_agent_response_event(
                AgentResponseStop()
            )
        )
        await super().terminate()
