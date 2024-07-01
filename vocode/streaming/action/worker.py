from __future__ import annotations

from typing import TYPE_CHECKING

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionInput, ActionResponse
from vocode.streaming.pipeline.worker import (
    AbstractWorker,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleWorker,
)

if TYPE_CHECKING:
    from vocode.streaming.pipeline.audio_pipeline import AudioPipeline


class ActionsWorker(InterruptibleWorker[InterruptibleEvent[ActionInput]]):
    consumer: AbstractWorker[InterruptibleEvent[ActionResponse]]
    pipeline: "AudioPipeline"

    def __init__(
        self,
        action_factory: AbstractActionFactory,
        interruptible_event_factory: InterruptibleEventFactory = InterruptibleEventFactory(),
    ):
        super().__init__(
            interruptible_event_factory=interruptible_event_factory,
        )
        self.action_factory = action_factory

    def attach_state(self, action: BaseAction):
        action.pipeline = self.pipeline

    async def process(self, item: InterruptibleEvent[ActionInput]):
        action_input = item.payload
        action = self.action_factory.create_action(action_input.action_config)
        self.attach_state(action)
        action_output = await action.run(action_input)
        self.consumer.consume_nonblocking(
            self.interruptible_event_factory.create_interruptible_event(
                ActionResponse(
                    conversation_id=action_input.conversation_id,
                    action_input=action_input,
                    action_output=action_output,
                    is_quiet=action.quiet,
                ),
                is_interruptible=False,
            )
        )
