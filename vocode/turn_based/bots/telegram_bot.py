from collections import defaultdict
from typing import Dict, List, Optional
import telegram
from vocode import getenv
from pydantic import BaseModel
from vocode.streaming import agent
from vocode.turn_based.agent.base_agent import BaseAgent
from vocode.turn_based.agent.chat_gpt_agent import ChatGPTAgent
from vocode.turn_based.transcriber.base_transcriber import BaseTranscriber
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from vocode.turn_based.transcriber.whisper_transcriber import WhisperTranscriber


class Chat(BaseModel):
    messages: List = []

    def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.messages.append({"role": "assistant", "content": message})

    @classmethod
    def from_initial_message(self, message: str):
        return Chat(messages=[{"role": "system", "content": message}])


DEFAULT_START_TEXT = """I'm a chatbot"""
INVALID_MESSAGE = """Sorry, I didn't understand that. Use /help for more information."""


class BaseTelegramBot:
    def __init__(
        self,
        agent: BaseAgent,
        transcriber: BaseTranscriber = WhisperTranscriber(),
        start_text: str = DEFAULT_START_TEXT,
    ):
        self.telegram_bot_key = getenv("TELEGRAM_BOT_KEY")
        if not self.telegram_bot_key:
            raise ValueError("TELEGRAM_BOT_KEY must be set in environment")
        self.transcriber = transcriber
        self.agent = agent
        self.db = defaultdict(lambda: Chat.from_initial_message(agent.system_prompt))
        self.start_text = start_text

    async def handle_telegram_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=self.start_text
        )

    async def handle_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message and update.message.voice:
            input = self.get_audio_segment_from_file(update.message.voice.file_id)
        elif update.message and update.message.text:
            input = update.message.text
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=INVALID_MESSAGE,
            )
            return
        self.db[update.effective_chat.id].add_user_message(input)
        response = await self.get_response(update.effective_chat.id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

    async def get_response(
        self,
        chat_id: int,
    ) -> str:
        agent_response = self.agent.respond_to_message_history(
            self.db[chat_id].messages
        )
        text_response = agent_response.choices[0].message.content

        self.db[chat_id].add_assistant_message(text_response)
        print(self.db[chat_id].messages)
        return text_response

    def start(self):
        application = ApplicationBuilder().token(self.telegram_bot_key).build()

        application.add_handler(CommandHandler("start", self.handle_telegram_start))
        application.add_handler(
            MessageHandler(~filters.COMMAND, self.handle_telegram_message)
        )
        application.run_polling()
