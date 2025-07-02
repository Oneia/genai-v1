from dotenv import load_dotenv
from agno.utils.pprint import pprint_run_response
from typing import Iterator

load_dotenv()

import os
import nest_asyncio
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

nest_asyncio.apply()

TOKEN = os.getenv("TELEGRAM_TOKEN")
group_chat_ids = set()

async def track_groups(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    print(update, 'upodate')
    if chat.type in ['group', 'supergroup']:
        group_chat_ids.add(chat.id)
        print(f"Tracked group: {chat.title} (ID: {chat.id})")

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.ChatType.GROUPS, track_groups))

# async def run_bot():
#     print("Bot is running. Add it to groups and send a message to track group IDs.")
#     await app.run_polling(close_loop=False)  # <-- This is the key change!

# # In a cell, run:
# await run_bot()
celpip_chat_id=-4891587211

async def send_message_to_all_groups(message: str):
    try:
        await app.bot.send_message(chat_id=celpip_chat_id, text=message, parse_mode=constants.ParseMode.MARKDOWN)
        print(f"Message sent to group ID: {celpip_chat_id}")
    except Exception as e:
        print(f"Failed to send message to {celpip_chat_id}: {e}")
