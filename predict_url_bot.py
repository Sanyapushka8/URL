import asyncio
import logging
import pickle
import re
import time
import numpy as np

from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, StateFilter
from joblib import executor
from keras.models import load_model
from keras.src.utils import pad_sequences
from telebot.states import StatesGroup, State


# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="7930769291:AAEiCjYWaSMu6xqWGdtgi_JGa0NhlMbi1pU")
# Диспетчер
dp = Dispatcher()
MAX_LEN = 150
logger = logging.getLogger(__name__)
#Загружаем готовый материал
MODEL = load_model('url_classifier_LSTM.keras')
with open('tokenizer.pkl', 'rb') as f:
    TOKENIZER = pickle.load(f)
with open('LabelEncoder.pkl', 'rb') as f:
      LE = pickle.load(f)

from datetime import datetime
dp["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

@dp.message(Command("info"))
async def cmd_info(message: types.Message, started_at: str):
    await message.answer(f"Бот запущен {started_at}")

# Хэндлер на команду /start
#@dp.message(Command("start"))
#async def cmd_start(message: types.Message):
#    await message.answer("Напиши мне URL и я проверю её на вредоность")

@dp.message(Command("dice"))
async def cmd_dice(message: types.Message):
    await message.answer_dice(emoji="🎲")

class URLs(StatesGroup):
     write_url_name = State()
     url_classifier = State()


def make_row_keyboard(items: list[str]) -> ReplyKeyboardMarkup:
    row = [KeyboardButton(text=item) for item in items]
    return ReplyKeyboardMarkup(keyboard=[row], resize_keyboard=True)
available_url_names = [
    "http://phishing.com",
    "https://google.com",
    "https://example.com"
]

def _is_valid_url(url):
    """Проверка валидности URL"""
    pattern = re.compile(
        r'^(?:http)s?://'  # Протокол
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # Домен
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'  # TLD
        r'(?:/?|[/?]\S+)$)',  # Путь и параметры
        re.IGNORECASE
    )
    return bool(re.match(pattern, url))


@dp.message(StateFilter(None), Command('start'))
async def cmd_start(message: types.Message, state: FSMContext):
    await message.answer(
        text='Напишите URL, либо выберите любую из списка🤩',
        reply_markup = make_row_keyboard(available_url_names)
        )
    await state.set_state(URLs.write_url_name)

    #@dp.message(URLs.write_url_name)
    #if _is_valid_url(url):

@dp.message(URLs.write_url_name)
async def _process_url(message: types.Message):
    url = message.text.split()
    user = message.from_user
    logger.info(f"Request from {user.id}: {url}")
    if not _is_valid_url(url):
        await message.answer(
            "❌ <b>Некорректный URL!</b>\n"
            "Пример правильного формата:\n"
            "<code>https://www.example.com/path?param=value</code>",
            parse_mode='HTML'
        )
        return


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())







