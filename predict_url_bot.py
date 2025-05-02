import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
import pickle
from tensorflow.keras.models import load_model

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="7930769291:AAEiCjYWaSMu6xqWGdtgi_JGa0NhlMbi1pU")
# Диспетчер
dp = Dispatcher()

from datetime import datetime
dp["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

@dp.message(Command("info"))
async def cmd_info(message: types.Message, started_at: str):
    await message.answer(f"Бот запущен {started_at}")

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Напиши мне URL и я проверю её на вредоность")

@dp.message(Command("dice"))
async def cmd_dice(message: types.Message):
    await message.answer_dice(emoji="🎲")

#Загружаем готовый материал
#model = load_model('url_classifier_LSTM.joblib')
#with open('tokenizer.pkl', 'rb') as f:
#    TOKENIZER = pickle.load(f)
#with open('label_encoder.pkl', 'rb') as f:
#      LE = pickle.load(f)

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

# импорты
from config_reader import config

# Для записей с типом Secret* необходимо
# вызывать метод get_secret_value(),
# чтобы получить настоящее содержимое вместо '*******'
bot = Bot(token=config.bot_token.get_secret_value())

from aiogram.filters import Command

