import asyncio
import logging
import pickle
import re
import numpy as np
from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.state import StatesGroup, State
from urllib.parse import urlparse

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="7930769291:AAEiCjYWaSMu6xqWGdtgi_JGa0NhlMbi1pU")
# Диспетчер
dp = Dispatcher()
MAX_LEN = 150
logger = logging.getLogger(__name__)
#Загружаем готовый материал
with open("url_classifierXG1.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open('tfidfXG.pkl', 'rb') as f:
      tfidf = pickle.load(f)
with open('label_encoderXG.pkl', 'rb') as f:
    le = pickle.load(f)

from datetime import datetime
dp["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

@dp.message(Command("info"))
async def cmd_info(message: types.Message, started_at: str):
    await message.answer(f"Бот запущен {started_at}")

@dp.message(Command("dice"))
async def cmd_dice(message: types.Message):
    await message.answer_dice(emoji="🎲")

def tokenize_url(url):
    try:
        # Удаление протокола и разделение URL на части
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query
        tokens = []

        # Разделение домена на части
        if parsed.netloc:
            tokens += re.split(r'[\.\-]', parsed.netloc)

        # Разделение пути и параметров
        tokens += re.split(r'[/&=?]', path + " " + query)

        # Фильтрация и очистка токенов
        tokens = [t.lower().strip() for t in tokens if t.strip() != '']
        return ' '.join(tokens)

    except:
        return ''

class URLs(StatesGroup):
     write_url_name = State()
     url_classifier = State()

def predict_url_type(url):
    try:
        # Токенизация
        tokens = tokenize_url(url)

        # Векторизация
        X_new = tfidf.transform([tokens])

        # Предсказание
        proba = MODEL.predict_proba(X_new)[0]
        pred_class = le.inverse_transform([np.argmax(proba)])[0]

        return {
            "url": url,
            "predicted_class": pred_class,
            "probabilities": dict(zip(le.classes_, proba.round(3)))
        }
    except Exception as e:
        # Логирование ошибки для отладки
        logger.error(f"Error processing URL {url}: {str(e)}")
        return {"error": str(e)}

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
        text=f"Привет!\n"
        "Я бот для анализа безопасности URL🤓.\n"
        "Просто отправь мне ссылку или выбери из списка, и я проверю её на🔎:\n"
        "• Безопасные сайты😇\n• Порча🧟‍♂️\n• Фишинг👿\n• Вредоносное ПО☠️",
        reply_markup = make_row_keyboard(available_url_names),
        )
    await state.set_state(URLs.write_url_name)

@dp.message(URLs.write_url_name)
async def _process_url(message: types.Message):
    url = message.text
    user = message.from_user
    logger.info(f"Request from {user.id}: {url}")
    if not _is_valid_url(url):
        await message.answer(
            "❌ <b>Некорректный URL!😔</b>\n"
            "Пример правильного формата✅:\n"
            "<code>https://www.example.com/path?param=value</code>",
            parse_mode='HTML'
        )
        return
    predict_url = predict_url_type(url)
    await message.answer(f"🔗 Ссылка🔎: {predict_url['url']}\n"
        f"📊 Класс🎲: {predict_url['predicted_class']}\n\n"
        f"📈 Вероятности👀🧠:\n"
        f"• Безопасная😇: {predict_url['probabilities'].get('benign', 0):.3f}\n"
        f"• Порча🧟‍♂️‍: {predict_url['probabilities'].get('defacement', 0):.3f}\n"
        f"• Фишинг👿: {predict_url['probabilities'].get('phishing', 0):.3f}\n"
        f"• Вирус☠️: {predict_url['probabilities'].get('malware', 0):.3f}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())







