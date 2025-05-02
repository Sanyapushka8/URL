import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Загрузка сохраненных артефактов
MODEL = load_model('url_classifier_lstm.1h5')
with open('tokenizer.pkl1', 'rb') as f:
    TOKENIZER = pickle.load(f)
with open('label_encoder.pkl1', 'rb') as f:
    LE = pickle.load(f)

# Конфигурация
MAX_LEN = 150
BOT_TOKEN = 'ВАШ_TELEGRAM_BOT_TOKEN'

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}!\n"
        "Я бот для анализа безопасности URL.\n"
        "Просто отправь мне ссылку, и я проверю её на:\n"
        "• Дефейсмент\n• Фишинг\n• Вредоносное ПО\n• Безопасные сайты"
    )


async def analyze_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик для анализа URL"""
    try:
        url = update.message.text

        # Валидация URL
        if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url):
            await update.message.reply_text("❌ Некорректный URL. Пожалуйста, отправьте полную ссылку с http/https")
            return

        # Предобработка и предсказание
        sequence = TOKENIZER.texts_to_sequences([url])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        proba = MODEL.predict(padded, verbose=0)[0]

        # Форматирование результатов
        response = "🔍 Результаты анализа:\n\n"
        for cls, prob in zip(LE.classes_, np.round(proba * 100, 1)):
            response += f"{cls.upper()}: {prob}%\n"

        # Определение главной угрозы
        main_threat = LE.classes_[np.argmax(proba)]
        if main_threat == 'benign':
            response += "\n✅ Эта ссылка безопасна"
        else:
            response += f"\n⚠️ Основная угроза: {main_threat.upper()}"

        await update.message.reply_text(response)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("😞 Произошла ошибка при анализе. Попробуйте другую ссылку")


def main():
    """Запуск бота"""
    app = Application.builder().token(BOT_TOKEN).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_url))

    # Запуск бота
    logging.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()