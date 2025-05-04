"""
Telegram URL Classifier Bot
Анализирует URL на 4 категории: Defacement, Phishing, Benign, Malware
"""

# 1. Импорт необходимых библиотек
import logging
import re
import time
import pickle
import numpy as np
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
from dotenv import load_dotenv
load_dotenv('token.env')

# 2. Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),  # Логи в файл
        logging.StreamHandler()  # Логи в консоль
    ]
)
logger = logging.getLogger(__name__)


class URLClassifierBot:
    def __init__(self):
        # Проверка токена
        self.dp = None
        self._init_bot()  # Создает self.dp
        self._register_handlers()
        self.API_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.API_TOKEN:
            logging.error("Токен не найден! Проверьте файл .env")
            exit(1)  # Должно соответствовать обученной модели

            # Проверка наличия файлов
        self._check_required_files()

    def _load_artifact(self, file_path, loader, is_model=False):
        if is_model:
            return loader(file_path)
        else:
            with open(file_path, 'rb') as f:
                return loader(f)

    # Загрузка артефактов
        self.model = self._load_artifact('url_classifier_LSTM.keras',load_model,is_model=True)
        self.tokenizer = self._load_artifact('tokenizer.pkl', pickle.load)
        self.label_encoder = self._load_artifact('LabelEncoder.pkl.pkl', pickle.load)
        # 5. Инициализация бота
        self.MAX_LEN = 150
        self.bot = Bot(token=self.API_TOKEN)
        self.dp = Dispatcher(self.bot)

        # 6. Регистрация обработчиков
    def _init_bot(self):
        self.bot = Bot(token="7930769291:AAEiCjYWaSMu6xqWGdtgi_JGa0NhlMbi1pU")
        self.dp = Dispatcher(self.bot)

    def _check_required_files(self):
        """Проверка наличия всех необходимых файлов"""
        required = {
            'model': 'url_classifier_LSTM.keras',
            'tokenizer': 'tokenizer.pkl',
            'label_encoder': 'LabelEncoder.pkl'
        }

        missing = [name for name, path in required.items() if not os.path.exists(path)]

        if missing:
            error_msg = "Отсутствуют необходимые файлы:\n" + "\n".join(missing)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _get_env_var(self, var_name):
        """Получение переменной окружения с проверкой"""
        value = os.getenv(var_name)
        if not value:
            logger.error(f"Не найдена переменная окружения: {var_name}")
            raise ValueError(f"Требуется установить {var_name} в переменных окружения")
        return value

    def _load_artifact(self, file_path, loader):
        """Загрузка артефактов модели с обработкой ошибок"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл {file_path} не найден")

            logger.info(f"Загрузка {file_path}...")
            with open(file_path, 'rb') as f:
                return loader(f)

        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {str(e)}")
            raise

    def _register_handlers(self):
        """Регистрация обработчиков команд"""
        self.dp.register_message_handler(self._cmd_start, commands=['start'])
        self.dp.register_message_handler(self._process_url)

    def _create_keyboard(self):
        """Создание клавиатуры с примерами URL"""
        keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
        examples = [
            "https://www.google.com",
            "http://example.com/login.php",
            "https://github.com",
            "http://malicious-site.com/download.exe"
        ]
        keyboard.add(*[KeyboardButton(ex) for ex in examples])
        return keyboard

    def _is_valid_url(self, url):
        """Проверка валидности URL"""
        pattern = re.compile(
            r'^(?:http)s?://'  # Протокол
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+)'  # Домен
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'  # TLD
            r'(?:/?|[/?]\S+)$',  # Путь и параметры
            re.IGNORECASE
        )
        return bool(re.match(pattern, url))

    async def _cmd_start(self, message: types.Message):
        """Обработчик команды /start"""
        welcome_text = (
            "🔍 <b>URL Classifier Bot</b>\n\n"
            "Я анализирую ссылки на:\n"
            "• 🛡️ Безопасные (Benign)\n"
            "• 🎭 Дефейсмент (Defacement)\n"
            "• 🎣 Фишинг (Phishing)\n"
            "• 🦠 Вредоносные (Malware)\n\n"
            "Отправьте мне ссылку или выберите пример:"
        )

        try:
            await message.answer(
                welcome_text,
                parse_mode='HTML',
                reply_markup=self._create_keyboard()
            )
            logger.info(f"New user: {message.from_user.id}")
        except Exception as e:
            logger.error(f"Start command error: {str(e)}")

    async def _process_url(self, message: types.Message):
        """Обработка URL сообщений"""
        try:
            url = message.text.strip()
            user = message.from_user
            logger.info(f"Request from {user.id}: {url}")

            # Валидация URL
            if not self._is_valid_url(url):
                await message.answer(
                    "❌ <b>Некорректный URL!</b>\n"
                    "Пример правильного формата:\n"
                    "<code>https://www.example.com/path?param=value</code>",
                    parse_mode='HTML'
                )
                return

            # Уведомление о начале обработки
            processing_msg = await message.answer("⏳ <b>Анализирую ссылку...</b>", parse_mode='HTML')

            # Препроцессинг URL
            start_time = time.time()
            sequence = self.tokenizer.texts_to_sequences([url])
            padded = pad_sequences(sequence, maxlen=self.MAX_LEN)

            # Предсказание
            probabilities = self.model.predict(padded, verbose=0)[0]
            processing_time = time.time() - start_time

            # Форматирование результатов
            results = []
            for cls, prob in zip(self.label_encoder.classes_, np.round(probabilities * 100, 1)):
                icon = "🟢" if cls == "benign" else "🔴"
                results.append(f"{icon} <b>{cls.upper()}</b>: <code>{prob}%</code>")

            main_category = self.label_encoder.classes_[np.argmax(probabilities)]
            result_text = (
                "🔍 <b>Результаты анализа:</b>\n\n" +
                "\n".join(results) +
                f"\n\n⏱ <b>Время обработки:</b> <code>{processing_time:.2f} сек</code>" +
                f"\n🚨 <b>Основная категория:</b> <code>{main_category.upper()}</code>"
            )

            # Отправка результатов
            await self.bot.delete_message(
                chat_id=message.chat.id,
                message_id=processing_msg.message_id
            )
            await message.answer(
                result_text,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            logger.info(f"Success: {url}")

        except Exception as e:
            error_text = "⚠️ <b>Ошибка обработки!</b> Попробуйте другую ссылку."
            await message.answer(error_text, parse_mode='HTML')
            logger.error(f"Processing error: {str(e)}", exc_info=True)

    def run(self):
        """Запуск бота"""
        logger.info("Starting bot...")
        try:
            executor.start_polling(self.dp, skip_updates=True)
        except Exception as e:
            logger.critical(f"Bot failed: {str(e)}", exc_info=True)
            raise


if __name__ == '__main__':
    # Проверка переменных окружения перед запуском
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        logger.error("Требуется установить TELEGRAM_BOT_TOKEN в переменных окружения!")
        exit(1)

    # Создание и запуск бота
    try:
        bot = URLClassifierBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Failed to initialize bot: {str(e)}")