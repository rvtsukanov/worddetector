import telebot
from PIL import Image
import requests
from io import BytesIO
from worddetector.models import Field
from worddetector.deck_leetcoded import findWords, words_list
from worddetector.settings import TelegramSecrets

telegram_secrets = TelegramSecrets()

FILE_URL_PATTERN = "https://api.telegram.org/file/bot{0}/{1}"
FIELD_SIZE = 5

bot = telebot.TeleBot(telegram_secrets.API_TOKEN.get_secret_value(), parse_mode=None)
words = words_list()
# print(words)


@bot.message_handler(content_types=["photo", "document"])
def proceed_with_photo(message):

    file_name = bot.get_file(message.document.file_id)
    response = requests.get(FILE_URL_PATTERN.format(telegram_secrets.API_TOKEN.get_secret_value(), file_name.file_path))
    bot.send_photo(message.chat.id, response.content)
    image = Image.open(BytesIO(response.content))
    field = Field(n=FIELD_SIZE, image=image)
    print(field.flatten())
    all_words = findWords(field.flatten(), words=words)
    print(all_words)
    all_words_filtered = sorted(all_words, key=lambda x: len(x), reverse=True)[:5]
    print(all_words_filtered)
    bot.reply_to(message, field.pretty())
    bot.reply_to(message, '\n'.join(all_words_filtered))

bot.infinity_polling()
