import logging
from telegram.ext  import Updater, CommandHandler, MessageHandler, Filters #, CallbackContext
from telegram import Update
import ml_and_other
from random import randint
from emoji import emojize


emoji_happy = {1: '&#128518',2: '&#128514', 3: '&#128517', 4: '&#128513', 5: '&#129315'}
emoji_upset = {1: '&#128542', 2: '&#129402', 3: '&#128557', 4: '&#128553', 5: '&#128546'}
emoji_argy = {1: '&#128548', 2: '&#128545', 3: '&#128544', 4: '&#129324', 5: '&#129327'}
emoji_neutral = {1: '&#128528', 2: '&#128578', 3: '&#58378', 4: '&#128580', 5: '&#128566'}

def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Добрый день')
    
def textMessage(bot, update):
    response = 'Ваше сообщение принял ' + update.message.text # формируем текст ответа
    bot.send_message(chat_id=update.message.chat_id, text=response)
    

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

def start(update: Update, context):
    update.message.reply_text('Привет')

def help(update: Update, context):
    update.message.reply_text('Помощь')

def echo(update: Update, context):
    txt = update.message.text.lower()
    answer = ''
 
    if 'погода' in txt:
        locations = ml_and_other.ner_extract(txt)
        for loc in locations:
            if loc != '':
                answer += f'По вашему запросу "{txt}" найдено вот что:\n <b>Погода в {loc.title()}:</b> \
                            \n <b>{ml_and_other.weather_by_loc2(loc).title()}</b>'
    else:
        answer += ml_and_other.bert_generator(txt)
        sent = ml_and_other.sentiment_check(answer)
        
        if sent[0]['label'] == 'POSITIVE':
            answer += emoji_happy[randint(1, 5)]
        elif sent[0]['label'] == 'NEGATIVE':
            answer += emoji_argy[randint(1, 5)]
        elif sent[0]['label'] == ['NEUTRAL']:
            answer += emoji_neutral[randint(1, 5)]
        
    
    if answer != '':
        update.message.reply_text(answer, parse_mode='html')
        
    else:
        answer = "Хмм.... &#129300 видимо что-то не то спросили. Или нет информации по вашему запросу. \
                  Попробуйте переформулировать вопрос."

        update.message.reply_text(answer, parse_mode='html')
        
    

def error(update: Update, context):
    logger.warning(f'Update {update} caused error {context.error}')


def main():
    updater = Updater(token='5832291397:AAFFU6vZS8ojjDoiFsTAxZQoejafHjdo7I8', use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    
    dispatcher.add_handler(CommandHandler('help', help))

    dispatcher.add_handler(MessageHandler(Filters.text, echo))

    dispatcher.add_error_handler(error)

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()