import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
from navec import Navec
from slovnet import NER
from ipymarkup import show_span_ascii_markup as show_markup
from bs4 import BeautifulSoup as bs
import gc


#clear cach memory cuda device
torch.cuda.empty_cache()
gc.collect()

# use transformers for bot talking
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium').to('cuda:1') # to cuda for faster answer 

model_tatyana = 'Tatyana/rubert-base-cased-sentiment-new'
pipe_tatyana = pipeline("text-classification", model=model_tatyana, tokenizer=model_tatyana, framework="pt")


def sentiment_check(input_text):
    return pipe_tatyana([input_text])
    

def bert_generator(input_text):
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda:1')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=3,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids ]
    answer = context_with_response[0].split('@@')[2]
    return answer    

# use natasha for NER, find location inside question if need weather
path_navec = '/home/medic/navec_news_v1_1B_250K_300d_100q.tar'
path_ner = '/home/medic/slovnet_ner_news_v1.tar'
navec = Navec.load(path_navec)
ner = NER.load(path_ner)
ner.navec(navec)

def ner_extract(input_text):
    location = ''
    #if 'погода' in input_text.lower():
    if input_text != '':
        markup_ner = ner(input_text.title())

        for mark in markup_ner.spans:
            if mark.type == 'LOC':
                location += input_text[mark.start:mark.stop] + ';'

    return location.split(';')


def weather_by_loc(location):
    yandex_api_geocode = "https://geocode-maps.yandex.ru/1.x/?format=json&apikey=b12656b6-a8a0-4f75-a9c3-66a19a79662b&results=1&geocode=" + location
    response = requests.get(yandex_api_geocode)
    geo_row_data = response.json()
    coord = geo_row_data['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos']
    r = requests.get(f'https://yandex.ru/pogoda/details/1-day-weather?lat={coord.split(" ")[1]}&lon={coord.split(" ")[0]}', headers = {'User-Agent':'Mozilla/5.0'})
    soup = bs(r.content, 'html.parser')

    table_weather = soup.find('table', class_='weather-table')
    rows = table_weather.find_all('tr')

    result = ""
    for row in rows:
        temps = row.find_all('div', class_='weather-table__temp')
        temp = [el.text.strip() for el in temps]
        part_days = row.find_all('div', class_='weather-table__daypart')
        part_day = [el.text.strip() for el in part_days]
        result += f'{part_day[0]} : {temp[0]}.\n'
        
    return result


def weather_by_loc2(location):
    yandex_api_geocode = "https://geocode-maps.yandex.ru/1.x/?format=json&apikey=b12656b6-a8a0-4f75-a9c3-66a19a79662b&results=1&geocode=" + location
    response = requests.get(yandex_api_geocode)
    geo_row_data = response.json()
    coord = geo_row_data['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos']
    r = requests.get(f'https://yandex.ru/pogoda/details/3-day-weather?lat={coord.split(" ")[1]}&lon={coord.split(" ")[0]}', headers = {'User-Agent':'Mozilla/5.0'})
    soup = bs(r.content, 'html.parser')
    
    result = ''
    cards_weather = soup.find_all('article', class_='card')

    for card_weather in cards_weather:
        date = card_weather.find('span', class_='a11y-hidden')
        if date:
            result += card_weather.find('span', class_='a11y-hidden').text.strip() + ':\n'
            table_weather = card_weather.find('table', class_='weather-table')
            rows = table_weather.find_all('tr')
            
            for row in rows:
                temps = row.find_all('div', class_='weather-table__temp')
                temp = [el.text.strip() for el in temps]
                part_days = row.find_all('div', class_='weather-table__daypart')
                part_day = [el.text.strip() for el in part_days]
                result += f'{part_day[0]} : {temp[0]}.\n'

    return result


#print(ner_extract('погода в перьми'))
#print(weather_by_loc2('Погода в перми'))
#print(sentiment_check('Анекдоты не люблю, но этот - зачетный.'))