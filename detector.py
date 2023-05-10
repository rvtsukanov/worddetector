import requests

YANDEX_DICT = 'https://dictionary.yandex.net/api/v1/dicservice.json/lookup?key={API}&lang=ru-ru&text={TEXT}'
KEY = 'dict.1.1.20230501T095415Z.9aaaf7efc4fe70c0.d2792ce5153f9f5eec941ae071c8c3584b321874'

text = 'кукурумецна'
# ans = requests.get(YANDEX_DICT.format(API=KEY, TEXT=text))

# print(ans.json())


import pandas as pd
d = pd.read_xml('~/Downloads/annot.opcorpora.xml')



class Detector:
    def check_word_exists(self, word):
        raise NotImplementedError


class YandexDetector(Detector):
    def __init__(self, api_wildcard, key):
        self.api_wildcard = api_wildcard
        self.key = key

    def check_word_exists(self, word):
        requests.get(self.api_wildcard.format(API=self.key, TEXT=word))



