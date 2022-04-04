import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def czyszczenie_tekstu(t: str) -> str:
    emotikony = re.findall(r'[;|:][-]?[\)|\(|<|>]', t)
    tekst_male = t.lower()
    usun_liczby = re.sub(r'\d', '', tekst_male)
    usun_html = re.sub(r'<.*?>', '', usun_liczby)
    usun_interp = re.sub(r'\W(?<!\s)', '', usun_html)
    usun_spacje = usun_interp.strip()
    oczyszczony = usun_spacje + ' '.join(emotikony)
    return oczyszczony


def usun_stop_words(t: str) -> str:
    stop_words = stopwords.words("english")
    word_tokens = word_tokenize(t)
    filtrowany_tekst = [w for w in word_tokens if not w.lower() in stop_words]
    return filtrowany_tekst
