import re
from nltk.stem import PorterStemmer
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


def usun_stop_words_wc(t: str) -> str:
    stop_words = stopwords.words("english")
    word_tokens = word_tokenize(t)
    filtrowany_tekst = [w for w in word_tokens if not w.lower() in stop_words]
    return filtrowany_tekst


def stemming_wc(t: str) -> list:
    ps = PorterStemmer()
    lista_stemming = []
    for w in t:
        lista_stemming.append(ps.stem(w))
    return lista_stemming


def usun_stop_words_t(t: str) -> str:
    stop_words = stopwords.words("english")
    filtrowany_tekst = [w for w in t if not w.lower() in stop_words]
    return filtrowany_tekst


def stemming_t(t: str) -> str:
    ps = PorterStemmer()
    return ps.stem(t)


def tokenizowanie_tekstu(t: str) -> list:
    oczyszczony = czyszczenie_tekstu(t)
    tokeny = word_tokenize(oczyszczony)
    bez_stop_words = usun_stop_words_t(tokeny)
    return [w for w in bez_stop_words if len(w) > 3]
