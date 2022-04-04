from nltk.stem import PorterStemmer


def tekst_stemming(t: str) -> list:
    ps = PorterStemmer()
    lista_stemming = []
    for w in t:
        lista_stemming.append(ps.stem(w))
    return lista_stemming
