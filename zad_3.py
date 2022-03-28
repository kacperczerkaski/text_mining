from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tekst = """ <p> Text mining, also :-) ;-) known as text data mining, is the process of transforming unstructured text 
into astructured format to identify meaningful patterns and new insights. By applying advanced analytical techniques, 
such as Naïve Bayes, Support Vector Machines (SVM), and 43 other deep learning algorithms, companies are able to 
<b>explore and discover ;) ;( hidden relationships within their unstructured data.<b>
123324 <p>
"""


def tekst_stemming(t: str) -> list:
    lista_stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(t)
    ps = PorterStemmer()
    lista_stemming = []
    bez_stop_words = []
    for w in word_tokens:
        if w not in lista_stop_words:
            bez_stop_words.append(w)

    for b in bez_stop_words:
        lista_stemming.append(ps.stem(b))
    return lista_stemming


print(tekst_stemming(tekst))
