from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tekst = """ <p> Text mining, also :-) ;-) known as text data mining, is the process of transforming unstructured text 
into astructured format to identify meaningful patterns and new insights. By applying advanced analytical techniques, 
such as Naïve Bayes, Support Vector Machines (SVM), and 43 other deep learning algorithms, companies are able to 
<b>explore and discover ;) ;( hidden relationships within their unstructured data.<b>
i me my<p>
"""


def usun_stop_words(t: str) -> list:
    stop_words = stopwords.words("english")
    word_tokens = word_tokenize(t)
    filtrowany_tekst = [w for w in word_tokens if not w.lower() in stop_words]
    return filtrowany_tekst


print(usun_stop_words(tekst))
