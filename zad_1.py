import re

tekst = """ <p> Text mining, also :-) ;-) known as text data mining, is the process of transforming unstructured text 
into astructured format to identify meaningful patterns and new insights. By applying advanced analytical techniques, 
such as Naïve Bayes, Support Vector Machines (SVM), and 43 other deep learning algorithms, companies are able to 
<b>explore and discover ;) ;( hidden relationships within their unstructured data.<b>
123324 <p>
"""


def czyszczenie_tekstu(t: str) -> str:
    emotikony = re.findall(r"[;|:][-]?[\)|\(|<|>]", t)
    tekst_male = t.lower()
    usun_liczby = re.sub(r"\d", "", tekst_male)
    usun_html = re.sub(r"<.*?>", "", usun_liczby)
    usun_interp = re.sub(r"[^\w\s]", '', usun_html)
    usun_spacje = usun_interp.strip()
    oczyszczony = usun_spacje + ' '.join(emotikony)
    return oczyszczony


print(czyszczenie_tekstu(tekst))
