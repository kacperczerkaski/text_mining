import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from funkcje import tokenizowanie_tekstu


nieprawdziwe_posty = pd.read_csv('./Data/Fake.csv', usecols=['title', 'text'])

probka = nieprawdziwe_posty['title'][:20]
caly_tekst = ' '.join(probka.to_list())
tokeny = tokenizowanie_tekstu(caly_tekst)
# print(tokeny)

count_vectorizer = CountVectorizer(tokenizer=tokenizowanie_tekstu)
X_transform = count_vectorizer.fit_transform(probka)
slownik = count_vectorizer.get_feature_names_out(probka)

# zadanie 2 top 10 najczęściej występujących tokenów
ilosc_slow = sum(X_transform.toarray())
indeks_z2 = np.argpartition(ilosc_slow, -10)[-10:]
print(slownik[indeks_z2])
print(ilosc_slow[indeks_z2])

# zadanie 3 top 10 najważniejszych tokenów
tfid_vectorizer = TfidfVectorizer(tokenizer=tokenizowanie_tekstu)
tfid_X_transform = tfid_vectorizer.fit_transform(probka)
najwazniejsze_slowa = sum(tfid_X_transform.toarray())
indeks_z3 = np.argpartition(najwazniejsze_slowa, -10)[-10:]
print(slownik[indeks_z3])
print(najwazniejsze_slowa[indeks_z3])

# zadanie 4 top 10 dokumentów, które zawierają najwięcej tokenów
dokumenty = np.sum(X_transform.toarray(), axis=1)
print(probka[dokumenty])

# print(X_transform.toarray())
