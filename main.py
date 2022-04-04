import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from funkcje import tokenizowanie_tekstu


nieprawdziwe_posty = pd.read_csv('./Data/Fake.csv', usecols=['title', 'text'])

probka = nieprawdziwe_posty['title'].sample(10)
caly_tekst = ' '.join(probka.to_list())
tokeny = tokenizowanie_tekstu(caly_tekst)
print(tokeny)

# vectorizer = TfidfVectorizer(tokenizer=tokenizowanie_tekstu)
# X_transform = vectorizer.fit_transform(true)
# print(np.asarray(X_transform))

vectorizer = CountVectorizer(tokenizer=tokenizowanie_tekstu)
X_transform = vectorizer.fit_transform(probka)

print(X_transform.toarray())
