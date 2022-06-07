import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from czyszczenie import tokenizowanie_tekstu


def najczesciej_wystepujace(t: str):
    # top 10 najczęściej występujących tokenów
    csv_data = pd.read_csv(t, usecols=['airline_sentiment', 'text'], sep=',', encoding='UTF-8')
    reviews = csv_data['text']
    print("top 10 najczęściej występujących tokenów")
    count_vectorizer = CountVectorizer(tokenizer=tokenizowanie_tekstu)
    x_transform = count_vectorizer.fit_transform(reviews)
    slownik = count_vectorizer.get_feature_names_out(reviews)
    slowa = sum(x_transform.toarray())
    i = np.argpartition(slowa, -10)[-10:]
    plt.barh(slownik[i], slowa[i], color='darkkhaki')
    plt.title('Najczęściej występujące słowa')
    plt.ylabel('Słowo')
    plt.xlabel('Ilość wystąpień')
    plt.show()
    columns = ["Słowo", "Ilość wystąpień"]
    new_table = PrettyTable()
    new_table.add_column(columns[0], slownik[i])
    new_table.add_column(columns[1], slowa[i])
    new_table.sortby = columns[1]
    print(new_table)


def najwazniejsze(t: str):
    # top 10 najważniejsze tokeny
    csv_data = pd.read_csv(t, usecols=['airline_sentiment', 'text'], sep=',', encoding='UTF-8')
    reviews = csv_data['text']
    print("top 10 najważniejsze tokeny")
    tfid_vectorizer = TfidfVectorizer(tokenizer=tokenizowanie_tekstu)
    tfid_x_transform = tfid_vectorizer.fit_transform(reviews)
    slownik = tfid_vectorizer.get_feature_names_out(reviews)
    najwazniejsze_slowa = sum(tfid_x_transform.toarray())
    i_2 = np.argpartition(najwazniejsze_slowa, -10)[-10:]
    plt.barh(slownik[i_2], najwazniejsze_slowa[i_2], color='darkkhaki')
    plt.title('Najważniejsze słowa')
    plt.ylabel('Słowo')
    plt.xlabel('TFIDF')
    plt.show()
    col = ["Słowo", "Wartosc TF-IDF"]
    new_table_2 = PrettyTable()
    new_table_2.add_column(col[0], slownik[i_2])
    new_table_2.add_column(col[1], najwazniejsze_slowa[i_2])
    new_table_2.sortby = col[1]
    print(new_table_2)
