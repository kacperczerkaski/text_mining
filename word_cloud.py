from wordcloud import WordCloud
import pandas
import text_cleaning
import stemming
import matplotlib.pyplot as plt


def word_cloud(t: str):
    csv_data = pandas.read_csv(t, usecols=['title', 'text'], encoding='UTF-8')
    #csv_data = csv_data.sample(20)
    csv_data = ''.join(csv_data['title'])
    csv_data = text_cleaning.czyszczenie_tekstu(csv_data)
    csv_data = text_cleaning.usun_stop_words(csv_data)
    csv_data = stemming.tekst_stemming(csv_data)
    unikalne = list(set(csv_data))
    bow = {u: csv_data.count(u) for u in unikalne}
    wc = WordCloud(width=3500, height=2500, background_color='black', colormap='prism')
    cloud = wc.generate_from_frequencies(bow)
    plt.axis("off")
    plt.imshow(cloud, interpolation='bilinear')
    plt.show()
