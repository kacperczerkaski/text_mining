from wordcloud import WordCloud
import pandas as pd
import czyszczenie
import matplotlib.pyplot as plt


def word_cloud(t: str):
    csv_data = pd.read_csv(t, usecols=['airline_sentiment', 'text'], encoding='UTF-8')
    # csv_data = csv_data.sample(20)
    csv_data = ''.join(csv_data['text'])
    csv_data = czyszczenie.czyszczenie_tekstu(csv_data)
    csv_data = czyszczenie.usun_stop_words_wc(csv_data)
    csv_data = czyszczenie.stemming_wc(csv_data)
    unikalne = list(set(csv_data))
    bow = {u: csv_data.count(u) for u in unikalne}
    wc = WordCloud(width=3500, height=2500, background_color='black', colormap='prism')
    cloud = wc.generate_from_frequencies(bow)
    plt.axis("off")
    plt.imshow(cloud, interpolation='bilinear')
    plt.show()
