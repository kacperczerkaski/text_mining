from wordcloud import WordCloud
import pandas as pd
import czyszczenie_tekstu
import matplotlib.pyplot as plt


def word_cloud(t: str):
    csv_data = pd.read_csv(t, usecols=['Review', 'Rating'], encoding='UTF-8')
    # csv_data = csv_data.sample(20)
    csv_data = ''.join(csv_data['Review'])
    csv_data = czyszczenie_tekstu.czyszczenie_tekstu(csv_data)
    csv_data = czyszczenie_tekstu.usun_stop_words_wc(csv_data)
    csv_data = czyszczenie_tekstu.stemming_wc(csv_data)
    unikalne = list(set(csv_data))
    bow = {u: csv_data.count(u) for u in unikalne}
    wc = WordCloud(width=3500, height=2500, background_color='white', colormap='winter')
    cloud = wc.generate_from_frequencies(bow)
    plt.axis("off")
    plt.imshow(cloud, interpolation='bilinear')
    plt.show()
