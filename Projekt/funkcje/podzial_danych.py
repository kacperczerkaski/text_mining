import pandas as pd

data_reviews = pd.read_csv('../Data/tripadvisor_hotel_reviews.csv', sep=',', encoding='UTF-8')

pozytywne = data_reviews[data_reviews['Rating'] > 3]
negatywne = data_reviews[data_reviews['Rating'] < 3]
neutralne = data_reviews[data_reviews['Rating'] == 3]

pozytywne.to_csv('./Data/pozytywne.csv')
negatywne.to_csv('./Data/negatywne.csv')
neutralne.to_csv('./Data/neutralne.csv')
