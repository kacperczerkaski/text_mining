import pandas as pd

airlines = pd.read_csv('../data/tweets_airline.csv', sep=',', encoding='UTF-8')

positive = airlines[airlines['airline_sentiment'] == 'positive']
negative = airlines[airlines['airline_sentiment'] == 'negative']
neutral = airlines[airlines['airline_sentiment'] == 'neutral']

positive.to_csv('../data/positive.csv')
negative.to_csv('../data/negative.csv')
neutral.to_csv('../data/neutral.csv')
