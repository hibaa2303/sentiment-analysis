import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tweets.csv', encoding='latin-1', names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'}).fillna('neutral')
df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.show()
