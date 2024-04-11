'''Download the movie_review.csv dataset from Kaggle by using the following link
:https://www.kaggle.com/nltkdata/movie-review/version/3?select=movie_review.csv to perform
sentiment analysis on above dataset and create a wordcloud. '''

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import requests

# Download the dataset from Kaggle
url = "https://www.kaggle.com/nltkdata/movie-review/version/3?select=movie_review.csv"
r = requests.get(url)
with open("movie_review.csv", "wb") as f:
    f.write(r.content)

# Load the dataset
df = pd.read_csv("movie_review.csv")

# Perform sentiment analysis
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['review'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Create wordcloud
text = ' '.join(df['review'])
stop_words = set(stopwords.words('english'))
wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)

# Plot the wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud of Movie Reviews')
plt.show()

# Display sentiment analysis results
print("Sentiment Analysis Results:")
print(df[['review', 'sentiment']])
