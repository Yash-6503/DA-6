'''Consider the following dataset : https://www.kaggle.com/datasets/seungguini/youtube-commentsfor-covid19-relatedvideos?select=covid_2021_1.csv
Write a Python script for the following :
i. Read the dataset and perform data cleaning operations on it.
ii. ii. Tokenize the comments in words. iii. Perform sentiment analysis and find the percentage of
positive, negative and neutral comments.. '''

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Read the dataset
df = pd.read_csv("covid_2021_1.csv")

# Data cleaning operations
# Drop rows with missing values
df.dropna(inplace=True)

# Tokenize the comments
df['tokenized_comments'] = df['comments'].apply(lambda x: nltk.word_tokenize(x.lower()))

# Perform sentiment analysis
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['comments'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Calculate the percentage of positive, negative, and neutral comments
positive_comments = (df['sentiment'] > 0).sum()
negative_comments = (df['sentiment'] < 0).sum()
neutral_comments = (df['sentiment'] == 0).sum()

total_comments = len(df)
percentage_positive = (positive_comments / total_comments) * 100
percentage_negative = (negative_comments / total_comments) * 100
percentage_neutral = (neutral_comments / total_comments) * 100

print("Percentage of Positive Comments:", percentage_positive)
print("Percentage of Negative Comments:", percentage_negative)
print("Percentage of Neutral Comments:", percentage_neutral)
