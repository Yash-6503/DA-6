'''Consider the following dataset : https://www.kaggle.com/datasets/datasnaek/youtubenew?select=INvideos.csv
Write a Python script for the following :
i. Read the dataset and perform data cleaning operations on it.
ii. ii. Find the total views, total likes, total dislikes and comment count'''

import pandas as pd

# Read the dataset
df = pd.read_csv("INvideos.csv")

# Data cleaning operations
# Drop rows with missing values
df.dropna(inplace=True)

# Find the total views, likes, dislikes, and comment count
total_views = df['views'].sum()
total_likes = df['likes'].sum()
total_dislikes = df['dislikes'].sum()
total_comments = df['comment_count'].sum()

print("Total Views:", total_views)
print("Total Likes:", total_likes)
print("Total Dislikes:", total_dislikes)
print("Total Comments:", total_comments)
