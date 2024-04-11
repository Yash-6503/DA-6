'''Consider any text paragraph. Remove the stopwords. Tokenize the paragraph to extract words and
sentences. Calculate the word frequency distribution and plot the frequencies. Plot the wordcloud of the
text.'''

import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Text paragraph
text = """
So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. So, keep moving, keep growing, keep learning. See you at work.
"""

# Remove stopwords
stop_words = set(stopwords.words('english'))
cleaned_text = ' '.join(word for word in text.split() if word.lower() not in stop_words)

# Tokenize words and sentences
words = word_tokenize(cleaned_text)
sentences = sent_tokenize(text)

# Calculate word frequency distribution
fdist = FreqDist(words)

# Plot word frequencies
plt.figure(figsize=(10, 6))
fdist.plot(20, cumulative=False)
plt.title('Word Frequency Distribution')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Generate and plot wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud')
plt.show()
