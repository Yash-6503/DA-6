'''Consider text paragraph.So, keep working. Keep striving. Never give up. Fall down seven times, get
up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than
hardship. So, keep moving, keep growing, keep learning. See you at work.Preprocess the text to remove
any special characters and digits. Generate the summary using extractive summarization process. give me code'''

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Text paragraph
text = """
So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. So, keep moving, keep growing, keep learning. See you at work.
"""

# Preprocess the text to remove special characters and digits
processed_text = re.sub(r'[^a-zA-Z\s]', '', text)

# Tokenize the processed text
tokens = word_tokenize(processed_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Calculate word frequency
fdist = FreqDist(filtered_tokens)

# Get the most frequent words
summary_words = fdist.most_common(10)

# Generate summary
summary = ' '.join(word for word, freq in summary_words)

# Print the summary
print("Original Text:")
print(text)
print("\nPreprocessed Text:")
print(processed_text)
print("\nSummary:")
print(summary)
