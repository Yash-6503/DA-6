'''Consider any text paragraph. Preprocess the text to remove any special characters and digits.
Generate the summary using extractive summarization process'''

import re
from nltk.tokenize import word_tokenize
from gensim.summarization import summarize

# Sample text paragraph
text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It enables computers to understand, interpret, and generate human language in a way that is both meaningful and useful. NLP algorithms are designed to analyze large amounts of natural language data and derive patterns and insights from it. This can be particularly useful in applications such as sentiment analysis, machine translation, chatbots, and more.
"""

# Preprocess the text to remove special characters and digits
processed_text = re.sub(r'[^a-zA-Z\s]', '', text)

# Tokenize the processed text
tokens = word_tokenize(processed_text)

# Generate the summary using extractive summarization
summary = summarize(processed_text)

# Print the summary
print("Original Text:")
print(text)
print("\nPreprocessed Text:")
print(processed_text)
print("\nSummary:")
print(summary)
