'''Consider text paragraph."""Hello all, Welcome to Python Programming Academy. Python
Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled
in this Academy."""Remove the stopwords. '''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords corpus if not already downloaded
nltk.download('stopwords')

# Sample text paragraph
text = """Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."""

# Tokenize the text
words = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_text = [word for word in words if word.lower() not in stop_words]

# Join the filtered words back into a sentence
filtered_sentence = ' '.join(filtered_text)

print("Text after removing stopwords:")
print(filtered_sentence)
