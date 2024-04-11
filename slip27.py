'''Create your own transactions dataset and apply the above process on your dataset'''

import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess the text
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

# Function to generate summary using extractive summarization
def generate_summary(text):
    # Tokenize the sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_sentence = [word for word in words if word not in stop_words]
        filtered_sentences.append(' '.join(filtered_sentence))
    
    # Convert sentences to vectors using CountVectorizer
    vectorizer = CountVectorizer().fit_transform(filtered_sentences)
    
    # Calculate sentence similarity using cosine similarity
    similarity_matrix = cosine_similarity(vectorizer, vectorizer)
    
    # Create dictionary to store sentence scores
    sentence_scores = {}
    for i in range(len(sentences)):
        sentence_scores[i] = similarity_matrix[i].mean()
    
    # Sort sentences by score in descending order
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # Generate summary with top ranked sentences
    summary = [sentences[i] for i in ranked_sentences[:2]]  # Top 2 sentences
    
    return ' '.join(summary)

# Read transactions from file
with open('transactions.txt', 'r') as file:
    transactions_text = file.read()

# Preprocess the text
preprocessed_text = preprocess_text(transactions_text)

# Generate summary
summary = generate_summary(preprocessed_text)
print("Summary:", summary)
