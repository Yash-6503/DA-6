'''Consider text paragraph. """Hello all, Welcome to Python Programming Academy. Python
Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled
in this Academy.""" Preprocess the text to remove any special characters and digits. Generate the
summary using extractive summarization process. '''

import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text paragraph
text = """Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."""

# Preprocess the text to remove special characters and digits
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

# Generate summary using extractive summarization
def generate_summary(text):
    # Tokenize the sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
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

# Preprocess the text
preprocessed_text = preprocess_text(text)

# Generate summary
summary = generate_summary(preprocessed_text)
print("Summary:", summary)
