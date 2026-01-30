import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data():
    """
    Load training and testing datasets from CSV files.
    Returns:
        train_df (DataFrame): Training dataset
        test_df (DataFrame): Testing dataset
    """
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    return train_df, test_df


# PREPROCESSING PIPELINE
REFUSAL_KEYWORDS = [
    'i cannot', 'i can\'t', 'i apologize', 'i\'m sorry', 'i cannot provide',
    'unable to', 'i cannot help', 'i don\'t', 'unfortunately', 'unable', 
    'i cannot assist', 'cannot assist', 'i cannot help', 'not able to',
    'i cannot answer', 'cannot answer', 'i cannot say', 'cannot say',
    'against my values', 'against my guidelines', 'not appropriate',
    'not allowed', 'not permitted', 'cannot provide', 'will not',
    'won\'t provide', 'won\'t help', 'refused', 'refuse', 'declining',
    'decline', 'don\'t think', 'don\'t feel', 'not comfortable'
]

def preprocess_text(text):
    """
    Apply comprehensive preprocessing to text:
    1. Lowercasing
    2. Noise removal and special character handling
    3. Extra whitespace removal
    4. Tokenization
    5. Lemmatization
    """
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep apostrophes and hyphens for contractions
    text = re.sub(r'[^a-zA-Z0-9\s\'\-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    processed_text = ' '.join(tokens)
    
    return processed_text, tokens

# FEATURE EXTRACTION 
# 1. Response Length Features
def extract_length_features(response_text):
    words = response_text.split()
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    return {
        'response_length': len(response_text),
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'char_per_word': len(response_text) / max(word_count, 1)
    }

# 2. Refusal Keywords Detection
def detect_refusal_keywords(response_text):
    """Detect if refusal keywords appear in the response, especially at the beginning"""
    text_lower = response_text.lower()
    
    # Check if any refusal keyword appears in the first 100 characters (beginning)
    beginning = text_lower[:100]
    has_keyword_at_start = sum(1 for keyword in REFUSAL_KEYWORDS if keyword in beginning)
    
    # Overall keyword presence
    has_keyword_overall = sum(1 for keyword in REFUSAL_KEYWORDS if keyword in text_lower)
    
    return {
        'refusal_keyword_at_start': has_keyword_at_start,
        'refusal_keyword_overall': has_keyword_overall,
        'has_any_refusal_keyword': 1 if has_keyword_overall > 0 else 0
    }

# 3. Sentiment Analysis Features
def extract_sentiment_features(response_text):
    """Extract sentiment features using TextBlob"""
    try:
        blob = TextBlob(response_text)
        polarity = blob.sentiment.polarity  # -1 to 1 (negative to positive)
        subjectivity = blob.sentiment.subjectivity  # 0 to 1 (objective to subjective)
        
        return {
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
            'is_negative_sentiment': 1 if polarity < -0.1 else 0,
            'is_neutral_sentiment': 1 if -0.1 <= polarity <= 0.1 else 0,
            'is_positive_sentiment': 1 if polarity > 0.1 else 0
        }
    except:
        return {
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0,
            'is_negative_sentiment': 0,
            'is_neutral_sentiment': 1,
            'is_positive_sentiment': 0
        }

# 4. Response Structure Features
def extract_structure_features(response_text):
    """Extract structural features of the response"""
    sentences = [s for s in response_text.split('.') if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentence_count > 0 else 0
    punctuation_count = sum(1 for c in response_text if c in string.punctuation)
    question_mark_count = response_text.count('?')
    exclamation_count = response_text.count('!')
    uppercase_ratio = sum(1 for c in response_text if c.isupper()) / max(len(response_text), 1)
    has_multiple_sentences = 1 if sentence_count > 1 else 0
    
    return {
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'punctuation_count': punctuation_count,
        'question_mark_count': question_mark_count,
        'exclamation_count': exclamation_count,
        'uppercase_ratio': uppercase_ratio,
        'has_multiple_sentences': has_multiple_sentences
    }

# 5. Apologetic Tone Features
def extract_apologetic_features(response_text):
    """Detect apologetic/formal tone indicators"""
    text_lower = response_text.lower()
    
    apologetic_words = ['sorry', 'apologize', 'apologies', 'regret', 'unfortunately']
    formal_indicators = ['unable', 'cannot', 'cannot help', 'assist', 'request', 'policy']
    
    has_apology = sum(1 for word in apologetic_words if word in text_lower)
    has_formal = sum(1 for word in formal_indicators if word in text_lower)
    
    return {
        'apology_word_count': has_apology,
        'formal_word_count': has_formal,
        'is_apologetic': 1 if has_apology > 0 else 0,
        'is_formal': 1 if has_formal > 0 else 0
    }

# Extract all features
def extract_all_features(train_df, test_df):
    print("Extracting length features...")
    train_length_features = train_df['response'].apply(extract_length_features).apply(pd.Series)
    test_length_features = test_df['response'].apply(extract_length_features).apply(pd.Series)

    print("Extracting refusal keyword features...")
    train_keyword_features = train_df['response'].apply(detect_refusal_keywords).apply(pd.Series)
    test_keyword_features = test_df['response'].apply(detect_refusal_keywords).apply(pd.Series)

    print("Extracting sentiment features...")
    train_sentiment_features = train_df['response'].apply(extract_sentiment_features).apply(pd.Series)
    test_sentiment_features = test_df['response'].apply(extract_sentiment_features).apply(pd.Series)

    print("Extracting structure features...")
    train_structure_features = train_df['response'].apply(extract_structure_features).apply(pd.Series)
    test_structure_features = test_df['response'].apply(extract_structure_features).apply(pd.Series)

    print("Extracting apologetic tone features...")
    train_apology_features = train_df['response'].apply(extract_apologetic_features).apply(pd.Series)
    test_apology_features = test_df['response'].apply(extract_apologetic_features).apply(pd.Series)

    print("\nFeature extraction complete!")

    train_engineered_features = pd.concat([
        train_length_features,
        train_keyword_features,
        train_sentiment_features,
        train_structure_features,
        train_apology_features
    ], axis=1)

    test_engineered_features = pd.concat([
        test_length_features,
        test_keyword_features,
        test_sentiment_features,
        test_structure_features,
        test_apology_features
    ], axis=1)
    
    return train_engineered_features, test_engineered_features

def vectorize_tfidf(train_df, test_df):
    print("Generating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_response'])
    test_tfidf = tfidf_vectorizer.transform(test_df['processed_response'])

    print(f"TF-IDF shape - Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")
    # Convert sparse matrices to dense for concatenation with other features
    train_tfidf_dense = train_tfidf.toarray()
    test_tfidf_dense = test_tfidf.toarray()

    # Create dataframes for vectorized features
    train_tfidf_df = pd.DataFrame(train_tfidf_dense, columns=[f'tfidf_{i}' for i in range(train_tfidf_dense.shape[1])])
    test_tfidf_df = pd.DataFrame(test_tfidf_dense, columns=[f'tfidf_{i}' for i in range(test_tfidf_dense.shape[1])])

    return train_tfidf_df, test_tfidf_df

def vectorize_count(train_df, test_df):
    print("\nGenerating Count Vectorizer features...")
    count_vectorizer = CountVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    train_count = count_vectorizer.fit_transform(train_df['processed_response'])
    test_count = count_vectorizer.transform(test_df['processed_response'])

    print(f"Count Vectorizer shape - Train: {train_count.shape}, Test: {test_count.shape}")

    # Convert sparse matrices to dense for concatenation with other features
    train_count_dense = train_count.toarray()
    test_count_dense = test_count.toarray()

    # Create dataframes for vectorized features
    train_count_df = pd.DataFrame(train_count_dense, columns=[f'count_{i}' for i in range(train_count_dense.shape[1])])
    test_count_df = pd.DataFrame(test_count_dense, columns=[f'count_{i}' for i in range(test_count_dense.shape[1])])

    return train_count_df, test_count_df


