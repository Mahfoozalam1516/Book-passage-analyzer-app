import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import os

# NLTK data path configuration
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

# Download and store NLTK data locally
def download_nltk_data():
    """Download required NLTK data to local directory"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', download_dir=NLTK_DATA_PATH)

    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

# Call the download function
download_nltk_data()

# Load API key from secrets.toml
api_key = st.secrets["general"]["api_key"]

# Linking to Google Books API
def search_books(passage):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        'q': passage,
        'key': api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        books = response.json()
        return books
    else:
        st.error(f"Error: {response.status_code}")
        return None

# Counts total number of words the passage
def count_words(passage):
    words = word_tokenize(passage)
    return len([word for word in words if word not in string.punctuation])

# Counts total number of words without considering stopwords
def count_words_without_stopwords(passage):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(passage)
    return len([word for word in words if word.lower() not in stop_words and word not in string.punctuation])

# Emotion analysis on the basis of compound score
def analyze_emotion(passage):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(passage)
    
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'joy'
    elif compound <= -0.05:
        return 'sadness'
    elif scores['pos'] > scores['neg']:
        return 'surprise'
    elif scores['neg'] > scores['pos']:
        return 'anger'
    elif scores['neg'] > 0.5:  # Threshold for disgust
        return 'disgust'
    elif scores['pos'] < 0.1 and scores['neg'] > 0.1:  # Threshold for fear
        return 'fear'
    else:
        return 'neutral'

# Summarizing the passage on the basis of LSA(Latent Semantic Analysis)
def summarize_with_lsa(passage):
    parser = PlaintextParser.from_string(passage, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join(str(sentence) for sentence in summary)

# Main Streamlit application
def main():
    st.title("Text Analysis Application")
    
    passage = st.text_area("Enter your passage here:", height=200)
    
    if st.button("Analyze"):
        if passage:
            st.subheader("=== Text Analysis Results ===")
            
            # 1. Word Count
            total_words = count_words(passage)
            total_words_no_stopwords = count_words_without_stopwords(passage)

            st.write(f"Total number of words: {total_words}")
            st.write(f"Total number of words (without stopwords): {total_words_no_stopwords}")
            
            # 2. Emotional Analysis
            emotion = analyze_emotion(passage)
            st.write(f"Predominant emotion: {emotion}\n")

            # 3. Book Search
            st.write("Possible books the passage might be from:")
            books = search_books(passage)
            if books:
                for item in books.get('items', [])[:3]:  # To get the first 3 possible books
                    title = item['volumeInfo'].get('title', 'No title found')
                    authors = item['volumeInfo'].get('authors', ['No authors found'])
                    st.write(f"- Title: {title}")
                    st.write(f"  Authors: {', '.join(authors)}\n")
            
            # 4. Summary using LSA
            lsa_summary = summarize_with_lsa(passage)
            st.subheader("Summary:")
            st.write(lsa_summary)
        else:
            st.warning("Please enter a passage to analyze.")

if __name__ == "__main__":
    main()
