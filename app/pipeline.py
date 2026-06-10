# app/pipeline.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor:
    def __init__(self, mode="lemmatize", lowercase=True, remove_stopwords=True):
        self.mode = mode
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Convert casing
        if self.lowercase:
            text = text.lower()
            
        # 2. Remove HTML tags, Twitter handles (@), and URLs
        text = re.sub(r'<.*?>|@\w+|https?://\S+|www\.\S+', '', text)
        
        # 3. Remove punctuation and numbers, keeping only alphabetic words
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def process(self, text):
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        
        # Filter stopwords
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words]
            
        # Apply Stemming vs Lemmatization
        if self.mode == "stem":
            tokens = [self.stemmer.stem(w) for w in tokens]
        elif self.mode == "lemmatize":
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
            
        return " ".join(tokens)