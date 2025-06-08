import pandas as pd 
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

class PreprocessReview:
    def __init__(self):
        # Download required NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_data(self, file_path):
        """
        Load review data from a CSV file
        Args:
            file_path (str): Path to the CSV file containing reviews
        Returns:
            pd.DataFrame: DataFrame containing the reviews
        """
        try:
            na_values = [
                    '', 'NA', 'N/A', 'NULL', 'null', 'NaN', 'nan', 'None', 'none',
                    '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                    '1.#IND', '1.#QNAN', '<NA>', 'NIL', 'nil', 'NULL', 'null', 'NaN', 'nan'
                    ]
            df = pd.read_csv(
                file_path,
                na_values=na_values,
                keep_default_na=True,  # Keep pandas default NA values
                encoding='utf-8'  # Specify encoding
            )
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, numbers, and extra whitespace
        Args:
            text (str): Input text to clean
        Returns:
            str: Cleaned text
        """
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        return ""
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        Args:
            text (str): Input text to tokenize
        Returns:
            list: List of tokens
        """
        if isinstance(text, str):
            return wordpunct_tokenize(text)
        return []
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokenized text
        Args:
            tokens (list): List of tokens
        Returns:
            list: List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form
        Args:
            tokens (list): List of tokens
        Returns:
            list: List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_review(self, text):
        """
        Complete preprocessing pipeline for a single review
        Args:
            text (str): Input review text
        Returns:
            str: Preprocessed text as a single string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def preprocess_reviews(self, df, text_column):
        """
        Preprocess all reviews in a DataFrame
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text
        Returns:
            pd.DataFrame: DataFrame with preprocessed reviews
        """
        df['preprocessed_review'] = df[text_column].apply(self.preprocess_review)
        return df
    