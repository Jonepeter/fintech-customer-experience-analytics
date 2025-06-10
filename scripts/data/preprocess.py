import pandas as pd 
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import emoji

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
            return df
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
    
    def preprocess_review(self, text, emoji_mode='convert'):
        """
        Complete preprocessing pipeline for a single review
        Args:
            text (str): Input review text
            emoji_mode (str): How to handle emojis - 'remove' or 'convert'
        Returns:
            str: Preprocessed text as a single string
        """
        # Handle emojis
        text = self.handle_emojis(text, mode=emoji_mode)
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
    
    def preprocess_reviews(self, df, text_column, emoji_mode='convert'):
        """
        Preprocess all reviews in a DataFrame
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text
            emoji_mode (str): How to handle emojis - 'remove' or 'convert'
        Returns:
            pd.DataFrame: DataFrame with preprocessed reviews
        """
        # Fill missing values in the text column with empty string
        df[text_column] = df[text_column].fillna('')
        
        # Apply preprocessing
        df['preprocessed_review'] = df[text_column].apply(
            lambda x: self.preprocess_review(x, emoji_mode=emoji_mode)
        )
        
        # Fill any remaining missing values in preprocessed_review with empty string
        df['preprocessed_review'] = df['preprocessed_review'].fillna('')
        
        return df
    
    def handle_emojis(self, text, mode='convert'):
        """
        Handle emojis in text by either removing them or converting to text descriptions
        Args:
            text (str): Input text containing emojis
            mode (str): 'remove' to remove emojis, 'convert' to convert to text descriptions
        Returns:
            str: Text with emojis handled according to mode
        """
        if not isinstance(text, str):
            return ""
            
        if mode == 'remove':
            # Remove all emojis
            return emoji.replace_emoji(text, replace='')
        elif mode == 'convert':
            # Convert emojis to their text descriptions
            return emoji.demojize(text, delimiters=(' ', ' '))
        else:
            raise ValueError("Mode must be either 'remove' or 'convert'")

    # filter emoji 
    def filter_emoji_rows(self, df, text_column):
        """
        Filter DataFrame to separate rows with and without emojis
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text
        Returns:
            tuple: (DataFrame with emoji rows, DataFrame without emoji rows)
        """
        # Function to check if text contains emoji
        def contains_emoji(text):
            if not isinstance(text, str):
                return False
            return any(char in emoji.EMOJI_DATA for char in text)
        
        # Create boolean mask for rows with emojis
        emoji_mask = df[text_column].apply(contains_emoji)
        
        # Split DataFrame into rows with and without emojis
        emoji_rows = df[emoji_mask].copy()
        
        return emoji_rows