"""
thematic_analysis.py
Script for performing thematic analysis on bank reviews.
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicAnalyzer:
    """Class for performing thematic analysis on text data."""
    
    def __init__(self):
        """Initialize NLP models and theme mappings."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.theme_mapping = {
                'Account Access': ['login', 'password', 'account access', 'authentication'],
                'Transaction Issues': ['transfer', 'transaction', 'payment', 'slow transfer'],
                'User Interface': ['app', 'interface', 'ui', 'ux', 'design'],
                'Customer Support': ['support', 'service', 'representative', 'response time'],
                'Fees': ['fee', 'charge', 'overdraft', 'monthly fee']
            }
            logger.info("Thematic analysis models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing thematic models: {str(e)}")
            raise

    def preprocess_text(self, text):
        """
        Preprocess text by lowercasing, lemmatizing, and removing stopwords.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Processed text
        """
        try:
            doc = self.nlp(text.lower())
            return " ".join([
                token.lemma_ for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and not token.is_space
                and token.is_alpha
            ])
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return ""

    def extract_keywords_tfidf(self, texts, ngram_range=(1, 2), max_features=100):
        """
        Extract keywords using TF-IDF.
        
        Args:
            texts (iterable): Collection of texts to analyze
            ngram_range (tuple): Range of n-grams to consider
            max_features (int): Maximum number of features to return
            
        Returns:
            list: Top keywords
        """
        try:
            tfidf = TfidfVectorizer(
                ngram_range=ngram_range, 
                max_features=max_features
            )
            tfidf_matrix = tfidf.fit_transform(texts)
            keywords = tfidf.get_feature_names_out()
            return keywords
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction: {str(e)}")
            return []

    def assign_themes_rulebased(self, text):
        """
        Assign themes to text based on keyword matching.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: Assigned theme
        """
        try:
            doc = self.nlp(text.lower())
            theme_scores = {theme: 0 for theme in self.theme_mapping}
            
            for token in doc:
                for theme, keywords in self.theme_mapping.items():
                    if token.lemma_ in [kw.lower() for kw in keywords]:
                        theme_scores[theme] += 1
                        
            if max(theme_scores.values()) == 0:
                return 'Other'
            return max(theme_scores.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"Error in rule-based theme assignment: {str(e)}")
            return 'Error'

    def analyze_with_bertopic(self, texts):
        """
        Perform topic modeling using BERTopic.
        
        Args:
            texts (iterable): Collection of texts to analyze
            
        Returns:
            tuple: (topics, probabilities, model)
        """
        try:
            topic_model = BERTopic(
                language="english", 
                calculate_probabilities=True
            )
            topics, probs = topic_model.fit_transform(texts)
            return topics, probs, topic_model
        except Exception as e:
            logger.error(f"Error in BERTopic analysis: {str(e)}")
            return None, None, None