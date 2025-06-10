"""
thematic_analysis.py
Script for performing thematic analysis on bank reviews.
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
# from bertopic import BERTopic
import pandas as pd
import logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicAnalyzer:
    """Class for performing thematic analysis on text data."""
    
    def __init__(self):
        """Initialize NLP models and theme mappings."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Enhanced theme mapping with more specific keywords and phrases
            self.theme_mapping = {
                'Account Access & Security': [
                    'login', 'password', 'account access', 'authentication', 'security',
                    'verification', 'biometric', 'face id', 'fingerprint', '2fa',
                    'locked out', 'access denied', 'secure', 'hack', 'breach'
                ],
                'Transaction & Payment Issues': [
                    'transfer', 'transaction', 'payment', 'slow transfer', 'failed',
                    'declined', 'pending', 'processing', 'fee', 'charge', 'overdraft',
                    'balance', 'deposit', 'withdrawal', 'atm', 'card'
                ],
                'User Interface & Experience': [
                    'app', 'interface', 'ui', 'ux', 'design', 'layout', 'navigation',
                    'menu', 'button', 'screen', 'responsive', 'crash', 'freeze',
                    'lag', 'slow', 'glitch', 'bug', 'update'
                ],
                'Customer Support & Service': [
                    'support', 'service', 'representative', 'response time', 'wait',
                    'hold', 'chat', 'email', 'phone', 'call', 'contact', 'help',
                    'assist', 'resolve', 'complaint', 'issue'
                ],
                'Fees & Charges': [
                    'fee', 'charge', 'overdraft', 'monthly fee', 'maintenance',
                    'transaction fee', 'atm fee', 'foreign fee', 'interest',
                    'penalty', 'cost', 'price', 'expensive', 'cheap'
                ]
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

    def extract_keywords_tfidf(self, texts, ngram_range=(1, 3), max_features=100):
        """
        Extract keywords using TF-IDF with enhanced n-gram support.
        
        Args:
            texts (iterable): Collection of texts to analyze
            ngram_range (tuple): Range of n-grams to consider (default: 1-3)
            max_features (int): Maximum number of features to return
            
        Returns:
            list: Top keywords with their TF-IDF scores
        """
        try:
            tfidf = TfidfVectorizer(
                ngram_range=ngram_range, 
                max_features=max_features,
                stop_words='english'
            )
            tfidf_matrix = tfidf.fit_transform(texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Calculate average TF-IDF scores for each feature
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create list of (keyword, score) tuples
            keywords_with_scores = list(zip(feature_names, avg_scores))
            return sorted(keywords_with_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction: {str(e)}")
            return []

    def assign_themes_rulebased(self, text):
        """
        Assign themes to text based on enhanced keyword matching with confidence scores.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            tuple: (primary_theme, theme_scores) where theme_scores is a dict of theme:score
        """
        try:
            doc = self.nlp(text.lower())
            theme_scores = {theme: 0 for theme in self.theme_mapping}
            
            # Count keyword matches for each theme
            for token in doc:
                for theme, keywords in self.theme_mapping.items():
                    if token.lemma_ in [kw.lower() for kw in keywords]:
                        theme_scores[theme] += 1
            
            # Calculate confidence scores
            total_matches = sum(theme_scores.values())
            if total_matches == 0:
                return 'Other', {'Other': 1.0}
            
            # Normalize scores
            theme_scores = {theme: score/total_matches for theme, score in theme_scores.items()}
            primary_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            
            return primary_theme, theme_scores
        except Exception as e:
            logger.error(f"Error in rule-based theme assignment: {str(e)}")
            return 'Error', {'Error': 1.0}

    # def analyze_with_bertopic(self, texts, n_topics=5):
    #     """
    #     Perform topic modeling using BERTopic with enhanced configuration.
        
    #     Args:
    #         texts (iterable): Collection of texts to analyze
    #         n_topics (int): Number of topics to extract
            
    #     Returns:
    #         tuple: (topics, probabilities, model, topic_keywords)
    #     """
    #     try:
    #         topic_model = BERTopic(
    #             language="english",
    #             calculate_probabilities=True,
    #             nr_topics=n_topics,
    #             min_topic_size=5
    #         )
    #         topics, probs = topic_model.fit_transform(texts)
            
    #         # Extract keywords for each topic
    #         topic_keywords = topic_model.get_topic_info()
            
    #         return topics, probs, topic_model, topic_keywords
    #     except Exception as e:
    #         logger.error(f"Error in BERTopic analysis: {str(e)}")
    #         return None, None, None, None

    def analyze_reviews(self, reviews_df):
        """
        Perform comprehensive thematic analysis on a DataFrame of reviews.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing review data
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with thematic analysis results
        """
        try:
            # Preprocess all reviews
            processed_texts = reviews_df['review_text'].apply(self.preprocess_text)
            
            # Extract keywords
            keywords = self.extract_keywords_tfidf(processed_texts)
            
            # Assign themes
            theme_results = reviews_df['review_text'].apply(self.assign_themes_rulebased)
            reviews_df['primary_theme'] = theme_results.apply(lambda x: x[0])
            reviews_df['theme_scores'] = theme_results.apply(lambda x: x[1])
            
            # Perform BERTopic analysis
            topics, probs, model, topic_keywords = self.analyze_with_bertopic(processed_texts)
            if topics is not None:
                reviews_df['topic_id'] = topics
                reviews_df['topic_probabilities'] = probs.tolist()
            
            return reviews_df
        except Exception as e:
            logger.error(f"Error in comprehensive review analysis: {str(e)}")
            return reviews_df