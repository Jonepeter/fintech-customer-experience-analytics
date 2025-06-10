"""
thematic_analysis.py
Script for performing thematic analysis on bank reviews.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicAnalyzer:
    """Class for performing thematic analysis on text data."""
    
    def __init__(self):
        """Initialize NLP models and theme mappings."""
        try:
            # Download required NLTK data
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
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
        Preprocess text by cleaning, tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Processed text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = wordpunct_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]
            
            return " ".join(processed_tokens)
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
            # Preprocess all texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            tfidf = TfidfVectorizer(
                ngram_range=ngram_range, 
                max_features=max_features,
                stop_words='english'
            )
            tfidf_matrix = tfidf.fit_transform(processed_texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Calculate average TF-IDF scores for each feature
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create list of (keyword, score) tuples
            keywords_with_scores = list(zip(feature_names, avg_scores))
            
            # Filter out low-scoring keywords
            keywords_with_scores = [(kw, score) for kw, score in keywords_with_scores if score > 0.01]
            
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
            # Preprocess text
            processed_text = self.preprocess_text(text)
            tokens = wordpunct_tokenize(processed_text)
            
            theme_scores = {theme: 0 for theme in self.theme_mapping}
            
            # Count keyword matches for each theme
            for token in tokens:
                for theme, keywords in self.theme_mapping.items():
                    if token in [kw.lower() for kw in keywords]:
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

    def analyze_with_lda(self, texts, n_topics=5, max_features=1000):
        """
        Perform topic modeling using Latent Dirichlet Allocation (LDA).
        
        Args:
            texts (iterable): Collection of texts to analyze
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features to consider
            
        Returns:
            tuple: (topic_assignments, topic_keywords, model)
        """
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Create document-term matrix
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            # Fit LDA model
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42,
                batch_size=128,
                verbose=0
            )
            
            # Get topic assignments
            topic_assignments = lda_model.fit_transform(doc_term_matrix)
            
            # Get top keywords for each topic
            feature_names = vectorizer.get_feature_names_out()
            topic_keywords = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_keywords_idx = topic.argsort()[:-10-1:-1]
                top_keywords = [(feature_names[i], topic[i]) for i in top_keywords_idx]
                topic_keywords.append(top_keywords)
            
            return topic_assignments, topic_keywords, lda_model
            
        except Exception as e:
            logger.error(f"Error in LDA analysis: {str(e)}")
            return None, None, None

    def analyze_reviews(self, reviews_df):
        """
        Perform comprehensive thematic analysis on a DataFrame of reviews.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing review data
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with thematic analysis results
        """
        try:
           
            # Extract keywords
            keywords = self.extract_keywords_tfidf(reviews_df['review'])
            
            # Assign themes
            theme_results = reviews_df['review'].apply(self.assign_themes_rulebased)
            reviews_df['primary_theme'] = theme_results.apply(lambda x: x[0])
            reviews_df['theme_scores'] = theme_results.apply(lambda x: x[1])
            
            # Perform LDA analysis
            topic_assignments, topic_keywords, lda_model = self.analyze_with_lda(reviews_df['review'])
            if topic_assignments is not None:
                # Get dominant topic for each review
                reviews_df['topic_id'] = topic_assignments.argmax(axis=1)
                reviews_df['topic_probabilities'] = topic_assignments.tolist()
                
                # Add topic keywords to the DataFrame
                reviews_df['topic_keywords'] = reviews_df['topic_id'].apply(
                    lambda x: [kw for kw, score in topic_keywords[x]]
                )
            
            return reviews_df
        except Exception as e:
            logger.error(f"Error in comprehensive review analysis: {str(e)}")
            return reviews_df