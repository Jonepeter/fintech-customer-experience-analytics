"""
sentiment_analysis.py
Script for performing sentiment analysis on bank reviews using DistilBERT and VADER.
"""

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for performing sentiment analysis using different models."""
    
    def __init__(self):
        """Initialize sentiment analysis models."""
        try:
            # Initialize DistilBERT
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english")
            
            # Initialize VADER
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info("Sentiment analysis models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {str(e)}")
            raise

    def analyze_with_distilbert(self, text):
        """
        Analyze sentiment using DistilBERT model.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing sentiment label and scores
        """
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = softmax(logits.numpy()[0])
            
            return {
                'sentiment_label': self.model.config.id2label[logits.argmax().item()],
                'positive_score': probs[1],
                'negative_score': probs[0]
            }
        except Exception as e:
            logger.error(f"Error in DistilBERT analysis: {str(e)}")
            return {
                'sentiment_label': 'ERROR',
                'positive_score': 0,
                'negative_score': 0
            }

    def analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER model.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing sentiment label and score
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'sentiment_label': (
                    'POSITIVE' if scores['compound'] > 0.05 
                    else 'NEGATIVE' if scores['compound'] < -0.05 
                    else 'NEUTRAL'
                ),
                'sentiment_score': scores['compound']
            }
        except Exception as e:
            logger.error(f"Error in VADER analysis: {str(e)}")
            return {
                'sentiment_label': 'ERROR',
                'sentiment_score': 0
            }

def aggregate_sentiment(df):
    """
    Aggregate sentiment scores by bank and rating.
    
    Args:
        df (pd.DataFrame): DataFrame containing review data
        
    Returns:
        pd.DataFrame: Aggregated sentiment statistics
    """
    try:
        return df.groupby(['bank_name', 'rating']).agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).reset_index()
    except Exception as e:
        logger.error(f"Error aggregating sentiment: {str(e)}")
        return pd.DataFrame()