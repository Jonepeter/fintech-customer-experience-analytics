"""
database_setup.py
Script for setting up and populating Oracle database with bank reviews data.
"""

import pandas as pd
import cx_Oracle
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Class for managing Oracle database operations."""
    
    def __init__(self, username: str, password: str, dsn: str):
        """
        Initialize database connection.
        
        Args:
            username (str): Oracle database username
            password (str): Oracle database password
            dsn (str): Oracle database connection string
        """
        try:
            self.connection = cx_Oracle.connect(username, password, dsn)
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to Oracle database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            # Create Banks table
            self.cursor.execute("""
                CREATE TABLE banks (
                    bank_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    bank_name VARCHAR2(100) NOT NULL,
                    bank_code VARCHAR2(20) UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create Reviews table
            self.cursor.execute("""
                CREATE TABLE reviews (
                    review_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    bank_id NUMBER,
                    review_text CLOB,
                    rating NUMBER(2,1),
                    sentiment_label VARCHAR2(20),
                    sentiment_score NUMBER(3,2),
                    primary_theme VARCHAR2(100),
                    theme_scores CLOB,
                    topic_id NUMBER,
                    topic_probabilities CLOB,
                    topic_keywords CLOB,
                    review_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
                )
            """)
            
            # Create indexes
            self.cursor.execute("""
                CREATE INDEX idx_reviews_bank_id ON reviews(bank_id)
            """)
            self.cursor.execute("""
                CREATE INDEX idx_reviews_rating ON reviews(rating)
            """)
            self.cursor.execute("""
                CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label)
            """)
            
            self.connection.commit()
            logger.info("Successfully created database tables")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            self.connection.rollback()
            raise

    def insert_bank(self, bank_name: str, bank_code: str) -> int:
        """
        Insert a new bank into the database.
        
        Args:
            bank_name (str): Name of the bank
            bank_code (str): Unique code for the bank
            
        Returns:
            int: ID of the inserted bank
        """
        try:
            self.cursor.execute("""
                INSERT INTO banks (bank_name, bank_code)
                VALUES (:1, :2)
                RETURNING bank_id INTO :3
            """, (bank_name, bank_code, self.cursor.var(cx_Oracle.NUMBER)))
            
            bank_id = self.cursor.getvalue(2)
            self.connection.commit()
            return bank_id
            
        except Exception as e:
            logger.error(f"Error inserting bank: {str(e)}")
            self.connection.rollback()
            raise

    def insert_review(self, review_data: Dict[str, Any], bank_id: int):
        """
        Insert a review into the database.
        
        Args:
            review_data (dict): Dictionary containing review information
            bank_id (int): ID of the associated bank
        """
        try:
            # Convert theme_scores and topic_probabilities to JSON strings
            theme_scores = str(review_data.get('theme_scores', {}))
            topic_probabilities = str(review_data.get('topic_probabilities', []))
            topic_keywords = str(review_data.get('topic_keywords', []))
            
            self.cursor.execute("""
                INSERT INTO reviews (
                    bank_id, review_text, rating, sentiment_label, sentiment_score,
                    primary_theme, theme_scores, topic_id, topic_probabilities,
                    topic_keywords, review_date
                ) VALUES (
                    :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11
                )
            """, (
                bank_id,
                review_data.get('review_text', ''),
                review_data.get('rating', 0),
                review_data.get('sentiment_label', ''),
                review_data.get('sentiment_score', 0),
                review_data.get('primary_theme', ''),
                theme_scores,
                review_data.get('topic_id', None),
                topic_probabilities,
                topic_keywords,
                review_data.get('review_date', datetime.now())
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error inserting review: {str(e)}")
            self.connection.rollback()
            raise

    def bulk_insert_reviews(self, reviews_df: pd.DataFrame):
        """
        Bulk insert reviews from a DataFrame.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing review data
        """
        try:
            # Get unique banks and insert them
            bank_ids = {}
            for bank_name in reviews_df['bank_name'].unique():
                bank_code = bank_name.lower().replace(' ', '_')
                bank_id = self.insert_bank(bank_name, bank_code)
                bank_ids[bank_name] = bank_id
            
            # Insert reviews
            for _, row in reviews_df.iterrows():
                review_data = {
                    'review_text': row['review_text'],
                    'rating': row['rating'],
                    'sentiment_label': row['sentiment_label'],
                    'sentiment_score': row['sentiment_score'],
                    'primary_theme': row['primary_theme'],
                    'theme_scores': row['theme_scores'],
                    'topic_id': row.get('topic_id'),
                    'topic_probabilities': row.get('topic_probabilities'),
                    'topic_keywords': row.get('topic_keywords'),
                    'review_date': row.get('review_date', datetime.now())
                }
                self.insert_review(review_data, bank_ids[row['bank_name']])
            
            logger.info(f"Successfully inserted {len(reviews_df)} reviews")
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {str(e)}")
            raise

    def export_schema(self, output_path: str):
        """
        Export database schema to SQL file.
        
        Args:
            output_path (str): Path to save the SQL file
        """
        try:
            # Get table creation scripts
            self.cursor.execute("""
                SELECT DBMS_METADATA.GET_DDL('TABLE', table_name)
                FROM user_tables
                WHERE table_name IN ('BANKS', 'REVIEWS')
            """)
            
            schema_scripts = self.cursor.fetchall()
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write("-- Oracle Database Schema for Bank Reviews\n\n")
                for script in schema_scripts:
                    f.write(script[0].read() + "\n\n")
            
            logger.info(f"Schema exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting schema: {str(e)}")
            raise

    def close(self):
        """Close database connection."""
        try:
            self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")

