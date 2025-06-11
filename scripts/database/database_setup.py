"""
database_setup.py
Script for setting up and populating PostgreSQL database with bank reviews data.
"""

import pandas as pd
import psycopg2
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Class for managing PostgreSQL database operations."""
    
    def __init__(self, username: str, password: str, host: str, port: str, database: str):
        """
        Initialize database connection.
        
        Args:
            username (str): PostgreSQL database username
            password (str): PostgreSQL database password
            host (str): PostgreSQL database host
            port (str): PostgreSQL database port
            database (str): PostgreSQL database name
        """
        try:
            self.connection = psycopg2.connect(
                user=username,
                password=password,
                host=host,
                port=port,
                database=database
            )
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            # Create Banks table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS banks (
                    bank_id SERIAL PRIMARY KEY,
                    bank_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create Reviews table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id SERIAL PRIMARY KEY,
                    bank_id INTEGER,
                    review_text TEXT,
                    rating NUMERIC(2,1),
                    sentiment_label VARCHAR(20),
                    sentiment_score NUMERIC(3,2),
                    primary_theme VARCHAR(100),
                    theme_scores TEXT,
                    topic_id INTEGER,
                    topic_probabilities TEXT,
                    topic_keywords TEXT,
                    review_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
                )
            """)
            
            # Create indexes
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id)
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating)
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label)
            """)
            
            self.connection.commit()
            logger.info("Successfully created database tables")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            self.connection.rollback()
            raise

    def insert_bank(self, bank_name: str) -> int:
        """
        Insert a new bank into the database.
        
        Args:
            bank_name (str): Name of the bank
            
        Returns:
            int: ID of the inserted bank
        """
        try:
            self.cursor.execute("""
                INSERT INTO banks (bank_name)
                VALUES (%s)
                RETURNING bank_id
            """, (bank_name,))
            
            bank_id = self.cursor.fetchone()[0]
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
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
            # Prepare the data for bulk insert
            review_data = []
            for _, row in reviews_df.iterrows():
                # Get bank_id for the current review
                bank_id = self.get_bank_id(row['a_name'])
                
                # Convert dictionary values to strings for JSON-like fields
                theme_scores = str(row.get('theme_scores', {}))
                topic_probabilities = str(row.get('topic_probabilities', []))
                topic_keywords = str(row.get('topic_keywords', []))
                
                # Handle review_date
                review_date = row.get('review_date')
                if isinstance(review_date, str):
                    review_date = datetime.datetime.strptime(review_date, '%Y-%m-%d %H:%M:%S')
                elif review_date is None:
                    review_date = datetime.datetime.now()
                
                review_data.append((
                    bank_id,
                    str(row['review']),
                    float(row['rating']),
                    str(row['sentiment_distilbert_label']),
                    float(row['sentiment_distilbert_score']),
                    str(row['primary_theme']),
                    theme_scores,
                    row.get('topic_id'),
                    topic_probabilities,
                    topic_keywords,
                    review_date
                ))
            
            # Bulk insert using executemany
            self.cursor.executemany("""
                INSERT INTO reviews (
                    bank_id, review_text, rating, sentiment_label, sentiment_score,
                    primary_theme, theme_scores, topic_id, topic_probabilities,
                    topic_keywords, review_date
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, review_data)
            
            self.connection.commit()
            logger.info(f"Successfully inserted {len(review_data)} reviews")
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {str(e)}")
            self.connection.rollback()
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
                SELECT table_name, column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name IN ('banks', 'reviews')
                ORDER BY table_name, ordinal_position
            """)
            
            schema_info = self.cursor.fetchall()
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write("-- PostgreSQL Database Schema for Bank Reviews\n\n")
                current_table = None
                for row in schema_info:
                    table_name, column_name, data_type, max_length = row
                    if current_table != table_name:
                        f.write(f"\nCREATE TABLE IF NOT EXISTS {table_name} (\n")
                        current_table = table_name
                    f.write(f"    {column_name} {data_type}")
                    if max_length:
                        f.write(f"({max_length})")
                    f.write(",\n")
                f.write(");\n")
            
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
    
    def get_bank_id(self, bank_name: str) -> int:
        """
        Get bank_id for a given bank name.
        
        Args:
            bank_name (str): Name of the bank
            
        Returns:
            int: ID of the bank
        """
        try:
            self.cursor.execute("""
                SELECT bank_id FROM banks WHERE bank_name = %s
            """, (bank_name,))
            
            result = self.cursor.fetchone()
            if result:
                return result[0]
            else:
                raise ValueError(f"Bank {bank_name} not found in database")
                
        except Exception as e:
            logger.error(f"Error getting bank_id: {str(e)}")
            raise