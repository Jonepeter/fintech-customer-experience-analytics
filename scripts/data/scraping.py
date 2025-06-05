from google_play_scraper import app, Sort, reviews_all
import pandas as pd
from datetime import datetime

def scrape_play_store_reviews(app_id, lang='en', country='us', max_reviews=1000):
    """
    Scrape reviews from Google Play Store
    
    Args:
        app_id (str): Package name of the app (e.g., 'com.instagram.android')
        lang (str): Language code (default 'en')
        country (str): Country code (default 'us')
        max_reviews (int): Maximum number of reviews to fetch
        
    Returns:
        pd.DataFrame: DataFrame containing reviews data
    """
    try:
        # Get app info first
        app_info = app(app_id, lang=lang, country=country)
        
        # Scrape reviews
        reviews = reviews_all(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=max_reviews,
            filter_score_with=None  # Get all scores (1-5)
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        
        # Add app info
        df['app_name'] = app_info.get('title', '')
        df['app_version'] = app_info.get('version', '')
        
        # Convert timestamp to readable date
        df['review_date'] = pd.to_datetime(df['at'], unit='ms')
        
        # Select relevant columns
        cols_to_keep = [
            'reviewId', 'content', 'score', 
            'thumbsUpCount', 'review_date', 'app_name', 'app_version'
        ]
        df = df[cols_to_keep]
        
        return df
    
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        return pd.DataFrame()