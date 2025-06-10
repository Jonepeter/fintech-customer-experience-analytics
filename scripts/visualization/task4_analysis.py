import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

# Set style for better visualizations
sns.set_palette("husl")


# Function to create word cloud
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=100,
                         contour_width=3,
                         contour_color='steelblue').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Function to plot sentiment trends
def plot_sentiment_trends(df, date, sentiment):
    plt.figure(figsize=(12, 6))
    sentiment_counts = df.groupby([date, sentiment]).size().unstack(fill_value=0)
    sentiment_counts.plot(kind='line', marker='o')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot rating distribution
def plot_rating_distribution(df, x_col):
    plt.figure(figsize=(12, 6))
    # Create a grouped bar plot
    sns.countplot(data=df, x=x_col, hue='rating', palette='viridis')
    plt.title(f'Rating Distribution by {x_col}')
    plt.xlabel(x_col)
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function to compare banks
def compare_banks(bank_df, col2_name):
    plt.figure(figsize=(12, 6))
    bank_sentiment = bank_df.groupby(['app_name', col2_name]).size().unstack(fill_value=0)
    bank_sentiment.plot(kind='bar', stacked=True)
    plt.title(f'Sentiment Distribution by {col2_name}')
    plt.xlabel('Bank')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

def plot_theme_frequency_by_bank(df, bank_col, theme_col, count_col='count', top_n=5):
    """
    Plot a grouped bar chart showing the frequency of the top N themes per bank.
    Args:
        df: DataFrame containing at least [bank_col, theme_col]
        bank_col: str, column name for bank/app name
        theme_col: str, column name for theme/category
        count_col: str, optional, name for count column if pre-aggregated
        top_n: int, number of top themes to show (by overall frequency)
    """
    # Aggregate theme counts per bank
    theme_counts = df.groupby([theme_col, bank_col]).size().reset_index(name='count')
    # Get top N themes overall
    top_themes = (
        theme_counts.groupby(theme_col)['count'].sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    theme_counts = theme_counts[theme_counts[theme_col].isin(top_themes)]
    # Pivot for plotting
    plot_df = theme_counts.pivot(index=theme_col, columns=bank_col, values='count').fillna(0)
    plot_df = plot_df.loc[top_themes]  # preserve order
    # Set up bar positions
    banks = plot_df.columns.tolist()
    themes = plot_df.index.tolist()
    n_banks = len(banks)
    x = np.arange(len(themes))
    width = 0.25
    # Color map for banks (match the example: blue, red, green)
    colors = ['dodgerblue', 'red', 'limegreen']
    if n_banks > 3:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('tab10')
        colors = [cmap(i) for i in range(n_banks)]
    plt.figure(figsize=(12, 7))
    for i, bank in enumerate(banks):
        plt.bar(x + i*width, plot_df[bank], width=width, label=bank, color=colors[i % len(colors)])
    plt.xticks(x + width, themes, rotation=20)
    plt.xlabel('')
    plt.ylabel('Number of Reviews')
    plt.title('Theme Frequency by Bank', fontsize=16, fontweight='bold', color='lightgray')
    plt.legend(title='', loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    # Set dark background to match the example
    plt.gca().set_facecolor('#181818')
    plt.gcf().patch.set_facecolor('#181818')
    plt.title('Theme Frequency by Bank', fontsize=16, fontweight='bold', color='lightgray')
    plt.ylabel('Number of Reviews', color='lightgray')
    plt.xlabel('', color='lightgray')
    plt.xticks(color='lightgray')
    plt.yticks(color='lightgray')
    plt.tight_layout()
    plt.show()

