# Bank Reviews Data Collection and Preprocessing Report

## Project Overview

This project focuses on collecting and preprocessing customer reviews from Google Play Store for three major banks such as Commercial Bank of Ethiopia, Bank of Abyssinia and Dashen bank. The goal is to create a clean, structured dataset for further analysis of customer experiences in the banking sector.

## Data Collection Process

### Data Sources and collection method

- **Google Play Store Reviews**
  - Collected using google-play-scraper
  - Target: 400+ reviews per bank
  - Total target: 1,200+ reviews

## Data Preprocessing

### Data Cleaning Steps

1. **Duplicate Removal**
   - Identified and removed duplicate reviews
   - Remaining unique reviews: [Number]

2. **Missing Data Handling**
   - Missing values percentage: [Percentage]

3. **Date Normalization**
   - Original format: [Format]
   - Normalized format: YYYY-MM-DD

### Final Dataset Structure

- Format: CSV
- Columns:
  1. review: Customer review text
  2. rating: Numerical rating (1-5)
  3. date: Normalized date (YYYY-MM-DD)
  4. bank: Bank name
  5. source: "Google Play Store"

## Technical Implementation

### Tools and Libraries

- Python 3.8+
- google-play-scraper: Web scraping
- pandas: Data manipulation
- numpy: Numerical operations
- datetime: Date processing

