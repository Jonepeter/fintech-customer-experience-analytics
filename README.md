# fintech-customer-experience-analytics

## Project Overview

This project analyzes customer satisfaction with mobile banking apps for three Ethiopian banks: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank. 
The goal is to scrape user reviews from the Google Play Store, preprocess the data, perform sentiment and thematic analysis, store the data in an Oracle database, and derive actionable insights to improve app performance and customer retention. 

## Task 1: Data Collection and Preprocessing

### Objective

Collect and preprocess user reviews from the Google Play Store for CBE, BOA, and Dashen Bank mobile apps to prepare a clean dataset for analysis.

## Task 2: Sentiment and Thematic Analysis

### Objective

Quantify the sentiment of user reviews (positive, negative, neutral) and identify recurring themes to uncover satisfaction drivers (e.g., intuitive UI) and pain points (e.g., login errors) for the mobile apps of CBE, BOA, and Dashen Bank.

## Task 3: Store Cleaned Data in Oracle

### Objective

Design and implement a relational database in Oracle XE to persistently store the cleaned and processed review data from Tasks 1 and 2, simulating enterprise data engineering workflows common in the banking sector.

### project structure

```
└──📁fintech-customer-experience-analytics
└── 📁.github
    └── 📁workflows
        └── tests.yml
└── 📁.vscode
    └── settings.json
└── 📁data
    └── 📁processed
        └── cleaned_review_data.csv
    └── 📁raw
        └── abyssinia_bank_review.csv
        └── cbe_review.csv
        └── dashen_bank_review.csv
└── 📁notebooks
    └── data_collection_preprocessing.ipynb
    └── README.md
└── 📁scripts
    └── __init__.py
    └── 📁data
        └── preprocess_review.py
        └── scraping.py
    └── README.md
    └── 📁utils
    └── 📁visualization
        └── basic_plots.py
└── 📁src
└── 📁tests
    └── __init__.py
└── .env
└── .gitignore
└── README.md
└── requirements.txt
```
