# Fintech Customer Experience Analytics

## Project Overview

This project focuses on analyzing customer reviews from mobile banking applications of three major Ethiopian banks: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank. The analysis combines web scraping, natural language processing (NLP), and data engineering to derive actionable insights for improving mobile banking experiences.


## Required Tools and Technologies

### Development Environment
- Python 3.8+
- Git for version control
- VS Code or preferred IDE
- Jupyter Notebook/Lab
- Oracle Database XE (Express Edition)

### Core Technologies
1. **Data Collection & Processing**
   - Google Play Store API
   - Web Scraping Tools
   - Oracle Database

2. **Data Analysis & NLP**
   - Natural Language Processing (NLP)
   - Sentiment Analysis
   - Text Mining
   - Data Visualization

### Python Dependencies

#### Core Data Analysis
- numpy==1.26.0
- pandas==2.2.0
- matplotlib==3.8.0
- seaborn==0.13.0
- scipy==1.12.0

#### Sentiment Analysis
- nltk==3.8.1
- textblob==0.17.1
- vaderSentiment==3.4.1
- spacy==3.7.0
- transformers==4.40.0

#### Utilities
- jupyter==1.0.0
- tqdm==4.66.0
- wordcloud==1.9.3

## Project Tasks

### 1. Data Collection and Preprocessing
- Scrape user reviews from Google Play Store for CBE, BOA, and Dashen Bank mobile apps
- Clean and preprocess the collected data
- Implement data validation and quality checks
- Expected output: Clean dataset with 1,200+ reviews

### 2. Sentiment and Thematic Analysis
- Perform sentiment analysis using multiple approaches:
  - VADER Sentiment
  - TextBlob
  - Custom NLP models
- Identify recurring themes and patterns
- Quantify satisfaction drivers and pain points
- Generate word clouds and topic models

### 3. Database Implementation
- Design and implement Oracle database schema
- Store processed review data
- Implement data access patterns
- Ensure data integrity and security

### 4. Insights and Visualization
- Create stakeholder-friendly visualizations:
  - Sentiment distribution charts
  - Theme frequency plots
  - Time series analysis
  - Comparative analysis between banks
- Generate actionable insights
- Prepare comprehensive analysis report

## Setup and Installation

### Prerequisites
1. Python 3.8 or higher
2. Oracle Database XE installed
3. Git installed
4. VS Code or preferred IDE

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/jonepeter/fintech-customer-experience-analytics.git
   cd fintech-customer-experience-analytics
   ```

2. Create and activate virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. Set up Oracle database connection:
   - Install Oracle Instant Client
   - Configure environment variables in `.env`:
     ```
     ORACLE_USER=your_username
     ORACLE_PASSWORD=your_password
     ORACLE_HOST=localhost
     ORACLE_PORT=1521
     ORACLE_SERVICE=XE
     ```

## Usage

### Data Collection
```bash
python scripts/data/scraping.py
```

### Data Preprocessing
```bash
python scripts/data/preprocess_review.py
```

### Analysis and Visualization
```bash
python scripts/visualization/basic_plots.py
```

### Running Tests
```bash
python -m pytest tests/
```

## Development Workflow

1. Create a new branch for each feature
2. Write unit tests for new functionality
3. Implement the feature
4. Run tests and ensure all pass
5. Create a pull request
6. Code review and merge

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Contact

Petros Abebe
- Email: petrosabebe3@gmail.com
- LinkedIn: [Petros Abebe](linkedin.com/in/petros-abebe-76668734a)
- GitHub: [Petros Abebe](https://github.com/jonepeter)
---


## Acknowledgments

- Google Play Store API
- NLTK and other open-source libraries
- Oracle Database Community

## Project Structure

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

