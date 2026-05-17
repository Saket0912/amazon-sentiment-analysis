"""
data_preprocessing.py
=====================
Handles:
    - Loading the Amazon CSV dataset
    - Applying VADER and TextBlob sentiment scoring
    - Encoding sentiment labels
    - Vectorizing review text for ML models
    - Train/test splitting
"""

import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK resource on first run
nltk.download("vader_lexicon", quiet=True)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Amazon reviews CSV into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to amazon.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.
    """
    df = pd.read_csv(filepath)
    # Drop rows where review_content is missing
    df = df.dropna(subset=["review_content"])
    df["review_content"] = df["review_content"].astype(str)
    print(f"[INFO] Loaded {len(df)} reviews from '{filepath}'")
    return df


# ---------------------------------------------------------------------------
# 2. Apply Sentiment Scores
# ---------------------------------------------------------------------------
def apply_textblob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TextBlob polarity and subjectivity columns to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'review_content' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new columns:
            - TextBlob_Polarity    : float in [-1, 1]
            - TextBlob_Subjectivity: float in [0, 1]
    """
    df["TextBlob_Polarity"] = df["review_content"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["TextBlob_Subjectivity"] = df["review_content"].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )
    print("[INFO] TextBlob scoring complete.")
    return df


def apply_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER compound score and categorical sentiment label.

    Labelling logic (continuous → categorical):
        compound >= 0.05  → 'Positive'
        compound <= -0.05 → 'Negative'
        otherwise         → 'Neutral'

    For ML targets (numeric):
        compound >  0.5  →  1  (Positive)
        compound >= 0.0  →  0  (Neutral)
        compound <  0.0  → -1  (Negative)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns:
            - Vader_Score       : continuous compound score
            - Vader_Label       : numeric label {-1, 0, 1}
            - Sentiment_Category: string label
    """
    sia = SentimentIntensityAnalyzer()

    df["Vader_Score"] = df["review_content"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    # Numeric label for ML
    def _numeric_label(score):
        if score > 0.5:
            return 1
        elif score >= 0.0:
            return 0
        else:
            return -1

    df["Vader_Label"] = df["Vader_Score"].apply(_numeric_label)

    # Human-readable category
    def _category(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment_Category"] = df["Vader_Score"].apply(_category)

    print("[INFO] VADER scoring complete.")
    print(df["Sentiment_Category"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
def vectorize_reviews(
    df: pd.DataFrame,
    max_features: int = 5000,
):
    """
    Convert review text to a Bag-of-Words feature matrix.

    Parameters
    ----------
    df          : pd.DataFrame  — must contain 'review_content'
    max_features: int           — vocabulary size cap

    Returns
    -------
    X : scipy sparse matrix  — BoW feature matrix
    y : pd.Series            — numeric sentiment labels
    vectorizer : CountVectorizer (fitted) — save for inference
    """
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(df["review_content"])
    y = df["Vader_Label"]
    print(f"[INFO] Vectorized: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, vectorizer


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[INFO] Split → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}"
    )
    return X_train, X_test, y_train, y_test
