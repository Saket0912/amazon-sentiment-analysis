# 🛒 Amazon Product Review — Sentiment Analysis

> A production-ready NLP pipeline that classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** using dual sentiment scoring, classical machine learning, and deep learning approaches.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.9-green)](https://nltk.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Results](#-live-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Visualizations](#-visualizations)
- [Quick Start](#-quick-start)
- [CLI Usage](#-cli-usage)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔍 Overview

This project performs **end-to-end sentiment analysis** on Amazon product reviews from the Electronics category. It combines:

| Component | Detail |
|-----------|--------|
| **Dual Scoring** | VADER (rule-based) + TextBlob (lexicon-based) |
| **Statistical Test** | One-Way ANOVA to validate sentiment group differences |
| **Feature Engineering** | Bag-of-Words with CountVectorizer (top 5,000 tokens) |
| **Classical ML** | Naive Bayes · SVM · Random Forest |
| **Deep Learning** | SimpleRNN · Conv1D (Keras/TensorFlow) |
| **Tuning** | GridSearchCV with Stratified K-Fold Cross Validation |
| **Visualizations** | Scatter · Bar · Grouped comparison · Learning curve |

---

## 📊 Live Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Naive Bayes | 0.8874 | 0.8363 | 0.8874 | 0.8604 |
| SVM | 0.8976 | 0.8599 | 0.8976 | 0.8530 |
| **Random Forest** | **0.9147** | **0.9179** | **0.9147** | **0.8914** |
| CNN ⁺ | 0.9100 | 0.9200 | 1.0000 | 0.9600 |
| RNN ⁺ | 0.9100 | 0.9200 | 1.0000 | 0.9600 |

> ⁺ CNN/RNN metrics sourced from original Google Colab training.<br>
> Classical ML metrics are from live local runs on the same dataset.

<br>

### ANOVA Statistical Test

| Metric | Value |
|--------|-------|
| F-value | 2461.53 |
| P-value | 0.000000 |
| Result | ✅ Statistically significant difference between sentiment groups (p < 0.05) |

<br>

### Sentiment Distribution — 1,465 Reviews

| Sentiment | Count | Share |
|-----------|------:|------:|
| Positive | 1,380 | 94.2% |
| Negative | 74 | 5.1% |
| Neutral | 11 | 0.7% |

---

## 📁 Project Structure

```
amazon-sentiment-analysis/
│
├── 📄 main.py                          # Pipeline entry point (CLI)
├── 📄 requirements.txt                 # All Python dependencies
├── 📄 README.md                        # Project documentation
├── 📄 .gitignore                       # Git exclusions
│
├── 📁 src/                             # Source modules
│   ├── __init__.py
│   ├── data_preprocessing.py           # Load · Score · Vectorize · Split
│   ├── ml_models.py                    # Naive Bayes · SVM · Random Forest
│   ├── dl_models.py                    # RNN · CNN (Keras/TensorFlow)
│   └── visualization.py                # All plotting functions → outputs/
│
├── 📁 data/
│   └── amazon.csv                      # Raw dataset (1,465 reviews)
│
└── 📁 outputs/                         # Auto-generated on every run
    ├── 01_textblob_vs_vader.png       # Sentiment tool correlation
    ├── 02_sentiment_distribution.png
    ├── 03_model_comparison.png         # All 5 models side-by-side
    └── 04_learning_curve.png           # Training vs validation loss
```

---

## 📦 Dataset

**Source:** Amazon Product Reviews — Electronics Category  
**Size:** 1,465 reviews · 16 columns

| Column | Type | Description |
|--------|------|-------------|
| `product_name` | str | Full product title |
| `category` | str | Category path (pipe-separated) |
| `discounted_price` | str | Sale price (₹) |
| `actual_price` | str | Original price (₹) |
| `discount_percentage` | str | Discount percentage |
| `rating` | float | Star rating (1.0 – 5.0) |
| `rating_count` | int | Total number of ratings |
| `review_content` | str | ⭐ **Primary feature** — full review text |
| `review_title` | str | Short review headline |

---

## 🧠 Methodology

### 1. Dual Sentiment Scoring

Two complementary NLP tools score each review independently:

```
Review Text
│
├──▶ VADER ──▶ compound score [-1.0, +1.0]
│   ├─ Positive ≥ 0.05
│   ├─ Negative ≤ -0.05
│   └─ Neutral (between)
│
└──▶ TextBlob ──▶ polarity [-1.0, +1.0]
    └─ subjectivity [0.0, 1.0]
```

| Tool | Strength |
|------|----------|
| **VADER** | Informal text, slang, punctuation, social media style |
| **TextBlob** | Subjectivity scoring, grammatically structured reviews |

Comparing both tools **validates label consistency** and provides richer features.

<br>

### 2. Statistical Validation — One-Way ANOVA

Before modelling, ANOVA confirms the three groups have **statistically different** VADER score distributions:

- H₀ : All group means are equal
- H₁ : At least one group mean differs

```
F = 2461.53 | p = 0.000000 → Reject H₀ ✅
```

This justifies using VADER compound score as the ground-truth label.

<br>

### 3. Feature Engineering

```python
CountVectorizer(max_features=5000, stop_words="english")
```

| Step | Detail |
|------|--------|
| Representation | Bag-of-Words (token counts) |
| Vocabulary | Top 5,000 terms |
| Stop words | English stop words removed |
| Split | Stratified 80% train / 20% test |

<br>

### 4. Classical ML Pipeline

```
Raw Text
    │
    ▼
CountVectorizer (BoW, 5000 features)
    │
    ▼
┌──────────────────────────────────┐
│  GridSearchCV — 5-fold CV        │
│  Scoring: weighted F1            │
├──────────────────────────────────┤
│  Naive Bayes  │ alpha, fit_prior │
│  SVM          │ C, kernel, gamma │
│  RandomForest │ n_estimators,    │
│               │ max_depth, etc.  │
└──────────────────────────────────┘
    │
    ▼
Best Estimator → Predictions → Metrics
```

<br>

### 5. Deep Learning Pipeline

```
Raw Text
    │
    ▼
Keras Tokenizer → Integer Sequences
    │
    ▼
Pad Sequences (maxlen = 500)
    │
    ▼
┌───────────────────────────────────────────────┐
│               Embedding Layer (dim=64)         │
└───────────────────────────────────────────────┘
    │                           │
    ▼                           ▼
┌──────────────┐       ┌──────────────────┐
│  SimpleRNN   │       │     Conv1D       │
│  units = 32  │       │  filters = 64   │
│  dropout=0.2 │       │  kernel_size = 3 │
└──────────────┘       └──────────────────┘
    │                           │
    ▼                           ▼
Dense(3, softmax)       MaxPooling1D
[Neg / Neu / Pos]           │
                        Flatten
                            │
                        Dense(64, relu)
                            │
                        Dropout(0.5)
                            │
                        Dense(3, softmax)
                       [Neg / Neu / Pos]
```

| Setting | Value |
|---------|-------|
| Label mapping | {-1, 0, 1} → {0, 1, 2} |
| Loss function | sparse_categorical_crossentropy |
| Optimizer | Adam |
| Epochs | 5 |

---

## 📈 Visualizations

All charts are auto-saved to `outputs/` on every pipeline run.

1. **TextBlob vs VADER Score Correlation** – Validates that both sentiment tools agree on review polarity direction
2. **Sentiment Category Distribution** – Reveals the significant class imbalance — 94% of reviews are Positive
3. **Algorithm Performance Comparison** – All 5 models benchmarked across Accuracy, Precision, Recall and F1-Score
4. **Random Forest Learning Curve** – Shows training vs validation loss as dataset size increases

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9 – 3.11 recommended
- Python 3.14 supported for classical ML only (TensorFlow not yet available)

### 1. Clone the repository
```bash
git clone https://github.com/Saket0912/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 5. Run the pipeline
```bash
# ⚡ Fastest — classical ML only, no tuning (~2 minutes)
python main.py --skip-dl --skip-tuning

# 🔧 Classical ML with GridSearchCV tuning (~10 minutes)
python main.py --skip-dl

# 🚀 Full pipeline with DL — Python 3.11 + TensorFlow required (~30 minutes)
python main.py
```

---

## 🖥️ CLI Usage

```
usage: main.py [-h] [--data DATA] [--skip-dl] [--skip-tuning]

Amazon Product Review Sentiment Analysis Pipeline

options:
  -h, --help      show this help message and exit

  --data DATA     Path to the input CSV file
                  default: data/amazon.csv

  --skip-dl       Skip RNN and CNN deep learning models
                  Use this on Python 3.14 (TensorFlow not yet supported)

  --skip-tuning   Skip GridSearchCV hyperparameter tuning
                  Useful for quick exploratory runs
```

| Command | What it does | Time |
|---------|--------------|------|
| `python main.py --skip-dl --skip-tuning` | Baseline ML only | ~2 min |
| `python main.py --skip-dl` | ML + GridSearchCV tuning | ~10 min |
| `python main.py` | Full pipeline with RNN + CNN | ~30 min |
| `python main.py --data path/to/file.csv` | Custom dataset | varies |

---

## 🛠️ Tech Stack

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| Data | pandas | 3.0+ | DataFrame operations |
| Data | numpy | 2.0+ | Numerical computing |
| Statistics | scipy | 1.17+ | One-Way ANOVA testing |
| NLP | nltk + vaderSentiment | 3.9+ | Rule-based sentiment scoring |
| NLP | textblob | 0.20+ | Lexicon-based sentiment + subjectivity |
| ML | scikit-learn | 1.8+ | Classifiers · Vectorizer · GridSearchCV |
| Deep Learning | tensorflow / keras | 2.13+ | SimpleRNN and Conv1D models |
| Visualization | matplotlib | 3.10+ | All chart generation |
| Visualization | seaborn | 0.13+ | Enhanced scatter plot styling |

---

## ⚠️ Limitations & Future Work

### Known Limitations

| # | Limitation | Impact |
|---|------------|--------|
| 1 | Class imbalance — 94% Positive reviews | Model biased toward majority class |
| 2 | Bag-of-Words features | Loses word order and semantic context |
| 3 | Python 3.14 + TensorFlow | DL models can only run on Colab / Python 3.11 |
| 4 | Fixed DL metrics | CNN/RNN values from Colab, not live local run |
| 5 | Small dataset | Only 1,465 reviews — may not generalise broadly |

### Planned Improvements

- SMOTE / class weighting — correct the 94% Positive class imbalance
- TF-IDF vectorisation — better term weighting than raw counts
- BERT / DistilBERT — transformer-based contextual embeddings
- LSTM / BiLSTM — replace SimpleRNN for richer sequence modelling
- Keras Tuner — automated hyperparameter search for DL models
- FastAPI endpoint — REST API for real-time review scoring
- Streamlit dashboard — interactive visualisation and live prediction

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <strong>Saket Verma</strong><br>
      <a href="https://linkedin.com/in/saket-verma-3b337a1a7">
        <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white" />
      </a>
      &nbsp;
      <a href="https://github.com/Saket0912">
        <img src="https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white" />
      </a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  ⭐ Star this repo if you found it useful

  <sub>Built with ❤️ for learning and portfolio purposes</sub>
</div>
