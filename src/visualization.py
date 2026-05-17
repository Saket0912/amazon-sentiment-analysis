"""
visualization.py
================
All plotting utilities for the Amazon Sentiment Analysis project.

Functions
---------
- plot_textblob_vs_vader     : Scatter plot of polarity scores
- plot_sentiment_distribution: Bar chart of sentiment categories
- plot_model_comparison      : Grouped bar chart of ML metrics
- plot_learning_curve        : Training vs validation loss (RF)
- plot_dl_history            : Loss/accuracy curves for DL models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Styling defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]


# ---------------------------------------------------------------------------
# 1. TextBlob vs VADER scatter
# ---------------------------------------------------------------------------
def plot_textblob_vs_vader(df, save_path: str = None):
    """
    Scatter plot comparing TextBlob polarity against VADER compound score.

    Parameters
    ----------
    df        : pd.DataFrame — must have 'TextBlob_Polarity', 'Vader_Score'
    save_path : str | None  — if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        x="TextBlob_Polarity", y="Vader_Score", data=df,
        hue="Sentiment_Category",
        palette={"Positive": "#4CAF50", "Neutral": "#FF9800", "Negative": "#F44336"},
        alpha=0.6, ax=ax,
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("TextBlob Polarity vs VADER Compound Score")
    ax.set_xlabel("TextBlob Polarity")
    ax.set_ylabel("VADER Compound Score")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Sentiment category distribution
# ---------------------------------------------------------------------------
def plot_sentiment_distribution(df, save_path: str = None):
    """
    Bar chart showing count of Positive / Neutral / Negative reviews.

    Parameters
    ----------
    df        : pd.DataFrame — must have 'Sentiment_Category'
    save_path : str | None
    """
    counts = df["Sentiment_Category"].value_counts()
    colors = {"Positive": "#4CAF50", "Neutral": "#FF9800", "Negative": "#F44336"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[colors.get(c, "#2196F3") for c in counts.index],
        edgecolor="white",
        width=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Sentiment Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Algorithm performance comparison
# ---------------------------------------------------------------------------
def plot_model_comparison(performance: dict = None, save_path: str = None):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score
    across multiple algorithms.

    Parameters
    ----------
    performance : dict
        Keys   = algorithm names
        Values = list of 4 floats [Accuracy, Precision, Recall, F1]
        Defaults to the reported results from this project.
    save_path : str | None
    """
    if performance is None:
        performance = {
            "Naive Bayes":    [0.91, 0.92, 1.00, 0.96],
            "SVM":            [0.91, 0.91, 1.00, 0.95],
            "Random Forest":  [0.92, 0.92, 1.00, 0.96],
            "CNN":            [0.91, 0.92, 1.00, 0.96],
            "RNN":            [0.91, 0.92, 1.00, 0.96],
        }

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_metrics = len(metrics)
    n_models = len(performance)
    bar_width = 0.15
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, scores) in enumerate(performance.items()):
        offsets = x + bar_width * (i - n_models / 2)
        bars = ax.bar(offsets, scores, width=bar_width,
                      label=model_name, color=PALETTE[i % len(PALETTE)],
                      edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.85, 1.05)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Algorithm Performance Comparison")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Learning curve (Random Forest)
# ---------------------------------------------------------------------------
def plot_learning_curve(X, y, save_path: str = None):
    """
    Plot training vs validation loss using sklearn's learning_curve.

    Parameters
    ----------
    X         : feature matrix
    y         : label array
    save_path : str | None
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    train_mean = -np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = -np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, label="Training Loss",   color="#2196F3")
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color="#2196F3")
    ax.plot(train_sizes, val_mean, label="Validation Loss", color="#F44336")
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color="#F44336")
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Random Forest Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Deep Learning training history
# ---------------------------------------------------------------------------
def plot_dl_history(history, model_name: str = "Model", save_path: str = None):
    """
    Plot training and validation accuracy / loss over epochs.

    Parameters
    ----------
    history    : Keras History object
    model_name : str — used in plot title
    save_path  : str | None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train", color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val",   color="#F44336")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train", color="#2196F3")
    axes[1].plot(history.history["val_loss"], label="Val",   color="#F44336")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Saved → {save_path}")
    plt.show()
"""
visualization.py
================
All plotting utilities for the Amazon Sentiment Analysis project.
Plots are saved to the 'outputs/' folder automatically.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ← non-interactive backend, fixes FigureCanvasAgg warning
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

# ---------------------------------------------------------------------------
# Create output directory automatically
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Styling defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]


# ---------------------------------------------------------------------------
# 1. TextBlob vs VADER scatter
# ---------------------------------------------------------------------------
def plot_textblob_vs_vader(df, save_path: str = None):
    """
    Scatter plot comparing TextBlob polarity against VADER compound score.

    Parameters
    ----------
    df        : pd.DataFrame — must have 'TextBlob_Polarity',
                'Vader_Score', 'Sentiment_Category'
    save_path : str | None  — overrides default output path
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "01_textblob_vs_vader.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x="TextBlob_Polarity",
        y="Vader_Score",
        data=df,
        hue="Sentiment_Category",
        palette={
            "Positive": "#4CAF50",
            "Neutral":  "#FF9800",
            "Negative": "#F44336",
        },
        alpha=0.6,
        ax=ax,
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("TextBlob Polarity vs VADER Compound Score")
    ax.set_xlabel("TextBlob Polarity")
    ax.set_ylabel("VADER Compound Score")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ---------------------------------------------------------------------------
# 2. Sentiment category distribution
# ---------------------------------------------------------------------------
def plot_sentiment_distribution(df, save_path: str = None):
    """
    Bar chart showing count of Positive / Neutral / Negative reviews.

    Parameters
    ----------
    df        : pd.DataFrame — must have 'Sentiment_Category'
    save_path : str | None
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "02_sentiment_distribution.png")

    counts = df["Sentiment_Category"].value_counts()
    color_map = {
        "Positive": "#4CAF50",
        "Neutral":  "#FF9800",
        "Negative": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[color_map.get(c, "#2196F3") for c in counts.index],
        edgecolor="white",
        width=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Sentiment Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ---------------------------------------------------------------------------
# 3. Algorithm performance comparison
# ---------------------------------------------------------------------------
def plot_model_comparison(performance: dict = None, save_path: str = None):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score
    across multiple algorithms.

    Parameters
    ----------
    performance : dict
        Keys   = algorithm names
        Values = list [Accuracy, Precision, Recall, F1]
    save_path : str | None
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "03_model_comparison.png")

    if performance is None:
        performance = {
            "Naive Bayes":   [0.91, 0.92, 1.00, 0.96],
            "SVM":           [0.91, 0.91, 1.00, 0.95],
            "Random Forest": [0.92, 0.92, 1.00, 0.96],
            "CNN":           [0.91, 0.92, 1.00, 0.96],
            "RNN":           [0.91, 0.92, 1.00, 0.96],
        }

    metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_metrics = len(metrics)
    n_models  = len(performance)
    bar_width = 0.15
    x         = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (model_name, scores) in enumerate(performance.items()):
        offsets = x + bar_width * (i - n_models / 2 + 0.5)
        bars = ax.bar(
            offsets,
            scores,
            width=bar_width,
            label=model_name,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,              # padding above bar
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)

    # ── Y-axis: start from 0.0, end at 1.10 to give room for labels ──
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks(np.arange(0.0, 1.10, 0.10))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.1f")
    )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Algorithm Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")

# ---------------------------------------------------------------------------
# 4. Learning curve (Random Forest)
# ---------------------------------------------------------------------------
def plot_learning_curve(X, y, save_path: str = None):
    """
    Plot training vs validation loss using sklearn's learning_curve.

    Parameters
    ----------
    X         : feature matrix
    y         : label array
    save_path : str | None
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "04_learning_curve.png")

    print("  [INFO] Computing learning curve (this may take ~30 seconds)...")

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    train_sizes, train_scores, val_scores = learning_curve(
        clf,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    train_mean = -np.mean(train_scores, axis=1)
    train_std  =  np.std(train_scores,  axis=1)
    val_mean   = -np.mean(val_scores,   axis=1)
    val_std    =  np.std(val_scores,    axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, label="Training Loss",   color="#2196F3", linewidth=2)
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#2196F3",
    )
    ax.plot(train_sizes, val_mean, label="Validation Loss", color="#F44336", linewidth=2)
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#F44336",
    )
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Random Forest — Learning Curve")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ---------------------------------------------------------------------------
# 5. Deep Learning training history
# ---------------------------------------------------------------------------
def plot_dl_history(history, model_name: str = "Model", save_path: str = None):
    """
    Plot training and validation accuracy / loss over epochs.

    Parameters
    ----------
    history    : Keras History object
    model_name : str — used in plot title and filename
    save_path  : str | None
    """
    if save_path is None:
        fname     = model_name.lower().replace(" ", "_")
        save_path = os.path.join(OUTPUT_DIR, f"05_{fname}_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],
                 label="Train", color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_accuracy"],
                 label="Val",   color="#F44336", linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],
                 label="Train", color="#2196F3", linewidth=2)
    axes[1].plot(history.history["val_loss"],
                 label="Val",   color="#F44336", linewidth=2)
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")