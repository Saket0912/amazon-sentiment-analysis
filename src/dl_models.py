"""
dl_models.py
============
Deep Learning classifiers built with Keras/TensorFlow:
    1. RNN  (SimpleRNN)
    2. CNN  (Conv1D)

Both models share the same tokenization pipeline.
Labels are expected to be {-1, 0, 1}; internally remapped to {0, 1, 2}
for softmax output.

Usage
-----
    from src.dl_models import prepare_sequences, build_rnn, build_cnn
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# GPU Memory Config (safe for Colab)
# ---------------------------------------------------------------------------
def configure_gpu():
    """Enable dynamic GPU memory growth to prevent OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU configured: {len(gpus)} device(s) found.")
        except RuntimeError as e:
            print(f"[WARN] GPU config error: {e}")
    else:
        print("[INFO] No GPU found — running on CPU.")


# ---------------------------------------------------------------------------
# Label remapping helpers
# ---------------------------------------------------------------------------
def remap_labels(y) -> np.ndarray:
    """
    Remap VADER labels {-1, 0, 1} → {0, 1, 2} for Keras.

    -1 → 0 (Negative)
     0 → 1 (Neutral)
     1 → 2 (Positive)
    """
    mapping = {-1: 0, 0: 1, 1: 2}
    return np.array([mapping[v] for v in y])


# ---------------------------------------------------------------------------
# Tokenization & Padding
# ---------------------------------------------------------------------------
def prepare_sequences(texts, labels, test_size=0.2, random_state=42):
    """
    Tokenize and pad review texts for RNN / CNN input.

    Parameters
    ----------
    texts       : list/Series of raw review strings
    labels      : array-like of numeric VADER labels {-1, 0, 1}
    test_size   : float
    random_state: int

    Returns
    -------
    X_train, X_test : padded integer sequences (numpy arrays)
    y_train, y_test : remapped labels (numpy arrays)
    vocab_size      : int  — vocabulary size + 1
    max_len         : int  — sequence length after padding
    """
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max(len(s) for s in sequences)
    # Cap at 500 tokens to keep memory manageable
    max_len = min(max_len, 500)

    X = pad_sequences(sequences, maxlen=max_len, truncating="post", padding="post")
    y = remap_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[INFO] Sequences ready | vocab={vocab_size} | maxlen={max_len} "
        f"| train={len(X_train)} | test={len(X_test)}"
    )
    return X_train, X_test, y_train, y_test, vocab_size, max_len


# ---------------------------------------------------------------------------
# 1. RNN Model
# ---------------------------------------------------------------------------
def build_rnn(vocab_size: int, max_len: int, embed_dim: int = 64,
              hidden_dim: int = 32, num_classes: int = 3) -> Sequential:
    """
    Build a Simple RNN classifier.

    Architecture
    ------------
    Embedding → SimpleRNN → Dense(softmax)

    Parameters
    ----------
    vocab_size  : int — vocabulary size
    max_len     : int — padded sequence length
    embed_dim   : int — embedding vector dimension
    hidden_dim  : int — RNN hidden units
    num_classes : int — output classes (3 for Neg/Neu/Pos)

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential(name="SimpleRNN_Classifier")
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim,
                        input_length=max_len))
    model.add(SimpleRNN(units=hidden_dim, dropout=0.2))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 2. CNN Model
# ---------------------------------------------------------------------------
def build_cnn(vocab_size: int, max_len: int, embed_dim: int = 64,
              num_filters: int = 64, kernel_size: int = 3,
              num_classes: int = 3) -> Sequential:
    """
    Build a 1D-CNN text classifier.

    Architecture
    ------------
    Embedding → Conv1D → MaxPool → Flatten → Dense(relu) → Dropout → Dense(softmax)

    Parameters
    ----------
    vocab_size  : int
    max_len     : int
    embed_dim   : int — embedding vector dimension
    num_filters : int — number of Conv1D filters
    kernel_size : int — convolution window size
    num_classes : int — output classes

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential(name="CNN_Classifier")
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim,
                        input_length=max_len))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                     activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------
def train_dl_model(model, X_train, y_train, X_test, y_test,
                   epochs: int = 5, batch_size: int = 32):
    """
    Fit a Keras model and evaluate on the test set.

    Returns
    -------
    history : Keras History object (for plotting loss curves)
    metrics : dict with 'loss' and 'accuracy' on test set
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[INFO] {model.name} — Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
    return history, {"loss": loss, "accuracy": accuracy}
