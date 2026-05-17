"""
ml_models.py
============
Classical Machine Learning classifiers:
    1. Naive Bayes  (MultinomialNB)   + GridSearchCV
    2. SVM          (SVC)             + GridSearchCV
    3. Random Forest                  + GridSearchCV

Each public function returns a dict with keys:
    'model'   : fitted estimator
    'y_pred'  : predictions on X_test
    'report'  : classification_report string
    'accuracy': float
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _evaluate(model, X_test, y_test, label: str) -> dict:
    """Run predictions and print a classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Classification Report:\n{report}")
    return {
        "model": model,
        "y_pred": y_pred,
        "report": report,
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# 1. Naive Bayes
# ---------------------------------------------------------------------------
def train_naive_bayes(X_train, y_train, X_test, y_test) -> dict:
    """
    Train a baseline MultinomialNB classifier.

    Returns
    -------
    dict with model, y_pred, report, accuracy
    """
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return _evaluate(nb, X_test, y_test, "Naive Bayes (Baseline)")


def tune_naive_bayes(X_train, y_train, X_test, y_test) -> dict:
    """
    Grid-search tuned Naive Bayes with 5-fold stratified CV.

    Hyperparameter grid:
        alpha      : [0.1, 1.0, 10.0]
        fit_prior  : [True, False]

    Returns
    -------
    dict with best estimator, predictions, report, best_params
    """
    param_grid = {"alpha": [0.1, 1.0, 10.0], "fit_prior": [True, False]}

    scoring = {
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "Recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "F1": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    gs = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring=scoring,
        refit="F1",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    result = _evaluate(gs.best_estimator_, X_test, y_test, "Naive Bayes (Tuned)")
    result["best_params"] = gs.best_params_
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")
    return result


# ---------------------------------------------------------------------------
# 2. Support Vector Machine
# ---------------------------------------------------------------------------
def train_svm(X_train, y_train, X_test, y_test) -> dict:
    """
    Train a baseline SVC with default hyperparameters.

    Returns
    -------
    dict with model, y_pred, report, accuracy
    """
    svc = SVC()
    svc.fit(X_train, y_train)
    return _evaluate(svc, X_test, y_test, "SVM (Baseline)")


def tune_svm(X_train, y_train, X_test, y_test) -> dict:
    """
    Grid-search tuned SVM.

    Hyperparameter grid:
        C            : [0.1, 1.0, 10.0]
        kernel       : ['linear', 'rbf']
        gamma        : ['scale', 'auto']
        class_weight : [None, 'balanced']

    Note: 'poly' kernel and degree variants removed for speed.
          Re-add to param_grid if compute allows.

    Returns
    -------
    dict with best estimator, predictions, report, best_params
    """
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"],
    }

    gs = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    result = _evaluate(gs.best_estimator_, X_test, y_test, "SVM (Tuned)")
    result["best_params"] = gs.best_params_
    print(f"  Best params : {gs.best_params_}")
    return result


# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """
    Train a baseline Random Forest classifier.

    Returns
    -------
    dict with model, y_pred, report, accuracy
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return _evaluate(rf, X_test, y_test, "Random Forest (Baseline)")


def tune_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """
    Grid-search tuned Random Forest.

    Hyperparameter grid:
        n_estimators     : [100, 200, 300]
        max_depth        : [10, 20, 30]
        min_samples_split: [2, 5, 10]
        min_samples_leaf : [1, 2, 4]
        bootstrap        : [True, False]

    Returns
    -------
    dict with best estimator, predictions, report, best_params, metrics
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    result = _evaluate(gs.best_estimator_, X_test, y_test, "Random Forest (Tuned)")
    result["best_params"] = gs.best_params_

    y_pred = result["y_pred"]
    result["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    result["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    result["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Precision   : {result['precision']:.4f}")
    print(f"  Recall      : {result['recall']:.4f}")
    print(f"  F1          : {result['f1']:.4f}")
    return result
"""
ml_models.py
============
Classical Machine Learning classifiers for sentiment analysis.

Models
------
1. Naive Bayes  (MultinomialNB)
2. SVM          (SVC)
3. Random Forest

Each public train_* function returns a dict:
    {
        'model'   : fitted estimator,
        'y_pred'  : np.ndarray of predictions on X_test,
        'report'  : classification_report string,
        'accuracy': float,
    }

Each public tune_* function additionally returns:
    {
        ...all above...,
        'best_params': dict of best hyperparameters from GridSearchCV,
    }
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)


# =============================================================================
# Shared helper
# =============================================================================
def _evaluate(model, X_test, y_test, label: str) -> dict:
    """
    Run predictions and print a formatted classification report.

    Parameters
    ----------
    model   : fitted sklearn estimator
    X_test  : feature matrix for test set
    y_test  : true labels for test set
    label   : str — display name for console output

    Returns
    -------
    dict with keys: model, y_pred, report, accuracy
    """
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'=' * 52}")
    print(f"  {label}")
    print(f"{'=' * 52}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Classification Report:\n{report}")

    return {
        "model":    model,
        "y_pred":   y_pred,
        "report":   report,
        "accuracy": acc,
    }


# =============================================================================
# 1. Naive Bayes
# =============================================================================
def train_naive_bayes(X_train, y_train, X_test, y_test) -> dict:
    """
    Train a baseline Multinomial Naive Bayes classifier.

    Parameters
    ----------
    X_train : sparse matrix — BoW features for training
    y_train : array-like    — training labels
    X_test  : sparse matrix — BoW features for testing
    y_test  : array-like    — test labels

    Returns
    -------
    dict : model, y_pred, report, accuracy
    """
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return _evaluate(nb, X_test, y_test, "Naive Bayes (Baseline)")


def tune_naive_bayes(X_train, y_train, X_test, y_test) -> dict:
    """
    Hyperparameter-tuned Naive Bayes using 5-fold stratified GridSearchCV.

    Hyperparameter grid
    -------------------
    alpha     : [0.1, 1.0, 10.0]  — Laplace smoothing parameter
    fit_prior : [True, False]      — whether to learn class prior probs

    Parameters
    ----------
    X_train : sparse matrix
    y_train : array-like
    X_test  : sparse matrix
    y_test  : array-like

    Returns
    -------
    dict : model, y_pred, report, accuracy, best_params
    """
    param_grid = {
        "alpha":     [0.1, 1.0, 10.0],
        "fit_prior": [True, False],
    }

    scoring = {
        "Accuracy":  make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "Recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
        "F1":        make_scorer(f1_score,        average="weighted", zero_division=0),
    }

    gs = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring=scoring,
        refit="F1",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    result = _evaluate(gs.best_estimator_, X_test, y_test, "Naive Bayes (Tuned)")
    result["best_params"] = gs.best_params_

    print(f"  Best params  : {gs.best_params_}")
    print(f"  Best CV F1   : {gs.best_score_:.4f}")
    return result


# =============================================================================
# 2. Support Vector Machine
# =============================================================================
def train_svm(X_train, y_train, X_test, y_test) -> dict:
    """
    Train a baseline Support Vector Classifier (SVC) with default settings.

    Parameters
    ----------
    X_train : sparse matrix
    y_train : array-like
    X_test  : sparse matrix
    y_test  : array-like

    Returns
    -------
    dict : model, y_pred, report, accuracy
    """
    svc = SVC(kernel="linear", random_state=42)
    svc.fit(X_train, y_train)
    return _evaluate(svc, X_test, y_test, "SVM (Baseline)")


def tune_svm(X_train, y_train, X_test, y_test) -> dict:
    """
    Hyperparameter-tuned SVM using 5-fold GridSearchCV.

    Hyperparameter grid
    -------------------
    C            : [0.1, 1.0, 10.0]       — regularisation strength
    kernel       : ['linear', 'rbf']       — kernel function
    gamma        : ['scale', 'auto']       — kernel coefficient
    class_weight : [None, 'balanced']      — handle class imbalance

    Parameters
    ----------
    X_train : sparse matrix
    y_train : array-like
    X_test  : sparse matrix
    y_test  : array-like

    Returns
    -------
    dict : model, y_pred, report, accuracy, best_params
    """
    param_grid = {
        "C":            [0.1, 1.0, 10.0],
        "kernel":       ["linear", "rbf"],
        "gamma":        ["scale", "auto"],
        "class_weight": [None, "balanced"],
    }

    gs = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    result = _evaluate(gs.best_estimator_, X_test, y_test, "SVM (Tuned)")
    result["best_params"] = gs.best_params_

    print(f"  Best params  : {gs.best_params_}")
    return result


# =============================================================================
# 3. Random Forest
# =============================================================================
def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """
    Train a baseline Random Forest classifier.

    Parameters
    ----------
    X_train : sparse matrix
    y_train : array-like
    X_test  : sparse matrix
    y_test  : array-like

    Returns
    -------
    dict : model, y_pred, report, accuracy
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return _evaluate(rf, X_test, y_test, "Random Forest (Baseline)")


def tune_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """
    Hyperparameter-tuned Random Forest using 5-fold GridSearchCV.

    Hyperparameter grid
    -------------------
    n_estimators      : [100, 200, 300]   — number of trees
    max_depth         : [10, 20, 30]      — max tree depth
    min_samples_split : [2, 5, 10]        — min samples to split a node
    min_samples_leaf  : [1, 2, 4]         — min samples in a leaf
    bootstrap         : [True, False]      — bootstrap sampling

    Parameters
    ----------
    X_train : sparse matrix
    y_train : array-like
    X_test  : sparse matrix
    y_test  : array-like

    Returns
    -------
    dict : model, y_pred, report, accuracy, best_params,
           precision, recall, f1
    """
    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "bootstrap":         [True, False],
    }

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    result = _evaluate(gs.best_estimator_, X_test, y_test, "Random Forest (Tuned)")
    result["best_params"] = gs.best_params_

    y_pred = result["y_pred"]
    result["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    result["recall"]    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    result["f1"]        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"  Best params  : {gs.best_params_}")
    print(f"  Precision    : {result['precision']:.4f}")
    print(f"  Recall       : {result['recall']:.4f}")
    print(f"  F1           : {result['f1']:.4f}")
    return result