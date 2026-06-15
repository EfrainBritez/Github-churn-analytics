"""
model.py
--------
Random Forest training module for the Customer Churn Predictor App.

Responsibilities
----------------
- Load a labelled, feature-selected dataset.
- Guard against target-leaking columns before training.
- Train a RandomForestClassifier with cross-validation.
- Evaluate with multiple metrics (accuracy, ROC-AUC, F1).
- Persist and reload the trained model.
- Expose a predict() helper for downstream inference.

NOT responsible for
-------------------
- Feature engineering  (features.py)
- Label creation       (label_builder.py)
- Feature selection    (feature_selection/)
- API serving          (FastAPI layer)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [model] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "models/model_filtered.pkl"

# Path to the filtered feature dataset used for training.
# Columns present: days_since_last_activity, account_age_days, repos_per_year,
#   followers_following_ratio, active_repo_ratio, inactive_repo_ratio,
#   avg_stars_per_repo, avg_forks_per_repo, repository_maintenance_ratio, churn
TRAINING_CSV_PATH = "data/filtered/filtered_features.csv"

# Columns that encode or directly derive from the churn label.
# Including ANY of these causes target leakage — the model learns the
# labelling rule instead of genuinely predictive patterns.
#
# Rule of thumb: if a column was used to *compute* the label, drop it.
#   churn = 1  iff  days_since_last_activity > 180
#   active_repo_ratio  uses the same 180-day push threshold
#   inactive_repo_ratio = 1 - active_repo_ratio  (redundant + leaking)
#
LEAKY_COLUMNS: list[str] = [
    "days_since_last_activity",   # direct source of the churn rule
    "active_repo_ratio",          # built from the same 180-day window
    "inactive_repo_ratio",        # complement of active_repo_ratio
]

# Non-feature identifier columns that must be dropped before training.
# The filtered_results dataset has no login/id column, so this is empty.
ID_COLUMNS: list[str] = []

# Cross-validation configuration
CV_FOLDS: int = 5
CV_SCORING: list[str] = ["accuracy", "f1", "roc_auc"]

# ---------------------------------------------------------------------------
# Data loading & leakage guard
# ---------------------------------------------------------------------------


def load_dataset(csv_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the labelled CSV and split into features and target.

    Drops identifier columns (e.g. ``login``) and any column listed in
    :data:`LEAKY_COLUMNS` to prevent target leakage.

    Parameters
    ----------
    csv_path:
        Path to the labelled CSV file.
    target_column:
        Name of the churn label column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` — feature matrix and target vector.
    """
    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Dataset shape: %d rows × %d columns.", *df.shape)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Drop identifier columns
    drop_ids = [c for c in ID_COLUMNS if c in df.columns]
    if drop_ids:
        logger.info("Dropping identifier column(s): %s", drop_ids)
        df = df.drop(columns=drop_ids)

    # Drop leaky columns
    drop_leaky = [c for c in LEAKY_COLUMNS if c in df.columns]
    if drop_leaky:
        logger.warning(
            "Dropping leaky column(s) to prevent target leakage: %s", drop_leaky
        )
        df = df.drop(columns=drop_leaky)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Drop any remaining rows where the target is null
    valid_mask = y.notna()
    dropped = (~valid_mask).sum()
    if dropped:
        logger.warning("Dropping %d rows with missing target values.", dropped)
        X, y = X[valid_mask], y[valid_mask]

    y = y.astype(int)

    logger.info(
        "Features: %d columns | Target distribution: %s",
        X.shape[1],
        y.value_counts().to_dict(),
    )
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    csv_path: str,
    target_column: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 10,
    min_samples_leaf: int = 5,
    class_weight: str | dict = "balanced",
) -> RandomForestClassifier:
    """Train a Random Forest classifier with cross-validation diagnostics.

    Parameters
    ----------
    csv_path:
        Path to the labelled, feature-selected CSV.
    target_column:
        Name of the churn label column.
    test_size:
        Proportion of data held out as the final test set.
    random_state:
        Seed for reproducibility.
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum tree depth.  ``None`` = unlimited (risks overfitting).
        Default is ``10`` — a reasonable regularising cap.
    min_samples_leaf:
        Minimum samples required at a leaf node (regularisation).
    class_weight:
        Passed to :class:`~sklearn.ensemble.RandomForestClassifier`.
        ``"balanced"`` compensates for class imbalance automatically.

    Returns
    -------
    RandomForestClassifier
        Fitted model trained on the full training split.
    """
    X, y = load_dataset(csv_path, target_column)

    # ------------------------------------------------------------------ #
    #  Hold-out split                                                       #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train / test split: %d train rows, %d test rows.",
        len(X_train),
        len(X_test),
    )

    # ------------------------------------------------------------------ #
    #  Cross-validation on the training set                                #
    # ------------------------------------------------------------------ #
    logger.info("Running %d-fold stratified cross-validation…", CV_FOLDS)

    cv_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        cv_model,
        X_train,
        y_train,
        cv=cv,
        scoring=CV_SCORING,
        return_train_score=True,
    )

    for metric in CV_SCORING:
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        logger.info(
            "CV %-12s | train %.4f ± %.4f | val %.4f ± %.4f",
            metric,
            train_scores.mean(),
            train_scores.std(),
            test_scores.mean(),
            test_scores.std(),
        )

        # Leakage warning: near-perfect CV accuracy signals remaining leakage
        if metric == "accuracy" and test_scores.mean() > 0.98:
            logger.warning(
                "CV accuracy=%.4f is suspiciously high. "
                "Check for remaining leaky features.",
                test_scores.mean(),
            )

    # ------------------------------------------------------------------ #
    #  Final model — retrain on full training split                        #
    # ------------------------------------------------------------------ #
    logger.info("Training final model on full training split…")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    #  Hold-out evaluation                                                 #
    # ------------------------------------------------------------------ #
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    try:
        roc_auc = roc_auc_score(y_test, proba)
    except ValueError:
        roc_auc = float("nan")

    logger.info("Hold-out Accuracy : %.4f", accuracy)
    logger.info("Hold-out ROC-AUC  : %.4f", roc_auc)

    if accuracy > 0.98:
        logger.warning(
            "Hold-out accuracy=%.4f — verify no leaky columns remain.", accuracy
        )

    print("\n" + "=" * 60)
    print("HOLD-OUT EVALUATION")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))

    # ------------------------------------------------------------------ #
    #  Feature importances                                                 #
    # ------------------------------------------------------------------ #
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print("Top-10 Feature Importances:")
    print(importances.head(10).to_string())
    print("=" * 60 + "\n")

    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: RandomForestClassifier, path: str = MODEL_PATH) -> None:
    """Serialise the trained model to disk with joblib.

    Parameters
    ----------
    model:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    path:
        Destination file path.
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved to %s.", model_path)


def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    """Load a serialised model from disk.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file produced by :func:`save_model`.

    Returns
    -------
    RandomForestClassifier
        Deserialised fitted model.
    """
    model = joblib.load(path)
    logger.info("Model loaded from %s.", path)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(model: RandomForestClassifier, features: dict[str, Any]) -> dict:
    """Predict churn and return probability for a single user.

    Parameters
    ----------
    model:
        Fitted model returned by :func:`train` or :func:`load_model`.
    features:
        Feature dictionary.  Keys must match the columns the model was
        trained on (leaky columns are silently dropped if present).

    Returns
    -------
    dict
        ``{"churned": bool, "churn_probability": float}``

    Example
    -------
    >>> result = predict(model, {"account_age_days": 730, "repos_per_year": 3.5})
    >>> result["churned"]
    False
    """
    X = pd.DataFrame([features])

    # Drop leaky columns if the caller accidentally includes them
    leaky_present = [c for c in LEAKY_COLUMNS if c in X.columns]
    if leaky_present:
        logger.warning(
            "Dropping leaky column(s) from prediction input: %s", leaky_present
        )
        X = X.drop(columns=leaky_present)

    # Align to training feature set
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]

        if missing:
            logger.warning(
                "Filling %d missing feature(s) with 0: %s", len(missing), missing
            )
            for c in missing:
                X[c] = 0

        if extra:
            logger.warning(
                "Dropping %d unrecognised feature(s): %s", len(extra), extra
            )
            X = X.drop(columns=extra)

        X = X[expected]

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "churned": bool(prediction),
        "churn_probability": round(probability, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = train(
        csv_path=TRAINING_CSV_PATH,
        target_column="churn",
    )
    save_model(model)