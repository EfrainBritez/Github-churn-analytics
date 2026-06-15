"""
label_builder.py
----------------
Churn label generation module for the Customer Churn Predictor App.

Single responsibility: apply the churn business rule to the feature dataset
and produce TWO output datasets:

    Dataset A — labeled_dataset  : all features + churn label (audit/inspection)
    Dataset B — training_dataset : leakage-free features + churn label (ML use)

The feature `days_since_last_activity` is used ONLY to generate the label
and is then removed from the training dataset to prevent target leakage.

Pipeline position
-----------------
GitHub API → scraper.py → features.py → label_builder.py
    → training_dataset.csv → feature_selection/ → model.py → FastAPI

NOT responsible for
-------------------
- Fetching or scraping data.
- Feature engineering.
- Model training or inference.
- API serving.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [label_builder] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Business rule: modify here to change the churn definition globally.
CHURN_THRESHOLD_DAYS: int = 180

# Feature used to generate the churn label.
# It MUST be removed from the training dataset after labelling.
LEAKAGE_FEATURE: str = "days_since_last_activity"

# All columns produced by features.py
FEATURE_COLUMNS: list[str] = [
    "login",
    "days_since_last_activity",
    "account_age_days",
    "repos_per_year",
    "followers_following_ratio",
    "active_repo_ratio",
    "inactive_repo_ratio",
    "avg_stars_per_repo",
    "avg_forks_per_repo",
    "repo_activity_density",
    "repository_maintenance_ratio",
]

# Dataset A: all features + label (audit only)
LABELED_COLUMNS: list[str] = FEATURE_COLUMNS + ["churn"]

# Dataset B: leakage-free features + label (ML training)
TRAINING_COLUMNS: list[str] = [
    c for c in LABELED_COLUMNS if c != LEAKAGE_FEATURE
]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitise an incoming feature DataFrame.

    Parameters
    ----------
    df:
        Raw feature DataFrame produced by ``features.py``.

    Returns
    -------
    pd.DataFrame
        Validated copy of ``df``.

    Raises
    ------
    TypeError
        If ``df`` is not a :class:`pandas.DataFrame`.
    ValueError
        If required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")

    df = df.copy()

    numeric_cols = [c for c in FEATURE_COLUMNS if c != "login"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(
                "Column '%s' has non-numeric dtype; coercing to float.", col
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    inf_mask = np.isinf(df[numeric_cols].values)
    if inf_mask.any():
        logger.warning(
            "%d infinite value(s) detected; replacing with NaN.", int(inf_mask.sum())
        )
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    nan_summary = df[numeric_cols].isnull().sum()
    for col, count in nan_summary[nan_summary > 0].items():
        logger.warning("Column '%s' contains %d NaN value(s).", col, count)

    neg_summary = (df[numeric_cols] < 0).sum()
    for col, count in neg_summary[neg_summary > 0].items():
        logger.warning("Column '%s' contains %d negative value(s).", col, count)

    logger.info(
        "Dataset validation complete: %d records, %d columns.", *df.shape
    )
    return df


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------


def create_churn_label(
    days_since_last_activity: Any,
    threshold: int = CHURN_THRESHOLD_DAYS,
) -> int | None:
    """Apply the churn business rule to a single inactivity value.

    Parameters
    ----------
    days_since_last_activity:
        Raw inactivity value.  May be ``None``, ``NaN``, float, or int.
    threshold:
        Inactivity days above which a user is considered churned.

    Returns
    -------
    int | None
        ``1`` (churned), ``0`` (active), or ``None`` (missing / invalid).
    """
    if days_since_last_activity is None:
        return None

    try:
        value = float(days_since_last_activity)
    except (TypeError, ValueError):
        logger.error(
            "Invalid days_since_last_activity value: %r", days_since_last_activity
        )
        return None

    if np.isnan(value) or np.isinf(value):
        return None

    if value < 0:
        logger.error(
            "Invalid days_since_last_activity value: %r (negative).",
            days_since_last_activity,
        )
        return None

    return 1 if value > threshold else 0


def apply_churn_labels(
    df: pd.DataFrame,
    threshold: int = CHURN_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """Add a ``churn`` column to the feature DataFrame (Dataset A).

    Parameters
    ----------
    df:
        Validated feature DataFrame.
    threshold:
        Inactivity threshold in days.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an additional ``churn`` column.
    """
    logger.info("Creating churn labels")
    logger.info("Threshold: %d days", threshold)

    df = df.copy()
    labels: list[int | None] = []

    for _, row in df.iterrows():
        raw = row.get(LEAKAGE_FEATURE)
        label = create_churn_label(raw, threshold=threshold)

        if label is None:
            login = row.get("login", "<unknown>")
            logger.warning("Missing inactivity value for user %s", login)

        labels.append(label)

    df["churn"] = pd.array(labels, dtype=pd.Int8Dtype())

    labelled_count = int(df["churn"].notna().sum())
    logger.info("Generated labels for %d users.", labelled_count)
    return df


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------


def detect_target_leakage(df: pd.DataFrame) -> None:
    """Raise if the leakage feature is still present in the training dataset.

    Parameters
    ----------
    df:
        DataFrame that will be used for model training.

    Raises
    ------
    ValueError
        If ``days_since_last_activity`` is found in ``df``.
    """
    if LEAKAGE_FEATURE in df.columns:
        message = (
            f"Target leakage detected. "
            f"Feature '{LEAKAGE_FEATURE}' cannot exist in the training dataset "
            f"because it was used to generate the label."
        )
        logger.error("Target leakage detected")
        raise ValueError(message)

    logger.info("Leakage validation passed")


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ``days_since_last_activity`` from a DataFrame.

    Parameters
    ----------
    df:
        Labelled DataFrame (Dataset A).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` without the leakage feature (Dataset B).
    """
    logger.info("Removing leakage feature: %s", LEAKAGE_FEATURE)
    df = df.copy()

    if LEAKAGE_FEATURE in df.columns:
        df = df.drop(columns=[LEAKAGE_FEATURE])

    return df


def build_training_dataset(
    df: pd.DataFrame,
    threshold: int = CHURN_THRESHOLD_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Dataset A (labeled) and Dataset B (training) from raw features.

    Parameters
    ----------
    df:
        Validated feature DataFrame from ``features.py``.
    threshold:
        Inactivity threshold in days.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(labeled_df, training_df)``

        - ``labeled_df``  — Dataset A: all features + churn (audit use).
        - ``training_df`` — Dataset B: leakage-free features + churn (ML use).
    """
    logger.info("Creating training dataset")

    # Dataset A — all features + label
    labeled_df = apply_churn_labels(df, threshold=threshold)

    # Dataset B — drop the leakage feature, then verify
    training_df = remove_leakage_features(labeled_df)
    detect_target_leakage(training_df)

    logger.info("Training dataset ready")
    logger.info(
        "Labeled dataset  : %d rows x %d columns", *labeled_df.shape
    )
    logger.info(
        "Training dataset : %d rows x %d columns", *training_df.shape
    )
    return labeled_df, training_df


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------


def calculate_class_distribution(df: pd.DataFrame) -> dict:
    """Compute class balance statistics from a labelled DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing a ``churn`` column.

    Returns
    -------
    dict
        Keys: ``total_users``, ``total_churned``, ``total_non_churned``,
        ``churn_rate``, ``non_churn_rate``.
    """
    if "churn" not in df.columns:
        raise ValueError("DataFrame does not contain a 'churn' column.")

    labelled = df["churn"].dropna()
    total = len(labelled)
    churned = int((labelled == 1).sum())
    non_churned = int((labelled == 0).sum())

    churn_rate = round(churned / total, 4) if total > 0 else 0.0
    non_churn_rate = round(non_churned / total, 4) if total > 0 else 0.0

    return {
        "total_users": total,
        "total_churned": churned,
        "total_non_churned": non_churned,
        "churn_rate": churn_rate,
        "non_churn_rate": non_churn_rate,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_labeling_report(
    labeled_df: pd.DataFrame,
    training_df: pd.DataFrame,
    threshold: int = CHURN_THRESHOLD_DAYS,
) -> dict:
    """Generate and log a labelling summary report.

    Parameters
    ----------
    labeled_df:
        Dataset A (all features + churn).
    training_df:
        Dataset B (leakage-free + churn).
    threshold:
        Inactivity threshold used during labelling.

    Returns
    -------
    dict
        Report with keys: ``threshold_days``, ``total_records``,
        ``churned_users``, ``active_users``, ``churn_percentage``,
        ``active_percentage``, ``leakage_removed``.
    """
    dist = calculate_class_distribution(labeled_df)
    total_records = len(labeled_df)
    labelled = dist["total_users"]
    churned = dist["total_churned"]
    active = dist["total_non_churned"]

    churn_pct = round(churned / labelled * 100, 2) if labelled > 0 else 0.0
    active_pct = round(active / labelled * 100, 2) if labelled > 0 else 0.0
    leakage_removed = LEAKAGE_FEATURE not in training_df.columns

    report: dict = {
        "threshold_days": threshold,
        "total_records": total_records,
        "churned_users": churned,
        "active_users": active,
        "churn_percentage": churn_pct,
        "active_percentage": active_pct,
        "leakage_removed": leakage_removed,
    }

    logger.info(
        "Report | threshold=%d | total=%d | churned=%d (%.1f%%) | "
        "active=%d (%.1f%%) | leakage_removed=%s",
        threshold, total_records, churned, churn_pct,
        active, active_pct, leakage_removed,
    )
    return report


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------


def save_labeled_dataset_csv(df: pd.DataFrame, path: str) -> None:
    """Save Dataset A to CSV.

    Parameters
    ----------
    df:
        Labeled DataFrame.
    path:
        Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("Labeled dataset saved to CSV: %s (%d rows)", dest, len(df))
    except OSError as exc:
        logger.error("Failed to write CSV [%s]: %s", dest, exc)


def save_labeled_dataset_parquet(df: pd.DataFrame, path: str) -> None:
    """Save Dataset A to Parquet.

    Parameters
    ----------
    df:
        Labeled DataFrame.
    path:
        Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        export = df.copy()
        export["churn"] = export["churn"].astype("float32")
        export.to_parquet(dest, index=False, engine="pyarrow")
        logger.info("Labeled dataset saved to Parquet: %s (%d rows)", dest, len(df))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write Parquet [%s]: %s", dest, exc)


def save_training_dataset_csv(df: pd.DataFrame, path: str) -> None:
    """Save Dataset B (leakage-free) to CSV.

    Parameters
    ----------
    df:
        Training DataFrame.
    path:
        Destination file path.
    """
    detect_target_leakage(df)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("Training dataset saved to CSV: %s (%d rows)", dest, len(df))
    except OSError as exc:
        logger.error("Failed to write CSV [%s]: %s", dest, exc)


def save_training_dataset_parquet(df: pd.DataFrame, path: str) -> None:
    """Save Dataset B (leakage-free) to Parquet.

    Parameters
    ----------
    df:
        Training DataFrame.
    path:
        Destination file path.
    """
    detect_target_leakage(df)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        export = df.copy()
        export["churn"] = export["churn"].astype("float32")
        export.to_parquet(dest, index=False, engine="pyarrow")
        logger.info(
            "Training dataset saved to Parquet: %s (%d rows)", dest, len(df)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write Parquet [%s]: %s", dest, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FEATURES_CSV = Path("data") / "features" / "github_features.csv"
    FEATURES_PARQUET = Path("data") / "features" / "github_features.parquet"

    LABELED_CSV     = Path("data") / "processed" / "labeled_dataset.csv"
    LABELED_PARQUET = Path("data") / "processed" / "labeled_dataset.parquet"
    TRAINING_CSV     = Path("data") / "processed" / "training_dataset.csv"
    TRAINING_PARQUET = Path("data") / "processed" / "training_dataset.parquet"

    # Load feature dataset
    if FEATURES_CSV.exists():
        logger.info("Loaded feature dataset from %s.", FEATURES_CSV)
        raw_df = pd.read_csv(FEATURES_CSV)
    elif FEATURES_PARQUET.exists():
        logger.info("Loaded feature dataset from %s.", FEATURES_PARQUET)
        raw_df = pd.read_parquet(FEATURES_PARQUET, engine="pyarrow")
    else:
        logger.error(
            "No feature dataset found at %s or %s. Run features.py first.",
            FEATURES_CSV, FEATURES_PARQUET,
        )
        raise SystemExit(1)

    # 1. Validate
    validated_df = validate_feature_dataset(raw_df)

    # 2. Build both datasets
    labeled_df, training_df = build_training_dataset(
        validated_df, threshold=CHURN_THRESHOLD_DAYS
    )

    # 3. Report
    report = generate_labeling_report(labeled_df, training_df)

    # 4. Export Dataset A
    save_labeled_dataset_csv(labeled_df, str(LABELED_CSV))
    save_labeled_dataset_parquet(labeled_df, str(LABELED_PARQUET))

    # 5. Export Dataset B
    save_training_dataset_csv(training_df, str(TRAINING_CSV))
    save_training_dataset_parquet(training_df, str(TRAINING_PARQUET))

    # 6. Print summary
    dist = calculate_class_distribution(labeled_df)

    print("\n" + "=" * 60)
    print("LABEL BUILDER SUMMARY")
    print("=" * 60)
    print(f"Threshold              : {CHURN_THRESHOLD_DAYS} days")
    print(f"Total records          : {report['total_records']}")
    print(f"Churned  (churn=1)     : {dist['total_churned']}  "
          f"({report['churn_percentage']}%)")
    print(f"Active   (churn=0)     : {dist['total_non_churned']}  "
          f"({report['active_percentage']}%)")
    print(f"Churn rate             : {dist['churn_rate']:.4f}")
    print(f"Leakage removed        : {report['leakage_removed']}")
    print("-" * 60)
    print("Dataset A (labeled)    :", LABELED_CSV)
    print("Dataset B (training)   :", TRAINING_CSV)
    print("=" * 60)