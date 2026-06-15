"""
features.py
-----------
Feature engineering module for the Customer Churn Predictor App.

Transforms raw GitHub API user records into machine-learning-ready features.
This module is completely independent of data collection, model training,
FastAPI, and prediction endpoints.

Responsibilities
----------------
- Parse and validate raw user/repository data.
- Compute all engineered features.
- Validate the resulting feature DataFrame.
- Export features to CSV and Parquet.

NOT responsible for
-------------------
- Fetching data from GitHub.
- Creating the churn label (days_since_last_activity > 180).
- Model training or inference.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [features] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVITY_THRESHOLD_DAYS: int = 180   # active  = pushed within this window
MAINTENANCE_THRESHOLD_DAYS: int = 90  # maintained = updated within this window
DAYS_PER_YEAR: float = 365.25
MIN_ACCOUNT_AGE_YEARS: float = 1 / DAYS_PER_YEAR  # prevent near-zero denominators

OUTPUT_COLUMNS: list[str] = [
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

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _today() -> date:
    """Return today's UTC date."""
    return datetime.now(tz=timezone.utc).date()


def parse_date(value: Any) -> date | None:
    """Parse a date/datetime value from several common representations.

    Parameters
    ----------
    value:
        Raw value to parse.  Accepts :class:`datetime`, :class:`date`,
        ISO-8601 strings (with or without timezone), or ``None``.

    Returns
    -------
    date | None
        Parsed UTC date, or ``None`` when parsing fails.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        # Normalise trailing Z
        cleaned = cleaned.replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(cleaned, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).date()
            except ValueError:
                continue
        logger.warning("Invalid date detected: %r", value)
        return None

    logger.warning("Unsupported date type %s: %r", type(value).__name__, value)
    return None


def _days_between(earlier: date | None, later: date | None) -> int | None:
    """Return the integer number of days between two dates, or ``None``."""
    if earlier is None or later is None:
        return None
    delta = (later - earlier).days
    if delta < 0:
        logger.warning("Negative day delta detected (%d); treating as 0.", delta)
        return 0
    return delta

# ---------------------------------------------------------------------------
# Individual feature calculators
# ---------------------------------------------------------------------------


def calculate_days_since_last_activity(user: dict) -> int:
    """Compute days since the user's profile was last updated.

    Parameters
    ----------
    user:
        Raw user record.

    Returns
    -------
    int
        Days since last activity; defaults to ``0`` on missing data.
    """
    updated = parse_date(user.get("updated_at"))
    if updated is None:
        logger.warning(
            "Missing 'updated_at' for user %r; defaulting days_since_last_activity=0.",
            user.get("login"),
        )
        return 0
    result = _days_between(updated, _today())
    return result if result is not None else 0


def calculate_account_age(user: dict) -> int:
    """Compute the account age in days from ``created_at`` to today.

    Parameters
    ----------
    user:
        Raw user record.

    Returns
    -------
    int
        Account age in days; defaults to ``1`` on missing data.
    """
    created = parse_date(user.get("created_at"))
    if created is None:
        logger.warning(
            "Missing 'created_at' for user %r; defaulting account_age_days=1.",
            user.get("login"),
        )
        return 1
    result = _days_between(created, _today())
    return result if result is not None and result > 0 else 1


def calculate_repos_per_year(public_repos: int, account_age_days: int) -> float:
    """Compute repository creation rate per year.

    Parameters
    ----------
    public_repos:
        Total public repositories owned by the user.
    account_age_days:
        Account age in days.

    Returns
    -------
    float
        Repositories created per year; ``0.0`` when age is negligible.
    """
    age_years = account_age_days / DAYS_PER_YEAR
    if age_years < MIN_ACCOUNT_AGE_YEARS:
        return 0.0
    return round(public_repos / age_years, 4)


def calculate_followers_ratio(followers: int, following: int) -> float:
    """Compute followers-to-following ratio.

    Uses ``following + 1`` in the denominator to avoid division by zero.

    Parameters
    ----------
    followers:
        Number of followers.
    following:
        Number of accounts the user follows.

    Returns
    -------
    float
        Followers / (following + 1), rounded to 4 decimal places.
    """
    return round(followers / (following + 1), 4)


def _classify_repositories(
    repositories: list[dict],
    today: date,
) -> tuple[int, int, int, float, float]:
    """Classify each repository and aggregate star/fork counts.

    Parameters
    ----------
    repositories:
        List of raw repository dicts.
    today:
        Reference date for threshold comparisons.

    Returns
    -------
    tuple[int, int, int, float, float]
        ``(active_count, inactive_count, maintained_count,
           total_stars, total_forks)``
    """
    active = inactive = maintained = 0
    total_stars: float = 0.0
    total_forks: float = 0.0

    for repo in repositories:
        if not isinstance(repo, dict):
            logger.warning("Malformed repository object skipped: %r", repo)
            continue

        # Stars / forks
        total_stars += float(repo.get("stargazers_count") or 0)
        total_forks += float(repo.get("forks_count") or 0)

        # Activity classification (pushed_at)
        pushed = parse_date(repo.get("pushed_at"))
        if pushed is not None:
            days_since_push = _days_between(pushed, today) or 0
            if days_since_push <= ACTIVITY_THRESHOLD_DAYS:
                active += 1
            else:
                inactive += 1
        else:
            inactive += 1

        # Maintenance classification (updated_at)
        updated = parse_date(repo.get("updated_at"))
        if updated is not None:
            days_since_update = _days_between(updated, today) or 0
            if days_since_update <= MAINTENANCE_THRESHOLD_DAYS:
                maintained += 1
        # repos without updated_at are simply not counted as maintained

    return active, inactive, maintained, total_stars, total_forks


def calculate_active_repo_ratio(active: int, total: int) -> float:
    """Compute the fraction of repositories that are active.

    Parameters
    ----------
    active:
        Number of active repositories.
    total:
        Total number of repositories.

    Returns
    -------
    float
        Ratio in ``[0.0, 1.0]``; ``0.0`` when ``total`` is zero.
    """
    if total == 0:
        return 0.0
    return round(active / total, 4)


def calculate_inactive_repo_ratio(inactive: int, total: int) -> float:
    """Compute the fraction of repositories that are inactive.

    Parameters
    ----------
    inactive:
        Number of inactive repositories.
    total:
        Total number of repositories.

    Returns
    -------
    float
        Ratio in ``[0.0, 1.0]``; ``0.0`` when ``total`` is zero.
    """
    if total == 0:
        return 0.0
    return round(inactive / total, 4)


def calculate_avg_stars_per_repo(total_stars: float, total: int) -> float:
    """Compute the average star count per repository.

    Parameters
    ----------
    total_stars:
        Sum of all stargazer counts.
    total:
        Total number of repositories.

    Returns
    -------
    float
        Average stars; ``0.0`` when ``total`` is zero.
    """
    if total == 0:
        return 0.0
    return round(total_stars / total, 4)


def calculate_avg_forks_per_repo(total_forks: float, total: int) -> float:
    """Compute the average fork count per repository.

    Parameters
    ----------
    total_forks:
        Sum of all fork counts.
    total:
        Total number of repositories.

    Returns
    -------
    float
        Average forks; ``0.0`` when ``total`` is zero.
    """
    if total == 0:
        return 0.0
    return round(total_forks / total, 4)


def calculate_repo_activity_density(active: int, account_age_days: int) -> float:
    """Compute active repository count per year of account age.

    Parameters
    ----------
    active:
        Number of active repositories.
    account_age_days:
        Account age in days.

    Returns
    -------
    float
        Active repos per year; ``0.0`` when age is negligible.
    """
    age_years = account_age_days / DAYS_PER_YEAR
    if age_years < MIN_ACCOUNT_AGE_YEARS:
        return 0.0
    return round(active / age_years, 4)


def calculate_repository_maintenance_ratio(maintained: int, total: int) -> float:
    """Compute the fraction of repositories that are actively maintained.

    Parameters
    ----------
    maintained:
        Number of maintained repositories.
    total:
        Total number of repositories.

    Returns
    -------
    float
        Ratio in ``[0.0, 1.0]``; ``0.0`` when ``total`` is zero.
    """
    if total == 0:
        return 0.0
    return round(maintained / total, 4)

# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


def build_feature_record(user: dict) -> dict | None:
    """Build a single flat feature record from a raw GitHub user dict.

    Parameters
    ----------
    user:
        Raw user record produced by the scraper.

    Returns
    -------
    dict | None
        Feature dictionary with keys matching :data:`OUTPUT_COLUMNS`,
        or ``None`` if the record is fundamentally unusable.
    """
    login: str = user.get("login") or ""
    if not login:
        logger.error("Feature generation failed: user record has no 'login' field.")
        return None

    logger.info("Processing user: %s", login)

    try:
        today = _today()

        # --- Temporal features ---
        days_since_last_activity = calculate_days_since_last_activity(user)
        account_age_days = calculate_account_age(user)

        # --- Social features ---
        followers = int(user.get("followers") or 0)
        following = int(user.get("following") or 0)
        followers_following_ratio = calculate_followers_ratio(followers, following)

        # --- Repository features ---
        public_repos = int(user.get("public_repos") or 0)
        repos_per_year = calculate_repos_per_year(public_repos, account_age_days)

        raw_repos = user.get("repositories")
        if raw_repos is None:
            logger.warning("Missing repository data for user: %s", login)
            raw_repos = []
        elif not isinstance(raw_repos, list):
            logger.warning(
                "Repository data for user %s is not a list (%s); treating as empty.",
                login,
                type(raw_repos).__name__,
            )
            raw_repos = []

        active, inactive, maintained, total_stars, total_forks = (
            _classify_repositories(raw_repos, today)
        )
        total_repos = len(raw_repos)

        active_repo_ratio = calculate_active_repo_ratio(active, total_repos)
        inactive_repo_ratio = calculate_inactive_repo_ratio(inactive, total_repos)
        avg_stars_per_repo = calculate_avg_stars_per_repo(total_stars, total_repos)
        avg_forks_per_repo = calculate_avg_forks_per_repo(total_forks, total_repos)
        repo_activity_density = calculate_repo_activity_density(
            active, account_age_days
        )
        repository_maintenance_ratio = calculate_repository_maintenance_ratio(
            maintained, total_repos
        )

        record: dict = {
            "login": login,
            "days_since_last_activity": days_since_last_activity,
            "account_age_days": account_age_days,
            "repos_per_year": repos_per_year,
            "followers_following_ratio": followers_following_ratio,
            "active_repo_ratio": active_repo_ratio,
            "inactive_repo_ratio": inactive_repo_ratio,
            "avg_stars_per_repo": avg_stars_per_repo,
            "avg_forks_per_repo": avg_forks_per_repo,
            "repo_activity_density": repo_activity_density,
            "repository_maintenance_ratio": repository_maintenance_ratio,
        }

        logger.info("Created %d features for user: %s", len(OUTPUT_COLUMNS) - 1, login)
        return record

    except Exception as exc:  # noqa: BLE001
        logger.error("Feature generation failed for user %r: %s", login, exc)
        return None

# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


def build_feature_dataframe(raw_users: list[dict]) -> pd.DataFrame:
    """Transform a list of raw GitHub user records into a feature DataFrame.

    Parameters
    ----------
    raw_users:
        List of raw user dicts from the scraper.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by :data:`OUTPUT_COLUMNS`.
        Rows for users that fail feature extraction are omitted.
    """
    records: list[dict] = []
    for user in raw_users:
        record = build_feature_record(user)
        if record is not None:
            records.append(record)

    if not records:
        logger.warning("No feature records were generated; returning empty DataFrame.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    df = _validate_features(df)
    logger.info("Feature DataFrame built: %d rows × %d columns.", *df.shape)
    return df

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitise the feature DataFrame.

    - Clamps ratio columns to ``[0.0, 1.0]``.
    - Replaces ``NaN`` / ``inf`` values with ``0``.
    - Enforces non-negative day counts.
    - Logs a warning for each anomaly found.

    Parameters
    ----------
    df:
        Feature DataFrame to validate.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.
    """
    ratio_cols = [
        "active_repo_ratio",
        "inactive_repo_ratio",
        "repository_maintenance_ratio",
    ]
    non_negative_cols = [
        "days_since_last_activity",
        "account_age_days",
        "repos_per_year",
        "avg_stars_per_repo",
        "avg_forks_per_repo",
        "repo_activity_density",
    ]

    # Replace infinities before further checks
    inf_mask = np.isinf(df.select_dtypes(include="number"))
    if inf_mask.any().any():
        logger.warning("Infinite values detected; replacing with 0.")
        df.replace([np.inf, -np.inf], 0, inplace=True)

    # NaN check
    nan_mask = df.isnull().any()
    if nan_mask.any():
        logger.warning("NaN values detected in columns: %s", nan_mask[nan_mask].index.tolist())
        df.fillna(0, inplace=True)

    # Clamp ratios
    for col in ratio_cols:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range:
                logger.warning("%d out-of-range values in ratio column '%s'; clamping.", out_of_range, col)
            df[col] = df[col].clip(lower=0.0, upper=1.0)

    # Non-negative enforcement
    for col in non_negative_cols:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives:
                logger.warning("%d negative values in column '%s'; clamping to 0.", negatives, col)
            df[col] = df[col].clip(lower=0)

    return df

# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------


def save_features_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Persist the feature DataFrame to a CSV file.

    Parameters
    ----------
    df:
        Feature DataFrame.
    path:
        Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("Features saved to CSV: %s  (%d rows)", dest, len(df))
    except OSError as exc:
        logger.error("Failed to write CSV [%s]: %s", dest, exc)


def save_features_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Persist the feature DataFrame to a Parquet file.

    Parameters
    ----------
    df:
        Feature DataFrame.
    path:
        Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(dest, index=False, engine="pyarrow")
        logger.info("Features saved to Parquet: %s  (%d rows)", dest, len(df))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write Parquet [%s]: %s", dest, exc)

# ---------------------------------------------------------------------------
# Entry point (executable example)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    RAW_JSON = Path("data") / "raw" / "github_users.json"
    FEATURES_CSV = Path("data") / "features" / "github_features.csv"
    FEATURES_PARQUET = Path("data") / "features" / "github_features.parquet"

    if not RAW_JSON.exists():
        logger.error(
            "Raw data file not found: %s  "
            "Run scraper.py first to generate raw data.",
            RAW_JSON,
        )
        sys.exit(1)

    with RAW_JSON.open("r", encoding="utf-8") as fh:
        raw_users: list[dict] = json.load(fh)

    logger.info("Loaded %d raw user records from %s.", len(raw_users), RAW_JSON)

    feature_df = build_feature_dataframe(raw_users)

    save_features_csv(feature_df, FEATURES_CSV)
    save_features_parquet(feature_df, FEATURES_PARQUET)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Users processed          : {len(feature_df)}")
    print(f"Features per user        : {len(OUTPUT_COLUMNS) - 1}")
    print(f"Output columns           : {OUTPUT_COLUMNS}")
    if not feature_df.empty:
        print("\nDescriptive statistics:")
        print(feature_df.describe().to_string())
    print("=" * 60)