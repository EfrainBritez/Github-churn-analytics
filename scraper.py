"""
scraper.py
----------
GitHub User Activity Data Scraper for the Customer Churn Predictor App.

Collects raw user, repository, and activity data from the GitHub REST API v3.
Raw data is persisted to data/raw/ as both JSON and CSV.

Usage:
    python app/scraper.py --input usernames.txt

Environment:
    GITHUB_TOKEN  Optional Personal Access Token for higher rate limits.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load environment variables from a .env file into os.environ.

    Existing environment variables are not overridden.
    """
    if not dotenv_path.exists():
        return

    with dotenv_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE: str = "https://api.github.com"
GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
RAW_DATA_DIR: Path = Path("data") / "raw"
DEFAULT_TIMEOUT: int = 15          # seconds per request
RATE_LIMIT_SLEEP: float = 1.0      # polite delay between requests
RETRY_SLEEP: float = 60.0          # wait when rate-limited
MAX_RETRIES: int = 3               # per-request retry cap
REPOS_PER_PAGE: int = 100          # maximum allowed by GitHub
CHECKPOINT_FREQUENCY: int = 50     # save progress after this many users

# Set a personal access token in the GITHUB_TOKEN environment variable to get higher GitHub API limits.
# On Windows PowerShell: $env:GITHUB_TOKEN = "<your_token>"
# On Windows CMD: set GITHUB_TOKEN=<your_token>

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def _build_session() -> requests.Session:
    """Build a :class:`requests.Session` with optional GitHub auth headers.

    Returns
    -------
    requests.Session
        Configured session instance.
    """
    session = requests.Session()
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        logger.info("Authenticated session created (token detected).")
    else:
        logger.warning(
            "No GITHUB_TOKEN found – using anonymous session (60 req/hr limit)."
        )
    session.headers.update(headers)
    return session


SESSION: requests.Session = _build_session()

# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------


def _get(url: str, params: dict | None = None) -> dict | list | None:
    """Perform a GET request against the GitHub API with retry / rate-limit logic.

    Parameters
    ----------
    url:
        Absolute URL to request.
    params:
        Optional query-string parameters.

    Returns
    -------
    dict | list | None
        Parsed JSON body, or ``None`` on unrecoverable error.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)

            if response.status_code == 404:
                logger.warning("Resource not found: %s", url)
                return None

            if response.status_code == 403:
                reset_ts: str = response.headers.get("X-RateLimit-Reset", "")
                remaining: str = response.headers.get("X-RateLimit-Remaining", "?")
                if remaining == "0" or "rate limit" in response.text.lower():
                    wait = _seconds_until_reset(reset_ts)
                    logger.warning(
                        "Rate limit exceeded. Sleeping %d s before retry %d/%d.",
                        wait,
                        attempt,
                        MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                logger.error("HTTP 403 Forbidden: %s", url)
                return None

            response.raise_for_status()
            time.sleep(RATE_LIMIT_SLEEP)
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(
                "Connection timeout on attempt %d/%d: %s", attempt, MAX_RETRIES, url
            )
        except requests.exceptions.ConnectionError:
            logger.error(
                "Network error on attempt %d/%d: %s", attempt, MAX_RETRIES, url
            )
        except requests.exceptions.JSONDecodeError:
            logger.error("Malformed JSON response from: %s", url)
            return None
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error [%s] from: %s", exc.response.status_code, url)
            return None

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP)

    logger.error("Exhausted %d retries for: %s", MAX_RETRIES, url)
    return None


def _seconds_until_reset(reset_timestamp: str) -> float:
    """Calculate how many seconds to sleep until a GitHub rate-limit window resets.

    Parameters
    ----------
    reset_timestamp:
        Unix timestamp string from the ``X-RateLimit-Reset`` header.

    Returns
    -------
    float
        Seconds to wait (minimum ``RETRY_SLEEP``).
    """
    try:
        reset_dt = datetime.fromtimestamp(int(reset_timestamp), tz=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        delta = (reset_dt - now).total_seconds()
        return max(delta + 5, RETRY_SLEEP)
    except (ValueError, TypeError):
        return RETRY_SLEEP

# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def get_user(username: str) -> dict:
    """Fetch raw profile data for a single GitHub user.

    Parameters
    ----------
    username:
        GitHub login name.

    Returns
    -------
    dict
        Subset of GitHub user fields relevant to churn analysis,
        or an empty dict if the user cannot be retrieved.
    """
    url = f"{GITHUB_API_BASE}/users/{username}"
    raw: dict | list | None = _get(url)

    if not raw or not isinstance(raw, dict):
        return {}

    return {
        "login": raw.get("login"),
        "id": raw.get("id"),
        "followers": raw.get("followers"),
        "following": raw.get("following"),
        "public_repos": raw.get("public_repos"),
        "public_gists": raw.get("public_gists"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "company": raw.get("company"),
        "location": raw.get("location"),
        "bio": raw.get("bio"),
    }


def get_user_repositories(username: str) -> list[dict]:
    """Fetch all public repositories for a GitHub user (handles pagination).

    Parameters
    ----------
    username:
        GitHub login name.

    Returns
    -------
    list[dict]
        List of repository records.  Returns an empty list on failure.
    """
    repos: list[dict] = []
    page: int = 1

    while True:
        url = f"{GITHUB_API_BASE}/users/{username}/repos"
        params: dict[str, int | str] = {
            "per_page": REPOS_PER_PAGE,
            "page": page,
            "type": "public",
        }
        batch: dict | list | None = _get(url, params=params)

        if batch is None:
            logger.warning("Failed to fetch repositories for user: %s", username)
            break

        if not isinstance(batch, list) or len(batch) == 0:
            break

        for repo in batch:
            repos.append(
                {
                    "repo_name": repo.get("name"),
                    "repo_id": repo.get("id"),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("updated_at"),
                    "pushed_at": repo.get("pushed_at"),
                    "stargazers_count": repo.get("stargazers_count"),
                    "forks_count": repo.get("forks_count"),
                    "watchers_count": repo.get("watchers_count"),
                    "language": repo.get("language"),
                    "size": repo.get("size"),
                    "open_issues_count": repo.get("open_issues_count"),
                    "default_branch": repo.get("default_branch"),
                }
            )

        if len(batch) < REPOS_PER_PAGE:
            break

        page += 1

    logger.info("Retrieved %d repositories for user: %s", len(repos), username)
    return repos


def _compute_activity_metrics(repositories: list[dict]) -> dict:
    """Derive raw activity counts from a user's repository list.

    No feature engineering is performed here – only direct counts.

    Parameters
    ----------
    repositories:
        List of repository dicts as returned by :func:`get_user_repositories`.

    Returns
    -------
    dict
        Raw activity metrics keyed by metric name.
    """
    total_stars: int = 0
    total_forks: int = 0
    active_repos: int = 0
    inactive_repos: int = 0
    now = datetime.now(tz=timezone.utc)
    activity_threshold_days: int = 180

    for repo in repositories:
        total_stars += repo.get("stargazers_count") or 0
        total_forks += repo.get("forks_count") or 0

        pushed_at_raw: str | None = repo.get("pushed_at")
        if pushed_at_raw:
            try:
                pushed_dt = datetime.fromisoformat(
                    pushed_at_raw.replace("Z", "+00:00")
                )
                days_since_push = (now - pushed_dt).days
                if days_since_push <= activity_threshold_days:
                    active_repos += 1
                else:
                    inactive_repos += 1
            except ValueError:
                inactive_repos += 1
        else:
            inactive_repos += 1

    return {
        "total_repositories": len(repositories),
        "total_stars_received": total_stars,
        "total_forks_received": total_forks,
        "active_repositories": active_repos,
        "inactive_repositories": inactive_repos,
    }


def collect_user_data(username: str) -> dict:
    """Collect the complete raw data record for a single GitHub user.

    Combines user profile, repository list, and raw activity metrics
    into a single flat dictionary ready for persistence.

    Parameters
    ----------
    username:
        GitHub login name.

    Returns
    -------
    dict
        Merged record, or an empty dict if the user profile is unavailable.
    """
    logger.info("Collecting user: %s", username)

    user_info = get_user(username)
    if not user_info:
        logger.warning("User not found or unreachable: %s", username)
        return {}

    repositories = get_user_repositories(username)
    activity_metrics = _compute_activity_metrics(repositories)

    record: dict = {
        **user_info,
        **activity_metrics,
        "repositories": repositories,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    return record


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_raw_data(data: list[dict], filename: str) -> None:
    """Persist a list of user records to JSON and CSV.

    Files are written to :data:`RAW_DATA_DIR`.  The ``repositories`` nested
    list is dropped from the CSV (it is retained in the JSON).

    Parameters
    ----------
    data:
        List of user record dicts produced by :func:`collect_user_data`.
    filename:
        Base filename **without** extension (e.g. ``"github_users"``).
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_path: Path = RAW_DATA_DIR / f"{filename}.json"
    csv_path: Path = RAW_DATA_DIR / f"{filename}.csv"

    # --- JSON (full nested structure) ---
    try:
        # If an existing JSON file is present, merge by `login` to avoid losing previous data.
        merged: dict[str, dict] = {}

        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                    if isinstance(existing, list):
                        for rec in existing:
                            login = rec.get("login")
                            if login:
                                merged[str(login).lower()] = rec
            except Exception:
                logger.warning("Existing JSON present but failed to load; overwriting.")

        # Add/replace with incoming records (new data takes precedence)
        for rec in data:
            login = rec.get("login")
            if login:
                merged[str(login).lower()] = rec

        merged_list = list(merged.values())

        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(merged_list, fh, ensure_ascii=False, indent=2, default=str)

        logger.info("Saved JSON (merged): %s  (%d total records)", json_path, len(merged_list))
    except OSError as exc:
        logger.error("Failed to write JSON [%s]: %s", json_path, exc)

    # --- CSV (flat, drop nested repositories list) ---
    try:
        flat_records = [
            {k: v for k, v in record.items() if k != "repositories"}
            for record in merged_list
        ]
        df = pd.DataFrame(flat_records)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("Saved CSV (merged): %s  (%d rows)", csv_path, len(df))
    except (OSError, ValueError) as exc:
        logger.error("Failed to write CSV [%s]: %s", csv_path, exc)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


SEARCH_USERS_URL: str = f"{GITHUB_API_BASE}/search/users"


def load_usernames(path: str) -> list[str]:
    """Load GitHub usernames from a plain-text file (one username per line).

    Lines beginning with ``#`` and blank lines are silently ignored.

    Parameters
    ----------
    path:
        File-system path to the usernames text file.

    Returns
    -------
    list[str]
        Deduplicated list of usernames in file order.
    """
    source = Path(path)
    if not source.exists():
        logger.error("Usernames file not found: %s", path)
        return []

    usernames: list[str] = []
    seen: set[str] = set()

    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lower = stripped.lower()
            if lower not in seen:
                seen.add(lower)
                usernames.append(stripped)

    logger.info("Loaded %d unique usernames from %s", len(usernames), path)
    return usernames


def search_users_by_location(location: str, max_pages: int = 5) -> list[str]:
    """Search GitHub users by location using the REST search API.

    Parameters
    ----------
    location:
        Location value for the GitHub search query.
    max_pages:
        Maximum number of search pages to request.

    Returns
    -------
    list[str]
        List of GitHub login names from the search results.
    """
    usernames: list[str] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        params = {
            "q": f"location:{location}",
            "per_page": 100,
            "page": page,
        }
        response = _get(SEARCH_USERS_URL, params=params)
        if not response or not isinstance(response, dict):
            logger.warning("Search request failed for location: %s", location)
            break

        items = response.get("items")
        if not isinstance(items, list) or len(items) == 0:
            break

        for item in items:
            login = item.get("login")
            if isinstance(login, str):
                lower = login.lower()
                if lower not in seen:
                    seen.add(lower)
                    usernames.append(login)

        if len(items) < 100:
            break

        total_count = response.get("total_count")
        if isinstance(total_count, int) and len(usernames) >= total_count:
            break

    logger.info(
        "Found %d unique users for location '%s' via GitHub search.",
        len(usernames),
        location,
    )
    return usernames


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_collection(usernames: list[str], checkpoint: int = CHECKPOINT_FREQUENCY) -> pd.DataFrame:
    """Run the full data-collection pipeline for a list of GitHub usernames.

    Parameters
    ----------
    usernames:
        GitHub login names to process.
    checkpoint:
        Save partial results after this many successful user records.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame of collected records (repositories column excluded).
        Failed users produce no row.
    """
    collected: list[dict] = []
    failures: int = 0

    for username in usernames:
        record = collect_user_data(username)
        if record:
            collected.append(record)
        else:
            failures += 1

        if len(collected) > 0 and len(collected) % checkpoint == 0:
            logger.info("Checkpoint reached at %d users, saving partial data.", len(collected))
            save_raw_data(collected, "github_users")

    if collected:
        logger.info("Final save of collected data.")
        save_raw_data(collected, "github_users")

    flat_records = [
        {k: v for k, v in r.items() if k != "repositories"} for r in collected
    ]
    df = pd.DataFrame(flat_records)

    total_repos: int = sum(
        r.get("total_repositories", 0) for r in collected
    )

    logger.info("=" * 50)
    logger.info("Collection complete.")
    logger.info("Users processed : %d", len(collected))
    logger.info("Repositories    : %d", total_repos)
    logger.info("Failures        : %d", failures)
    logger.info("=" * 50)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape GitHub user activity data for the Churn Predictor project."
    )
    parser.add_argument(
        "--location",
        type=str,
        default="argentina",
        help="GitHub location to search for users (default: argentina).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to a text file with one GitHub username per line. Overrides location search if provided.",
    )
    args = parser.parse_args()

    if args.input:
        usernames_list = load_usernames(args.input)
    else:
        usernames_list = search_users_by_location(args.location)

    if not usernames_list:
        logger.error("No usernames to process. Exiting.")
        sys.exit(1)

    dataframe = run_collection(usernames_list)

    # Summary statistics printed to stdout
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Users processed     : {len(dataframe)}")
    if not dataframe.empty and "total_repositories" in dataframe.columns:
        print(f"Repositories total  : {int(dataframe['total_repositories'].sum())}")
        print(f"Avg repos / user    : {dataframe['total_repositories'].mean():.1f}")
        print(f"Avg followers       : {dataframe['followers'].mean():.1f}")
        print(f"Active repos total  : {int(dataframe['active_repositories'].sum())}")
        print(f"Inactive repos total: {int(dataframe['inactive_repositories'].sum())}")
    print("=" * 50)
    print(f"\nRaw data saved to : {RAW_DATA_DIR.resolve()}")