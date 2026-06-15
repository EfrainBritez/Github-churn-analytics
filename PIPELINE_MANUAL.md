# GitHub Churn Analytics Pipeline Manual

This guide shows how to run the complete data pipeline from querying GitHub to building filtered feature sets for machine learning. It documents recent updates: automatic `.env` loading, scraper checkpointing and merging behavior, and standardised filtered/model output paths.

---

## Pipeline Overview

```
1. Query (scraper.py)
   ↓
   data/raw/github_users.json   (merged, appended)
   ↓
2. Features (features.py)
   ↓
   data/features/github_features.csv
   ↓
3. Labels (label_builder.py)
   ↓
   data/processed/labeled_dataset.csv
   ↓
4. Filter (filter_selection.py | rfe_selection.py | rf_selection.py | dt_selection.py)
   ↓
   data/filtered/*.csv
   ↓
5. Train / Model (model.py)
   ↓
   data/filtered/model.pkl
```

---

## Step 1: Query GitHub Data

**Script:** `scraper.py`

**Purpose:** Fetch GitHub user data from the GitHub REST API (default: `location:paraguay`).

### Key behaviors (updated)

- `.env` is automatically loaded at startup if present in the repo root. Use it to store `GITHUB_TOKEN` (this file is ignored by Git via `.gitignore`).
- The scraper saves a checkpointed merged JSON/CSV instead of blindly overwriting:
  - New runs are merged into `data/raw/github_users.json` by `login` (new records override existing ones).
  - The flat CSV `data/raw/github_users.csv` is regenerated from the merged JSON.
- Periodic checkpoint saves occur every 50 successfully collected users to avoid losing progress.

### Basic Usage

```bash
python scraper.py
```

### Advanced Usage

Search by different location:

```bash
python scraper.py --location "san francisco"
```

Load from a custom usernames file instead of searching:

```bash
python scraper.py --input usernames.txt
```

### Environment Setup

Place your GitHub Personal Access Token in one of these places:

- Environment variable (PowerShell):

```powershell
$env:GITHUB_TOKEN = "your_personal_access_token"
```

- Environment variable (CMD):

```cmd
set GITHUB_TOKEN=your_personal_access_token
```

- Or add a `.env` file at the repo root with a single line:

```
GITHUB_TOKEN=your_personal_access_token
```

The script will automatically load `.env` (without overwriting existing environment variables).

### Output

- `data/raw/github_users.json` — merged raw user records (updated on each run)
- `data/raw/github_users.csv` — flat user records derived from the merged JSON
- Checkpoints are written every 50 users during long runs

---

## Step 2: Engineer Features

**Script:** `features.py`

**Purpose:** Transform raw GitHub data into ML-ready features.

### Basic Usage

```bash
python features.py
```

### How It Works

Reads from: `data/raw/github_users.json`

Generates these features:
- `days_since_last_activity`
- `account_age_days`
- `repos_per_year`
- `followers_following_ratio`
- `active_repo_ratio`
- `inactive_repo_ratio`
- `avg_stars_per_repo`
- `avg_forks_per_repo`
- `repo_activity_density`
- `repository_maintenance_ratio`

### Output

- `data/features/github_features.csv`
- `data/features/github_features.parquet`

---

## Step 3: Create Churn Labels

**Script:** `label_builder.py`

**Purpose:** Add churn labels using the business rule (default 180 days).

### Basic Usage

```bash
python label_builder.py
```

### How It Works

Reads from `data/features/github_features.csv` (or the Parquet equivalent) and outputs labeled datasets:

- `data/processed/labeled_dataset.csv`
- `data/processed/labeled_dataset.parquet`

Churn rule:

```
churn = 1  if  days_since_last_activity > 180
churn = 0  otherwise
```

---

## Step 4: Feature Filtering Options

The repository provides several feature-selection scripts. All of them default to writing outputs into `data/filtered/` and will create the directory automatically.

### 4a: Variance & Correlation Filter (`filter_selection.py`)

```bash
python filter_selection.py \
  --input data/processed/labeled_dataset.csv \
  --target churn
```

- Default thresholds: `--variance 0.01` and `--correlation 0.90`.
- Use `--output` to change the output filename; by default the script writes to the provided path (recommended under `data/filtered/`).

### 4b: Recursive Feature Elimination (`rfe_selection.py`)

```bash
python rfe_selection.py \
  --input data/processed/labeled_dataset.csv \
  --target churn \
  --task classification \
  --estimator rf \
  --features 6
```

- Default output: `data/filtered/rfe_selected.csv` and `data/filtered/rfe_rankings.csv`.

### 4c: Random Forest Importance (`rf_selection.py`)

```bash
python rf_selection.py \
  --input data/processed/labeled_dataset.csv \
  --target churn \
  --task classification \
  --top-k 6
```

- Default output: `data/filtered/rf_selected.csv` and `data/filtered/rf_feature_importance.csv`.
- Use `--min-importance` to keep features above a given importance threshold.

### 4d: Decision Tree Importance (`dt_selection.py`)

```bash
python dt_selection.py \
  --input data/processed/labeled_dataset.csv \
  --target churn \
  --task classification \
  --top-k 6
```

- Default output: `data/filtered/dt_selected.csv` and `data/filtered/dt_feature_importance.csv`.

---

## Step 5: Train Model

**Script:** `model.py`

**Purpose:** Train a Random Forest model on a filtered feature set and save the trained model to `data/filtered/model.pkl`.

### Basic Usage (example uses RF-selected features):

```bash
python model.py
```

This script will read the example filtered CSV (by default `data/filtered/rf_selected.csv`), train a Random Forest classifier, and save the trained model to:

- `data/filtered/model.pkl`

### Programmatic training (one-liner):

```bash
python -c "from model import train, save_model; m=train('data/filtered/rf_selected.csv','churn'); save_model(m)"
```

---

## API / Prediction

**Script:** `api.py` (example usage file)

- `api.py` now constructs a prediction payload using the RF-selected feature names by default (see `data/filtered/rf_selected.csv` for the exact column list).
- `model.predict()` has been hardened to align incoming feature dicts with the model's `feature_names_in_` (it fills missing features with zeros and drops extras), but you should pass the expected feature names when possible.

### Quick test

```bash
python api.py
```

---

## Output Directory Structure (updated)

```
data/
├── raw/
│   ├── github_users.json       ← Merged raw data (appended by login)
│   └── github_users.csv
├── features/
│   ├── github_features.csv
│   └── github_features.parquet
├── processed/
│   ├── labeled_dataset.csv
│   └── labeled_dataset.parquet
└── filtered/
    ├── filtered_features.csv   ← Variance & correlation filtered
    ├── rfe_selected.csv        ← RFE selected features
    ├── rfe_rankings.csv        ← RFE rankings
    ├── rf_selected.csv         ← Random Forest selected features
    ├── rf_feature_importance.csv
    ├── dt_selected.csv         ← Decision Tree selected features
    ├── dt_feature_importance.csv
    └── model.pkl               ← Trained model (example location)
```

---

## Troubleshooting (updated)

- "No usernames to process": search returned no users — try a different `--location` or supply `--input` usernames.
- "No feature dataset found": run `features.py` first and verify `data/raw/github_users.json` exists.
- "Target column not found": verify `--target churn` and the CSV header.
- "Rate limit exceeded": set `GITHUB_TOKEN` in env or `.env`.
- "Feature names mismatched at prediction": retrain the model using the exact filtered CSV you'll use in production (e.g. `data/filtered/rf_selected.csv`) so `feature_names_in_` matches the API payload.

---

## Tips

- Keep a copy of your `.env` locally; never commit it to the repository.
- Use `--top-k` or `--min-importance` on the RF/DT/RFE scripts to create compact, reproducible feature sets for model training.
- If you want incremental raw-data versioning instead of merging, I can add an option to the scraper to write timestamped snapshots.
