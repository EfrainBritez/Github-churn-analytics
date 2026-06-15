# rfe_selection.py

import argparse
from pathlib import Path

import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def get_estimator(task_type, estimator_name):
    """
    Returns the estimator to use for RFE.
    """

    if task_type == "classification":

        if estimator_name == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

        elif estimator_name == "logistic":
            return LogisticRegression(
                max_iter=5000,
                random_state=42
            )

    elif task_type == "regression":

        if estimator_name == "rf":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

        elif estimator_name == "linear":
            return LinearRegression()

    raise ValueError(
        f"Unsupported combination: task={task_type}, estimator={estimator_name}"
    )


def run_rfe(X, y, estimator, n_features):
    """
    Perform Recursive Feature Elimination.
    """

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=1
    )

    selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()

    rankings = pd.DataFrame({
        "feature": X.columns,
        "ranking": selector.ranking_
    }).sort_values("ranking")

    return selected_features, rankings


def main():

    parser = argparse.ArgumentParser(
        description="Recursive Feature Elimination (RFE)"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Problem type"
    )

    parser.add_argument(
        "--estimator",
        default="rf",
        help="rf, logistic, or linear"
    )

    parser.add_argument(
        "--features",
        type=int,
        required=True,
        help="Number of features to keep"
    )

    parser.add_argument(
        "--output",
        default="data/filtered/rfe_selected.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    print("Loading dataset...")

    df = pd.read_csv(args.input)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found."
        )

    y = df[args.target]

    X = df.drop(columns=[args.target])

    # Use only numeric columns
    X = X.select_dtypes(include=["number"])

    print(f"Initial features: {X.shape[1]}")

    if args.features > X.shape[1]:
        raise ValueError(
            f"Requested {args.features} features but only "
            f"{X.shape[1]} available."
        )

    estimator = get_estimator(
        args.task,
        args.estimator
    )

    print("Running RFE...")

    selected_features, rankings = run_rfe(
        X,
        y,
        estimator,
        args.features
    )

    print(f"Selected {len(selected_features)} features")

    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered dataset
    result = X[selected_features].copy()
    result[args.target] = y

    result.to_csv(args.output, index=False)

    # Save rankings
    rankings_file = output_dir / "rfe_rankings.csv"
    rankings.to_csv(rankings_file, index=False)

    print(f"\nSaved selected dataset: {args.output}")
    print(f"Saved rankings: {rankings_file}")


if __name__ == "__main__":
    main()