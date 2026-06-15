# rf_selection.py

import argparse
from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


def get_model(task_type, n_estimators=100):

    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

    elif task_type == "regression":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Random Forest Feature Selection"
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
        "--top-k",
        type=int,
        default=None,
        help="Keep top K features"
    )

    parser.add_argument(
        "--min-importance",
        type=float,
        default=None,
        help="Minimum feature importance threshold"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest"
    )

    parser.add_argument(
        "--output",
        default="data/filtered/rf_selected.csv",
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

    # Numeric features only
    X = X.select_dtypes(include=["number"])

    print(f"Initial features: {X.shape[1]}")

    model = get_model(
        args.task,
        args.n_estimators
    )

    print(
        f"Training Random Forest ({args.n_estimators} trees)..."
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nFeature Importance Ranking:")
    print(importance_df)

    # Feature selection
    if args.top_k is not None:

        selected_features = (
            importance_df
            .head(args.top_k)["feature"]
            .tolist()
        )

        print(
            f"\nSelecting Top {args.top_k} features"
        )

    elif args.min_importance is not None:

        selected_features = (
            importance_df[
                importance_df["importance"]
                >= args.min_importance
            ]["feature"]
            .tolist()
        )

        print(
            f"\nSelecting features with "
            f"importance >= {args.min_importance}"
        )

    else:
        raise ValueError(
            "Specify either --top-k or --min-importance"
        )

    print(
        f"\nSelected {len(selected_features)} features"
    )

    print("\nSelected Features:")

    for feature in selected_features:
        print(feature)

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reduced dataset
    result = X[selected_features].copy()
    result[args.target] = y

    result.to_csv(
        args.output,
        index=False
    )

    # Save importance ranking
    ranking_file = output_dir / "rf_feature_importance.csv"

    importance_df.to_csv(
        ranking_file,
        index=False
    )

    print(f"\nSaved dataset: {args.output}")
    print(f"Saved ranking: {ranking_file}")


if __name__ == "__main__":
    main()