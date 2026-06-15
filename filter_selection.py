# filter_selection.py

import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Feature used to generate labels - must be excluded from training
LEAKAGE_FEATURE = "days_since_last_activity"


def variance_threshold_selection(X, threshold=0.01):
    """
    Remove features with variance below the threshold.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    selected_features = X.columns[selector.get_support()].tolist()

    return X[selected_features], selected_features


def correlation_filter(X, threshold=0.9):
    """
    Remove one feature from highly correlated pairs.
    """
    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    X_filtered = X.drop(columns=to_drop)

    return X_filtered, to_drop


def main():
    parser = argparse.ArgumentParser(
        description="Feature selection using Variance Threshold and Correlation Matrix"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--variance",
        type=float,
        default=0.01,
        help="Variance threshold"
    )

    parser.add_argument(
        "--correlation",
        type=float,
        default=0.90,
        help="Correlation threshold"
    )

    parser.add_argument(
        "--output",
        default="data/filtered/filtered_features.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")

    y = df[args.target]

    X = df.drop(columns=[args.target])

    # Remove leakage feature if present
    if LEAKAGE_FEATURE in X.columns:
        print(f"Removing leakage feature: {LEAKAGE_FEATURE}")
        X = X.drop(columns=[LEAKAGE_FEATURE])

    # Keep numeric features only
    X = X.select_dtypes(include=["number"])

    print(f"Initial features: {X.shape[1]}")

    # Variance Threshold
    X_var, kept_variance = variance_threshold_selection(
        X,
        threshold=args.variance
    )

    print(
        f"After variance threshold ({args.variance}): "
        f"{X_var.shape[1]} features"
    )

    # Correlation Filter
    X_final, removed_corr = correlation_filter(
        X_var,
        threshold=args.correlation
    )

    print(
        f"After correlation filtering ({args.correlation}): "
        f"{X_final.shape[1]} features"
    )

    print(f"Removed by correlation: {len(removed_corr)}")

    # Reattach target
    result = X_final.copy()
    result[args.target] = y

    result.to_csv(args.output, index=False)

    print(f"Saved filtered dataset to: {args.output}")

    print("\nSelected Features:")
    for feature in X_final.columns:
        print(feature)


if __name__ == "__main__":
    main()