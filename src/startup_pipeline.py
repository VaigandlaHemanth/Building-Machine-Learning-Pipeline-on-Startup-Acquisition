"""End-to-end ML pipeline for predicting startup status from Crunchbase data.

The original notebook stopped at EDA/cleaning and a hand-written labelling rule;
no model was ever trained. This module supplies the actual pipeline:

    load_data -> build_pipeline -> train_and_evaluate

Target: the dataset's ``status`` column (typically operating / acquired /
closed / ipo). The pipeline is dataset-agnostic: it auto-detects numeric vs
categorical features, drops high-cardinality identifier-like columns, imputes
and scales/encodes inside a single sklearn Pipeline (so there is no train/test
leakage), and handles class imbalance with balanced class weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that are identifiers / free text / leak the label — never features.
DROP_ALWAYS = {
    "name", "normalized_name", "permalink", "domain", "homepage_url",
    "twitter_username", "logo_url", "logo_width", "logo_height",
    "short_description", "description", "overview", "tag_list", "id", "entity_id",
    "created_by", "created_at", "updated_at", "first_investment_at",
    "last_investment_at", "first_funding_at", "last_funding_at",
    "first_milestone_at", "last_milestone_at", "closed_at",
}


def load_data(path: str, target: str = "status") -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV and return (features X, target y), rows with a known label only."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(
            f"target column {target!r} not found. Available: {list(df.columns)[:20]}..."
        )
    df = df[df[target].notna()].copy()
    y = df[target].astype(str)
    X = df.drop(columns=[target])
    return X, y


def select_features(X: pd.DataFrame, max_cardinality: int = 40):
    """Split usable columns into numeric and low-cardinality categorical lists.

    High-cardinality text/ID columns are dropped: one-hot encoding them explodes
    dimensionality and they are usually identifiers, not signal.
    """
    cols = [c for c in X.columns if c not in DROP_ALWAYS]
    numeric, categorical = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric.append(c)
        elif X[c].nunique(dropna=True) <= max_cardinality:
            categorical.append(c)
        # else: high-cardinality string column -> dropped
    return numeric, categorical


def build_pipeline(numeric, categorical, model):
    """A single Pipeline: impute + scale/encode -> classifier (no leakage)."""
    numeric_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])
    pre = ColumnTransformer([
        ("num", numeric_tf, numeric),
        ("cat", categorical_tf, categorical),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def models():
    """The classifiers we compare (balanced for the imbalanced status classes)."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def train_and_evaluate(X, y, test_size=0.2, random_state=42, verbose=True):
    """Stratified split, fit each model, return {name: metrics dict}."""
    numeric, categorical = select_features(X)
    if not numeric and not categorical:
        raise ValueError("no usable feature columns found after selection")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    for name, model in models().items():
        pipe = build_pipeline(numeric, categorical, model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = (y_pred == y_test.values).mean()
        results[name] = {
            "pipeline": pipe,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "labels": sorted(y.unique()),
        }
        if verbose:
            print(f"\n=== {name} ===")
            print(f"accuracy: {acc:.3f}   macro-F1: {macro_f1:.3f}")
            print(results[name]["report"])

    return results, (numeric, categorical)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="companies.csv")
    p.add_argument("--target", default="status")
    args = p.parse_args()

    X, y = load_data(args.data, target=args.target)
    print(f"Loaded {len(X):,} rows; class balance:\n{y.value_counts()}\n")
    train_and_evaluate(X, y)
