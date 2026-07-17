"""Offline tests — synthetic data, no companies.csv needed. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.startup_pipeline import (
    DROP_ALWAYS, build_pipeline, models, select_features, train_and_evaluate,
)


def _synthetic(n=400, seed=0):
    """A frame that mimics the Crunchbase schema: numeric funding, a category,
    a high-cardinality id/name, and a learnable `status` target."""
    rng = np.random.default_rng(seed)
    funding = rng.exponential(1.0, n)
    rounds = rng.integers(0, 6, n)
    cats = rng.choice(["web", "games_video", "biotech", "mobile"], n)
    # status is correlated with funding so a model can actually learn something
    status = np.where(funding > 1.5, "acquired", "operating")
    status = np.where(funding < 0.2, "closed", status)
    X = pd.DataFrame({
        "funding_total_usd": funding,
        "funding_rounds": rounds,
        "category_code": cats,
        "name": [f"company_{i}" for i in range(n)],   # high-cardinality -> dropped
        "twitter_username": [f"@u{i}" for i in range(n)],  # in DROP_ALWAYS
    })
    return X, pd.Series(status, name="status")


def test_select_features_drops_ids_and_high_cardinality():
    X, _ = _synthetic()
    numeric, categorical = select_features(X)
    assert "funding_total_usd" in numeric
    assert "funding_rounds" in numeric
    assert "category_code" in categorical
    assert "name" not in numeric + categorical          # high-cardinality dropped
    assert "twitter_username" not in numeric + categorical  # in DROP_ALWAYS


def test_pipeline_builds_fits_predicts():
    X, y = _synthetic()
    numeric, categorical = select_features(X)
    pipe = build_pipeline(numeric, categorical, models()["logistic_regression"])
    pipe.fit(X, y)
    preds = pipe.predict(X.head(10))
    assert len(preds) == 10
    assert set(preds).issubset(set(y.unique()))


def test_train_and_evaluate_returns_metrics_and_beats_random():
    X, y = _synthetic(n=600)
    results, (numeric, categorical) = train_and_evaluate(X, y, verbose=False)
    assert set(results) == {"logistic_regression", "random_forest"}
    for name, m in results.items():
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        # target is a deterministic function of funding, so a real model should
        # comfortably beat the majority-class baseline (~0.5 here)
        assert m["accuracy"] > 0.7, f"{name} underperformed: {m['accuracy']}"


def test_missing_target_raises():
    X, y = _synthetic()
    df = X.copy()
    df.to_csv  # noqa - just ensuring import path is used elsewhere
    from src.startup_pipeline import load_data
    tmp = "._tmp_no_status.csv"
    X.assign(other=1).to_csv(tmp, index=False)
    with pytest.raises(KeyError):
        load_data(tmp, target="status")
    import os
    os.remove(tmp)
