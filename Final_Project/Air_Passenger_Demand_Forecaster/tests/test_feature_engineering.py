# These tests just sanity check that feature creation behaves.

import pandas as pd

from src import feature_engineering
from src.config import ProjectConfig


def sample_cleaned_df():
    # Fake 24-month series so lag features have enough history.
    dates = pd.date_range("2021-01-01", periods=24, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "origin_airport": ["ZRH"] * 24,
            "destination_airport": ["LHR"] * 24,
            "passengers": range(24),
        }
    )


def test_prepare_features_generates_lags():
    # Lags should exist and not be NaN once enough history is there.
    df = sample_cleaned_df()
    config = ProjectConfig(routes=[("ZRH", "LHR")])
    feats = feature_engineering.prepare_features(df, config=config)
    lag_cols = [col for col in feats.columns if col.startswith("lag_")]
    assert lag_cols, "Expected lag features to be created."
    assert feats[lag_cols].notna().all().all()


def test_split_features_target_shapes():
    # X and y should align, and passengers shouldn't live in X.
    df = sample_cleaned_df()
    config = ProjectConfig(routes=[("ZRH", "LHR")])
    feats = feature_engineering.prepare_features(df, config=config)
    X, y = feature_engineering.split_features_target(feats)
    assert len(X) == len(y)
    assert "passengers" not in X.columns

