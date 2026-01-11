# Tests for the models (ML + SARIMA).
# Idea: check that they can train and forecast on simple fake data.

import pandas as pd

from src import models_ml, models_sarima
from src.config import ProjectConfig


def dummy_feature_set():
    # Fake features for 36 months on one route.
    # Values are simple 0..35, just to have something numeric.
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame(
        {
            "date": dates,
            "origin_airport": ["ZRH"] * 36,
            "destination_airport": ["LHR"] * 36,
            "passengers": range(36),
            "lag_1": range(36),
            "lag_3": range(36),
            "lag_6": range(36),
        }
    )
    return df


def test_ml_training(tmp_path):
    # Check that the three ML models can train without errors.
    df = dummy_feature_set()
    config = ProjectConfig(
        base_dir=tmp_path,
        data_path=tmp_path / "data.csv",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "results" / "figures",
        models_dir=tmp_path / "results" / "models",
    )

    X = df[["lag_1", "lag_3", "lag_6"]]
    y = df["passengers"]

    models = models_ml.train_models(X, y, config=config)

    # I expect the three main models to be there
    assert set(models.keys()) == {"random_forest", "mlp", "xgboost"}


def test_sarima_forecast():
    # Fit a SARIMA model and ask it to predict a few months ahead.
    df = dummy_feature_set()
    series = df.set_index("date")["passengers"]
    config = ProjectConfig()

    fitted = models_sarima.fit_sarima_model(series, config.model_config)
    forecast = models_sarima.forecast_route(fitted, steps=3)

    # we just check the length, not the accuracy
    assert len(forecast) == 3
