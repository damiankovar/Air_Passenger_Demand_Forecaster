# Tests for the evaluation helpers.
# I only check that metrics are returned and look reasonable.

import numpy as np

from src import evaluation


def test_evaluate_predictions():
    # Simple fake predictions that are slightly off.
    y_true = [100, 120, 140]
    y_pred = [110, 115, 130]

    metrics = evaluation.evaluate_predictions(y_true, y_pred)

    # we expect these three keys at least
    assert set(metrics.keys()) == {"mae", "rmse", "mape"}
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0


