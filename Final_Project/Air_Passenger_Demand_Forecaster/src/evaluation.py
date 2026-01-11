"""Model evaluation and visualization.

Regression and time-series metrics for model comparison, plus helper tools

to summarise results.

"""

import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, but skip routes with really low passenger counts.
    
    If a route has less than 1 passenger/month, the percentage errors get crazy
    (like 1000% error). So I just skip those and return NaN instead.
    """
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    
    # 1 passenger is used as minimum threshold (not machine epsilon)
    # This makes sense because we can't have 0.001 passengers anyway
    min_threshold = 1.0
    
    # Only calculate MAPE for months where we had meaningful passenger counts
    # This avoids crazy percentages from routes that were discontinued or had near-zero traffic
    meaningful_mask = np.abs(y_true) >= min_threshold
    
    # If no meaningful values, return NaN (can't calculate meaningful percentage)
    if meaningful_mask.sum() == 0:
        return np.nan
    
    # Calculate MAPE only for meaningful values
    errors = np.abs(y_true[meaningful_mask]-y_pred[meaningful_mask])
    actuals = np.abs(y_true[meaningful_mask])
    mape = np.mean(errors/actuals)*100

    return mape


def root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y_true-y_pred)**2)))


def mean_absolute_error(y_true, y_pred):
    # Compute mean absolute error
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_predictions(
    y_true,
    y_pred,
):
    """Return a dict with MAE, RMSE, and MAPE scores."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }