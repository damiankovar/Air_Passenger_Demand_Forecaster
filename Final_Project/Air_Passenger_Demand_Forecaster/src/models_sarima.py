'''Model definitions and training.

SARIMA models using statsmodels.

I am using SARIMA as a time-series model with seasonality to compare with the ML models.
'''


from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from .config import CONFIG, ModelConfig


def fit_sarima_model(
    series: pd.Series,
    model_config: ModelConfig,
) -> SARIMAXResults:
    """Fit a SARIMA model on one route's time series."""

    # I use the orders from the config. If they are not perfect, at least 
    # they give a reasonable starting point for monthly air-traffic data.

    model = SARIMAX(
        series,
        order=model_config.sarima_order,
        seasonal_order=model_config.sarima_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    return fit


def forecast_route(
    fitted_model: SARIMAXResults,
    steps: int,
) -> pd.Series:
    forecast = fitted_model.forecast(steps=steps)
    #It is important that we don't predict negative passengers (that's impossible!)
    forecast = forecast.clip(lower=0.0)
    return forecast


def seasonal_naive_forecast(
    series: pd.Series,
    steps: int,
    seasonality: int = 12,
) -> pd.Series:

    """Seasonal naive forecast: predict using same month from previous year.
    
    For each forecast step h=1..steps:
        y_hat(t+h) = y_true(t+h-12)  (same month one year earlier)
    
    If y_true(t+h-12) is missing (not enough history), it resorts to
    y_hat(t+h) = y_true(t) (last observed value).
    
    Args:
        series: time series with date index and passenger values
        steps: number of steps ahead to forecast
        seasonality: seasonal period (12 for monthly data)
    
    Returns:
        Forecast series with same index as would be expected
    """
    if len(series) < seasonality:
        # Not enough history for seasonal naive, use last value
        last_value = series.iloc[-1] if len(series) > 0 else 0.0
        return pd.Series([max(0.0, last_value)] * steps)
    
    # Get the last date to determine future dates
    last_date = series.index[-1]
    
    # For each step:
    # For step h: predict month t+h using month (t+h-12) = same month one year earlier
    # If t is the last training month (index len-1), then:
    #   h=1: use index (len-1) - 12 + 1 = len - 12 (12 months ago)
    #   h=2: use index (len-1) - 12 + 2 = len - 11 
    #   ...
    #   h=6: use index (len-1) - 12 + 6 = len - 7 
    
    forecasts = []
    for h in range(1, steps + 1):
        if len(series) >= seasonality:
            # Calculate index: for step h, use value from (seasonality - h) months before the last
            # This gives us the same month from one year earlier
            idx = len(series) - seasonality + (h - 1)
            if idx >= 0 and idx < len(series):
                pred = series.iloc[idx]
            else:
                # Fallback: use value from exactly seasonality months ago
                idx = len(series) - seasonality
                if idx >= 0:
                    pred = series.iloc[idx]
                else:
                    pred = series.iloc[-1]
        else:
            # Not enough history for seasonal naive, use last observed value
            pred = series.iloc[-1]
        
        # Ensure non-negative
        forecasts.append(max(0.0, pred))
    
    return pd.Series(forecasts)
