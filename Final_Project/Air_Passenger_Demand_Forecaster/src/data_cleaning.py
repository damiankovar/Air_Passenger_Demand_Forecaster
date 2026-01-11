# Cleaning helpers so the models don't freak out due to messy data.

import numpy as np
import pandas as pd

from .config import CONFIG


def handle_missing_values(df):
    """Fill gaps in each route's passenger series."""
    # Interpolate within each route, then forward/back fill anything left.

    df = df.copy()
    df["passengers"] = (
        df.groupby(["origin_airport", "destination_airport"])["passengers"]
        .apply(lambda series: series.interpolate(method="linear").ffill().bfill())
        .reset_index(level=[0, 1], drop=True)
    )
    return df


def detect_anomalies(series, z_thresh=3.5):
    # Modified z-score is nice because it uses the median instead of mean.
    median = series.median()
    mad = np.median(np.abs(series-median)) or 1.0
    modified_z = 0.6745*(series-median)/mad
    return np.abs(modified_z) > z_thresh


def cap_anomalies(df, z_thresh=3.5):
    # I go route by route so they don't influence each other.
    capped = df.copy()
    for (origin, dest), group in capped.groupby(["origin_airport", "destination_airport"]):
        mask = detect_anomalies(group["passengers"], z_thresh=z_thresh)
        if mask.any():
            replacement = group["passengers"].rolling(window=3, center=True, min_periods=1).median()
            capped.loc[group.index[mask], "passengers"] = replacement.loc[group.index[mask]]
    return capped


def clean_passenger_data(
    df,
    config=CONFIG,
):
    """Run the full cleaning pipeline."""

    cleaned = handle_missing_values(df)
    cleaned = cap_anomalies(cleaned)
    return cleaned.reset_index(drop=True)



