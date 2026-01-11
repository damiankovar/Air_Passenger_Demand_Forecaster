# Functions for creating features from the passenger data.
# I keep everything in one place so it's easier to understand what is going on.

import pandas as pd

from .config import CONFIG


def sort_by_route(df):
    """
    I still keep a tiny docstring here because it's really central.
    This just sorts by route and date so that lags and rolling windows make sense.
    """
    return (
        df.sort_values(["origin_airport", "destination_airport", "date"])
        .reset_index(drop=True)
    )


def add_time_features(df, feature_config):
    # Time-based features: month, year, and season.
    # Air travel is very seasonal so these are important.
    result = df.copy()

    if feature_config.include_month:
        result["month"] = result["date"].dt.month

    if feature_config.include_year:
        result["year"] = result["date"].dt.year

    if feature_config.include_season:
        # quick mapping from month number to season name
        month_to_season = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
        result["season"] = result["date"].dt.month.map(month_to_season)

    return result


def add_lag_features(df, lags):
    # Lag features = how many passengers there were X months ago.
    # I do this by route so Zurich–London does not mix with Geneva–London.
    result = df.copy()

    for lag in lags:
        col_name = f"lag_{lag}"
        result[col_name] = (
            result.groupby(["origin_airport", "destination_airport"])["passengers"]
            .shift(lag)
        )

    return result


def add_rolling_features(df, windows):
    # Rolling mean and std over a few months.
    result = df.copy()
    grouped = result.groupby(["origin_airport", "destination_airport"])["passengers"]

    for window in windows:
        mean_col = f"roll_mean_{window}"
        std_col = f"roll_std_{window}"

        result[mean_col] = grouped.transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
        )
        # std can be NaN for very short series, so I just replace it with 0
        result[std_col] = grouped.transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).std()
        ).fillna(0.0)

    return result


def encode_categoricals(df, feature_config, drop_first=False):
    # Convert text columns like airport codes into dummy variables
    result = df.copy()
    categorical_cols = []

    if feature_config.include_airport_encoding:
        categorical_cols.append("origin_airport")
    if feature_config.include_destination_encoding:
        categorical_cols.append("destination_airport")

    if feature_config.include_season and "season" in result.columns:
        categorical_cols.append("season")

    if not categorical_cols:
        return result

    dummies = pd.get_dummies(
        result[categorical_cols],
        drop_first=drop_first,
        prefix=categorical_cols,
    )

    result = pd.concat(
        [result.drop(columns=categorical_cols), dummies],
        axis=1,
    )

    return result


def drop_na_rows(df, min_required_lags=12):
    # When I create lags, the first rows have NaN because there is no history.
    # Here I drop the rows that have too many missing feature values.
    result = df.copy()

    # I don't look at date / airport columns when deciding what to drop.
    non_feature_cols = {"date", "origin_airport", "destination_airport"}
    feature_cols = [c for c in result.columns if c not in non_feature_cols]


    # "a row must have at least this many non-NaN values among the features".
    filtered = result.dropna(subset=feature_cols, thresh=min_required_lags)

    return filtered.reset_index(drop=True)


def prepare_features(df, config=CONFIG):
    # This is the main function I use in the pipeline.
    # It calls the different feature steps in a fixed order.

    feature_cfg = config.feature_config

    # Always start sorted by route and date
    feats = sort_by_route(df)

    # Time info
    feats = add_time_features(feats, feature_cfg)

    # Lags (for example: last month, last year)
    feats = add_lag_features(feats, feature_cfg.lag_features)

    # Rolling averages and std
    feats = add_rolling_features(feats, feature_cfg.rolling_windows)

    # Turn categories (airports, season) into dummy variables
    feats = encode_categoricals(feats, feature_cfg)

    # Finally drop rows with too many missing values
    feats = drop_na_rows(feats, min_required_lags=len(feature_cfg.lag_features))

    return feats.reset_index(drop=True)


def split_features_target(df):
    # I remove the columns that are not supposed to be used as numeric features.
# Split into X (features) and y (target = passengers)
    target = df["passengers"].astype(float)

    excluded = {"passengers", "date", "origin_airport", "destination_airport"}
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].astype(float)
    return X, target
