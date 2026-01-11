"""Data loading and preprocessing."""

# Helpers for reading the BFS CSV and getting it into a predictable shape.

import pandas as pd

from .config import CONFIG, ColumnConfig


class DataValidationError(RuntimeError):
    """Raised when the CSV is missing something important."""


def _resolve_paths(config):
    """Figure out which data files to load.
    
    I have 2 files ( separate for GVA and ZRH). That's why I use those.
    Otherwise only one single data_path can be used as well.
    """
    # If multiple files are specified, use those
    if hasattr(config, 'data_paths') and config.data_paths:
        paths = config.data_paths
    # Otherwise, use the single file path
    else:
        paths = [config.data_path]
    
    # Check that all files exist before trying to load them
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset file(s) not found: {missing}. Please make sure the files exist "
            "in the data directory."
        )
    
    return paths


def _map_icao_to_iata(airport_code):
    """Convert airport codes to IATA format (ZRH, GVA).
    
    The BFS data sometimes uses ICAO codes (LSZH, LSGG) or full names (Zurich, Geneva),
    but I need IATA codes for consistency.
    """
    mapping = {
        "LSZH": "ZRH",  # Zurich ICAO
        "LSGG": "GVA",  # Geneva ICAO
        "Zurich": "ZRH",  # Zurich full name
        "Geneva": "GVA",  # Geneva full name
    }
    # If it's already an IATA code or unknown, return as-is
    return mapping.get(airport_code, airport_code)


def _ensure_columns(df, column_config):
    # Make sure all required columns are present, then rename them to simple names.
    missing = [
        actual_name
        for actual_name in column_config.as_dict().values()
        if actual_name not in df.columns
    ]
    if missing:
        raise DataValidationError(
            "The following required columns are missing from the dataset: "
            f"{missing}. Please update `config.ColumnConfig` to match the CSV."
        )

    renamed_df = df.rename(columns=column_config.as_dict())
    # Select only the columns we need and rename them to internal names
    result_df = renamed_df[
        [
            column_config.date,
            column_config.origin_airport,
            column_config.destination_airport,
            column_config.passengers,
        ]
    ].rename(
        columns={
            column_config.date: "date",
            column_config.origin_airport: "origin_airport",
            column_config.destination_airport: "destination_airport",
            column_config.passengers: "passengers",
        }
    )
    
    # Convert ICAO codes to IATA codes for origin airports
    result_df["origin_airport"] = result_df["origin_airport"].apply(_map_icao_to_iata)
    
    return result_df


def load_raw_data(config=CONFIG):
    """Load the CSV files and clean up the columns.
    

    My two files are combined into one dataframe.
    Only loads the 4 columns I actually need - makes it faster for big files.
    """

    paths = _resolve_paths(config)
    
    # Get the actual CSV column names we need (from the config)
    required_csv_columns = [
        config.column_config.date,
        config.column_config.origin_airport,
        config.column_config.destination_airport,
        config.column_config.passengers,
    ]
    
    # Load each file and combine them
    all_dataframes = []
    for path in paths:
        # Read only the columns we need from the CSV file (optimization)
        df = pd.read_csv(path, usecols=required_csv_columns)
        # Make sure it has the right columns and rename them to our internal names
        df = _ensure_columns(df, config.column_config)
        all_dataframes.append(df)
    
    # If we have multiple files, combine them into one dataframe
    if len(all_dataframes) > 1:
        df = pd.concat(all_dataframes, ignore_index=True)
    else:
        df = all_dataframes[0]

    
    try:
        df["date"] = pd.to_datetime(df["date"])
    except ValueError as exc:
        raise DataValidationError(
            "Failed to parse the date column. Please ensure the column "
            "contains ISO-formatted dates or update the parsing logic."
        ) from exc

    # Convert passengers to numbers and drop any rows that still look weird
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
    df = df.dropna(subset=["passengers"])

    # Normalize destination names for duplicates
    df = _normalize_destination_names(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def filter_routes(
    df,
    routes=None,
):
    """Filter down to just the routes I care about."""

    if not routes:
        return df.copy()

    route_set = set(routes)
    df['route_tuple'] = list(zip(df['origin_airport'], df['destination_airport']))
    mask = df['route_tuple'].isin(route_set)
    result = df[mask].drop(columns=['route_tuple']).reset_index(drop=True)
    return result


def normalize_destination_name(dest):
    """
    Normalize destination names to avoid duplicates.
    Returns None for destinations that should be filtered out.
    """
    dest_str = str(dest).strip()
    
    # Drop country names that overlap with their capital cities
    if dest_str == 'Albania' or dest_str == 'North Macedonia':
        return None  # Will be filtered out - keep only city names
    
    # Combine duplicate airport names for the same city
    if 'Berlin' in dest_str or 'Brandenburg' in dest_str:
        return 'Berlin'
    if 'Cuba' in dest_str or dest_str == 'Havana':
        return 'Cuba'
    if dest_str == 'Qatar' or 'Hamad' in dest_str:
        return 'Qatar'
        
    return dest


def _normalize_destination_names(df):
    """
    Remove country names that overlap with their capital cities.
    Keep only the city names to avoid double-counting passengers.
    """
    df = df.copy()
    df['destination_airport'] = df['destination_airport'].apply(normalize_destination_name)
    # Filter out None values (dropped country names)
    df = df[df['destination_airport'].notna()].copy()
    return df


def aggregate_route_monthly(df):
    # Sum passenger counts per route/month so each row is unique.
    # Note: destination names are already normalized in load_raw_data()
    grouped = (
        df.groupby(["date", "origin_airport", "destination_airport"], dropna=False)
        .agg({"passengers": "sum"})
        .reset_index()
    )
    grouped = grouped.sort_values(["origin_airport", "destination_airport", "date"])
    return grouped.reset_index(drop=True)


def load_and_preprocess(
    config=CONFIG,
    routes=None,
):
    """Load data and preprocess it for modeling.
    """
    df = load_raw_data(config)
    # Filter routes only if explicitly provided (for visualization)
    # For model training, pass routes=None to use all available routes
    if routes is not None:
        df = filter_routes(df, routes)
    df = aggregate_route_monthly(df)
    return df


def unique_routes(df):
    """Return all unique (origin, destination) pairs."""
    routes = (
        df[["origin_airport", "destination_airport"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return list(routes)


def ensure_minimum_history(
    df, min_months=24, verbose=False
):

    counts = (
        df.groupby(["origin_airport", "destination_airport"])["date"]
        .count()
        .reset_index(name="months")
    )
    valid_routes = counts.loc[counts["months"] >= min_months, ["origin_airport", "destination_airport"]]
    merged = df.merge(valid_routes, on=["origin_airport", "destination_airport"], how="inner")

    if verbose:
        dropped = set(unique_routes(df)) - set(unique_routes(merged))
        if dropped:
            print(
                "Dropped the following routes due to insufficient history: "
                f"{sorted(dropped)}"
            )
    return merged.reset_index(drop=True)


