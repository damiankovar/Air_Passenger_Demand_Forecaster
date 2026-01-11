# Tests for data loading.
# Tests with the actual GVA and ZRH data files.

import pandas as pd
from pathlib import Path

from src import data_loading
from src.config import ColumnConfig, ProjectConfig


def build_config_with_real_files():
    """Create config pointing to the actual data files."""
    base_dir = Path(__file__).parent.parent  # Go up from tests/ to project-name/
    
    return ProjectConfig(
        base_dir=base_dir,
        data_paths=[
            base_dir / "data" / "GVA Data.csv",
            base_dir / "data" / "ZRH Data.csv"
        ],
        results_dir=base_dir / "results",
        figures_dir=base_dir / "results" / "figures",
        models_dir=base_dir / "results" / "models",
        routes=[],  # Don't filter routes for testing
        column_config=ColumnConfig(),
    )


def test_load_raw_data():
    """Test loading the actual GVA and ZRH data files."""
    config = build_config_with_real_files()
    df = data_loading.load_raw_data(config)

    # Verify basic structure
    assert "date" in df.columns
    assert "origin_airport" in df.columns
    assert "destination_airport" in df.columns
    assert "passengers" in df.columns
    
    # Verify date parsing
    assert str(df["date"].dtype).startswith("datetime64")
    
    # Verify we have data from both airports
    origins = df["origin_airport"].unique()
    assert "GVA" in origins or "Geneva" in origins
    assert "ZRH" in origins or "Zurich" in origins
    
    # Verify we have passenger data
    assert len(df) > 0
    assert df["passengers"].sum() > 0
    
    print(f"✓ Loaded {len(df):,} rows")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✓ Unique routes: {df[['origin_airport', 'destination_airport']].drop_duplicates().shape[0]}")


def test_filter_routes():
    """Test filtering routes from the actual data."""
    config = build_config_with_real_files()
    df = data_loading.load_and_preprocess(config)

    # Get some actual routes from the data
    unique_routes = df[["origin_airport", "destination_airport"]].drop_duplicates()
    
    if len(unique_routes) > 0:
        # Test filtering with the first route found
        test_route = tuple(unique_routes.iloc[0])
        filtered = data_loading.filter_routes(df, [test_route])
        
        assert len(filtered) > 0
        assert len(filtered["origin_airport"].unique()) <= 1
        assert len(filtered["destination_airport"].unique()) <= 1
        
        print(f"✓ Filtered route {test_route}: {len(filtered)} rows")
    else:
        print("⚠ No routes found in data to test filtering")


def test_data_structure():
    """Test that the data has the expected structure after preprocessing."""
    config = build_config_with_real_files()
    df = data_loading.load_and_preprocess(config)
    
    # Verify columns
    required_columns = ["date", "origin_airport", "destination_airport", "passengers"]
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Verify no missing critical values
    assert df["date"].notna().all(), "Some dates are missing"
    assert df["passengers"].notna().all(), "Some passenger counts are missing"
    
    # Verify passenger counts are positive
    assert (df["passengers"] > 0).any(), "No positive passenger counts found"
    
    print(f"✓ Data structure is valid")
    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Sample routes:")
    sample_routes = df[["origin_airport", "destination_airport"]].drop_duplicates().head(5)
    for _, row in sample_routes.iterrows():
        print(f"    {row['origin_airport']} -> {row['destination_airport']}")
