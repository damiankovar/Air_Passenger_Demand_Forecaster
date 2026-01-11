# Configuration classes for the project.
# I used dataclasses to keep everything better organized.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class ColumnConfig:
    """Configuration for CSV column name mapping."""
    
    date: str = "TIME_PERIOD"
    origin_airport: str = "Airport"
    destination_airport: str = "Final destination"
    passengers: str = "OBS_VALUE"
    
    def as_dict(self):
        """Return a dictionary mapping internal names to CSV column names."""
        return {
            "date": self.date,
            "origin_airport": self.origin_airport,
            "destination_airport": self.destination_airport,
            "passengers": self.passengers,
        }


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    lag_features: Tuple[int, ...] = (1, 3, 6, 12)
    rolling_windows: Tuple[int, ...] = (3, 6, 12)
    include_month: bool = True
    include_year: bool = True
    include_season: bool = True
    include_airport_encoding: bool = True
    include_destination_encoding: bool = True
    min_avg_monthly_passengers: float = 50.0  # Filter routes with < 50 passengers/month from training


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters."""
    
    sarima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
    random_forest_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,  # Original complexity restored
        "max_depth": 12,
        "random_state": 42,
        "min_samples_split": 4,
    })
    xgboost_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,  # Original complexity restored
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
    })
    mlp_params: dict = field(default_factory=lambda: {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-3,
        "random_state": 42,
        "max_iter": 500,  # Original complexity restored
    })


@dataclass
class ValidationConfig:
    """Configuration for validation and evaluation."""
    
    min_train_periods: int = 24
    # How many months ahead to predict. Can be 3, 6, 12, 24, etc.
    # The recursive forecasting approach allows for longer horizons.
    forecast_horizon: int = 6  # 6 months ahead


@dataclass
class ProjectConfig:
    """Main project configuration with all nested configs."""
    
    base_dir: Path = field(default_factory=lambda: Path("."))
    data_path: Path = field(default_factory=lambda: Path("data/bfs_air_passengers.csv"))
    
    data_paths: List[Path] = field(default_factory=lambda: [])
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("results/figures"))
    models_dir: Path = field(default_factory=lambda: Path("results/models"))  # Prepared for future model saving (currently unused)
    # Routes for focused visualization/analysis (optional)
    # Models always train on ALL available routes regardless of this setting
    # If empty, routes will be auto-selected or all routes will be used for visualizations
    routes: List[Tuple[str, str]] = field(default_factory=lambda: [])
    column_config: ColumnConfig = field(default_factory=ColumnConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    def __post_init__(self):
        """Convert string paths to Path objects and resolve relative paths."""
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.figures_dir, str):
            self.figures_dir = Path(self.figures_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
        
        # Handle data_paths list (for multiple files like GVA and ZRH)
        if isinstance(self.data_paths, list):
            self.data_paths = [
                Path(p) if isinstance(p, str) else p 
                for p in self.data_paths
            ]
            # Resolve relative paths for each file in the list
            self.data_paths = [
                p if p.is_absolute() else self.base_dir / p 
                for p in self.data_paths
            ]
        
        # Resolve relative paths against base_dir
        if not self.data_path.is_absolute():
            self.data_path = self.base_dir / self.data_path
        if not self.results_dir.is_absolute():
            self.results_dir = self.base_dir / self.results_dir
        if not self.figures_dir.is_absolute():
            self.figures_dir = self.base_dir / self.figures_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.base_dir / self.models_dir
    
    def ensure_directories(self):
        """Create all necessary output directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def describe(self):
        """Return a string description of the configuration."""
        # Show which data files will be used
        if self.data_paths:
            data_info = f"Data files: {[str(p) for p in self.data_paths]}"
        else:
            data_info = f"Data path: {self.data_path}"
        
        return f"""
Project Configuration:
  {data_info}
  Results directory: {self.results_dir}
  Figures directory: {self.figures_dir}
  Models directory: {self.models_dir}
  Routes: {len(self.routes)} routes configured
  Forecast horizon: {self.validation_config.forecast_horizon} months
  Min training periods: {self.validation_config.min_train_periods} months
"""


# Global default configuration instance
CONFIG = ProjectConfig(
    data_paths=[
        Path("data/raw/GVA_Data_cleaned.csv"),
        Path("data/raw/ZRH_Data_cleaned.csv"),
    ],
    routes=[],  # Will be auto-selected by route_analysis.py
)
