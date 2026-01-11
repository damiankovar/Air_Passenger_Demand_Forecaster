## Research Question

How can we forecast monthly air-passenger demand on Swiss routes (departing from Zurich ZRH and Geneva GVA) using both classical statistical models (SARIMA) and modern machine-learning models (Random Forest, XGBoost, MLP)?

The goal is to provide route-level insights for aviation planning and compare forecasting performance across models.

## Setup

### Create environment

Ensure Python 3.11 is installed.

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate air-passenger-forecasting
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
# Activate environment
conda activate iris-project

# Run the project
python main.py
```

This will:
- Automatically select interesting routes (if not already configured)
- Load and preprocess data
- Create features (lags, rolling averages, seasonality)
- Train ML models (Random Forest, XGBoost, MLP) and SARIMA models
- Evaluate Seasonal Naive baseline
- Make predictions for all routes
- Evaluate model performance
- Generate visualizations

### Route Selection (Optional)

```bash
python scripts/route_analysis.py
```

This analyzes all BFS routes and selects a diverse set (high-volume, growing, declining, seasonal).

### Run Tests

```bash
pytest
```

## Project Structure

```
air-passenger-forecasting/
├── main.py                 # Main entry point (runs full pipeline)
├── src/                    # Source code
│   ├── data_loading.py     # Data loading/preprocessing
│   ├── models_ml.py        # ML model training (RF, XGBoost, MLP)
│   ├── models_sarima.py    # SARIMA model training
│   ├── evaluation.py       # Evaluation metrics and model comparison
│   ├── visualization.py    # Plotting and charts
│   ├── feature_engineering.py  # Feature creation
│   ├── data_cleaning.py    # Data cleaning pipeline
│   ├── run_pipeline.py     # Full forecasting pipeline
│   └── config.py          # Configuration settings
├── scripts/
│   └── route_analysis.py   # Route selection and analysis
├── data/
│   └── raw/                # Data files (GVA_Data_cleaned.csv, ZRH_Data_cleaned.csv)
├── results/                # Output plots and metrics
│   ├── figures/            # Visualization charts
│   ├── ml_metrics.csv      # ML model performance
│   ├── sarima_metrics.csv # SARIMA model performance
│   ├── seasonal_naive_metrics.csv  # Seasonal Naive baseline performance
│   └── fastest_*_routes_*.csv  # Route analysis results
└── environment.yml         # Dependencies
```

## Results

# Evaluation Metrics

Model performance is evaluated using standard regression metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual passengers. Lower is better. Units: passengers/month.
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences. Penalizes large errors more than MAE. Lower is better. Units: passengers/month.
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Scale-independent, useful for comparing routes with different volumes. Lower is better. Units: percentage (%).

### Model Performance (6-month forecast horizon)

**Performance Summary (averaged across 19 routes):**
- Random Forest: 43.5% MAPE, 408.5 MAE, 569.6 RMSE
- SARIMA: 70.0% MAPE, 333.0 MAE, 420.7 RMSE
- XGBoost: 78.5% MAPE, 479.9 MAE, 606.4 RMSE
- MLP: 73.4% MAPE, 464.5 MAE, 611.1 RMSE
- **Seasonal Naive (baseline)**: 77.3% MAPE, 311.3 MAE, 375.3 RMSE

*Forecast horizon is 6 months ahead, otherwise runtime increase significantly (possibly +10 minutes from what has been tried) Metrics are averaged across 19 auto-selected routes. Seasonal Naive uses the same month from the previous year as a simple baseline for comparison.*

# Detailed Results

Model performance metrics are saved in `results/`:
- **ML Models**: Per-route performance in `ml_metrics.csv` and `ml_metrics_summary.csv`
- **SARIMA Models**: Per-route performance in `sarima_metrics.csv`
- **Seasonal Naive Baseline**: Per-route performance in `seasonal_naive_metrics.csv`
- **Route Analysis**: Fastest growing/declining routes for GVA and ZRH in separate CSV files
- **Visualizations**: Model comparisons, trend charts, and analysis plots in `results/figures/`

## Requirements

**Swiss Air Passenger Demand Forecasting**

Python 3.11

- scikit-learn, pandas, numpy, matplotlib, seaborn, scipy
- statsmodels (for SARIMA)
- xgboost (for XGBoost model)
- pytest (for testing)

All dependencies are specified in `environment.yml`.
