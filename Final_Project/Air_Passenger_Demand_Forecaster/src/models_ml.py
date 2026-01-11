"""Model definitions and training."""

# Non-SARIMA models, written in a pretty straightforward way.

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    # If xgboost isn't installed, I just skip it and fallback to GradientBoostingRegressor.
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

from .config import CONFIG

def train_random_forest(X_train, y_train, model_config):
    """Train a Random Forest regressor using the project config."""
    # Just printing so I know this part runs
    print("Training Random Forest...")
    model = RandomForestRegressor(**model_config.random_forest_params)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, model_config):
    print("Training XGBoost...")
    model = XGBRegressor(**model_config.xgboost_params)
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train, model_config):
    """Train an MLP (neural network) regressor, with feature scaling."""
    from sklearn.preprocessing import StandardScaler
    
    print("Training MLP...")
    # MLP doesn't accept NaN values, so we need to handle them
    X_clean = X_train.fillna(0.0)
    y_clean = y_train.fillna(0.0) if hasattr(y_train, 'fillna') else y_train
    
    # Scale features: this puts all features on the same scale (mean=0, std=1)
    # This is important for neural networks to work well
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    model = MLPRegressor(**model_config.mlp_params)
    model.fit(X_scaled, y_clean)
    
    # Store the scaler with the model so we can use it later for predictions
    # (we need to scale new data the same way we scaled training data)
    model.scaler = scaler
    
    return model

def train_models(X, y, config=CONFIG):
    """Train all models and return them."""
    model_config = config.model_config
    
    # Clean NaN values before training (MLP especially needs this)
    # Fill with 0 for features (safer than mean for lag/rolling features)
    X_clean = X.fillna(0.0)
    y_clean = y.fillna(0.0) if hasattr(y, 'fillna') else y
    
    models = {}
    models["random_forest"] = train_random_forest(X_clean, y_clean, model_config)
    models["xgboost"] = train_xgboost(X_clean,y_clean, model_config)
    models["mlp"] = train_mlp(X_clean, y_clean, model_config)
    return models


def recursive_forecast(model, route_data, horizon, feature_config, route):
    """Make predictions for multiple months ahead.
    
    Can't use future data, so I predict month 1, then use that to predict month 2, etc.
    This was tricky to get right - had to rebuild features for each future month.
    
    route_data should already be filtered to just this route's data.
    """
    origin, dest = route
    
    if len(route_data) == 0:
        # If there's no data for this route, return zeros
        return np.zeros(horizon)
    
    # Sort by date to make sure we have the right order
    route_data = route_data.sort_values('date').reset_index(drop=True)
    
    # Keep track of passenger values (real ones + predictions as we go)
    passenger_history = route_data['passengers'].tolist()
    date_history = route_data['date'].tolist()
    
    # Get the last date so I know where to start predicting
    last_date = route_data['date'].max()
    
    # Create the future dates I want to predict
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'  #MS=month start
    )
    
    # This will store all my predictions
    predictions = []
    
    # I need to know what feature columns the model expects
    # So I'll look at what columns exist in the historical data (excluding non-feature ones)
    excluded_cols = {'passengers', 'date', 'origin_airport', 'destination_airport', 'season'}
    feature_cols = [c for c in route_data.columns if c not in excluded_cols]
    
    # Now predict month by month
    for i, future_date in enumerate(future_dates):
        # Build features for this future month
        # Start with basic info
        row_features = {}
        
        # Time-based features (these I can calculate from the date)
        if feature_config.include_month:
            row_features['month'] = future_date.month
        if feature_config.include_year:
            row_features['year'] = future_date.year
        if feature_config.include_season:
            # Map month to season
            month = future_date.month
            if month in [12, 1, 2]:
                season = 'winter'
            elif month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            else:
                season = 'autumn'
            row_features['season'] = season
        
        # Lag features - these use the passenger_history
        # lag_1 means "passengers 1 month ago", lag_3 means "3 months ago", etc.
        for lag in feature_config.lag_features:
            lag_col = f'lag_{lag}'
            if len(passenger_history) >= lag:
                # I have enough history, use the value from 'lag' months ago
                row_features[lag_col] = passenger_history[-lag]
            else:
                # Not enough history, just use the first value I have
                row_features[lag_col] = passenger_history[0] if passenger_history else 0
        
        # Rolling statistics - mean and std over a window
        for window in feature_config.rolling_windows:
            mean_col = f'roll_mean_{window}'
            std_col = f'roll_std_{window}'
            
            if len(passenger_history) >= window:
                # I have enough data for this window
                window_data = passenger_history[-window:]
                row_features[mean_col] = np.mean(window_data)
                row_features[std_col] = np.std(window_data) if len(window_data)>1 else 0.0
            else:
                # Not enough data, use what I have
                if len(passenger_history) > 0:
                    row_features[mean_col] = np.mean(passenger_history)
                    row_features[std_col] = np.std(passenger_history) if len(passenger_history) > 1 else 0.0
                else:
                    row_features[mean_col] = 0.0
                    row_features[std_col] = 0.0
        
        # Now I need to handle categorical encoding (airports, seasons)
        # The model was trained with dummy variables, so I need to create the same ones
        # Look at what dummy columns exist in the training data
        for col in feature_cols:
            if col.startswith('origin_airport_'):
                # This is a dummy column for an origin airport
                # Set it to 1 if it matches our origin, 0 otherwise
                row_features[col] = 1.0 if col == f'origin_airport_{origin}' else 0.0
            elif col.startswith('destination_airport_'):
                #Same for destination
                row_features[col] = 1.0 if col == f'destination_airport_{dest}' else 0.0
            elif col.startswith('season_'):
                # Same for season
                if 'season' in row_features:
                    row_features[col] = 1.0 if col == f'season_{row_features["season"]}' else 0.0
                else:
                    row_features[col] = 0.0
            elif col not in row_features:
                # Some other feature column I haven't set yet
                # Check if it exists in the historical data and use a default
                if col in route_data.columns:
                    # Use the last known value
                    row_features[col] = route_data[col].iloc[-1] if len(route_data) > 0 else 0.0
                else:
                    row_features[col] = 0.0
        
        # Create a DataFrame with just this one row, with all the feature columns
        # Make sure the columns are in the same order as the model expects
        pred_row = pd.DataFrame([row_features])
        
        # Select only the feature columns that the model knows about
        # (in case there are some extra columns)
        available_features = [c for c in feature_cols if c in pred_row.columns]
        X_pred = pred_row[available_features].astype(float)
        
        # Make sure all expected columns are there
        # If a column is missing, add it with 0
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0.0
        
        # Reorder to match training data
        X_pred = X_pred[feature_cols]
        
        # Now make the prediction!
        try:
            # If this is an MLP model, we need to scale the features first
            if hasattr(model, 'scaler'):
                X_pred_scaled = model.scaler.transform(X_pred)
                pred = model.predict(X_pred_scaled)[0]
            else:
                pred = model.predict(X_pred)[0]
            
            # Important: passengers can't be negative! Clip to zero if needed
            pred = max(0.0, pred)
            
            predictions.append(pred)
            # Add this prediction to history so I can use it for the next month
            passenger_history.append(pred)
            date_history.append(future_date)
        except Exception as e:
            # If something goes wrong, use the last known value as a fallback
            print(f"Warning: prediction failed for {origin}-{dest} month {i+1}: {e}")
            if len(passenger_history) > 0:
                fallback = max(0.0, passenger_history[-1])  # Also clip fallback to zero
            else:
                fallback = 0.0
            predictions.append(fallback)
            passenger_history.append(fallback)
    
    return np.array(predictions)

