# This is the main script I use to run the whole project.

import numpy as np
import pandas as pd

from . import data_loading, data_cleaning, feature_engineering
from . import models_ml, models_sarima, evaluation, visualization
from .config import CONFIG

def run_pipeline():
    # 1. Make sure output folders exist
    CONFIG.ensure_directories()
    print("Running with the following configuration:")
    print(CONFIG.describe())

    # 2. Load ALL data for model training (no route filtering)
    # Models learn better from all available routes
    print("Loading data (all routes for model training)...")
    df = data_loading.load_and_preprocess(CONFIG, routes=None)  # None = load all routes

    # Drop routes with too few months of history
    df = data_loading.ensure_minimum_history(df)
    print(f"Data loaded. Total rows: {len(df)} (all routes)")

    # 3. Cleaning step
    print("Cleaning data...")
    cleaned_df = data_cleaning.clean_passenger_data(df, config=CONFIG)

    # 3.2. Exclude COVID years (2020-01 to 2021-12) from training and testing
    # COVID years have abnormal patterns that would confuse the models
    print("Excluding COVID years (2020-01 to 2021-12) from training and testing...")
    original_rows = len(cleaned_df)
    
    # Filter out dates from 2020-01-01 to 2021-12-31
    covid_start = pd.Timestamp('2020-01-01')
    covid_end = pd.Timestamp('2021-12-31')
    covid_mask = (cleaned_df['date'] >= covid_start) & (cleaned_df['date'] <= covid_end)
    cleaned_df = cleaned_df[~covid_mask].copy()
    
    excluded_rows = original_rows - len(cleaned_df)
    print(f"  Excluded {excluded_rows} rows from COVID period (2020-2021)")
    print(f"  Remaining data: {len(cleaned_df)} rows")

    # 3.2. Filter out Swiss destinations (domestic routes)
    print("Filtering out Swiss destinations...")
    swiss_destinations = {'ZRH', 'GVA', 'BRN', 'LUG', 'BSL', 'SIR', 'Zurich', 'Geneva', 'Bern', 'Lugano', 'Basel', 'Sion', 'Switzerland'}
    original_rows_swiss = len(cleaned_df)
    cleaned_df = cleaned_df[~cleaned_df['destination_airport'].isin(swiss_destinations)].copy()
    excluded_swiss = original_rows_swiss - len(cleaned_df)
    if excluded_swiss > 0:
        print(f"  Excluded {excluded_swiss} rows with Swiss destinations")
        print(f"  Remaining data: {len(cleaned_df)} rows")
    else:
        print(f"  No Swiss destinations found in data")

    # 3.25. Filter out low-volume routes from training (improves model performance)
    # Calculate average monthly passengers per route
    route_volumes = cleaned_df.groupby(['origin_airport', 'destination_airport'])['passengers'].mean()
    min_volume = CONFIG.feature_config.min_avg_monthly_passengers
    
    # Get routes that meet the minimum volume threshold
    high_volume_routes = set(route_volumes[route_volumes>=min_volume].index)

    if len(high_volume_routes) < len(route_volumes):
        print(f"Filtering out low-volume routes (< {min_volume} passengers/month)...")
        original_routes = len(route_volumes)
        filtered_routes = len(high_volume_routes)
        
        # Filter cleaned_df to only include high-volume routes
        cleaned_df['route_tuple'] = list(zip(cleaned_df['origin_airport'], cleaned_df['destination_airport']))
        route_mask = cleaned_df['route_tuple'].isin(high_volume_routes)
        cleaned_df = cleaned_df[route_mask].drop(columns=['route_tuple']).copy()

        print(f"  Filtered from {original_routes} routes to {filtered_routes} routes")
        print(f"  Removed {original_routes - filtered_routes} low-volume routes from training")
        print(f"  (Models will learn from cleaner, higher-volume routes only)")
    else:
        print(f"All routes meet minimum volume threshold ({min_volume} passengers/month)")

    # Save full data for visualizations (all routes, before route filtering)
    all_routes_data = cleaned_df.copy()
    
    # Filter to selected routes for ML training (same as SARIMA) to ensure fair comparison
    # Both ML and SARIMA models will train and evaluate on the same routes
    if CONFIG.routes:
        selected_routes_set = set(CONFIG.routes)
        print(f"Filtering to {len(selected_routes_set)} selected routes for ML training (same as SARIMA)...")
        cleaned_df['route_tuple'] = list(zip(cleaned_df['origin_airport'], cleaned_df['destination_airport']))
        route_mask = cleaned_df['route_tuple'].isin(selected_routes_set)
        cleaned_df = cleaned_df[route_mask].drop(columns=['route_tuple']).copy()
        print(f"  Filtered to {len(cleaned_df)} rows for selected routes")
    else:
        print("No routes selected. Using all available routes for training.")

    # 4. Feature engineering
    print("Creating features...")
    features = feature_engineering.prepare_features(cleaned_df, config=CONFIG)

    # 5. Train ML models
    print("Training ML models...")
    X, y = feature_engineering.split_features_target(features)

    # Split data: use everything except the last 'horizon' months for training
    # The last 'horizon' months will be used to test/evaluate predictions
    horizon = CONFIG.validation_config.forecast_horizon
    X_train = X.iloc[:-horizon]
    y_train = y.iloc[:-horizon]

    # Keep the full features dataframe (with all columns) for recursive forecasting
    # I need this because recursive forecasting needs to rebuild features month by month
    historical_features = features.iloc[:-horizon].copy()

    # train the models
    ml_models = models_ml.train_models(X_train, y_train, config=CONFIG)

    # Now make recursive predictions for each route
    # The old way was to predict directly using X_test, but that only works if you
    # already have the data. For real future predictions, we need to do it recursively.
    # Predictions are made for the same routes used for training (consistent with SARIMA)
    print(f"Making recursive forecasts for {horizon} months ahead...")
    
    # Determine which routes to use for predictions (same routes as training)
    if CONFIG.routes:
        # Use selected routes (from auto-selection or config)
        prediction_routes = set(CONFIG.routes)
        print(f"Making predictions for {len(prediction_routes)} selected routes...")
    else:
        # No routes selected: use all routes (fallback)
        prediction_routes = set(data_loading.unique_routes(cleaned_df))
        print(f"No routes selected. Making predictions for all {len(prediction_routes)} routes...")
    
    # Store predictions per route and per model
    all_route_predictions = {}
    
    for route in prediction_routes:
        origin, dest = route
        route_predictions = {}
        
        # Filter this route's data once (instead of filtering 3 times inside recursive_forecast)
        # After encoding, airports become dummy variables, so check for those first
        origin_col = f'origin_airport_{origin}'
        dest_col = f'destination_airport_{dest}'
        
        if origin_col in historical_features.columns and dest_col in historical_features.columns:
            # Use dummy variables to filter
            route_data = historical_features[
                (historical_features[origin_col] == 1) &
                (historical_features[dest_col] == 1)
            ].copy()
        elif 'origin_airport' in historical_features.columns:
            # Fallback: use original columns if encoding wasn't done
            route_data = historical_features[
                (historical_features['origin_airport'] == origin) &
                (historical_features['destination_airport'] == dest)
            ].copy()
        else:
            # Can't filter, skip this route
            print(f"Warning: Could not filter data for route {origin}-{dest}")
            continue
        
        if len(route_data) == 0:
            print(f"Warning: No data found for route {origin}-{dest}")
            continue
        
        # Now make predictions with each model using the pre-filtered data
        for model_name, model in ml_models.items():
            try:
                preds = models_ml.recursive_forecast(
                    model=model,
                    route_data=route_data,
                    horizon=horizon,
                    feature_config=CONFIG.feature_config,
                    route=route
                )
                route_predictions[model_name] = preds
            except Exception as e:
                print(f"Model {model_name} failed for route {origin}-{dest}: {e}")
        
        if route_predictions:
            all_route_predictions[route] = route_predictions
    
    # Now I am evaluating predictions route by route
    # This is simpler than trying to match everything up globally
    ml_results = []
    
    for route in prediction_routes:
        origin, dest = route
        route_name = f"{origin}-{dest}"
        
        # Get the actual test values for this route
        route_test_data = cleaned_df[
            (cleaned_df['origin_airport'] == origin) &
            (cleaned_df['destination_airport'] == dest)
        ].sort_values('date')
        
        if len(route_test_data) < horizon:
            # Not enough test data for this route
            continue
        
        # Get the last 'horizon' months as test data
        route_test_series = route_test_data['passengers'].iloc[-horizon:].values
        
        # Evaluate each model's predictions for this route
        if route in all_route_predictions:
            for model_name, preds in all_route_predictions[route].items():
                if len(preds) == len(route_test_series):
                    metrics = evaluation.evaluate_predictions(route_test_series, preds)
                    metrics['model'] = model_name
                    metrics['origin'] = origin
                    metrics['destination'] = dest
                    metrics['route'] = route_name
                    ml_results.append(metrics)
    
    # Save per-route metrics
    if ml_results:
        ml_results_df = pd.DataFrame(ml_results)
        ml_metrics_path = CONFIG.results_dir / "ml_metrics.csv"
        ml_results_df.to_csv(ml_metrics_path, index=False)
        print(f"Saved ML metrics to {ml_metrics_path}")
        
        # Also create a summary by model (average across routes)
        # Calculate mean metrics, excluding NaN values (routes with < 1 passenger/month)
        summary = ml_results_df.groupby('model')[['mae', 'rmse', 'mape']].mean()
        
        # Add count of routes used for each metric (helps understand if some routes were excluded)
        summary['n_routes_mae'] = ml_results_df.groupby('model')['mae'].count()
        summary['n_routes_mape'] = ml_results_df.groupby('model')['mape'].count()  # NaN routes excluded
        
        summary_path = CONFIG.results_dir / "ml_metrics_summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved ML metrics summary to {summary_path}")
    else:
        print("Warning: No ML predictions to evaluate")

    # 6. Train SARIMA models (on same routes as ML models for fair comparison)
    # Both ML and SARIMA train and evaluate on the same routes
    
    # Use the same routes as ML predictions
    sarima_routes = prediction_routes
    print(f"Training SARIMA on {len(sarima_routes)} routes (same as ML models)...")
    
    # Train SARIMA only on the selected routes
    sarima_results = []
    for (origin, dest), group in cleaned_df.groupby(["origin_airport", "destination_airport"]):
        # Skip routes not in our SARIMA training set
        if (origin, dest) not in sarima_routes:
            continue
            
        series = group.set_index("date")["passengers"]
        if len(series) <= horizon + CONFIG.validation_config.min_train_periods:
            # skip very short series
            continue

        train_series = series.iloc[:-horizon]
        test_series = series.iloc[-horizon:]

        try:
            model = models_sarima.fit_sarima_model(train_series, CONFIG.model_config)
            # Use forecast_route() which clips negative predictions to zero
            forecast = models_sarima.forecast_route(model, steps=horizon)
            metrics = evaluation.evaluate_predictions(test_series, forecast)
            metrics["origin"] = origin
            metrics["destination"] = dest
            metrics["model"] = "sarima"  # Add model name for consistency
            sarima_results.append(metrics)
        except Exception as e:
            print(f"SARIMA failed for {origin}-{dest}: {e}")

    sarima_df = pd.DataFrame(sarima_results)
    sarima_path = CONFIG.results_dir / "sarima_metrics.csv"
    sarima_df.to_csv(sarima_path, index=False)
    print(f"Saved SARIMA metrics to {sarima_path}")

    # 6.5. Evaluate Seasonal Naive baseline (on same routes as other models)
    print(f"Evaluating Seasonal Naive baseline on {len(sarima_routes)} routes...")
    
    seasonal_naive_results = []
    for (origin, dest), group in cleaned_df.groupby(["origin_airport", "destination_airport"]):
        # Skip routes not in our evaluation set
        if (origin, dest) not in sarima_routes:
            continue
            
        series = group.set_index("date")["passengers"]
        if len(series) <= horizon + CONFIG.validation_config.min_train_periods:
            # skip very short series
            continue

        train_series = series.iloc[:-horizon]
        test_series = series.iloc[-horizon:]

        try:
            # Seasonal naive forecast: use same month from previous year
            forecast = models_sarima.seasonal_naive_forecast(
                train_series, 
                steps=horizon, 
                seasonality=12
            )
            metrics = evaluation.evaluate_predictions(test_series, forecast)
            metrics["origin"] = origin
            metrics["destination"] = dest
            metrics["model"] = "seasonal_naive"  # Add model name for consistency
            seasonal_naive_results.append(metrics)
        except Exception as e:
            print(f"Seasonal Naive failed for {origin}-{dest}: {e}")

    seasonal_naive_df = pd.DataFrame(seasonal_naive_results)
    if len(seasonal_naive_df) > 0:
        seasonal_naive_path = CONFIG.results_dir / "seasonal_naive_metrics.csv"
        seasonal_naive_df.to_csv(seasonal_naive_path, index=False)
        print(f"Saved Seasonal Naive metrics to {seasonal_naive_path}")
        
        # Print summary for Seasonal Naive
        seasonal_naive_summary = seasonal_naive_df[['mae', 'rmse', 'mape']].mean()
        print(f"\nSeasonal Naive Performance (averaged across {len(seasonal_naive_df)} routes):")
        print(f"  MAE: {seasonal_naive_summary['mae']:.1f}")
        print(f"  RMSE: {seasonal_naive_summary['rmse']:.1f}")
        print(f"  MAPE: {seasonal_naive_summary['mape']:.1f}%")
    else:
        print("Warning: No Seasonal Naive predictions to evaluate")

    # 7. Generate visualizations

    # Most visualizations use all routes, but some focus on selected routes. 
    # For example, all the fastest growing/delcining routes are based on all routes, since the script only selected 19 routes.
    print("Generating visualizations...")
    
    # Seasonality plot (uses all routes)
    visualization.plot_seasonality(all_routes_data, config=CONFIG)
    
    # Seasonality strength CDF (uses all routes)
    visualization.plot_seasonality_strength_cdf(all_routes_data, config=CONFIG)
    
    # 8. Model comparison visualizations (uses all routes from metrics)

    print("Generating model comparison visualizations...")
    try:
        ml_results_df = pd.read_csv(CONFIG.results_dir / "ml_metrics.csv")
        
        visualization.plot_model_ranking(ml_results_df, metric='mape', config=CONFIG)
        visualization.plot_model_ranking(ml_results_df, metric='mae', config=CONFIG)
        visualization.plot_model_comparison_heatmap(ml_results_df, config=CONFIG)
        visualization.plot_best_model_per_route(ml_results_df, config=CONFIG)
        
        sarima_path = CONFIG.results_dir / "sarima_metrics.csv"
        seasonal_naive_path = CONFIG.results_dir / "seasonal_naive_metrics.csv"
        if sarima_path.exists() and seasonal_naive_path.exists():
            visualization.plot_model_performance_summary(
                CONFIG.results_dir / "ml_metrics.csv",
                sarima_path,
                seasonal_naive_path,
                config=CONFIG
            )
        elif sarima_path.exists():
            # Fallback if seasonal naive not available
            visualization.plot_model_performance_summary(
                CONFIG.results_dir / "ml_metrics.csv",
                sarima_path,
                None,
                config=CONFIG
            )
    except Exception as e:
        print(f"Warning: Model comparison visualization failed: {e}")
    
    # 10. Fastest growing/declining routes (uses all routes from dataset to show top 15)
    print("Generating fastest growing and declining routes analysis...")
    try:
        # Use all routes (not just selected) to ensure we can show top 15 per airport
        # Tables still use selected routes for focused analysis
        if CONFIG.routes:
            all_routes_data['route_tuple'] = list(zip(all_routes_data['origin_airport'], all_routes_data['destination_airport']))
            route_mask = all_routes_data['route_tuple'].isin(set(CONFIG.routes))
            selected_data_for_tables = all_routes_data[route_mask].drop(columns=['route_tuple']).copy()
        else:
            selected_data_for_tables = all_routes_data
        
        visualization.create_growing_declining_tables(selected_data_for_tables, config=CONFIG)
        
        # Generate charts for top 15 routes per airport (use all routes, not just selected)
        visualization.plot_fastest_growing_routes(all_routes_data, top_n=15, origin_airport='GVA', config=CONFIG)
        visualization.plot_fastest_growing_routes(all_routes_data, top_n=15, origin_airport='ZRH', config=CONFIG)
        visualization.plot_fastest_declining_routes(all_routes_data, top_n=15, origin_airport='GVA', config=CONFIG)
        visualization.plot_fastest_declining_routes(all_routes_data, top_n=15, origin_airport='ZRH', config=CONFIG)
        
        # Cluster destinations by seasonal behavior
        visualization.cluster_destinations_by_seasonality(all_routes_data, n_clusters=4, config=CONFIG)
        
        # Generate detailed trend charts for all selected routes
        # These charts show the historical trend and predictions for each selected route
        print("  Generating detailed trend charts for selected routes...")
        
        if not CONFIG.routes:
            print("    No routes selected. Skipping enhanced trend charts.")
        else:
            # Clean up old enhanced trend charts first (remove charts from previous runs)
            figures_dir = CONFIG.figures_dir
            old_charts = list(figures_dir.glob("enhanced_trend_*.png"))
            if old_charts:
                print(f"    Cleaning up {len(old_charts)} old enhanced trend charts...")
                for chart_file in old_charts:
                    chart_file.unlink()
            
            # Read the selected routes CSV file
            selected_routes_path = CONFIG.results_dir / "selected_routes.csv"
            if not selected_routes_path.exists():
                print(f"    Warning: {selected_routes_path} not found. Skipping enhanced trend charts.")
            else:
                selected_routes_df = pd.read_csv(selected_routes_path)
                
                charts_generated = 0
                
                # Generate a chart for each selected route
                for _, row in selected_routes_df.iterrows():
                    origin = row['origin']
                    destination = row['destination']
                    route = (origin, destination)
                    
                    # Double check to make surewe skip routes with extreme declines (>-75%) as alrady done by  route selection script
                    growth_pct = row.get('recent_growth_pct', 0)
                    if pd.notna(growth_pct) and growth_pct < -75:
                        print(f"      Skipping {origin}-{destination}: extreme decline ({growth_pct:.1f}%)")
                        continue
                    
                    # Even though some of them have already been removed, I still add this here as safety measure...
                    geopolitical_destinations = ['belarus', 'minsk', 'russia', 'moscow', 'ukraine', 'kiev', 'istanbul', 'tripoli']
                    if any(geo in str(destination).lower() for geo in geopolitical_destinations):
                        print(f"      Skipping {origin}-{destination}: geopolitical route")
                        continue
                    
                    # Generate the chart
                    try:
                        output = visualization.plot_enhanced_route_trend(
                            selected_data_for_tables, route, config=CONFIG
                        )
                        if output:
                            charts_generated += 1
                    except Exception as e:
                        print(f"      Warning: Could not generate trend chart for {origin}-{destination}: {e}")
                
                print(f"  Generated {charts_generated} detailed trend charts")
    except Exception as e:
        print(f"Warning: Growing/declining routes analysis failed: {e}")
    
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
