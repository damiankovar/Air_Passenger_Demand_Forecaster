# 1. Problem Statement and Motivation

Forecasting air travel demand is critical for airports, airlines, and aviation planners. It influences fleet allocation, revenue planning, and route scheduling decisions. Switzerland's two major international hubs (Zurich (ZRH) and Geneva (GVA)) both exhibit distinct traffic patterns shaped by business travel, tourism cycles, and seasonal peaks.

The main source will be the publicly available monthly "Air passengers on scheduled and charter flights" dataset from the Swiss Federal Statistical Office (BFS) : https://www.bfs.admin.ch/asset/en/36144222.

This project aims to build a machine learning system that predicts future demand for scheduled passengers traveling between Zurich/Geneva and a auto-selected set of key destinations.

The goal is not to model airline profitability but to construct a robust, data-driven forecasting framework rooted entirely in real, high-quality Swiss aviation data. Understanding and predicting demand at the route level provides meaningful insights into market dynamics, seasonal fluctuations, and airport-specific trends.

# 2. Planned Approach and Technologies

The project will focus on 20 routes from ZRH and GVA. By including both, the ML models will learn more diverse seasonal and structural behaviours, improving evaluation and making the results more meaningful.

For each route, the system will:

Ingest and clean BFS monthly series using pandas and NumPy

Conduct exploratory data analysis (seasonality, trends, shocks)

Create features: lag variables, rolling averages, seasonal indicators, airport and destination encoding

Train multiple forecasting models:

- SARIMA (Seasonal AutoRegressive Integrated Moving Average)

- Random Forest Regressor

- XGBoost Regressor

- Neural Network (MLP)

- Evaluate results using rolling time-series validation

- Visualize results using matplotlib and seaborn

The repository will follow a modular structure with clear separation between data, features, models, evaluation, and visualizations. A comprehensive suite of pytest tests will ensure code reliability.

# 3. Expected Challenges and Mitigation

- COVID anomalies: treat 2020 - 2021 separately or downweight them.

- Time-series modeling complexities.

- Strong seasonality: addressed through time-based features and SARIMA.

- Data gaps: cross-validation designed to avoid leakage.

- Multiple routes: modular pipeline to avoid code duplication.

# 4. Success Criteria

- Produces reliable forecasts for each selected ZRH/GVA route

- Identify declining vs growing routes

- Generates clear visual insights into airport and route patterns

- Demonstrates comparative ML performance across models

# 5. Stretch Goals

- Clustering destinations by seasonal behavior

- Compare demand dynamics of the two airports and highlight the different roles these airports play in Swiss aviation.

- External features (tourism statistics, exchange rates)

