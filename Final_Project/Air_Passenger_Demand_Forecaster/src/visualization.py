# Here are the plotting helpers used to help see what's going on beyond the metrics.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sklearn.cluster import KMeans

from .config import CONFIG
from .data_loading import normalize_destination_name

sns.set_style("whitegrid")


def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_seasonality(df, config=CONFIG):
    """Heatmap showing monthly passenger counts by year - rows are months, columns are years."""
    df = df.copy()
    
    # Extract year and month from date column
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Aggregate passengers by year and month
    monthly_data = df.groupby(['year', 'month'])['passengers'].sum().reset_index()
    
    # Pivot: rows = months (Jan-Dec), columns = years
    heatmap_data = monthly_data.pivot(index='month', columns='year', values='passengers')
    
    # Create month names for better readability
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_data.index = [month_names[i-1] for i in heatmap_data.index]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use seaborn heatmap with sequential colormap
    sns.heatmap(
        heatmap_data,
        annot=True,  # Fill each cell with the value
        fmt='.0f',   # Format as integers (no decimals)
        cmap='rocket',  # Colormap (dark to bright)
        cbar_kws={'label': 'Passengers'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Month', fontsize=12, fontweight='bold')
    ax.set_title('Heat Map of Monthly Passengers', fontsize=14, fontweight='bold', pad=15)
    
    # Rotate year labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output = config.figures_dir / "seasonality.png"
    _save_figure(fig, output)
    return output


# MODEL COMPARISON VISUALIZATIONS

def plot_model_ranking(metrics_df, metric='mape', config=CONFIG):
    """Bar chart ranking models by performance (lower is better)."""
    summary = metrics_df.groupby('model')[metric].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['gold', 'silver', 'peru', 'steelblue']  # Gold, Silver, Bronze, Blue
    bars = ax.barh(range(len(summary)), summary.values, color=colors[:len(summary)])
    
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([m.replace('_', ' ').title() for m in summary.index], fontsize=12)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Model Ranking by {metric.upper()} (Lower = Better)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(summary.items()):
        ax.text(val, i, f'  {val:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    output = config.figures_dir / f"model_ranking_{metric}.png"
    _save_figure(fig, output)
    return output


def plot_model_comparison_heatmap(metrics_df, config=CONFIG):
    """Heatmap showing performance (MAPE) for each model × route."""
    pivot = metrics_df.pivot_table(
        values='mape',
        index='route',
        columns='model',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                linewidths=0.5, cbar_kws={'label': 'MAPE (%)'}, ax=ax)
    ax.set_title("Model Performance Heatmap (MAPE % - Lower is Better)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Route", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output = config.figures_dir / "model_comparison_heatmap.png"
    _save_figure(fig, output)
    return output


def plot_best_model_per_route(metrics_df, config=CONFIG):
    """Bar chart showing which model performs best for each route."""
    # Filter out rows with NaN MAPE (routes where MAPE couldn't be calculated)
    metrics_df_clean = metrics_df[metrics_df['mape'].notna()].copy()
    
    if len(metrics_df_clean) == 0:
        print("Warning: No routes with valid MAPE values to plot")
        return None
    
    # Find best model (lowest MAPE) for each route
    best_models = metrics_df_clean.loc[metrics_df_clean.groupby('route')['mape'].idxmin()]
    best_models = best_models[['route', 'model', 'mape']].sort_values('mape')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = {'random_forest': 'steelblue', 'xgboost': 'purple', 'mlp': 'orange', 'sarima': 'darkred'}
    
    for _, row in best_models.iterrows():
        model = row['model']
        mape = row['mape']
        color = colors.get(model, 'gray')
        ax.barh(row['route'], mape, color=color, alpha=0.8)
    
    ax.set_xlabel("MAPE (%)", fontsize=12)
    ax.set_ylabel("Route", fontsize=12)
    ax.set_title("Best Model per Route (Ranked by MAPE)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], label=m.replace('_', ' ').title()) 
                      for m in colors.keys() if m in best_models['model'].values]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output = config.figures_dir / "best_model_per_route.png"
    _save_figure(fig, output)
    return output


def plot_model_performance_summary(ml_metrics_path, sarima_metrics_path, seasonal_naive_metrics_path=None, config=CONFIG):
    """Create comprehensive summary visualization comparing all models."""
    ml_df = pd.read_csv(ml_metrics_path)
    sarima_df = pd.read_csv(sarima_metrics_path)
    
    sarima_df['model'] = 'sarima'
    sarima_df['route'] = sarima_df['origin'] + '-' + sarima_df['destination']
    
    # Combine all metrics
    metrics_list = [
        ml_df[['mae', 'rmse', 'mape', 'model', 'route']],
        sarima_df[['mae', 'rmse', 'mape', 'model', 'route']]
    ]
    
    # Add seasonal naive if available
    if seasonal_naive_metrics_path and Path(seasonal_naive_metrics_path).exists():
        seasonal_naive_df = pd.read_csv(seasonal_naive_metrics_path)
        if 'route' not in seasonal_naive_df.columns:
            seasonal_naive_df['route'] = seasonal_naive_df['origin'] + '-' + seasonal_naive_df['destination']
        metrics_list.append(seasonal_naive_df[['mae', 'rmse', 'mape', 'model', 'route']])
    
    all_metrics = pd.concat(metrics_list, ignore_index=True)
    
    summary = all_metrics.groupby('model')[['mae', 'rmse', 'mape']].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    summary.plot(kind='bar', ax=axes[0, 0], color=['coral', 'turquoise', 'lightblue'])
    axes[0, 0].set_title("Average Performance Across All Routes", fontweight='bold')
    axes[0, 0].set_ylabel("Error")
    axes[0, 0].legend(title="Metric")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    all_metrics.boxplot(column='mape', by='model', ax=axes[0, 1])
    axes[0, 1].set_title("MAPE Distribution by Model", fontweight='bold')
    axes[0, 1].set_xlabel("Model")
    axes[0, 1].set_ylabel("MAPE (%)")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    best_per_route = all_metrics.loc[all_metrics.groupby('route')['mape'].idxmin()]
    wins = best_per_route['model'].value_counts()
    axes[1, 0].bar(wins.index, wins.values, color=['gold', 'silver', 'peru', 'steelblue'])
    axes[1, 0].set_title("Number of Routes Where Model is Best", fontweight='bold')
    axes[1, 0].set_xlabel("Model")
    axes[1, 0].set_ylabel("Number of Routes")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for model in all_metrics['model'].unique():
        model_data = all_metrics[all_metrics['model'] == model]
        axes[1, 1].scatter(model_data['mae'], model_data['mape'], 
                          label=model, alpha=0.6, s=50)
    axes[1, 1].set_xlabel("MAE")
    axes[1, 1].set_ylabel("MAPE (%)")
    axes[1, 1].set_title("MAE vs MAPE Scatter", fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Complete Model Performance Analysis", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output = config.figures_dir / "model_performance_summary.png"
    _save_figure(fig, output)
    return output


# NEW VISUALIZATIONS FOR TREND ANALYSIS (GROWTH/DECLINE)

def calculate_route_trends(df):
    """Calculate growth/decline stats for each route. Skip COVID period to avoid weird numbers."""
    # Normalize duplicate destinations - combine destinations that represent the same place
    df = df.copy()
    df['destination_airport'] = df['destination_airport'].apply(normalize_destination_name)
    # Filter out None values (dropped country names)
    df = df[df['destination_airport'].notna()].copy()
    
    # I'm excluding routes affected by conflicts, sanctions, or geopolitical issues
    # Aditionally, I am excluding Swiss destinations (domestic routes)

    exclude_destinations = [
        'Belarus', 'Minsk', 'Russia', 'Russian', 'Moskva', 'Moscow', 
        'Domodedovo', 'Sheremetyevo', 'Vnukovo', 'Pulkovo', 'St Petersburg',
        'Ukraine', 'Kiev', 'Kiev Borispol', 'Lugano', 'Lugano-Agno', 'Zürich', 'Zurich', 'Genève', 'Geneva', 'Basel', 'Bern'
    ]
    
    # Vectorized exclusion check
    dest_lower = df['destination_airport'].astype(str).str.lower()
    exclude_mask = pd.Series(False, index=df.index)
    for exclude_word in exclude_destinations:
        exclude_mask |= dest_lower.str.contains(exclude_word.lower(), na=False)
    df = df[~exclude_mask].copy()
    
    # Re-aggregate in case we combined destinations - sum passengers if same month/origin/dest
    df = df.groupby(['date', 'origin_airport', 'destination_airport'])['passengers'].sum().reset_index()
    
    trends = []
    
    for (origin, dest), group in df.groupby(['origin_airport', 'destination_airport']):
        group = group.sort_values('date').copy()
        
        if len(group) < 6:
            continue
        
        # Compare 2023+ vs pre-COVID (2017-2019), skip 2022 (recovery bounce)
        pre_covid_baseline = group[(group['date'] >= pd.Timestamp('2017-01-01')) & 
                                   (group['date'] < pd.Timestamp('2020-01-01'))]
        post_covid_performance = group[group['date'] >= pd.Timestamp('2023-01-01')]
        
        recent_growth_pct = None
        recent_growth_abs = None
        pre_avg = None
        post_avg = None
        
        if len(pre_covid_baseline) >= 6 and len(post_covid_performance) >= 6:
            pre_avg = pre_covid_baseline['passengers'].mean()
            post_avg = post_covid_performance['passengers'].mean()
            recent_growth_abs = post_avg - pre_avg
            
            # Only use percentage if pre-COVID had at least 100 passengers/month
            if pre_avg >= 100:
                recent_growth_pct = ((post_avg - pre_avg) / pre_avg) * 100
        
        trends.append({
            'origin': origin,
            'destination': dest,
            'route': f"{origin}-{dest}",
            'recent_growth_pct': recent_growth_pct,  # Percentage change (None if baseline < 100)
            'recent_growth_abs': recent_growth_abs if recent_growth_abs is not None else 0,  # Absolute change (passengers/month)
            'pre_covid_avg': pre_avg,  # Store pre-COVID average for reference
            'post_covid_avg': post_avg,  # Store post-COVID (2023+) average for reference
        })
    
    # Clean up: Replace any infinite or NaN values with NaN (safely handled in plots)
    trends_df = pd.DataFrame(trends)
    trends_df = trends_df.replace([np.inf, -np.inf], np.nan)
    return trends_df


def calculate_seasonality_strength(df):
    """Calculate seasonality strength for each route using CV of monthly averages.
    
    Higher CV = stronger seasonality (more variation between months).
    """
    df = df.copy()
    df['month'] = df['date'].dt.month
    
    seasonality_data = []
    
    for (origin, dest), group in df.groupby(['origin_airport', 'destination_airport']):
        # Get average passengers per month for this route
        monthly_avgs = group.groupby('month')['passengers'].mean()
        
        # Need at least 6 months of data and positive mean
        if len(monthly_avgs) >= 6 and monthly_avgs.mean() > 0:
            # Coefficient of variation = std / mean
            cv = monthly_avgs.std() / monthly_avgs.mean()
            seasonality_data.append({
                'origin': origin,
                'destination': dest,
                'route': f"{origin}-{dest}",
                'seasonality_strength': cv
            })
    
    return pd.DataFrame(seasonality_data)


def plot_seasonality_strength_cdf(df, config=CONFIG):
    """Plot cumulative distribution of seasonality strength for GVA vs ZRH routes."""
    seasonality_df = calculate_seasonality_strength(df)
    
    if len(seasonality_df) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by airport
    for airport in ['GVA', 'ZRH']:
        airport_data = seasonality_df[seasonality_df['origin'] == airport]['seasonality_strength']
        
        if len(airport_data) > 0:
            # Sort for CDF
            sorted_data = np.sort(airport_data)
            # Calculate cumulative probability
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            ax.plot(sorted_data, y, label=airport, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Seasonality Strength (CV of Monthly Averages)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Seasonality Strength Distribution: GVA vs ZRH', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    output = config.figures_dir / "seasonality_strength_cdf.png"
    _save_figure(fig, output)
    return output


# GVA VS ZRH COMPARISON

def plot_enhanced_route_trend(df, route, config=CONFIG):
    """Enhanced route trend plot with annotations and trend lines."""
    origin, dest = route
    subset = df[(df["origin_airport"] == origin) & (df["destination_airport"] == dest)].copy()
    
    if len(subset) == 0:
        return None
    
    subset = subset.sort_values("date")
    subset = subset.reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Main passenger line
    ax.plot(subset["date"], subset["passengers"], label="Passengers", linewidth=2, color='steelblue', alpha=0.8)
    
    # Rolling mean
    subset["rolling_mean"] = subset["passengers"].rolling(window=12, min_periods=1).mean()
    ax.plot(subset["date"], subset["rolling_mean"], label="12-Month Rolling Mean", 
            linewidth=2.5, color='red', linestyle='--', alpha=0.7)
    
    # Trend line
    x_numeric = np.arange(len(subset))
    slope, intercept, r_value, p_value, std_err = linregress(x_numeric, subset["passengers"].values)
    trend_line = intercept + slope * x_numeric
    ax.plot(subset["date"], trend_line, label=f"Trend (slope: {slope:.1f}/month)", 
            linewidth=2, color='green', linestyle=':', alpha=0.8)
    
    # Highlight COVID period
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2021-12-31')
    ax.axvspan(covid_start, covid_end, alpha=0.2, color='red', label='COVID Period')
    
    # Annotate key points
    if len(subset) > 0:
        max_idx = subset["passengers"].idxmax()
        min_idx = subset["passengers"].idxmin()
        ax.annotate(f'Peak: {subset.loc[max_idx, "passengers"]:.0f}', 
                   xy=(subset.loc[max_idx, "date"], subset.loc[max_idx, "passengers"]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_title(f"{origin}-{dest} Passenger Trend (Enhanced Analysis)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Passengers", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    safe_dest = dest.replace('/', '_').replace(' ', '_')
    output = config.figures_dir / f"enhanced_trend_{origin}_{safe_dest}.png"
    _save_figure(fig, output)
    return output


# SIMPLE GROWING AND DECLINING ROUTES - TABLES AND CHARTS

def create_growing_declining_tables(df, config=CONFIG):
    """
    Create simple CSV tables showing fastest growing and declining routes.
    Creates separate files for GVA and ZRH, each with top 15 routes.
    This matches the charts which also show 15 routes per airport.
    """
    trends_df = calculate_route_trends(df)
    
    if len(trends_df) == 0:
        return None, None
    
    # Only use routes with meaningful baseline (>= 100 passengers/month pre-COVID)
    routes_with_pct = trends_df[trends_df['recent_growth_pct'].notna()].copy()
    
    if len(routes_with_pct) == 0:
        print("No routes with meaningful pre-COVID baseline found for tables")
        return None, None
    
    # Separate routes by airport FIRST (matching chart logic)
    gva_routes = routes_with_pct[routes_with_pct['origin'] == 'GVA'].copy()
    zrh_routes = routes_with_pct[routes_with_pct['origin'] == 'ZRH'].copy()
    
    saved_files = []
    
    def format_and_save_table(routes_df, file_path, sort_ascending=False):
        """Format routes dataframe and save to CSV."""
        if len(routes_df) == 0:
            return None
        
        formatted = routes_df[['route', 'origin', 'destination', 'recent_growth_pct', 
                                'post_covid_avg', 'pre_covid_avg']].copy()
        formatted['recent_growth_pct'] = formatted['recent_growth_pct'].round(1)
        formatted['post_covid_avg'] = formatted['post_covid_avg'].round(0)
        formatted['pre_covid_avg'] = formatted['pre_covid_avg'].round(0)
        formatted.columns = ['Route', 'Origin', 'Destination', 'Growth (%)', 
                            'Post-COVID Avg Passengers/Month (2023+)', 'Pre-COVID Avg Passengers/Month (2017-2019)']
        
        formatted = formatted.sort_values('Growth (%)', ascending=sort_ascending)
        formatted.to_csv(file_path, index=False)
        return file_path
    
    # Process all 4 combinations: GVA/ZRH × growing/declining
    for airport, airport_routes in [('GVA', gva_routes), ('ZRH', zrh_routes)]:
        for route_type, is_growing in [('growing', True), ('declining', False)]:
            if is_growing:
                filtered = airport_routes[airport_routes['recent_growth_pct'] > 0].copy()
                if len(filtered) > 0:
                    # Always show top 15 routes (or all if fewer available)
                    top_routes = filtered.nlargest(15, 'recent_growth_pct')
                else:
                    top_routes = pd.DataFrame()
            else:
                # For declining routes, exclude routes with > -75% decline (likely cancellations)
                filtered = airport_routes[airport_routes['recent_growth_pct'] < 0].copy()
                filtered = filtered[filtered['recent_growth_pct'] > -75].copy()
                if len(filtered) > 0:
                    # Always show top 15 routes (or all if fewer available)
                    top_routes = filtered.nsmallest(15, 'recent_growth_pct')
                else:
                    top_routes = pd.DataFrame()
            
            if len(top_routes) > 0:
                path = config.results_dir / f"fastest_{route_type}_routes_{airport}.csv"
                format_and_save_table(top_routes, path, sort_ascending=not is_growing)
                saved_files.append(path)
                print(f"Saved {airport} fastest {route_type} routes table ({len(top_routes)} routes) to {path}")
    
    return saved_files


def _plot_route_chart(df, top_n, origin_airport, is_growing, config=CONFIG):
    """Helper function to plot growing or declining routes."""
    trends_df = calculate_route_trends(df)
    
    if len(trends_df) == 0:
        return None
    
    if origin_airport:
        trends_df = trends_df[trends_df['origin'] == origin_airport].copy()
        if len(trends_df) == 0:
            return None
    
    routes_with_pct = trends_df[trends_df['recent_growth_pct'].notna()].copy()
    if len(routes_with_pct) == 0:
        return None
    
    if is_growing:
        filtered_routes = routes_with_pct[routes_with_pct['recent_growth_pct'] > 0].copy()
        if len(filtered_routes) == 0:
            return None
        # Always show top 15 routes (or all if fewer available)
        top_routes = filtered_routes.nlargest(top_n, 'recent_growth_pct')
        color = 'green'
        edge_color = 'darkgreen'
        text_offset = 1
        text_ha = 'left'
        chart_type = 'growing'
        explanation = 'Growth rate shows percentage change in average monthly passengers (2023+ vs pre-COVID baseline).'
    else:
        # For declining routes, exclude routes with > -75% decline (likely cancellations)
        filtered_routes = routes_with_pct[routes_with_pct['recent_growth_pct'] < 0].copy()
        filtered_routes = filtered_routes[filtered_routes['recent_growth_pct'] > -75].copy()
        if len(filtered_routes) == 0:
            return None
        # Always show top 15 routes (or all if fewer available)
        top_routes = filtered_routes.nsmallest(top_n, 'recent_growth_pct')
        color = 'red'
        edge_color = 'darkred'
        text_offset = -1
        text_ha = 'right'
        chart_type = 'declining'
        explanation = 'Negative growth means fewer passengers now compared to pre-COVID baseline.'
    
    if len(top_routes) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(top_routes) * 0.5)))
    
    if origin_airport:
        route_labels = [f"{row['destination'][:45]}" for _, row in top_routes.iterrows()]
        title_airport = origin_airport
    else:
        route_labels = [f"{row['origin']} → {row['destination'][:40]}" for _, row in top_routes.iterrows()]
        title_airport = "All Airports"
    
    ax.barh(range(len(top_routes)), top_routes['recent_growth_pct'], 
            color=color, alpha=0.8, edgecolor=edge_color, linewidth=1)
    
    for i, (idx, row) in enumerate(top_routes.iterrows()):
        growth_pct = row['recent_growth_pct']
        ax.text(growth_pct + text_offset, i, f'{growth_pct:.1f}%', 
                va='center', ha=text_ha, fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(top_routes)))
    ax.set_yticklabels(route_labels, fontsize=11)
    ax.set_xlabel('Growth Rate (%)', fontsize=13, fontweight='bold')
    chart_title = 'Growing' if is_growing else 'Declining'
    ax.set_title(f'Top {len(top_routes)} Fastest {chart_title} Routes - {title_airport}\n(Comparing 2023+ vs Pre-COVID 2017-2019)', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    if origin_airport:
        output = config.figures_dir / f"fastest_{chart_type}_routes_{origin_airport}.png"
    else:
        output = config.figures_dir / f"fastest_{chart_type}_routes_chart.png"
    _save_figure(fig, output)
    return output


def plot_fastest_growing_routes(df, top_n=15, origin_airport=None, config=CONFIG):
    """Simple bar chart showing the fastest growing routes."""
    return _plot_route_chart(df, top_n, origin_airport, is_growing=True, config=config)


def plot_fastest_declining_routes(df, top_n=15, origin_airport=None, config=CONFIG):
    """Simple bar chart showing the fastest declining routes."""
    return _plot_route_chart(df, top_n, origin_airport, is_growing=False, config=config)


def extract_seasonal_patterns(df):
    """Extract normalized monthly patterns for each route (12 values: Jan-Dec as percentages)."""
    df = df.copy()
    df['month'] = df['date'].dt.month
    
    patterns = []
    
    for (origin, dest), group in df.groupby(['origin_airport', 'destination_airport']):
        # Get average passengers per month
        monthly_avgs = group.groupby('month')['passengers'].mean()
        
        # Need at least 6 months of data
        if len(monthly_avgs) >= 6 and monthly_avgs.sum() > 0:
            # Reindex to ensure we have all 12 months (fill missing with 0)
            all_months = pd.Series(index=range(1, 13), dtype=float).fillna(0)
            monthly_avgs_full = monthly_avgs.reindex(all_months.index, fill_value=0)
            
            # Normalize to percentages (so volume doesn't matter, only pattern)
            total = monthly_avgs_full.sum()
            if total > 0:
                full_pattern = (monthly_avgs_full / total * 100).values
            else:
                full_pattern = np.zeros(12)
            
            patterns.append({
                'origin': origin,
                'destination': dest,
                'route': f"{origin}-{dest}",
                'pattern': full_pattern
            })
    
    return pd.DataFrame(patterns)


def cluster_destinations_by_seasonality(df, n_clusters=4, config=CONFIG):
    """Cluster destinations by their seasonal patterns and visualize."""
    
    # Extract seasonal patterns
    patterns_df = extract_seasonal_patterns(df)
    
    if len(patterns_df) < n_clusters:
        print(f"Not enough routes for {n_clusters} clusters. Skipping clustering.")
        return None
    
    # Prepare data for clustering (12 monthly percentages per route)
    X = np.array([p for p in patterns_df['pattern']])
    
    # Cluster routes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    patterns_df['cluster'] = clusters
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Destination Clusters by Seasonal Behavior', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        cluster_routes = patterns_df[patterns_df['cluster'] == cluster_id]
        
        if len(cluster_routes) == 0:
            ax.text(0.5, 0.5, f'No routes in cluster {cluster_id}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cluster {cluster_id + 1}', fontsize=12, fontweight='bold')
            continue
        
        # Plot average pattern for this cluster
        cluster_patterns = np.array([p for p in cluster_routes['pattern']])
        avg_pattern = cluster_patterns.mean(axis=0)
        
        ax.plot(months, avg_pattern, marker='o', linewidth=2, markersize=6, color=f'C{cluster_id}')
        ax.fill_between(months, avg_pattern, alpha=0.3, color=f'C{cluster_id}')
        ax.set_title(f'Cluster {cluster_id + 1} ({len(cluster_routes)} routes)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('Passengers (%)', fontsize=10)
        ax.set_xticks(months[::2])  # Show every other month
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output = config.figures_dir / "seasonal_clusters.png"
    _save_figure(fig, output)
    
    # Save cluster assignments to CSV
    cluster_csv = patterns_df[['route', 'origin', 'destination', 'cluster']].copy()
    cluster_csv['cluster'] = cluster_csv['cluster'] + 1  # Make clusters 1-indexed
    cluster_csv = cluster_csv.sort_values(['cluster', 'route'])
    csv_output = config.results_dir / "seasonal_clusters.csv"
    cluster_csv.to_csv(csv_output, index=False)
    
    print(f"Saved seasonal clusters to {output}")
    print(f"Saved cluster assignments to {csv_output}")
    
    return output
