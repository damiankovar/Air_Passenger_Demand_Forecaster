# Script to analyze routes in the BFS dataset and select interesting ones
# The idea: instead of picking routes because they're famous (like London, New York),
# I analyze the actual data to find routes that are interesting for different reasons:
# - Big routes with lots of passengers 
# - Growing routes 
# - Declining routes 
# - Small but fast-growing routes (surprising picks)
# - Very seasonal routes 

# Usage: Run this once when you first get your BFS datasets
# It will analyze everything and suggest a mix of routes
# Then copy the suggested routes into src/config.py

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import CONFIG
from src import data_loading
from src.data_loading import normalize_destination_name


def is_valid_route_destination(destination):
    """Check if destination is a real route, not just a region or country."""
    dest_str = str(destination).strip()
    
    if dest_str.startswith('Other airports:'):
        return False
    
    # Filter out generic regions and country names
    invalid_names = {
        'Europe', 'Asia', 'Africa', 'North America', 'South America', 'Middle East', 
        'Oceania', 'Antarctica', 'United Kingdom', 'UK', 'France', 'Germany', 'Italy', 
        'Spain', 'Switzerland', 'Netherlands', 'Belgium', 'Austria', 'Greece', 'Portugal', 
        'Poland', 'Czech Republic', 'Hungary', 'Romania', 'Bulgaria', 'Croatia', 'Slovenia', 
        'Slovakia', 'Denmark', 'Sweden', 'Norway', 'Finland', 'Ireland', 'United States', 
        'USA', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia', 
        'China', 'Japan', 'India', 'South Korea', 'Thailand', 'Malaysia', 'Singapore', 
        'Indonesia', 'Philippines', 'Vietnam', 'Australia', 'New Zealand', 'South Africa', 
        'Egypt', 'Morocco', 'Turkey', 'Israel', 'Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Bahrain'
    }
    
    return dest_str not in invalid_names


def calculate_route_statistics(df):
    # Go route by route and compute all the things I care about:
    # trend (without covid years), seasonality, growth, etc.
    results = []
    from scipy.stats import linregress
    
    # Normalize duplicate destinations - combine destinations that represent the same place
    df = df.copy()
    df['destination_airport'] = df['destination_airport'].apply(normalize_destination_name)
    # Filter out None values (dropped country names)
    df = df[df['destination_airport'].notna()].copy()
    # Re-aggregate in case we combined destinations - sum passengers if same month/origin/dest
    df = df.groupby(['date', 'origin_airport', 'destination_airport'])['passengers'].sum().reset_index()
    
    for (origin, dest), group in df.groupby(['origin_airport', 'destination_airport']):
        if not is_valid_route_destination(dest):
            continue
        route_data = group.sort_values('date').copy()
        
        months_count = len(route_data)
        total_passengers = route_data['passengers'].sum()
        mean_passengers = route_data['passengers'].mean()
        
        # Calculate trend (excluding 2020-2021)
        normal_data = route_data[(route_data['date'] < pd.Timestamp('2020-01-01')) | 
                                 (route_data['date'] >= pd.Timestamp('2022-01-01'))]
        
        if len(normal_data) >= 24:
            x = np.arange(len(normal_data))
            y = normal_data['passengers'].values
            slope, _, _, _, _ = linregress(x, y)
            avg_passengers = normal_data['passengers'].mean()
            trend = (slope / max(avg_passengers, 1e-6)) * 100 if avg_passengers > 0 else 0
        elif len(normal_data) >= 12:
            mid = len(normal_data) // 2
            first_half = normal_data.head(mid)['passengers'].mean()
            second_half = normal_data.tail(len(normal_data) - mid)['passengers'].mean()
            trend = ((second_half - first_half) / max(first_half, 1e-6)) * 100 if first_half > 0 else 0
        else:
            trend = 0
        
        # Seasonality: how much variation by month?
        route_data['month'] = route_data['date'].dt.month
        monthly_avg = route_data.groupby('month')['passengers'].mean()
        seasonality_strength = (monthly_avg.std() / monthly_avg.mean()) * 100 if monthly_avg.mean() > 0 else 0
        
        # Recent growth: 2017-2019 vs 2023+
        baseline_period = route_data[(route_data['date'] >= pd.Timestamp('2017-01-01')) & 
                                     (route_data['date'] < pd.Timestamp('2020-01-01'))]
        recent_period = route_data[route_data['date'] >= pd.Timestamp('2023-01-01')]
        
        recent_growth_pct = None
        recent_avg = None
        if len(baseline_period) >= 12 and len(recent_period) >= 12:
            baseline_avg = baseline_period['passengers'].mean()
            recent_avg = recent_period['passengers'].mean()
            # Skip if baseline too small - avoids fake 9900% growth from tiny numbers
            if baseline_avg >= 100:
                recent_growth_pct = ((recent_avg - baseline_avg) / baseline_avg) * 100
        
        # Test period average: last 6 months (what we'll evaluate on)
        test_period_avg = route_data.tail(6)['passengers'].mean() if len(route_data) >= 6 else None
        
        # Volatility: how consistent month-to-month
        if len(route_data) > 1:
            volatility = (route_data['passengers'].std() / route_data['passengers'].mean()) * 100 if route_data['passengers'].mean() > 0 else 0
        else:
            volatility = 0
        
        results.append({
            'origin': origin,
            'destination': dest,
            'route': f"{origin}-{dest}",
            'months_count': months_count,
            'total_passengers': total_passengers,
            'mean_passengers': mean_passengers,
            'trend': trend,
            'seasonality_strength': seasonality_strength,
            'recent_growth_pct': recent_growth_pct,  # Percentage change (None if baseline < 100)
            'recent_avg': recent_avg,  # 2023+ average passengers/month (None if < 12 months of data)
            'test_period_avg': test_period_avg,  # Last 6 months average (for evaluation)
            'volatility': volatility,
        })
    
    return pd.DataFrame(results)


def classify_routes(stats_df):
    # Put routes into categories like 'high_volume_stable_growth_low_seasonality'
    stats_df = stats_df.copy()
    
    # I use percentiles so it works no matter how many routes I end up with
    volume_p75 = stats_df['mean_passengers'].quantile(0.75)
    volume_p50 = stats_df['mean_passengers'].quantile(0.50)
    volume_p25 = stats_df['mean_passengers'].quantile(0.25)
    
    trend_p75 = stats_df['trend'].quantile(0.75)
    seasonality_p75 = stats_df['seasonality_strength'].quantile(0.75)
    
    categories = []
    
    for _, row in stats_df.iterrows():
        route_type = []
        
        if row['mean_passengers'] >= volume_p75:
            route_type.append('high_volume')
        elif row['mean_passengers'] >= volume_p50:
            route_type.append('mid_volume')
        else:
            route_type.append('low_volume')
        
        if row['trend'] >= trend_p75:
            route_type.append('high_growth')
        elif row['trend'] >= 0:
            route_type.append('stable_growth')
        else:
            route_type.append('declining')
        
        if row['seasonality_strength'] >= seasonality_p75:
            route_type.append('high_seasonality')
        else:
            route_type.append('low_seasonality')
        
        categories.append('_'.join(route_type))
    
    stats_df['category'] = categories
    return stats_df


def find_common_routes(stats_df):
    # Find destinations served by both airports - these are good for comparison
    zrh_routes = set(stats_df[stats_df['origin'] == 'ZRH']['destination'].unique())
    gva_routes = set(stats_df[stats_df['origin'] == 'GVA']['destination'].unique())
    common_destinations = zrh_routes.intersection(gva_routes)
    return list(common_destinations)


def calculate_comparison_interest(stats_df):
    # Score routes by how different ZRH and GVA are
    # Bigger differences = more interesting to compare
    common_destinations = find_common_routes(stats_df)
    
    if len(common_destinations) == 0:
        return pd.DataFrame()
    
    comparison_results = []
    
    for dest in common_destinations:
        zrh_route = stats_df[(stats_df['origin'] == 'ZRH') & (stats_df['destination'] == dest)]
        gva_route = stats_df[(stats_df['origin'] == 'GVA') & (stats_df['destination'] == dest)]
        
        if len(zrh_route) == 0 or len(gva_route) == 0:
            continue
        
        zrh_row = zrh_route.iloc[0]
        gva_row = gva_route.iloc[0]
        
        trend_diff = abs(zrh_row['trend'] - gva_row['trend'])
        seasonality_diff = abs(zrh_row['seasonality_strength'] - gva_row['seasonality_strength'])
        
        # This scoring is a bit arbitrary but it gives me a reasonable ranking
        comparison_score = (trend_diff + seasonality_diff) / 2
        
        comparison_results.append({
            'destination': dest,
            'zrh_origin': 'ZRH',
            'gva_origin': 'GVA',
            'zrh_trend': zrh_row['trend'],
            'gva_trend': gva_row['trend'],
            'trend_divergence': trend_diff,
            'seasonality_diff': seasonality_diff,
            'comparison_interest_score': comparison_score,
            'zrh_mean_passengers': zrh_row['mean_passengers'],
            'gva_mean_passengers': gva_row['mean_passengers'],
        })
    
    return pd.DataFrame(comparison_results)


def select_route_mix(stats_df, n_routes=20):
    # Simple approach: pick a mix of routes that are interesting
    # - Some high-growth routes (show potential)
    # - Some declining routes (show what lost popularity)
    # - Some comparison routes (destinations served by both GVA and ZRH with biggest differences)
    # - Rest: routes with interesting mix of volume, trends, volatility, seasonality
    stats_df = stats_df.copy()
    stats_df = stats_df[stats_df['destination'].apply(is_valid_route_destination)].copy()
    
    # Need at least 5 years of data
    stats_df = stats_df[stats_df['months_count'] >= 60].copy()
    
    # Exclude routes affected by conflicts, sanctions, or geopolitical issues
    # Also exclude Swiss destinations (domestic routes) 
    exclude_destinations = [
        'Belarus', 'Minsk', 'Russia', 'Russian', 'Moskva', 'Moscow', 
        'Domodedovo', 'Sheremetyevo', 'Vnukovo', 'Pulkovo', 'St Petersburg',
        'Ukraine', 'Kiev', 'Istanbul', 'Kleyate', 'Tripoli',
        'Lugano', 'Lugano-Agno', 'Zürich', 'Zurich', 'Genève', 'Geneva', 'Basel', 'Bern'
    ]
    
    exclude_mask = stats_df['destination'].apply(
        lambda dest: any(exclude_word.lower() in str(dest).lower() for exclude_word in exclude_destinations)
    )
    stats_df = stats_df[~exclude_mask].copy()
    
    if len(stats_df) == 0:
        print("Warning: No routes with enough data (need at least 60 months)")
        return pd.DataFrame()
    
    selected = []
    
    # First: get some high-growth routes (using recent_growth_pct)
    # These show routes that are really growing compared to pre-COVID
    growth_routes = stats_df[
        (stats_df['recent_growth_pct'].notna()) & 
        (stats_df['recent_growth_pct'] > 20) &  # At least 20% growth
        (stats_df['mean_passengers'] >= 200) &  # Decent volume
        ((stats_df['test_period_avg'].isna()) | (stats_df['test_period_avg'] >= 50))  # Test period has sufficient passengers
    ].copy()
    
    if len(growth_routes) > 0:
        growth_sorted = growth_routes.sort_values('recent_growth_pct', ascending=False)
        n_growth = min(5, len(growth_sorted))  # Get up to 5 high-growth routes
        selected.extend(growth_sorted.head(n_growth).index.tolist())
    
    # Second: get some declining routes (using recent_growth_pct)
    # These show destinations that lost popularity
    # But exclude routes with >75% decline - these are likely airline-driven, not market-driven
    selected_indices = set(selected)
    declining_routes = stats_df[
        (stats_df['recent_growth_pct'].notna()) & 
        (stats_df['recent_growth_pct'] < -20) &  # At least 20% decline
        (stats_df['recent_growth_pct'] > -75) &  # But not >75% decline (airline-driven, not market-driven)
        (stats_df['mean_passengers'] >= 200) &  # Decent volume
        ((stats_df['test_period_avg'].isna()) | (stats_df['test_period_avg'] >= 50)) &  # Test period has sufficient passengers
        (~stats_df.index.isin(selected_indices))
    ].copy()
    
    if len(declining_routes) > 0:
        declining_sorted = declining_routes.sort_values('recent_growth_pct')  # Most negative first
        n_declining = min(5, len(declining_sorted))  # Get up to 5 declining routes
        selected.extend(declining_sorted.head(n_declining).index.tolist())
    
    # Step 2.5: Select interesting comparison routes (GVA vs ZRH)
    # Find destinations served by both airports with biggest differences
    selected_indices = set(selected)
    comparison_df = calculate_comparison_interest(stats_df)
    
    if len(comparison_df) > 0:
        # Sort by comparison score (biggest differences = most interesting)
        comparison_sorted = comparison_df.sort_values('comparison_interest_score', ascending=False)
        
        # Select top 3 comparison destinations (this gives us 6 routes: 3 destinations × 2 airports)
        n_comparisons = min(3, len(comparison_sorted))
        top_comparison_dests = comparison_sorted.head(n_comparisons)['destination'].tolist()
        
        # Add both GVA and ZRH routes for each selected destination
        for dest in top_comparison_dests:
            zrh_route = stats_df[(stats_df['origin'] == 'ZRH') & (stats_df['destination'] == dest)]
            gva_route = stats_df[(stats_df['origin'] == 'GVA') & (stats_df['destination'] == dest)]
            
            # Only add if both routes exist and aren't already selected
            if len(zrh_route) > 0 and len(gva_route) > 0:
                zrh_idx = zrh_route.index[0]
                gva_idx = gva_route.index[0]
                if zrh_idx not in selected_indices and gva_idx not in selected_indices:
                    selected.extend([zrh_idx, gva_idx])
    
    # Third: fill rest with interesting routes
    # Score routes by multiple factors: volume, trend, seasonality, volatility
    # This gives us routes that are interesting for different reasons, not just big
    remaining_slots = n_routes - len(selected)
    
    if remaining_slots > 0:
        selected_indices = set(selected)
        all_routes = stats_df[~stats_df.index.isin(selected_indices)].copy()
        
        # Filter out routes with too little volume (otherwise we get routes with 0-5 passengers)
        all_routes = all_routes[all_routes['mean_passengers'] >= 100].copy()
        
        # Apply same test-period filter as other categories: require test_period_avg >= 50 passengers/month
        # This filter improves stability and comparability of model evaluation by avoiding very low-volume
        # routes that would distort metrics and plots. Routes with < 50 passengers/month in test period
        # are essentially discontinued and not suitable for meaningful evaluation.
        all_routes = all_routes[
            (all_routes['test_period_avg'].isna()) |  # No test period data (keep it)
            (all_routes['test_period_avg'] >= 50)     # Or test period avg >= 50 passengers/month
        ].copy()
        
        if len(all_routes) > 0:
            # Score routes by combining different factors
            def score_route_interest(route_df):
                route_df = route_df.copy()
                
                # Normalize each factor to 0-1 scale so we can combine them
                max_volume = route_df['mean_passengers'].max()
                max_trend_abs = route_df['trend'].abs().max()
                max_seasonality = route_df['seasonality_strength'].max()
                max_volatility = route_df['volatility'].max()
                
                volume_score = route_df['mean_passengers'] / max_volume if max_volume > 0 else 0
                trend_score = route_df['trend'].abs() / max_trend_abs if max_trend_abs > 0 else 0
                seasonality_score = route_df['seasonality_strength'] / max_seasonality if max_seasonality > 0 else 0
                volatility_score = route_df['volatility'] / max_volatility if max_volatility > 0 else 0
                
                # Equal weight to all four - simple but works
                route_df['interest_score'] = (volume_score + trend_score + seasonality_score + volatility_score) / 4
                return route_df
            
            all_routes_scored = score_route_interest(all_routes)
            interesting_sorted = all_routes_scored.sort_values('interest_score', ascending=False)
            selected.extend(interesting_sorted.head(remaining_slots).index.tolist())
    
    selected = list(set(selected))
    
    # Trim to exactly n_routes if we got too many
    if len(selected) > n_routes:
        selected = selected[:n_routes]
    
    if len(selected) > 0:
        return stats_df.loc[selected]
    else:
        return pd.DataFrame()


def analyze_and_select_routes(config=CONFIG, output_path=None, n_routes=20):
    # Main function - does everything: load data, analyze routes, suggest selection
    print("=" * 60)
    print("Route Analysis")
    print("=" * 60)
    print("\nLoading data...")
    try:
        df = data_loading.load_raw_data(config)
        print(f"Loaded {len(df)} rows of data")
        
        required_cols = ['date', 'origin_airport', 'destination_airport', 'passengers']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"ERROR: Missing columns: {missing_cols}")
            return None, None, None
        
        unique_origins = df['origin_airport'].nunique()
        unique_destinations = df['destination_airport'].nunique()
        date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
        
        print(f"  Date range: {date_range}")
        print(f"  Origins: {unique_origins}, Destinations: {unique_destinations}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    print("\nFiltering routes by volume (speeds up analysis)...")
    # Only analyze routes with decent volume - filters out tiny routes that aren't interesting
    # This speeds up route selection significantly
    min_volume = 50  # Only analyze routes with at least 50 passengers/month on average
    
    # Calculate average passengers per route
    route_volumes = df.groupby(['origin_airport', 'destination_airport'])['passengers'].mean().reset_index()
    original_route_count = len(route_volumes)
    
    # Get routes that meet the volume threshold
    valid_routes = route_volumes[route_volumes['passengers'] >= min_volume][['origin_airport', 'destination_airport']]
    
    # Filter dataframe to only include routes with decent volume (using merge - fast!)
    df_filtered = df.merge(valid_routes, on=['origin_airport', 'destination_airport'], how='inner')
    
    filtered_route_count = len(valid_routes)
    
    print(f"  Filtered from {original_route_count} routes to {filtered_route_count} routes")
    print(f"  (Only analyzing routes with ≥{min_volume} passengers/month average)")
    
    print("\nCalculating route statistics...")
    stats_df = calculate_route_statistics(df_filtered)  # Use filtered data
    print(f"Found {len(stats_df)} unique routes")
    
    print("Classifying routes...")
    stats_df = classify_routes(stats_df)
    
    print(f"\nSelecting {n_routes} diverse routes...")
    selected_df = select_route_mix(stats_df, n_routes=n_routes)
    
    if len(selected_df) == 0:
        print("No routes selected. Check your data.")
        return stats_df, pd.DataFrame(), []
    
    routes_list = [
        (row['origin'], row['destination'])
        for _, row in selected_df.iterrows()
    ]
    
    if output_path is None:
        output_path = config.results_dir / "route_analysis.csv"
    
    config.results_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    selected_path = config.results_dir / "selected_routes.csv"
    selected_df.to_csv(selected_path, index=False)
    
    # Note: We no longer generate declining_routes.csv since we have
    # fastest_declining_routes_GVA.csv and fastest_declining_routes_ZRH.csv
    # which match the charts (15 routes per airport)
    declining_path = None
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nTotal routes in dataset: {len(stats_df)}")
    print(f"Selected routes: {len(selected_df)}")
    
    print(f"\n{'='*60}")
    print("Selected Routes with Stats:")
    print(f"{'-'*80}")
    for _, row in selected_df.iterrows():
        is_declining = pd.notna(row.get('recent_growth_pct')) and row['recent_growth_pct'] < -20
        is_growing = pd.notna(row.get('recent_growth_pct')) and row['recent_growth_pct'] > 20
        
        markers = []
        if is_declining:
            markers.append("[DECLINING]")
        if is_growing:
            markers.append("[GROWING]")
        marker_str = " " + " ".join(markers) if markers else ""
        
        growth_info = f" | Growth: {row['recent_growth_pct']:+.1f}%" if pd.notna(row.get('recent_growth_pct')) else ""
        
        print(f"  {row['route']:30s}{marker_str} | "
              f"Avg: {row['mean_passengers']:8.0f}/month | "
              f"Trend: {row['trend']:6.1f}%{growth_info}")
    
    # Show summary of what we selected
    declining_count = len(selected_df[(selected_df['recent_growth_pct'].notna()) & (selected_df['recent_growth_pct'] < -20)])
    growing_count = len(selected_df[(selected_df['recent_growth_pct'].notna()) & (selected_df['recent_growth_pct'] > 20)])
    
    if declining_count > 0:
        print(f"\nDeclining routes included (lost popularity): {declining_count}")
        declining_routes = selected_df[(selected_df['recent_growth_pct'].notna()) & (selected_df['recent_growth_pct'] < -20)]
        for _, row in declining_routes.iterrows():
            print(f"  {row['route']:30s} | {row['recent_growth_pct']:+.1f}% growth | {row['mean_passengers']:.0f} passengers/month")
    
    if growing_count > 0:
        print(f"\nGrowing routes included: {growing_count}")
        growing_routes = selected_df[(selected_df['recent_growth_pct'].notna()) & (selected_df['recent_growth_pct'] > 20)]
        for _, row in growing_routes.iterrows():
            print(f"  {row['route']:30s} | {row['recent_growth_pct']:+.1f}% growth | {row['mean_passengers']:.0f} passengers/month")
    
    print("\n" + "=" * 60)
    print("Routes list for config.py:")
    print("=" * 60)
    print("Copy this into src/config.py, in the ProjectConfig class, routes field:")
    print()
    print("routes: List[Tuple[str, str]] = field(default_factory=lambda: [")
    for route in routes_list:
        print(f'    {route},')
    print("])")
    print()
    print(f"\nResults saved to:")
    print(f"  - {output_path} (all routes with stats)")
    print(f"  - {selected_path} (just the selected ones)")
    print("=" * 60)
    
    return stats_df, selected_df, routes_list


if __name__ == "__main__":
    # Create a custom config with your actual data files
    from pathlib import Path
    from src.config import ProjectConfig
    
    custom_config = ProjectConfig(
        base_dir=Path("."),
        data_paths=[
            Path("data/raw/GVA_Data_cleaned.csv"),  # Geneva destinations
            Path("data/raw/ZRH_Data_cleaned.csv"),   # Zurich destinations
        ]
    )
    
    # Run the analysis with the custom config
    stats_df, selected_df, routes_list = analyze_and_select_routes(config=custom_config)
    
    if routes_list:
        print("Analysis successful! Copy the routes list above into src/config.py")
