"""Main script to run the full forecasting pipeline.

Main entry point for Swiss Air Passenger Demand Forecasting.

This script:
- Automatically selects interesting routes (20 routes by default)
- Loads and preprocesses all available routes
- Trains ML models (Random Forest, XGBoost, MLP) on selected routes
- Trains SARIMA models (one per route) on selected routes
- Makes predictions for selected routes
- Evaluates model performance (consistent comparison between ML and SARIMA)
- Generates visualizations

Usage:

    python main.py

Note: Both ML and SARIMA models train and evaluate on the same selected routes
      to ensure a fair and consistent comparison.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CONFIG
from src.run_pipeline import run_pipeline


def main():
    """Main entry point"""
    
    # Step 1: Auto-select routes for model training and evaluation
    # Both ML and SARIMA models train on the same selected routes for comparison
    if not CONFIG.routes:
        print("=" * 70)
        print("Auto-selecting routes...")
        print("=" * 70)
        print("\nNo routes configured. Analyzing all routes to find interesting ones.")
        print("This will select a mix of:")
        print("  - High-growth routes")
        print("  - Declining routes")
        print("  - High-volume routes")
        print("  - Routes with interesting patterns\n")
        
        try:
            # Import the route analysis script
            from scripts.route_analysis import analyze_and_select_routes
            
            # Run analysis and select 20 interesting routes
            stats_df, selected_df, routes_list = analyze_and_select_routes(
                config=CONFIG, 
                n_routes=20
            )
            
            if routes_list:
                CONFIG.routes = routes_list
                print(f"\n✓ Selected {len(routes_list)} routes for model training and evaluation")
                print("  (Both ML and SARIMA models will train on these same routes)")
                print("\nSelected routes:")
                for i, (origin, dest) in enumerate(routes_list[:10], 1):
                    print(f"  {i}. {origin} → {dest}")
                if len(routes_list) > 10:
                    print(f"  ... and {len(routes_list) - 10} more")
                print("\n" + "=" * 70 + "\n")
            else:
                print("\nNote: No routes selected for focused analysis.")
                print("Pipeline will use all available routes (this is fine).\n")
                
        except Exception as e:
            print(f"\nNote: Route auto-selection failed: {e}")
            print("Pipeline will proceed with all available routes (this is fine).\n")
    
    # Step 2: Run the main pipeline
    print("=" * 70)
    print("Starting main pipeline...")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Load and clean all available routes")
    print("  2. Filter to selected routes (for consistent ML vs SARIMA comparison)")
    print("  3. Create features (lags, rolling averages, etc.)")
    print("  4. Train ML models on selected routes (Random Forest, XGBoost, MLP)")
    print("  5. Train SARIMA models (one per route, same selected routes)")
    print("  6. Make predictions for selected routes")
    print("  7. Evaluate model performance")
    print("  8. Generate visualizations")
    print("\n" + "=" * 70 + "\n")
    
    try:
        run_pipeline()
        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print("=" * 70)
        print("\nResults saved to:")
        print(f"  - {CONFIG.results_dir}/ml_metrics.csv")
        print(f"  - {CONFIG.results_dir}/sarima_metrics.csv")
        print(f"  - {CONFIG.results_dir}/figures/")
        print(f"  - {CONFIG.results_dir}/fastest_growing_routes_GVA.csv")
        print(f"  - {CONFIG.results_dir}/fastest_growing_routes_ZRH.csv")
        print(f"  - {CONFIG.results_dir}/fastest_declining_routes_GVA.csv")
        print(f"  - {CONFIG.results_dir}/fastest_declining_routes_ZRH.csv")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Pipeline failed")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

