#!/usr/bin/env python3
"""
Command-line interface for the Cryptocurrency Volatility Forecasting Toolkit
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Volatility Forecasting Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  crypto-forecast --config config.json --target BTC
  crypto-forecast --quick-run --target ETH --days 30
        """
    )
    
    parser.add_argument(
        "--config", 
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--target",
        default="BTC", 
        help="Target cryptocurrency (default: BTC)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Lookback days for data collection (default: 90)"
    )
    
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Run quick example with reduced parameters"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna optimization trials (default: 50)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="OutputData",
        help="Output directory for results (default: OutputData)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a config.json file or specify a different path with --config")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Import modules (do this after arg parsing to show help faster)
        from .config import load_config_from_file
        from .data.collectors import CryptoDataCollector
        from .features.engineering import CryptoFeatureEngineer
        from .models.pipeline import CryptoVolatilityMLPipeline
        from .utils.dask_helpers import create_optimized_dask_client, cleanup_dask_client
        
        print("Cryptocurrency Volatility Forecasting Toolkit")
        print("=" * 50)
        
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config_from_file(args.config)
        
        # Override config with CLI arguments
        config.data.target_coin = args.target
        config.data.lookback_days = args.days
        
        if args.quick_run:
            config.ml.n_trials = 5
            config.data.top_n = 10
            print("Quick run mode enabled")
        else:
            config.ml.n_trials = args.trials
        
        print(f"Target: {config.data.target_coin}")
        print(f"Lookback: {config.data.lookback_days} days")
        print(f"Optimization trials: {config.ml.n_trials}")
        
        # Initialize components
        print("\nInitializing data collector...")
        collector = CryptoDataCollector(
            timezone=config.data.timezone,
            top_n=config.data.top_n,
            lookback_days=config.data.lookback_days
        )
        
        print("Initializing feature engineer...")
        engineer = CryptoFeatureEngineer(
            extraction_settings=config.tsfresh.extraction_settings,
            fdr_level=config.tsfresh.fdr_level,
            time_window=config.tsfresh.time_window,
            random_seed=config.tsfresh.random_seed
        )
        
        print("Initializing ML pipeline...")
        ml_pipeline = CryptoVolatilityMLPipeline(
            n_trials=config.ml.n_trials,
            n_rounds=config.ml.n_rounds,
            eval_metric=config.ml.eval_metric,
            early_stopping_rounds=config.ml.early_stopping_rounds,
            splits=config.ml.splits,
            random_seed=config.ml.random_seed
        )
        
        # Create Dask client
        print("Setting up Dask cluster...")
        client = create_optimized_dask_client(
            n_workers=config.dask.n_workers,
            threads_per_worker=config.dask.threads_per_worker,
            memory_limit=config.dask.memory_limit,
            dashboard_port=config.dask.dashboard_port,
            processes=config.dask.processes
        )
        
        # Run the pipeline
        print("\nStarting data collection...")
        data_sources = collector.collect_all_data()
        
        if not any(df.empty for df in data_sources.values()):
            unified_data = collector.combine_data_sources(data_sources)
            print(f"Unified dataset: {unified_data.shape}")
            
            print("Preparing features and target...")
            X, y = engineer.prepare_target_variable(unified_data, target_coin=config.data.target_coin)
            
            if not X.empty and not y.empty:
                print("Adding technical indicators...")
                ta_indicators = engineer.compute_ta_indicators(X, price_prefix="prices_")
                
                if not ta_indicators.empty:
                    X_with_ta = X.join(ta_indicators, how='left').dropna()
                    common_idx = X_with_ta.index.intersection(y.index)
                    X = X_with_ta.loc[common_idx]
                    y = y.loc[common_idx]
                
                print("Running TSFresh feature extraction...")
                tsfresh_features = engineer.run_tsfresh_pipeline(X, y, client)
                
                print("Creating final feature set...")
                final_features = engineer.create_final_feature_set(
                    X_base=X,
                    y=y,
                    tsfresh_features=tsfresh_features,
                    include_ta_indicators=True
                )
                
                print(f"Final features: {final_features.shape}")
                
                print("Running ML pipeline...")
                results = ml_pipeline.run_complete_pipeline(
                    final_features=final_features,
                    client=client,
                    target_coin=config.data.target_coin,
                    optimize=True
                )
                
                # Display results
                print("\nRESULTS")
                print("=" * 30)
                study = results['study']
                metrics = results['metrics']
                
                print(f"Best {config.ml.eval_metric.upper()}: {study.best_value:.6f}")
                print(f"Test R²: {metrics['r2_score']:.6f}")
                print(f"Test MASE: {metrics['mase']:.6f}")
                
                # Save results
                results_file = os.path.join(args.output_dir, f"results_{args.target}_{args.days}days.txt")
                with open(results_file, 'w') as f:
                    f.write(f"Cryptocurrency Volatility Forecasting Results\\n")
                    f.write(f"Target: {args.target}\\n")
                    f.write(f"Lookback Days: {args.days}\\n")
                    f.write(f"Best {config.ml.eval_metric.upper()}: {study.best_value:.6f}\\n")
                    f.write(f"Test R²: {metrics['r2_score']:.6f}\\n")
                    f.write(f"Test MASE: {metrics['mase']:.6f}\\n")
                    f.write(f"\\nBest Parameters:\\n")
                    for param, value in study.best_params.items():
                        f.write(f"  {param}: {value}\\n")
                
                print(f"Results saved to: {results_file}")
                
            else:
                print("Error: Could not prepare features and target")
                sys.exit(1)
        else:
            print("Error: Data collection failed")
            sys.exit(1)
        
        # Cleanup
        cleanup_dask_client(client)
        print("\\nPipeline completed successfully")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all requirements are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()