"""
Sample Data Generator
=====================
Generates sample price data CSV for testing without internet connection.

Usage:
    python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_correlated_returns(n_days, n_assets, correlation=0.3, seed=42):
    """
    Generate correlated returns for multiple assets.
    
    Args:
        n_days: Number of trading days
        n_assets: Number of assets
        correlation: Average correlation between assets
        seed: Random seed
    
    Returns:
        DataFrame of returns
    """
    np.random.seed(seed)
    
    # Create correlation matrix
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated random returns
    uncorrelated = np.random.normal(0, 1, (n_days, n_assets))
    
    # Apply correlation
    correlated = uncorrelated @ L.T
    
    return correlated


def generate_sample_prices(
    start_date='2020-01-01',
    end_date='2025-10-30',
    tickers=None,
    initial_prices=None,
    annual_returns=None,
    annual_vols=None,
    correlation=0.3
):
    """
    Generate realistic sample price data.
    
    Args:
        start_date: Start date
        end_date: End date
        tickers: List of ticker symbols
        initial_prices: Initial prices for each ticker
        annual_returns: Expected annual returns
        annual_vols: Annual volatilities
        correlation: Average correlation
    
    Returns:
        DataFrame of prices
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'VTI', 'VOO', 'BND', 'GLD', 'BTC-USD', 'SPY']
    
    if initial_prices is None:
        initial_prices = [150.0, 300.0, 200.0, 400.0, 80.0, 180.0, 45000.0, 420.0]
    
    if annual_returns is None:
        # Expected annual returns (stocks ~10%, bonds ~3%, gold ~5%, crypto ~50%)
        annual_returns = [0.10, 0.12, 0.09, 0.10, 0.03, 0.05, 0.50, 0.10]
    
    if annual_vols is None:
        # Annual volatilities
        annual_vols = [0.30, 0.28, 0.18, 0.18, 0.06, 0.20, 0.80, 0.19]
    
    # Generate date range (business days only)
    dates = pd.bdate_range(start_date, end_date)
    n_days = len(dates)
    n_assets = len(tickers)
    
    # Convert annual stats to daily
    daily_returns = np.array(annual_returns) / 252
    daily_vols = np.array(annual_vols) / np.sqrt(252)
    
    # Generate correlated returns
    standardized_returns = generate_correlated_returns(n_days, n_assets, correlation)
    
    # Scale by volatility and add drift
    returns = daily_returns + standardized_returns * daily_vols
    
    # Generate prices via geometric Brownian motion
    prices = np.zeros((n_days, n_assets))
    prices[0, :] = initial_prices
    
    for i in range(1, n_days):
        prices[i, :] = prices[i-1, :] * (1 + returns[i, :])
    
    # Create DataFrame
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    
    # Add some realistic features
    
    # 1. Market crash (simulate COVID crash in March 2020)
    crash_start = pd.Timestamp('2020-02-20')
    crash_end = pd.Timestamp('2020-03-23')
    if crash_start in df.index:
        crash_mask = (df.index >= crash_start) & (df.index <= crash_end)
        crash_indices = df.index[crash_mask]
        
        # Stocks crash 30-35%, bonds stable, gold up, crypto crashes
        crash_factors = [0.70, 0.68, 0.72, 0.70, 1.02, 1.10, 0.50, 0.70]
        
        for i, ticker in enumerate(tickers):
            if len(crash_indices) > 0:
                df.loc[crash_indices, ticker] *= crash_factors[i]
    
    # 2. Recovery rally
    rally_start = pd.Timestamp('2020-04-01')
    rally_end = pd.Timestamp('2020-12-31')
    if rally_start in df.index:
        rally_mask = (df.index >= rally_start) & (df.index <= rally_end)
        rally_indices = df.index[rally_mask]
        
        # Stocks recover strongly
        rally_factors = [1.5, 1.6, 1.4, 1.4, 1.0, 1.1, 2.0, 1.4]
        
        for i, ticker in enumerate(tickers):
            if len(rally_indices) > 0:
                df.loc[rally_indices, ticker] *= rally_factors[i]
    
    # 3. Add some missing data (simulate trading halts or data gaps)
    for ticker in ['GLD', 'BTC-USD']:
        if ticker in df.columns:
            # Random 2-3 day gaps
            gap_start = np.random.randint(100, len(df) - 10)
            df.loc[df.index[gap_start:gap_start+3], ticker] = np.nan
    
    return df


def main():
    """Generate and save sample data."""
    print("Generating sample price data...")
    
    # Create sample_data directory
    sample_dir = Path('sample_data')
    sample_dir.mkdir(exist_ok=True)
    
    # Generate data
    prices = generate_sample_prices()
    
    # Save to CSV
    output_file = sample_dir / 'prices_sample.csv'
    prices.to_csv(output_file)
    
    print(f"✓ Generated {len(prices)} days of data for {len(prices.columns)} tickers")
    print(f"✓ Saved to {output_file}")
    print(f"\nDate range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\nTickers: {', '.join(prices.columns)}")
    print(f"\nSample data (first 5 rows):")
    print(prices.head())
    
    # Print statistics
    print(f"\nPrice statistics:")
    print(prices.describe())
    
    # Save summary
    summary = {
        'n_days': len(prices),
        'n_tickers': len(prices.columns),
        'date_range': f"{prices.index[0]} to {prices.index[-1]}",
        'tickers': list(prices.columns),
        'missing_data': prices.isnull().sum().to_dict()
    }
    
    import json
    with open(sample_dir / 'data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Summary saved to {sample_dir / 'data_summary.json'}")
    
    # Test reading the data
    print(f"\n✓ Testing data load...")
    test_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
    assert len(test_df) == len(prices)
    print("✓ Data loads correctly!")
    
    print("\nYou can now use this sample data with:")
    print("  python data_fetcher.py --portfolio sample_data/example_portfolio.json \\")
    print("    --source sample_data/prices_sample.csv \\")
    print("    --start 2020-01-01 --end 2025-10-30 \\")
    print("    --output data/prices.parquet")


if __name__ == '__main__':
    main()