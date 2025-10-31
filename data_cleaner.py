"""
Data Cleaner Module
===================
Cleans, aligns, and preprocesses price data for risk analysis.
Handles missing data, computes returns, and provides resampling.

Usage:
    python data_cleaner.py --input data/prices.parquet \\
        --output data/cleaned_prices.parquet --returns-output data/returns.parquet
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_price_data(
    prices: pd.DataFrame,
    max_missing_pct: float = 0.2,
    forward_fill_limit: int = 5,
    backfill_limit: int = 2
) -> pd.DataFrame:
    """
    Clean and align price data, handling missing values.
    
    Args:
        prices: Raw price DataFrame (dates x tickers)
        max_missing_pct: Maximum allowed missing data ratio per ticker (0-1)
        forward_fill_limit: Max consecutive days to forward fill
        backfill_limit: Max consecutive days to backfill
    
    Returns:
        Cleaned price DataFrame
    """
    logger.info(f"Cleaning price data: {prices.shape}")
    logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Check for missing data
    missing_pct = prices.isnull().sum() / len(prices)
    logger.info(f"Missing data per ticker:\n{missing_pct[missing_pct > 0]}")
    
    # Drop tickers with too much missing data
    tickers_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
    if tickers_to_drop:
        logger.warning(f"Dropping tickers with >{max_missing_pct*100}% missing data: {tickers_to_drop}")
        prices = prices.drop(columns=tickers_to_drop)
    
    if prices.empty:
        raise ValueError("No tickers remaining after cleaning")
    
    # Forward fill limited gaps (e.g., market holidays)
    prices = prices.fillna(method='ffill', limit=forward_fill_limit)
    
    # Backfill limited gaps at the start
    prices = prices.fillna(method='bfill', limit=backfill_limit)
    
    # Drop any remaining rows with NaN
    before_drop = len(prices)
    prices = prices.dropna()
    after_drop = len(prices)
    
    if before_drop > after_drop:
        logger.warning(f"Dropped {before_drop - after_drop} rows with remaining NaN values")
    
    logger.info(f"Cleaned data shape: {prices.shape}")
    
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = 'log',
    freq: str = 'daily'
) -> pd.DataFrame:
    """
    Compute returns from price data.
    
    Args:
        prices: Price DataFrame (dates x tickers)
        method: 'log' for log returns or 'simple' for simple returns
        freq: Resampling frequency ('daily', 'weekly', 'monthly')
    
    Returns:
        Returns DataFrame
    """
    logger.info(f"Computing {method} returns at {freq} frequency")
    
    # Resample if needed
    if freq == 'weekly':
        prices_resampled = prices.resample('W-FRI').last()
    elif freq == 'monthly':
        prices_resampled = prices.resample('M').last()
    else:
        prices_resampled = prices
    
    # Compute returns
    if method == 'log':
        returns = np.log(prices_resampled / prices_resampled.shift(1))
    elif method == 'simple':
        returns = prices_resampled.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    # Drop first row (NaN)
    returns = returns.dropna()
    
    logger.info(f"Returns shape: {returns.shape}")
    logger.info(f"Sample statistics:\n{returns.describe()}")
    
    return returns


def align_data_with_benchmark(
    prices: pd.DataFrame,
    benchmark_ticker: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align portfolio prices with benchmark, ensuring same dates.
    
    Args:
        prices: Price DataFrame including benchmark
        benchmark_ticker: Ticker symbol of benchmark
    
    Returns:
        Tuple of (portfolio_prices, benchmark_prices)
    """
    if benchmark_ticker not in prices.columns:
        raise ValueError(f"Benchmark {benchmark_ticker} not in price data")
    
    # Extract benchmark
    benchmark = prices[benchmark_ticker]
    portfolio = prices.drop(columns=[benchmark_ticker])
    
    # Align to common dates
    common_dates = portfolio.index.intersection(benchmark.index)
    portfolio = portfolio.loc[common_dates]
    benchmark = benchmark.loc[common_dates]
    
    logger.info(f"Aligned to {len(common_dates)} common dates")
    
    return portfolio, benchmark


def compute_portfolio_value(
    prices: pd.DataFrame,
    positions: dict,
    cash: float = 0.0
) -> pd.Series:
    """
    Compute total portfolio value over time.
    
    Args:
        prices: Price DataFrame (dates x tickers)
        positions: Dict of {ticker: quantity}
        cash: Cash position
    
    Returns:
        Series of portfolio values
    """
    portfolio_value = pd.Series(0.0, index=prices.index)
    
    for ticker, quantity in positions.items():
        if ticker in prices.columns:
            portfolio_value += prices[ticker] * quantity
        else:
            logger.warning(f"Ticker {ticker} not in price data")
    
    portfolio_value += cash
    
    return portfolio_value


def winsorize_returns(
    returns: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.DataFrame:
    """
    Winsorize extreme returns to reduce outlier impact.
    
    Args:
        returns: Returns DataFrame
        lower_pct: Lower percentile (0-1)
        upper_pct: Upper percentile (0-1)
    
    Returns:
        Winsorized returns DataFrame
    """
    from scipy.stats import mstats
    
    returns_winsorized = returns.copy()
    
    for col in returns.columns:
        returns_winsorized[col] = mstats.winsorize(
            returns[col],
            limits=(lower_pct, 1 - upper_pct)
        )
    
    logger.info(f"Winsorized returns at {lower_pct:.1%} and {upper_pct:.1%} percentiles")
    
    return returns_winsorized


def detect_outliers(
    returns: pd.DataFrame,
    n_std: float = 4.0
) -> pd.DataFrame:
    """
    Detect outliers in returns (> n standard deviations from mean).
    
    Args:
        returns: Returns DataFrame
        n_std: Number of standard deviations
    
    Returns:
        Boolean DataFrame indicating outliers
    """
    mean = returns.mean()
    std = returns.std()
    
    outliers = (returns - mean).abs() > (n_std * std)
    
    outlier_count = outliers.sum()
    if outlier_count.sum() > 0:
        logger.warning(f"Detected outliers:\n{outlier_count[outlier_count > 0]}")
    
    return outliers


def main():
    """Command-line interface for data cleaner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean and preprocess price data')
    parser.add_argument('--input', required=True, help='Input price file (.parquet or .csv)')
    parser.add_argument('--output', required=True, help='Output cleaned price file')
    parser.add_argument('--returns-output', help='Output returns file (optional)')
    parser.add_argument('--max-missing', type=float, default=0.2, 
                       help='Max missing data ratio (0-1)')
    parser.add_argument('--return-method', choices=['log', 'simple'], default='log',
                       help='Return computation method')
    parser.add_argument('--freq', choices=['daily', 'weekly', 'monthly'], default='daily',
                       help='Return frequency')
    parser.add_argument('--winsorize', action='store_true',
                       help='Apply winsorization to returns')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if input_path.suffix == '.parquet':
        prices = pd.read_parquet(input_path)
    else:
        prices = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Clean prices
    cleaned_prices = clean_price_data(
        prices,
        max_missing_pct=args.max_missing
    )
    
    # Save cleaned prices
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        cleaned_prices.to_parquet(output_path)
    else:
        cleaned_prices.to_csv(output_path)
    
    logger.info(f"Saved cleaned prices to {output_path}")
    
    # Compute and save returns if requested
    if args.returns_output:
        returns = compute_returns(
            cleaned_prices,
            method=args.return_method,
            freq=args.freq
        )
        
        if args.winsorize:
            returns = winsorize_returns(returns)
        
        returns_path = Path(args.returns_output)
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        
        if returns_path.suffix == '.parquet':
            returns.to_parquet(returns_path)
        else:
            returns.to_csv(returns_path)
        
        logger.info(f"Saved returns to {returns_path}")


if __name__ == '__main__':
    main()