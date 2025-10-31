"""
Data Fetcher Module
===================
Fetches historical price data from Yahoo Finance, Alpha Vantage, or local CSV files.
Includes robust retry logic, rate limiting, and caching.

Usage:
    python data_fetcher.py --portfolio sample_data/example_portfolio.json \\
        --start 2020-01-01 --end 2025-10-30 --output data/prices.parquet
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
CACHE_DIR = Path(os.getenv('CACHE_DIR', './cache'))
CACHE_DIR.mkdir(exist_ok=True)
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
RATE_LIMIT_DELAY = float(os.getenv('YAHOO_RATE_LIMIT_DELAY', 1))
CACHE_EXPIRY_HOURS = int(os.getenv('CACHE_EXPIRY_HOURS', 24))


def fetch_yahoo(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    cache_file = CACHE_DIR / f"yahoo_{'_'.join(sorted(tickers))}_{start}_{end}.parquet"
    
    # Check cache
    if use_cache and cache_file.exists():
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age < CACHE_EXPIRY_HOURS * 3600:
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
    
    logger.info(f"Fetching data for {len(tickers)} tickers from Yahoo Finance")
    
    all_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        retry_count = 0
        success = False
        
        while retry_count < MAX_RETRIES and not success:
            try:
                logger.info(f"Fetching {ticker} (attempt {retry_count + 1}/{MAX_RETRIES})")
                
                # Fetch data
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start=start, end=end, auto_adjust=True)
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    failed_tickers.append(ticker)
                    break
                
                # Use Close price (already adjusted)
                all_data[ticker] = df['Close']
                success = True
                
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error fetching {ticker}: {e}")
                if retry_count < MAX_RETRIES:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    failed_tickers.append(ticker)
    
    if not all_data:
        raise ValueError("Failed to fetch data for any tickers")
    
    # Combine into DataFrame
    prices_df = pd.DataFrame(all_data)
    
    # Ensure timezone-naive for consistency
    if prices_df.index.tz is not None:
        prices_df.index = prices_df.index.tz_localize(None)
    
    # Sort by date
    prices_df = prices_df.sort_index()
    
    if failed_tickers:
        logger.warning(f"Failed to fetch data for: {failed_tickers}")
    
    # Cache the results
    prices_df.to_parquet(cache_file)
    logger.info(f"Cached data to {cache_file}")
    
    return prices_df


def fetch_alpha_vantage(
    tickers: List[str],
    start: str,
    end: str,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical data from Alpha Vantage (optional premium source).
    
    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        api_key: Alpha Vantage API key (or read from env)
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    api_key = api_key or os.getenv('ALPHA_VANTAGE_KEY')
    if not api_key:
        raise ValueError("Alpha Vantage API key not provided")
    
    base_url = "https://www.alphavantage.co/query"
    all_data = {}
    
    for ticker in tickers:
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': api_key
        }
        
        try:
            logger.info(f"Fetching {ticker} from Alpha Vantage")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"No data for {ticker}: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                continue
            
            # Parse time series
            ts_data = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Use adjusted close
            all_data[ticker] = pd.to_numeric(df['5. adjusted close'])
            
            # Alpha Vantage free tier: 5 calls/minute
            time.sleep(12)
            
        except Exception as e:
            logger.error(f"Error fetching {ticker} from Alpha Vantage: {e}")
    
    if not all_data:
        raise ValueError("Failed to fetch data from Alpha Vantage")
    
    prices_df = pd.DataFrame(all_data)
    
    # Filter by date range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    prices_df = prices_df[(prices_df.index >= start_dt) & (prices_df.index <= end_dt)]
    
    return prices_df


def read_local_csv(path: str, date_col: str = 'Date') -> pd.DataFrame:
    """
    Read historical prices from a local CSV file.
    
    CSV format: Date, Ticker1, Ticker2, ...
    
    Args:
        path: Path to CSV file
        date_col: Name of the date column
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    logger.info(f"Reading local CSV from {path}")
    
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()
    
    # Ensure timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df


def fetch_portfolio_data(
    portfolio_path: str,
    start: str,
    end: str,
    source: str = 'yahoo',
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch data for all tickers in a portfolio JSON file.
    
    Args:
        portfolio_path: Path to portfolio JSON
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        source: Data source ('yahoo', 'alpha_vantage', or CSV path)
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with prices for all portfolio tickers
    """
    # Load portfolio
    with open(portfolio_path, 'r') as f:
        portfolio = json.load(f)
    
    tickers = [pos['ticker'] for pos in portfolio['positions']]
    
    # Add benchmark if specified
    if 'benchmark' in portfolio and portfolio['benchmark']:
        if portfolio['benchmark'] not in tickers:
            tickers.append(portfolio['benchmark'])
    
    logger.info(f"Fetching data for portfolio with {len(tickers)} tickers")
    
    # Fetch based on source
    if source == 'yahoo':
        return fetch_yahoo(tickers, start, end, use_cache)
    elif source == 'alpha_vantage':
        return fetch_alpha_vantage(tickers, start, end)
    else:
        # Assume it's a CSV path
        return read_local_csv(source)


def main():
    """Command-line interface for data fetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch historical price data')
    parser.add_argument('--portfolio', required=True, help='Path to portfolio JSON')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', required=True, help='Output path (.parquet or .csv)')
    parser.add_argument('--source', default='yahoo', help='Data source (yahoo, alpha_vantage, or CSV path)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    
    args = parser.parse_args()
    
    # Fetch data
    prices = fetch_portfolio_data(
        args.portfolio,
        args.start,
        args.end,
        args.source,
        use_cache=not args.no_cache
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        prices.to_parquet(output_path)
    else:
        prices.to_csv(output_path)
    
    logger.info(f"Saved {len(prices)} days of data for {len(prices.columns)} tickers to {output_path}")
    logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    logger.info(f"Sample data:\n{prices.head()}")


if __name__ == '__main__':
    main()