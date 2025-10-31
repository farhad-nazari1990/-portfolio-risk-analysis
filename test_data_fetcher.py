"""
Unit tests for data_fetcher module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetcher import (
    fetch_yahoo,
    read_local_csv,
    fetch_portfolio_data
)


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample price data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("Date,AAPL,MSFT,SPY\n")
        f.write("2024-01-01,150.0,300.0,400.0\n")
        f.write("2024-01-02,151.0,301.0,401.0\n")
        f.write("2024-01-03,152.0,302.0,402.0\n")
        f.write("2024-01-04,153.0,303.0,403.0\n")
        f.write("2024-01-05,154.0,304.0,404.0\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def sample_portfolio_file():
    """Create a temporary portfolio JSON file."""
    portfolio = {
        "owner": "Test User",
        "currency": "USD",
        "positions": [
            {"ticker": "AAPL", "quantity": 10},
            {"ticker": "MSFT", "quantity": 5}
        ],
        "cash": 1000,
        "benchmark": "SPY"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(portfolio, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


def test_read_local_csv(sample_csv_file):
    """Test reading price data from local CSV."""
    prices = read_local_csv(sample_csv_file)
    
    assert isinstance(prices, pd.DataFrame)
    assert len(prices) == 5
    assert list(prices.columns) == ['AAPL', 'MSFT', 'SPY']
    assert prices.index.name == 'Date' or isinstance(prices.index, pd.DatetimeIndex)
    
    # Check data types
    assert prices['AAPL'].dtype in [np.float64, np.float32]
    
    # Check values
    assert prices['AAPL'].iloc[0] == 150.0
    assert prices['MSFT'].iloc[-1] == 304.0


def test_read_local_csv_date_parsing(sample_csv_file):
    """Test that dates are properly parsed."""
    prices = read_local_csv(sample_csv_file)
    
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index[0] == pd.Timestamp('2024-01-01')
    assert prices.index[-1] == pd.Timestamp('2024-01-05')


def test_read_local_csv_sorted(sample_csv_file):
    """Test that data is sorted by date."""
    prices = read_local_csv(sample_csv_file)
    
    assert prices.index.is_monotonic_increasing


def test_read_local_csv_timezone_naive(sample_csv_file):
    """Test that returned data is timezone-naive."""
    prices = read_local_csv(sample_csv_file)
    
    assert prices.index.tz is None


def test_fetch_yahoo_mock(monkeypatch):
    """Test Yahoo Finance fetching with mocked data."""
    # Mock yfinance Ticker class
    class MockTicker:
        def __init__(self, ticker):
            self.ticker = ticker
        
        def history(self, start, end, auto_adjust=True):
            dates = pd.date_range(start, end, freq='D')
            data = pd.DataFrame({
                'Close': np.random.uniform(100, 200, len(dates))
            }, index=dates)
            return data
    
    # Monkeypatch yfinance
    import yfinance as yf
    monkeypatch.setattr(yf, 'Ticker', MockTicker)
    
    # Test fetching
    prices = fetch_yahoo(
        ['AAPL', 'MSFT'],
        '2024-01-01',
        '2024-01-10',
        use_cache=False
    )
    
    assert isinstance(prices, pd.DataFrame)
    assert 'AAPL' in prices.columns
    assert 'MSFT' in prices.columns
    assert len(prices) > 0


def test_fetch_portfolio_data_from_csv(sample_csv_file, sample_portfolio_file):
    """Test fetching portfolio data from CSV."""
    prices = fetch_portfolio_data(
        sample_portfolio_file,
        '2024-01-01',
        '2024-01-05',
        source=sample_csv_file
    )
    
    assert isinstance(prices, pd.DataFrame)
    assert 'AAPL' in prices.columns
    assert 'MSFT' in prices.columns
    assert 'SPY' in prices.columns  # Benchmark should be included
    assert len(prices) == 5


def test_fetch_yahoo_empty_tickers():
    """Test that empty ticker list raises appropriate error."""
    with pytest.raises((ValueError, Exception)):
        fetch_yahoo([], '2024-01-01', '2024-01-10', use_cache=False)


def test_fetch_yahoo_invalid_dates():
    """Test behavior with invalid date range."""
    # Future dates should return empty or raise error
    result = fetch_yahoo(
        ['AAPL'],
        '2030-01-01',
        '2030-01-10',
        use_cache=False
    )
    # Either empty DataFrame or exception is acceptable
    assert isinstance(result, pd.DataFrame)


def test_data_shape_consistency(sample_csv_file):
    """Test that data shape is consistent."""
    prices = read_local_csv(sample_csv_file)
    
    # All columns should have same length
    assert all(len(prices[col]) == len(prices) for col in prices.columns)
    
    # No completely empty columns
    assert all(prices[col].notna().any() for col in prices.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])