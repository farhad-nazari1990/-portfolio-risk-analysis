"""
Unit tests for risk_engine module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from risk_engine import (
    annualized_return,
    annualized_volatility,
    cov_matrix,
    rolling_volatility,
    calc_var,
    calc_cvar,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    beta_vs_benchmark,
    omega_ratio
)


@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    return returns


@pytest.fixture
def sample_prices():
    """Generate sample price series."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    prices = pd.Series(100, index=dates)
    
    # Generate random walk
    returns = np.random.normal(0.0005, 0.01, 252)
    for i in range(1, len(prices)):
        prices.iloc[i] = prices.iloc[i-1] * (1 + returns[i])
    
    return prices


def test_annualized_return_positive(sample_returns):
    """Test annualized return calculation."""
    annual_ret = annualized_return(sample_returns, freq='daily')
    
    assert isinstance(annual_ret, float)
    assert -1.0 < annual_ret < 10.0  # Reasonable range


def test_annualized_return_zero_returns():
    """Test with zero returns."""
    zero_returns = pd.Series([0.0] * 252)
    annual_ret = annualized_return(zero_returns, freq='daily')
    
    assert abs(annual_ret) < 1e-10  # Should be very close to 0


def test_annualized_volatility(sample_returns):
    """Test volatility calculation."""
    vol = annualized_volatility(sample_returns, freq='daily')
    
    assert isinstance(vol, float)
    assert vol > 0
    assert 0.01 < vol < 2.0  # Reasonable annual volatility


def test_annualized_volatility_zero_variance():
    """Test volatility with constant returns."""
    constant_returns = pd.Series([0.001] * 252)
    vol = annualized_volatility(constant_returns, freq='daily')
    
    assert vol == 0.0


def test_cov_matrix_basic():
    """Test covariance matrix calculation."""
    np.random.seed(42)
    returns = pd.DataFrame({
        'A': np.random.normal(0, 0.01, 100),
        'B': np.random.normal(0, 0.015, 100)
    })
    
    cov = cov_matrix(returns)
    
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (2, 2)
    assert all(cov.columns == ['A', 'B'])
    
    # Covariance matrix should be symmetric
    assert np.allclose(cov.values, cov.values.T)
    
    # Diagonal elements (variances) should be positive
    assert all(np.diag(cov.values) > 0)


def test_cov_matrix_with_shrinkage():
    """Test covariance matrix with shrinkage."""
    np.random.seed(42)
    returns = pd.DataFrame({
        'A': np.random.normal(0, 0.01, 100),
        'B': np.random.normal(0, 0.015, 100)
    })
    
    cov = cov_matrix(returns, shrinkage=0.5)
    
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (2, 2)


def test_rolling_volatility(sample_returns):
    """Test rolling volatility calculation."""
    rolling_vol = rolling_volatility(sample_returns, window_days=30)
    
    assert isinstance(rolling_vol, pd.Series)
    assert len(rolling_vol) == len(sample_returns)
    
    # First 29 values should be NaN
    assert rolling_vol[:29].isna().all()
    
    # Remaining values should be positive
    assert (rolling_vol[29:] > 0).all()


def test_calc_var_historical(sample_returns):
    """Test VaR calculation with historical method."""
    var = calc_var(sample_returns, confidence_level=0.95, method='historical')
    
    assert isinstance(var, float)
    assert var > 0  # VaR should be positive (representing potential loss)
    
    # VaR should be close to the 5th percentile
    percentile_5 = -sample_returns.quantile(0.05)
    assert abs(var - percentile_5) < 0.001


def test_calc_var_parametric(sample_returns):
    """Test VaR calculation with parametric method."""
    var = calc_var(sample_returns, confidence_level=0.95, method='parametric')
    
    assert isinstance(var, float)
    assert var > 0


def test_calc_var_cornish_fisher(sample_returns):
    """Test VaR calculation with Cornish-Fisher method."""
    var = calc_var(sample_returns, confidence_level=0.95, method='cornish_fisher')
    
    assert isinstance(var, float)
    assert var > 0


def test_calc_var_different_confidence_levels(sample_returns):
    """Test VaR at different confidence levels."""
    var_90 = calc_var(sample_returns, confidence_level=0.90, method='historical')
    var_95 = calc_var(sample_returns, confidence_level=0.95, method='historical')
    var_99 = calc_var(sample_returns, confidence_level=0.99, method='historical')
    
    # Higher confidence should give higher VaR
    assert var_90 < var_95 < var_99


def test_calc_cvar_historical(sample_returns):
    """Test CVaR calculation."""
    var = calc_var(sample_returns, confidence_level=0.95, method='historical')
    cvar = calc_cvar(sample_returns, confidence_level=0.95, method='historical')
    
    assert isinstance(cvar, float)
    assert cvar > 0
    
    # CVaR should be greater than or equal to VaR
    assert cvar >= var


def test_calc_cvar_parametric(sample_returns):
    """Test CVaR calculation with parametric method."""
    cvar = calc_cvar(sample_returns, confidence_level=0.95, method='parametric')
    
    assert isinstance(cvar, float)
    assert cvar > 0


def test_max_drawdown(sample_prices):
    """Test maximum drawdown calculation."""
    dd, peak_date, trough_date = max_drawdown(sample_prices)
    
    assert isinstance(dd, float)
    assert 0 <= dd <= 1  # Drawdown should be between 0 and 1 (100%)
    assert isinstance(peak_date, pd.Timestamp)
    assert isinstance(trough_date, pd.Timestamp)
    assert peak_date < trough_date  # Peak should come before trough


def test_max_drawdown_monotonic_increase():
    """Test max drawdown with always increasing prices."""
    prices = pd.Series([100, 101, 102, 103, 104])
    dd, _, _ = max_drawdown(prices)
    
    # Should be very small or zero
    assert dd < 0.01


def test_max_drawdown_monotonic_decrease():
    """Test max drawdown with always decreasing prices."""
    prices = pd.Series([100, 95, 90, 85, 80])
    dd, _, _ = max_drawdown(prices)
    
    # Should be 20%
    assert abs(dd - 0.20) < 0.01


def test_sharpe_ratio(sample_returns):
    """Test Sharpe ratio calculation."""
    sharpe = sharpe_ratio(sample_returns, risk_free_rate=0.02, freq='daily')
    
    assert isinstance(sharpe, float)
    assert -5 < sharpe < 5  # Reasonable range


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility."""
    constant_returns = pd.Series([0.001] * 252)
    sharpe = sharpe_ratio(constant_returns, risk_free_rate=0.0, freq='daily')
    
    assert sharpe == 0.0


def test_sortino_ratio(sample_returns):
    """Test Sortino ratio calculation."""
    sortino = sortino_ratio(sample_returns, risk_free_rate=0.02, freq='daily')
    
    assert isinstance(sortino, float)
    
    # Sortino should be higher than or equal to Sharpe (penalizes only downside)
    sharpe = sharpe_ratio(sample_returns, risk_free_rate=0.02, freq='daily')
    assert sortino >= sharpe


def test_sortino_ratio_no_downside():
    """Test Sortino ratio with no negative returns."""
    positive_returns = pd.Series(np.abs(np.random.normal(0.001, 0.005, 252)))
    sortino = sortino_ratio(positive_returns, risk_free_rate=0.0, freq='daily')
    
    assert sortino == 0.0  # No downside volatility


def test_beta_vs_benchmark():
    """Test beta calculation."""
    np.random.seed(42)
    
    # Create correlated returns
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    portfolio_returns = 0.8 * benchmark_returns + pd.Series(np.random.normal(0, 0.005, 252))
    
    beta = beta_vs_benchmark(portfolio_returns, benchmark_returns)
    
    assert isinstance(beta, float)
    assert 0.5 < beta < 1.5  # Should be close to 0.8


def test_beta_vs_benchmark_perfect_correlation():
    """Test beta with perfect correlation."""
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    
    beta = beta_vs_benchmark(returns, returns)
    
    # Beta should be 1.0 when comparing to itself
    assert abs(beta - 1.0) < 0.01


def test_beta_vs_benchmark_zero_variance():
    """Test beta with zero variance benchmark."""
    portfolio_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    benchmark_returns = pd.Series([0.001] * 252)
    
    beta = beta_vs_benchmark(portfolio_returns, benchmark_returns)
    
    assert np.isnan(beta)


def test_omega_ratio(sample_returns):
    """Test Omega ratio calculation."""
    omega = omega_ratio(sample_returns, threshold=0.0)
    
    assert isinstance(omega, float)
    assert omega > 0


def test_omega_ratio_all_positive():
    """Test Omega ratio with all positive returns."""
    positive_returns = pd.Series(np.abs(np.random.normal(0.001, 0.005, 100)))
    omega = omega_ratio(positive_returns, threshold=0.0)
    
    # Should be infinite (no losses)
    assert np.isinf(omega)


def test_omega_ratio_all_negative():
    """Test Omega ratio with all negative returns."""
    negative_returns = pd.Series(-np.abs(np.random.normal(0.001, 0.005, 100)))
    omega = omega_ratio(negative_returns, threshold=0.0)
    
    # Should be 0 (no gains)
    assert omega == 0.0


def test_return_metrics_consistency():
    """Test that different return metrics are consistent."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))
    
    # Calculate various metrics
    annual_ret = annualized_return(returns)
    annual_vol = annualized_volatility(returns)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
    
    # Manual Sharpe calculation
    expected_sharpe = (annual_ret / annual_vol)
    
    # Should be close
    assert abs(sharpe - expected_sharpe) < 0.1


def test_empty_series_handling():
    """Test handling of empty series."""
    empty_returns = pd.Series([])
    
    with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
        annualized_return(empty_returns)


def test_single_value_handling():
    """Test handling of single value."""
    single_return = pd.Series([0.01])
    
    # Should handle gracefully
    vol = annualized_volatility(single_return)
    assert vol == 0.0  # Only one value, no variance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])