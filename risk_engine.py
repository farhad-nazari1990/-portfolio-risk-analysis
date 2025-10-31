"""
Risk Engine Module
==================
Computes portfolio risk metrics including VaR, CVaR, drawdowns, volatility,
Sharpe ratio, and other risk-adjusted return measures.

Usage:
    python risk_engine.py --prices data/prices.parquet \\
        --portfolio sample_data/example_portfolio.json \\
        --output output/risk_summary.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def annualized_return(returns: pd.Series, freq: str = 'daily') -> float:
    """
    Calculate annualized return from a return series.
    
    Args:
        returns: Return series
        freq: Frequency ('daily', 'weekly', 'monthly')
    
    Returns:
        Annualized return
    
    Plain language: The average yearly gain, accounting for compounding.
    """
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}[freq]
    
    cumulative_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    
    if n_periods < periods_per_year:
        logger.warning(f"Less than one year of data ({n_periods} periods)")
    
    years = n_periods / periods_per_year
    annual_return = (1 + cumulative_return) ** (1 / years) - 1
    
    return annual_return


def annualized_volatility(returns: pd.Series, freq: str = 'daily') -> float:
    """
    Calculate annualized volatility (standard deviation of returns).
    
    Args:
        returns: Return series
        freq: Frequency ('daily', 'weekly', 'monthly')
    
    Returns:
        Annualized volatility
    
    Plain language: A measure of how much your returns bounce around.
    Higher volatility means larger swings in value.
    """
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}[freq]
    return returns.std() * np.sqrt(periods_per_year)


def cov_matrix(
    returns: pd.DataFrame,
    shrinkage: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate covariance matrix of returns with optional shrinkage.
    
    Args:
        returns: Returns DataFrame (dates x assets)
        shrinkage: Shrinkage factor (0-1), None for sample covariance
    
    Returns:
        Covariance matrix
    
    Plain language: Shows how assets move together. High values mean
    they tend to go up or down at the same time.
    """
    if shrinkage is None:
        return returns.cov()
    
    # Ledoit-Wolf shrinkage
    sample_cov = returns.cov()
    
    # Target is diagonal matrix (assumes zero correlation)
    target = np.diag(np.diag(sample_cov))
    
    # Shrink towards target
    shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
    
    return shrunk_cov


def rolling_volatility(
    returns: pd.Series,
    window_days: int = 30,
    freq: str = 'daily'
) -> pd.Series:
    """
    Calculate rolling annualized volatility.
    
    Args:
        returns: Return series
        window_days: Rolling window size
        freq: Frequency
    
    Returns:
        Series of rolling volatility
    """
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}[freq]
    rolling_std = returns.rolling(window=window_days).std()
    return rolling_std * np.sqrt(periods_per_year)


def calc_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'historical', 'parametric', or 'cornish_fisher'
    
    Returns:
        VaR value (positive number representing potential loss)
    
    Plain language: The maximum loss you'd expect on a typical bad day.
    For example, VaR95 of $500 means there's only a 5% chance you'll
    lose more than $500 in a day.
    """
    if method == 'historical':
        # Use actual historical percentile
        var = -returns.quantile(1 - confidence_level)
    
    elif method == 'parametric':
        # Assume normal distribution
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mean + z_score * std)
    
    elif method == 'cornish_fisher':
        # Adjust for skewness and kurtosis
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        z = stats.norm.ppf(1 - confidence_level)
        cf_adj = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * (skew**2) / 36
        
        var = -(mean + cf_adj * std)
    
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    
    return var


def calc_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    Args:
        returns: Return series
        confidence_level: Confidence level
        method: 'historical' or 'parametric'
    
    Returns:
        CVaR value (positive number)
    
    Plain language: The average loss when things get worse than VaR.
    If your VaR is $500 and CVaR is $750, when you do hit that worst 5%
    of days, you typically lose $750.
    """
    if method == 'historical':
        # Average of worst (1-confidence_level) returns
        var_threshold = returns.quantile(1 - confidence_level)
        cvar = -returns[returns <= var_threshold].mean()
    
    elif method == 'parametric':
        # Closed form for normal distribution
        mean = returns.mean()
        std = returns.std()
        z = stats.norm.ppf(1 - confidence_level)
        cvar = -(mean - std * stats.norm.pdf(z) / (1 - confidence_level))
    
    else:
        raise ValueError(f"Unknown CVaR method: {method}")
    
    return cvar


def max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its dates.
    
    Args:
        prices: Price series
    
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    
    Plain language: The biggest peak-to-trough decline. For example,
    a 20% max drawdown means at worst, your portfolio fell 20% from
    its highest point before recovering.
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    
    # Find peak before trough
    peak_date = running_max[:trough_date].idxmax()
    
    return abs(max_dd), peak_date, trough_date


def drawdown_table(prices: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """
    Generate table of top N drawdowns.
    
    Args:
        prices: Price series
        top_n: Number of top drawdowns to return
    
    Returns:
        DataFrame with drawdown details
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Find all drawdown periods
    is_drawdown = drawdown < 0
    drawdown_periods = []
    
    in_drawdown = False
    start_date = None
    
    for date, dd in drawdown.items():
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_date = date
        elif dd == 0 and in_drawdown:
            in_drawdown = False
            if start_date:
                period_dd = drawdown[start_date:date]
                min_dd = period_dd.min()
                trough_date = period_dd.idxmin()
                
                drawdown_periods.append({
                    'Peak': start_date,
                    'Trough': trough_date,
                    'Recovery': date,
                    'Drawdown': abs(min_dd),
                    'Duration': (date - start_date).days
                })
    
    df = pd.DataFrame(drawdown_periods)
    if not df.empty:
        df = df.sort_values('Drawdown', ascending=False).head(top_n)
    
    return df


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    freq: str = 'daily'
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        freq: Frequency
    
    Returns:
        Sharpe ratio
    
    Plain language: Reward per unit of risk. Higher is better.
    A Sharpe of 1.0 means you earn 1% extra return for each 1% of volatility.
    """
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}[freq]
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    freq: str = 'daily'
) -> float:
    """
    Calculate Sortino ratio (like Sharpe, but only penalizes downside volatility).
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        freq: Frequency
    
    Returns:
        Sortino ratio
    
    Plain language: Similar to Sharpe, but only counts bad volatility
    (days when you lose money) as risk, ignoring upside volatility.
    """
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}[freq]
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def beta_vs_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate beta relative to a benchmark.
    
    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
    
    Returns:
        Beta coefficient
    
    Plain language: How much your portfolio moves with the market.
    Beta of 1.0 means you move in sync with the benchmark.
    Beta > 1 means you're more volatile than the market.
    """
    # Align series
    aligned = pd.DataFrame({'portfolio': returns, 'benchmark': benchmark_returns}).dropna()
    
    if len(aligned) < 2:
        return np.nan
    
    covariance = aligned['portfolio'].cov(aligned['benchmark'])
    benchmark_var = aligned['benchmark'].var()
    
    if benchmark_var == 0:
        return np.nan
    
    return covariance / benchmark_var


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio.
    
    Args:
        returns: Return series
        threshold: Minimum acceptable return (MAR)
    
    Returns:
        Omega ratio
    
    Plain language: Ratio of gains to losses relative to a threshold.
    Higher is better. Omega > 1 means gains outweigh losses.
    """
    returns_excess = returns - threshold
    gains = returns_excess[returns_excess > 0].sum()
    losses = abs(returns_excess[returns_excess < 0].sum())
    
    if losses == 0:
        return np.inf
    
    return gains / losses


def calculate_portfolio_risk_metrics(
    prices: pd.DataFrame,
    portfolio_config: dict,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.04
) -> Dict:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        prices: Price DataFrame (dates x tickers)
        portfolio_config: Portfolio configuration dict
        confidence_level: VaR/CVaR confidence level
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary of risk metrics
    """
    logger.info("Calculating portfolio risk metrics")
    
    # Extract positions
    positions = {pos['ticker']: pos['quantity'] 
                for pos in portfolio_config['positions']}
    cash = portfolio_config.get('cash', 0.0)
    benchmark_ticker = portfolio_config.get('benchmark', 'SPY')
    
    # Calculate portfolio value
    portfolio_value = pd.Series(cash, index=prices.index)
    
    for ticker, quantity in positions.items():
        if ticker in prices.columns:
            portfolio_value += prices[ticker] * quantity
    
    # Calculate returns
    portfolio_returns = portfolio_value.pct_change().dropna()
    
    # Get benchmark returns if available
    benchmark_returns = None
    if benchmark_ticker in prices.columns:
        benchmark_returns = prices[benchmark_ticker].pct_change().dropna()
    
    # Calculate metrics
    metrics = {
        'portfolio_value_start': float(portfolio_value.iloc[0]),
        'portfolio_value_end': float(portfolio_value.iloc[-1]),
        'total_return': float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1),
        'annualized_return': float(annualized_return(portfolio_returns)),
        'annualized_volatility': float(annualized_volatility(portfolio_returns)),
        'sharpe_ratio': float(sharpe_ratio(portfolio_returns, risk_free_rate)),
        'sortino_ratio': float(sortino_ratio(portfolio_returns, risk_free_rate)),
        'max_drawdown': float(max_drawdown(portfolio_value)[0]),
        'max_drawdown_peak_date': str(max_drawdown(portfolio_value)[1]),
        'max_drawdown_trough_date': str(max_drawdown(portfolio_value)[2]),
        'var_95_daily_pct': float(calc_var(portfolio_returns, confidence_level, 'historical')),
        'var_95_daily_dollar': float(calc_var(portfolio_returns, confidence_level, 'historical') * portfolio_value.iloc[-1]),
        'cvar_95_daily_pct': float(calc_cvar(portfolio_returns, confidence_level, 'historical')),
        'cvar_95_daily_dollar': float(calc_cvar(portfolio_returns, confidence_level, 'historical') * portfolio_value.iloc[-1]),
        'omega_ratio': float(omega_ratio(portfolio_returns)),
        'skewness': float(portfolio_returns.skew()),
        'kurtosis': float(portfolio_returns.kurtosis()),
    }
    
    # Monthly VaR (approximation)
    monthly_returns = portfolio_value.resample('M').last().pct_change().dropna()
    if len(monthly_returns) > 0:
        metrics['var_95_monthly_pct'] = float(calc_var(monthly_returns, confidence_level, 'historical'))
        metrics['var_95_monthly_dollar'] = float(metrics['var_95_monthly_pct'] * portfolio_value.iloc[-1])
    
    # Benchmark metrics
    if benchmark_returns is not None:
        metrics['beta_vs_benchmark'] = float(beta_vs_benchmark(portfolio_returns, benchmark_returns))
        metrics['benchmark_ticker'] = benchmark_ticker
    
    logger.info(f"Calculated {len(metrics)} risk metrics")
    
    return metrics


def main():
    """Command-line interface for risk engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate portfolio risk metrics')
    parser.add_argument('--prices', required=True, help='Price file (.parquet or .csv)')
    parser.add_argument('--portfolio', required=True, help='Portfolio JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk-free rate')
    
    args = parser.parse_args()
    
    # Load data
    if Path(args.prices).suffix == '.parquet':
        prices = pd.read_parquet(args.prices)
    else:
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    
    with open(args.portfolio, 'r') as f:
        portfolio_config = json.load(f)
    
    # Calculate metrics
    metrics = calculate_portfolio_risk_metrics(
        prices,
        portfolio_config,
        args.confidence,
        args.risk_free_rate
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved risk metrics to {output_path}")
    
    # Print summary
    print("\n=== Portfolio Risk Summary ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"VaR (95%, daily): ${metrics['var_95_daily_dollar']:,.2f}")
    print(f"CVaR (95%, daily): ${metrics['cvar_95_daily_dollar']:,.2f}")


if __name__ == '__main__':
    main()