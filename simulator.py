"""
Simulator Module
================
Monte Carlo simulation with correlated returns and stress testing.

Usage:
    python simulator.py --prices data/prices.parquet \\
        --portfolio sample_data/example_portfolio.json \\
        --output output/simulation_results.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import cholesky

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def monte_carlo_correlated(
    returns: pd.DataFrame,
    n_sims: int = 10000,
    n_days: int = 252,
    initial_weights: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation with correlated returns using Cholesky decomposition.
    
    Args:
        returns: Historical returns DataFrame (dates x assets)
        n_sims: Number of simulation paths
        n_days: Number of days to simulate
        initial_weights: Portfolio weights (if None, equal-weighted)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (simulated_portfolio_returns, simulated_asset_returns)
        - simulated_portfolio_returns: (n_sims, n_days) array
        - simulated_asset_returns: (n_sims, n_days, n_assets) array
    
    Plain language: Projects thousands of possible future price paths based on
    how assets moved historically, accounting for correlations between them.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    logger.info(f"Running Monte Carlo: {n_sims} simulations x {n_days} days")
    
    # Calculate statistics from historical data
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(mean_returns)
    
    # Default to equal weights
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets
    
    # Cholesky decomposition for correlated random variables
    try:
        chol_matrix = cholesky(cov_matrix, lower=True)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix not positive definite, using diagonal")
        chol_matrix = np.diag(np.sqrt(np.diag(cov_matrix)))
    
    # Generate random returns
    # Shape: (n_sims, n_days, n_assets)
    random_returns = np.random.normal(0, 1, size=(n_sims, n_days, n_assets))
    
    # Apply correlation structure
    # For each simulation and day, apply Cholesky matrix
    correlated_returns = np.zeros_like(random_returns)
    for i in range(n_sims):
        for j in range(n_days):
            correlated_returns[i, j, :] = mean_returns + chol_matrix @ random_returns[i, j, :]
    
    # Calculate portfolio returns
    # Shape: (n_sims, n_days)
    portfolio_returns = np.sum(correlated_returns * initial_weights, axis=2)
    
    logger.info(f"Simulation complete. Portfolio return range: "
               f"[{portfolio_returns.min():.4f}, {portfolio_returns.max():.4f}]")
    
    return portfolio_returns, correlated_returns


def simulate_portfolio_paths(
    portfolio_returns: np.ndarray,
    initial_value: float
) -> np.ndarray:
    """
    Convert simulated returns to portfolio value paths.
    
    Args:
        portfolio_returns: Array of daily returns (n_sims, n_days)
        initial_value: Starting portfolio value
    
    Returns:
        Portfolio value paths (n_sims, n_days+1) including initial value
    """
    n_sims, n_days = portfolio_returns.shape
    
    # Initialize with starting value
    portfolio_values = np.zeros((n_sims, n_days + 1))
    portfolio_values[:, 0] = initial_value
    
    # Compound returns
    for i in range(n_days):
        portfolio_values[:, i + 1] = portfolio_values[:, i] * (1 + portfolio_returns[:, i])
    
    return portfolio_values


def calculate_simulation_statistics(
    portfolio_returns: np.ndarray,
    initial_value: float,
    confidence_levels: List[float] = [0.01, 0.05, 0.10]
) -> Dict:
    """
    Calculate statistics from Monte Carlo simulation.
    
    Args:
        portfolio_returns: Simulated daily returns (n_sims, n_days)
        initial_value: Starting portfolio value
        confidence_levels: List of confidence levels for VaR
    
    Returns:
        Dictionary of simulation statistics
    """
    # Total returns over simulation period
    total_returns = np.sum(portfolio_returns, axis=1)
    final_values = initial_value * (1 + total_returns)
    
    # Calculate percentiles
    stats_dict = {
        'mean_final_value': float(np.mean(final_values)),
        'median_final_value': float(np.median(final_values)),
        'std_final_value': float(np.std(final_values)),
        'min_final_value': float(np.min(final_values)),
        'max_final_value': float(np.max(final_values)),
        'percentile_5': float(np.percentile(final_values, 5)),
        'percentile_25': float(np.percentile(final_values, 25)),
        'percentile_75': float(np.percentile(final_values, 75)),
        'percentile_95': float(np.percentile(final_values, 95)),
    }
    
    # Probability of loss
    prob_loss = np.mean(final_values < initial_value)
    stats_dict['probability_of_loss'] = float(prob_loss)
    
    # Probability of large losses
    for threshold in [0.05, 0.10, 0.20]:
        loss_threshold = initial_value * (1 - threshold)
        prob = np.mean(final_values < loss_threshold)
        stats_dict[f'probability_loss_gt_{int(threshold*100)}pct'] = float(prob)
    
    # VaR at different confidence levels
    for cl in confidence_levels:
        var_value = np.percentile(final_values, cl * 100)
        var_loss = initial_value - var_value
        stats_dict[f'var_{int(cl*100)}'] = float(var_loss)
    
    return stats_dict


def stress_test_scenario(
    returns: pd.DataFrame,
    positions: Dict[str, float],
    shocks: Dict[str, float]
) -> Dict:
    """
    Apply stress test shocks to portfolio.
    
    Args:
        returns: Historical returns DataFrame
        positions: Dict of {ticker: quantity}
        shocks: Dict of {ticker: shock_percentage} (e.g., {'AAPL': -0.20})
    
    Returns:
        Dictionary with stress test results
    
    Plain language: Shows how your portfolio would perform if specific
    assets suddenly dropped (or rose) by a given percentage.
    """
    logger.info(f"Running stress test with {len(shocks)} shocks")
    
    results = {
        'shocks_applied': shocks,
        'asset_impacts': {},
        'total_impact': 0.0
    }
    
    for ticker, shock in shocks.items():
        if ticker in positions:
            # Calculate impact on this position
            quantity = positions[ticker]
            # Use last available price as reference
            if ticker in returns.columns:
                # Estimate position value (simplified)
                avg_return = returns[ticker].mean()
                impact = quantity * shock  # Simplified impact
                results['asset_impacts'][ticker] = float(impact)
                results['total_impact'] += impact
            else:
                logger.warning(f"Ticker {ticker} not in returns data")
    
    return results


def factor_stress_test(
    returns: pd.DataFrame,
    positions: Dict[str, float],
    factor_shock: float,
    equity_tickers: List[str]
) -> Dict:
    """
    Apply a factor shock (e.g., market crash) to equity positions.
    
    Args:
        returns: Historical returns DataFrame
        positions: Dict of {ticker: quantity}
        factor_shock: Shock to apply (e.g., -0.30 for -30%)
        equity_tickers: List of tickers to apply shock to
    
    Returns:
        Stress test results
    
    Plain language: Simulates a market-wide event, like a 30% crash
    in all equity positions simultaneously.
    """
    logger.info(f"Running factor stress test: {factor_shock:.1%} shock")
    
    shocks = {ticker: factor_shock for ticker in equity_tickers if ticker in positions}
    
    return stress_test_scenario(returns, positions, shocks)


def historical_scenario_replay(
    returns: pd.DataFrame,
    scenario_name: str,
    scenario_returns: pd.Series
) -> Dict:
    """
    Replay a historical crisis scenario.
    
    Args:
        returns: Historical returns DataFrame
        scenario_name: Name of scenario (e.g., "2008 Financial Crisis")
        scenario_returns: Series of returns during crisis period
    
    Returns:
        Scenario analysis results
    """
    logger.info(f"Replaying historical scenario: {scenario_name}")
    
    # Calculate statistics for the scenario
    results = {
        'scenario_name': scenario_name,
        'total_return': float(scenario_returns.sum()),
        'max_drawdown': float(scenario_returns.min()),
        'volatility': float(scenario_returns.std()),
        'worst_day': float(scenario_returns.min()),
        'best_day': float(scenario_returns.max())
    }
    
    return results


def get_sample_simulation_paths(
    portfolio_values: np.ndarray,
    n_paths: int = 10
) -> np.ndarray:
    """
    Extract a sample of simulation paths for visualization.
    
    Args:
        portfolio_values: All simulated paths (n_sims, n_days+1)
        n_paths: Number of paths to sample
    
    Returns:
        Sampled paths (n_paths, n_days+1)
    """
    n_sims = portfolio_values.shape[0]
    indices = np.random.choice(n_sims, min(n_paths, n_sims), replace=False)
    return portfolio_values[indices, :]


def main():
    """Command-line interface for simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation and stress tests')
    parser.add_argument('--prices', required=True, help='Price file (.parquet or .csv)')
    parser.add_argument('--portfolio', required=True, help='Portfolio JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--n-sims', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--n-days', type=int, default=252, help='Simulation horizon (days)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load data
    if Path(args.prices).suffix == '.parquet':
        prices = pd.read_parquet(args.prices)
    else:
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    
    with open(args.portfolio, 'r') as f:
        portfolio_config = json.load(f)
    
    # Compute returns
    returns = prices.pct_change().dropna()
    
    # Get portfolio positions
    positions = {pos['ticker']: pos['quantity'] 
                for pos in portfolio_config['positions']}
    
    # Calculate current portfolio value
    initial_value = portfolio_config.get('cash', 0.0)
    for ticker, quantity in positions.items():
        if ticker in prices.columns:
            initial_value += prices[ticker].iloc[-1] * quantity
    
    logger.info(f"Initial portfolio value: ${initial_value:,.2f}")
    
    # Run Monte Carlo simulation
    portfolio_returns, _ = monte_carlo_correlated(
        returns[list(positions.keys())],
        n_sims=args.n_sims,
        n_days=args.n_days,
        random_seed=args.seed
    )
    
    # Calculate simulation statistics
    sim_stats = calculate_simulation_statistics(
        portfolio_returns,
        initial_value
    )
    
    # Run stress tests
    market_crash_shock = -0.30  # 30% market crash
    equity_tickers = [t for t in positions.keys() if t != 'BND']  # Assume BND is bonds
    
    stress_results = factor_stress_test(
        returns,
        positions,
        market_crash_shock,
        equity_tickers
    )
    
    # Combine results
    results = {
        'initial_value': float(initial_value),
        'simulation_params': {
            'n_sims': args.n_sims,
            'n_days': args.n_days,
            'random_seed': args.seed
        },
        'simulation_statistics': sim_stats,
        'stress_test_market_crash_30pct': stress_results
    }
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved simulation results to {output_path}")
    
    # Print summary
    print("\n=== Monte Carlo Simulation Summary ===")
    print(f"Initial Value: ${initial_value:,.2f}")
    print(f"Expected Final Value: ${sim_stats['mean_final_value']:,.2f}")
    print(f"5th Percentile: ${sim_stats['percentile_5']:,.2f}")
    print(f"95th Percentile: ${sim_stats['percentile_95']:,.2f}")
    print(f"Probability of Loss: {sim_stats['probability_of_loss']:.2%}")
    print(f"Probability of >10% Loss: {sim_stats['probability_loss_gt_10pct']:.2%}")


if __name__ == '__main__':
    main()