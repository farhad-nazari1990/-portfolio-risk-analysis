"""
Main Pipeline Orchestrator
===========================
Runs the complete portfolio risk analysis pipeline end-to-end.

Usage:
    python run_all.py --portfolio sample_data/example_portfolio.json
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from data_fetcher import fetch_portfolio_data
from data_cleaner import clean_price_data, compute_returns
from risk_engine import calculate_portfolio_risk_metrics
from simulator import monte_carlo_correlated, calculate_simulation_statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    portfolio_path: str,
    start_date: str = None,
    end_date: str = None,
    output_dir: str = './output'
):
    """
    Run the complete risk analysis pipeline.
    
    Args:
        portfolio_path: Path to portfolio JSON
        start_date: Start date (default: 3 years ago)
        end_date: End date (default: today)
        output_dir: Output directory for results
    """
    logger.info("=" * 60)
    logger.info("RETAIL RISK KIT - Portfolio Analysis Pipeline")
    logger.info("=" * 60)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load portfolio
    logger.info(f"\n[1/6] Loading portfolio: {portfolio_path}")
    with open(portfolio_path, 'r') as f:
        portfolio_config = json.load(f)
    
    logger.info(f"Portfolio owner: {portfolio_config.get('owner', 'Unknown')}")
    logger.info(f"Number of positions: {len(portfolio_config['positions'])}")
    logger.info(f"Cash: ${portfolio_config.get('cash', 0):,.2f}")
    
    # Set date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Step 2: Fetch data
    logger.info(f"\n[2/6] Fetching price data...")
    try:
        prices = fetch_portfolio_data(
            portfolio_path,
            start_date,
            end_date,
            source='yahoo',
            use_cache=True
        )
        
        # Save raw prices
        prices.to_parquet(output_path / 'prices_raw.parquet')
        logger.info(f"Fetched {len(prices)} days for {len(prices.columns)} tickers")
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise
    
    # Step 3: Clean data
    logger.info(f"\n[3/6] Cleaning price data...")
    try:
        cleaned_prices = clean_price_data(prices, max_missing_pct=0.2)
        cleaned_prices.to_parquet(output_path / 'prices_cleaned.parquet')
        logger.info(f"Cleaned data: {len(cleaned_prices)} days, {len(cleaned_prices.columns)} tickers")
        
        # Compute returns
        returns = compute_returns(cleaned_prices, method='log', freq='daily')
        returns.to_parquet(output_path / 'returns.parquet')
        
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise
    
    # Step 4: Calculate risk metrics
    logger.info(f"\n[4/6] Calculating risk metrics...")
    try:
        confidence_level = portfolio_config.get('confidence_level', 0.95)
        risk_free_rate = portfolio_config.get('risk_free_rate', 0.04)
        
        risk_metrics = calculate_portfolio_risk_metrics(
            cleaned_prices,
            portfolio_config,
            confidence_level,
            risk_free_rate
        )
        
        # Save risk metrics
        with open(output_path / 'risk_summary.json', 'w') as f:
            json.dump(risk_metrics, f, indent=2)
        
        logger.info("Risk metrics calculated successfully")
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise
    
    # Step 5: Run Monte Carlo simulation
    logger.info(f"\n[5/6] Running Monte Carlo simulation...")
    try:
        # Get portfolio positions
        positions = {pos['ticker']: pos['quantity'] 
                    for pos in portfolio_config['positions']}
        
        # Calculate initial portfolio value
        initial_value = portfolio_config.get('cash', 0.0)
        for ticker, quantity in positions.items():
            if ticker in cleaned_prices.columns:
                initial_value += cleaned_prices[ticker].iloc[-1] * quantity
        
        # Filter returns to portfolio tickers only
        portfolio_tickers = [t for t in positions.keys() if t in returns.columns]
        portfolio_returns_df = returns[portfolio_tickers]
        
        # Run simulation
        n_sims = 10000
        n_days = 252
        
        portfolio_sim_returns, _ = monte_carlo_correlated(
            portfolio_returns_df,
            n_sims=n_sims,
            n_days=n_days,
            random_seed=42
        )
        
        # Calculate simulation statistics
        sim_stats = calculate_simulation_statistics(
            portfolio_sim_returns,
            initial_value
        )
        
        # Save simulation results
        simulation_results = {
            'initial_value': float(initial_value),
            'n_simulations': n_sims,
            'horizon_days': n_days,
            'statistics': sim_stats
        }
        
        with open(output_path / 'simulation_results.json', 'w') as f:
            json.dump(simulation_results, f, indent=2)
        
        logger.info("Monte Carlo simulation completed")
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        # Don't fail the pipeline, just log warning
        logger.warning("Continuing without simulation results")
        simulation_results = None
    
    # Step 6: Generate summary report
    logger.info(f"\n[6/6] Generating summary report...")
    try:
        generate_text_report(
            output_path,
            portfolio_config,
            risk_metrics,
            simulation_results
        )
        
        logger.info(f"Report saved to {output_path / 'report.txt'}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"\nOutputs saved to: {output_path.absolute()}")
    logger.info(f"  - prices_raw.parquet")
    logger.info(f"  - prices_cleaned.parquet")
    logger.info(f"  - returns.parquet")
    logger.info(f"  - risk_summary.json")
    logger.info(f"  - simulation_results.json")
    logger.info(f"  - report.txt")
    logger.info("\nNext steps:")
    logger.info("  1. Review risk_summary.json for detailed metrics")
    logger.info("  2. Run 'streamlit run dashboard.py' to view interactive dashboard")
    logger.info("  3. Customize portfolio and re-run analysis")
    
    return risk_metrics, simulation_results


def generate_text_report(
    output_path: Path,
    portfolio_config: dict,
    risk_metrics: dict,
    simulation_results: dict = None
):
    """Generate a plain text summary report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PORTFOLIO RISK ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Portfolio Owner: {portfolio_config.get('owner', 'N/A')}")
    report_lines.append(f"Currency: {portfolio_config.get('currency', 'USD')}")
    
    # Portfolio composition
    report_lines.append("\n" + "-" * 80)
    report_lines.append("PORTFOLIO COMPOSITION")
    report_lines.append("-" * 80)
    for pos in portfolio_config['positions']:
        report_lines.append(f"  {pos['ticker']:8s} - {pos['quantity']:>10.4f} shares")
    report_lines.append(f"  {'CASH':8s} - ${portfolio_config.get('cash', 0):>10,.2f}")
    
    # Risk metrics
    report_lines.append("\n" + "-" * 80)
    report_lines.append("RISK METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"\nPortfolio Value:")
    report_lines.append(f"  Start:  ${risk_metrics['portfolio_value_start']:>15,.2f}")
    report_lines.append(f"  End:    ${risk_metrics['portfolio_value_end']:>15,.2f}")
    report_lines.append(f"  Return: {risk_metrics['total_return']:>15.2%}")
    
    report_lines.append(f"\nReturn & Volatility:")
    report_lines.append(f"  Annualized Return:     {risk_metrics['annualized_return']:>10.2%}")
    report_lines.append(f"  Annualized Volatility: {risk_metrics['annualized_volatility']:>10.2%}")
    
    report_lines.append(f"\nRisk-Adjusted Returns:")
    report_lines.append(f"  Sharpe Ratio:  {risk_metrics['sharpe_ratio']:>10.2f}")
    report_lines.append(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:>10.2f}")
    report_lines.append(f"  Omega Ratio:   {risk_metrics['omega_ratio']:>10.2f}")
    
    report_lines.append(f"\nValue at Risk (VaR) - 95% Confidence:")
    report_lines.append(f"  Daily VaR:   ${risk_metrics['var_95_daily_dollar']:>12,.2f} ({risk_metrics['var_95_daily_pct']:>6.2%})")
    if 'var_95_monthly_dollar' in risk_metrics:
        report_lines.append(f"  Monthly VaR: ${risk_metrics['var_95_monthly_dollar']:>12,.2f} ({risk_metrics['var_95_monthly_pct']:>6.2%})")
    
    report_lines.append(f"\nConditional VaR (CVaR) - 95% Confidence:")
    report_lines.append(f"  Daily CVaR:  ${risk_metrics['cvar_95_daily_dollar']:>12,.2f} ({risk_metrics['cvar_95_daily_pct']:>6.2%})")
    
    report_lines.append(f"\nDrawdown Analysis:")
    report_lines.append(f"  Maximum Drawdown: {risk_metrics['max_drawdown']:>10.2%}")
    report_lines.append(f"  Peak Date:        {risk_metrics['max_drawdown_peak_date']}")
    report_lines.append(f"  Trough Date:      {risk_metrics['max_drawdown_trough_date']}")
    
    if 'beta_vs_benchmark' in risk_metrics:
        report_lines.append(f"\nBenchmark Comparison ({risk_metrics['benchmark_ticker']}):")
        report_lines.append(f"  Beta: {risk_metrics['beta_vs_benchmark']:>10.2f}")
    
    # Monte Carlo results
    if simulation_results:
        report_lines.append("\n" + "-" * 80)
        report_lines.append("MONTE CARLO SIMULATION")
        report_lines.append("-" * 80)
        stats = simulation_results['statistics']
        report_lines.append(f"\nSimulation Parameters:")
        report_lines.append(f"  Simulations: {simulation_results['n_simulations']:,}")
        report_lines.append(f"  Horizon:     {simulation_results['horizon_days']} days")
        report_lines.append(f"  Initial:     ${simulation_results['initial_value']:,.2f}")
        
        report_lines.append(f"\nProjected Final Value:")
        report_lines.append(f"  Mean:   ${stats['mean_final_value']:>15,.2f}")
        report_lines.append(f"  Median: ${stats['median_final_value']:>15,.2f}")
        report_lines.append(f"  5th %:  ${stats['percentile_5']:>15,.2f}")
        report_lines.append(f"  95th %: ${stats['percentile_95']:>15,.2f}")
        
        report_lines.append(f"\nProbabilities:")
        report_lines.append(f"  Loss:        {stats['probability_of_loss']:>10.2%}")
        report_lines.append(f"  Loss > 10%:  {stats['probability_loss_gt_10pct']:>10.2%}")
        report_lines.append(f"  Loss > 20%:  {stats['probability_loss_gt_20pct']:>10.2%}")
    
    # Plain language explanation
    report_lines.append("\n" + "=" * 80)
    report_lines.append("PLAIN LANGUAGE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("""
This report measures how much your portfolio historically fluctuated (volatility),
how severe the worst losses were (drawdowns), and estimates how large a loss could
be under normal and stressed conditions (VaR and CVaR).

KEY METRICS EXPLAINED:

• VaR (Value at Risk): The maximum loss you'd expect on a typical bad day.
  For example, VaR95 of $500 means there's only a 5% chance you'll lose more
  than $500 in a day.

• CVaR (Conditional VaR): The average loss when things get worse than VaR.
  This tells you how bad things get when you hit that worst 5% of days.

• Sharpe Ratio: Reward per unit of risk. Higher is better. A Sharpe of 1.0
  means you earn 1% extra return for each 1% of volatility.

• Maximum Drawdown: The biggest peak-to-trough decline. A 20% max drawdown
  means your portfolio fell 20% from its highest point before recovering.

RECOMMENDATIONS:

• Use these numbers to match your risk tolerance and investment horizon
• Lower VaR/CVaR and narrower drawdowns typically imply less risk
• Consider diversification if single-asset risk is high
• Re-run this analysis monthly to track changes in risk profile
• Use stress tests to understand tail risk scenarios
""")
    
    report_lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path / 'report.txt', 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete portfolio risk analysis pipeline'
    )
    parser.add_argument(
        '--portfolio',
        default='sample_data/example_portfolio.json',
        help='Path to portfolio JSON file'
    )
    parser.add_argument(
        '--start',
        help='Start date (YYYY-MM-DD), default: 3 years ago'
    )
    parser.add_argument(
        '--end',
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--output',
        default='./output',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        run_pipeline(
            args.portfolio,
            args.start,
            args.end,
            args.output
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()