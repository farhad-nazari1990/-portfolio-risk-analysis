# Module Documentation

## data_fetcher.py

### Purpose
Fetches historical price data from multiple sources (Yahoo Finance, Alpha Vantage, or local CSV).

### Key Functions

- `fetch_yahoo(tickers, start, end, use_cache=True)` - Fetch from Yahoo Finance
- `fetch_alpha_vantage(tickers, start, end, api_key=None)` - Fetch from Alpha Vantage
- `read_local_csv(path, date_col='Date')` - Read from local CSV
- `fetch_portfolio_data(portfolio_path, start, end, source='yahoo')` - Fetch all tickers in portfolio

### Usage Example

```bash
# Fetch data for a portfolio
python data_fetcher.py \
  --portfolio sample_data/example_portfolio.json \
  --start 2020-01-01 \
  --end 2025-10-30 \
  --output data/prices.parquet

# Use Alpha Vantage instead
export ALPHA_VANTAGE_KEY=your_key_here
python data_fetcher.py \
  --portfolio sample_data/example_portfolio.json \
  --start 2020-01-01 \
  --end 2025-10-30 \
  --output data/prices.parquet \
  --source alpha_vantage

# Use local CSV
python data_fetcher.py \
  --portfolio sample_data/example_portfolio.json \
  --start 2020-01-01 \
  --end 2025-10-30 \
  --output data/prices.parquet \
  --source sample_data/prices_sample.csv
```

### Expected Output

Creates a parquet file with:
- Index: DatetimeIndex (timezone-naive)
- Columns: One per ticker symbol
- Values: Adjusted close prices

```
Date        AAPL    MSFT    VTI     ...
2020-01-01  75.09   160.62  162.02  ...
2020-01-02  75.35   160.49  162.48  ...
...
```

---

## data_cleaner.py

### Purpose
Cleans and preprocesses price data, handles missing values, computes returns.

### Key Functions

- `clean_price_data(prices, max_missing_pct=0.2)` - Clean and fill gaps
- `compute_returns(prices, method='log', freq='daily')` - Calculate returns
- `align_data_with_benchmark(prices, benchmark_ticker)` - Align with benchmark
- `compute_portfolio_value(prices, positions, cash)` - Calculate portfolio value
- `winsorize_returns(returns, lower_pct=0.01, upper_pct=0.99)` - Handle outliers

### Usage Example

```bash
# Clean price data and compute returns
python data_cleaner.py \
  --input data/prices.parquet \
  --output data/cleaned_prices.parquet \
  --returns-output data/returns.parquet \
  --max-missing 0.2 \
  --return-method log \
  --freq daily

# With winsorization
python data_cleaner.py \
  --input data/prices.parquet \
  --output data/cleaned_prices.parquet \
  --returns-output data/returns.parquet \
  --winsorize
```

### Expected Output

**cleaned_prices.parquet**: Same structure as input, but with:
- Missing values filled (forward/backfill)
- Tickers with >20% missing data removed
- No NaN values

**returns.parquet**: Returns DataFrame
```
Date        AAPL      MSFT      VTI       ...
2020-01-02  0.0035    -0.0008   0.0028    ...
2020-01-03  0.0042    0.0015    0.0019    ...
...
```

---

## risk_engine.py

### Purpose
Calculates comprehensive portfolio risk metrics.

### Key Functions

- `annualized_return(returns, freq='daily')` - Annualized return
- `annualized_volatility(returns, freq='daily')` - Annualized volatility
- `calc_var(returns, confidence_level=0.95, method='historical')` - Value at Risk
- `calc_cvar(returns, confidence_level=0.95, method='historical')` - Conditional VaR
- `max_drawdown(prices)` - Maximum drawdown
- `sharpe_ratio(returns, risk_free_rate=0.0, freq='daily')` - Sharpe ratio
- `sortino_ratio(returns, risk_free_rate=0.0, freq='daily')` - Sortino ratio
- `beta_vs_benchmark(returns, benchmark_returns)` - Beta coefficient
- `calculate_portfolio_risk_metrics(prices, portfolio_config, ...)` - Complete analysis

### Usage Example

```bash
python risk_engine.py \
  --prices data/cleaned_prices.parquet \
  --portfolio sample_data/example_portfolio.json \
  --output output/risk_summary.json \
  --confidence 0.95 \
  --risk-free-rate 0.04
```

### Expected Output

**risk_summary.json**:
```json
{
  "portfolio_value_start": 50000.00,
  "portfolio_value_end": 57500.00,
  "total_return": 0.15,
  "annualized_return": 0.12,
  "annualized_volatility": 0.18,
  "sharpe_ratio": 0.67,
  "sortino_ratio": 0.89,
  "max_drawdown": 0.22,
  "max_drawdown_peak_date": "2022-01-04",
  "max_drawdown_trough_date": "2022-10-12",
  "var_95_daily_pct": 0.025,
  "var_95_daily_dollar": 1437.50,
  "cvar_95_daily_pct": 0.037,
  "cvar_95_daily_dollar": 2127.50,
  "var_95_monthly_pct": 0.089,
  "var_95_monthly_dollar": 5117.50,
  "omega_ratio": 1.34,
  "beta_vs_benchmark": 0.95,
  "benchmark_ticker": "SPY",
  "skewness": -0.23,
  "kurtosis": 3.45
}
```

### Interpretation Guide

- **VaR95 Daily = $1,437**: On a typical bad day (worst 5%), expect to lose up to $1,437
- **CVaR95 Daily = $2,127**: When you hit that worst 5%, average loss is $2,127
- **Max Drawdown = 22%**: Worst peak-to-trough decline was 22%
- **Sharpe = 0.67**: You earn 0.67% extra return per 1% of volatility
- **Beta = 0.95**: Portfolio moves slightly less than the market

---

## simulator.py

### Purpose
Monte Carlo simulation with correlated returns and stress testing.

### Key Functions

- `monte_carlo_correlated(returns, n_sims=10000, n_days=252, ...)` - Run MC simulation
- `simulate_portfolio_paths(portfolio_returns, initial_value)` - Convert to value paths
- `calculate_simulation_statistics(portfolio_returns, initial_value, ...)` - Compute stats
- `stress_test_scenario(returns, positions, shocks)` - Apply specific shocks
- `factor_stress_test(returns, positions, factor_shock, equity_tickers)` - Market crash scenario

### Usage Example

```bash
python simulator.py \
  --prices data/cleaned_prices.parquet \
  --portfolio sample_data/example_portfolio.json \
  --output output/simulation_results.json \
  --n-sims 10000 \
  --n-days 252 \
  --seed 42
```

### Expected Output

**simulation_results.json**:
```json
{
  "initial_value": 57500.00,
  "simulation_params": {
    "n_sims": 10000,
    "n_days": 252,
    "random_seed": 42
  },
  "simulation_statistics": {
    "mean_final_value": 63250.00,
    "median_final_value": 62100.00,
    "std_final_value": 8900.00,
    "min_final_value": 32400.00,
    "max_final_value": 98700.00,
    "percentile_5": 48900.00,
    "percentile_95": 79200.00,
    "probability_of_loss": 0.32,
    "probability_loss_gt_5pct": 0.22,
    "probability_loss_gt_10pct": 0.14,
    "probability_loss_gt_20pct": 0.06,
    "var_1": -15200.00,
    "var_5": -8600.00,
    "var_10": -5500.00
  },
  "stress_test_market_crash_30pct": {
    "shocks_applied": {
      "AAPL": -0.30,
      "MSFT": -0.30,
      "VTI": -0.30
    },
    "total_impact": -15000.00
  }
}
```

### Interpretation Guide

- **Mean Final Value = $63,250**: Expected value after 1 year
- **5th Percentile = $48,900**: In worst 5% of scenarios, end with $48,900 or less
- **Probability of Loss = 32%**: 32% chance of ending below starting value
- **Probability Loss >10% = 14%**: 14% chance of losing more than 10%

---

## run_all.py

### Purpose
Main orchestrator - runs complete pipeline end-to-end.

### Usage Example

```bash
# Run full pipeline with defaults
python run_all.py

# Custom portfolio and date range
python run_all.py \
  --portfolio my_portfolio.json \
  --start 2018-01-01 \
  --end 2025-10-30 \
  --output my_results/

# Using Docker
docker run -v $(pwd)/output:/app/output retail-risk-kit
```

### Pipeline Steps

1. **Load Portfolio**: Reads portfolio JSON
2. **Fetch Data**: Downloads historical prices
3. **Clean Data**: Handles missing values, computes returns
4. **Calculate Risk**: Computes VaR, CVaR, Sharpe, drawdowns
5. **Run Simulation**: Monte Carlo with 10,000 simulations
6. **Generate Report**: Creates text report and JSON outputs

### Expected Outputs

```
output/
├── prices_raw.parquet           # Raw price data
├── prices_cleaned.parquet       # Cleaned prices
├── returns.parquet              # Daily returns
├── risk_summary.json            # All risk metrics
├── simulation_results.json      # Monte Carlo results
└── report.txt                   # Human-readable summary
```

---

## Common Workflows

### Workflow 1: Quick Analysis

```bash
# One command to analyze sample portfolio
python run_all.py

# View results
cat output/report.txt
python -m json.tool output/risk_summary.json
```

### Workflow 2: Custom Portfolio Analysis

1. Create your portfolio JSON:
```json
{
  "owner": "Jane Doe",
  "currency": "USD",
  "positions": [
    {"ticker": "AAPL", "quantity": 50},
    {"ticker": "GOOGL", "quantity": 10}
  ],
  "cash": 10000,
  "benchmark": "SPY"
}
```

2. Run analysis:
```bash
python run_all.py --portfolio my_portfolio.json
```

### Workflow 3: Update Existing Analysis

```bash
# Re-fetch latest data (no cache)
python data_fetcher.py \
  --portfolio my_portfolio.json \
  --start 2020-01-01 \
  --end 2025-10-30 \
  --output data/prices.parquet \
  --no-cache

# Re-run analysis
python run_all.py --portfolio my_portfolio.json
```

### Workflow 4: Stress Testing

```bash
# Run simulation with custom scenarios
python simulator.py \
  --prices data/prices.parquet \
  --portfolio my_portfolio.json \
  --output stress_test.json \
  --n-sims 50000  # More simulations for better tail estimates
```

---

## Testing

Run unit tests:
```bash
# All tests
pytest

# Specific module
pytest tests/test_risk_engine.py -v

# With coverage
pytest --cov=. --cov-report=html
```

---

## Troubleshooting

### Issue: "Failed to fetch data for any tickers"

**Solution**: 
- Check internet connection
- Verify ticker symbols are correct
- Try with `--no-cache` flag
- Check Yahoo Finance is accessible

### Issue: "Too much missing data"

**Solution**:
- Increase `--max-missing` threshold
- Use longer date range
- Remove problematic tickers from portfolio

### Issue: "Covariance matrix not positive definite"

**Solution**:
- This can happen with highly correlated or short time series
- System automatically falls back to diagonal matrix
- Consider longer historical period

### Issue: API Rate Limits

**Solution**:
- System includes automatic rate limiting and retries
- Use cache (default enabled)
- For Alpha Vantage, wait 1 minute between runs (free tier limit)

---

## Performance Tips

1. **Use Parquet**: Faster than CSV for large datasets
2. **Enable Caching**: Speeds up repeated runs (default: on)
3. **Reduce Simulations**: Start with 1,000 sims for testing, use 10,000+ for production
4. **Shorter Date Ranges**: If you only need recent risk metrics

---

## Next Steps

1. Review generated `report.txt` for high-level summary
2. Examine `risk_summary.json` for detailed metrics
3. Run `streamlit run dashboard.py` for interactive exploration
4. Customize portfolio and re-run
5. Set up monthly scheduled runs to track risk evolution