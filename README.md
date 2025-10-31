# Retail Risk Kit

A comprehensive portfolio risk analysis system designed for small retail investors. Supports 5-50 positions including stocks, ETFs, and crypto with fractional shares.

## Features

- **Data Ingestion**: Yahoo Finance (free), Alpha Vantage (optional), CSV imports
- **Risk Metrics**: VaR, CVaR, drawdowns, volatility, Sharpe/Sortino ratios
- **Simulations**: Monte Carlo with correlated returns, stress testing
- **Optimization**: Mean-variance, maximum Sharpe, CVaR-constrained
- **Backtesting**: Strategy comparison with transaction costs
- **Reporting**: JSON summaries, PDF/HTML reports, interactive dashboard

## Quick Start

### Local Installation

```bash
# Clone and setup
git clone <repo-url>
cd retail-risk-kit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (optional for Alpha Vantage)
echo "ALPHA_VANTAGE_KEY=your_key_here" > .env

# Run on sample portfolio
python run_all.py --portfolio sample_data/example_portfolio.json

# View results
ls -la output/
```

### Docker

```bash
# Build image
docker build -t retail-risk-kit .

# Run analysis
docker run -v $(pwd)/output:/app/output retail-risk-kit

# Run dashboard
docker run -p 8501:8501 retail-risk-kit streamlit run dashboard.py
```

## Project Structure

```
retail-risk-kit/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── run_all.py              # Main pipeline orchestrator
├── data_fetcher.py         # Fetch price data
├── data_cleaner.py         # Clean and align data
├── risk_engine.py          # Compute risk metrics
├── simulator.py            # Monte Carlo & stress tests
├── optimizer.py            # Portfolio optimization
├── backtester.py           # Strategy backtesting
├── report_generator.py     # Generate reports
├── dashboard.py            # Streamlit dashboard
├── sample_data/
│   ├── example_portfolio.json
│   └── prices_sample.csv
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_data_cleaner.py
│   └── ...
├── cache/                  # Cached API responses
└── output/                 # Generated reports
```

## Usage Examples

### Fetch Data

```bash
python data_fetcher.py --portfolio sample_data/example_portfolio.json \
  --start 2020-01-01 --end 2025-10-30 --output data/prices.parquet
```

### Run Risk Analysis

```bash
python risk_engine.py --prices data/prices.parquet \
  --portfolio sample_data/example_portfolio.json \
  --output output/risk_summary.json
```

### Launch Dashboard

```bash
streamlit run dashboard.py
# Navigate to http://localhost:8501
```

## Understanding the Results

### What is Value at Risk (VaR)?
VaR estimates the maximum loss you might face over a specific time period with a given confidence level. For example, a daily VaR95 of $500 means there's only a 5% chance you'll lose more than $500 in a single day.

### What is Conditional VaR (CVaR)?
CVaR is the average loss when things get worse than VaR. If your VaR is $500 and CVaR is $750, it means when you do exceed that 5% worst case, you typically lose $750 on average.

### What is Maximum Drawdown?
The largest peak-to-trough decline in your portfolio value. A 20% max drawdown means at worst, your portfolio fell 20% from its highest point before recovering.

### Plain Language Summary
This system measures how much your portfolio historically fluctuated (volatility), how severe the worst losses were (drawdowns), and estimates how large a loss could be under normal and stressed conditions (VaR and CVaR). Use these numbers to match your risk tolerance and investment horizon; lower VaR/CVaR and narrower drawdowns typically imply less risk but often lower expected return.

## Configuration

### Environment Variables (.env)
```
ALPHA_VANTAGE_KEY=your_key_here
CACHE_DIR=./cache
OUTPUT_DIR=./output
```

### Portfolio JSON Format
```json
{
  "owner": "Sample Investor",
  "currency": "USD",
  "positions": [
    {"ticker": "AAPL", "quantity": 10},
    {"ticker": "MSFT", "quantity": 5}
  ],
  "cash": 2500,
  "benchmark": "SPY",
  "risk_horizon_days": 252
}
```

## Security & Privacy

⚠️ **Important Security Notes:**
- Never commit `.env` files with API keys
- Don't share portfolio JSON files publicly (contains your holdings)
- Use environment variables for all secrets
- Be cautious with third-party data providers

## Model Assumptions & Limitations

- **Returns Distribution**: Assumes log-normal returns (may underestimate tail risk)
- **Stationarity**: Assumes past statistics predict future (breaks during regime changes)
- **Correlations**: Uses historical correlations (can change rapidly in crisis)
- **VaR Limitations**: Parametric VaR assumes normality; use CVaR for tail risk
- **Rebalancing**: Ignores market impact for large positions

**Recommendation**: Re-calibrate risk metrics monthly; use stress tests for tail scenarios.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific module
pytest tests/test_risk_engine.py -v
```

## Next Steps for Production

1. **Enhanced Data**: Integrate paid data providers (Bloomberg, Refinitiv)
2. **Real-time**: Add streaming price updates
3. **Alerts**: Email/SMS notifications for breaches
4. **Multi-currency**: Handle FX conversions
5. **Tax**: Add tax-loss harvesting analysis
6. **Machine Learning**: Regime detection, return forecasting

## Support & Contributing

For issues, feature requests, or contributions, please open an issue on GitHub.

## License

MIT License - See LICENSE file for details
