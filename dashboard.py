"""
Interactive Risk Dashboard
==========================
Streamlit dashboard for portfolio risk analysis.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Retail Risk Kit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load analysis results if available."""
    output_dir = Path('./output')
    
    data = {
        'prices': None,
        'returns': None,
        'risk_summary': None,
        'simulation': None
    }
    
    # Load price data
    if (output_dir / 'prices_cleaned.parquet').exists():
        data['prices'] = pd.read_parquet(output_dir / 'prices_cleaned.parquet')
    
    # Load returns
    if (output_dir / 'returns.parquet').exists():
        data['returns'] = pd.read_parquet(output_dir / 'returns.parquet')
    
    # Load risk summary
    if (output_dir / 'risk_summary.json').exists():
        with open(output_dir / 'risk_summary.json', 'r') as f:
            data['risk_summary'] = json.load(f)
    
    # Load simulation results
    if (output_dir / 'simulation_results.json').exists():
        with open(output_dir / 'simulation_results.json', 'r') as f:
            data['simulation'] = json.load(f)
    
    return data


def plot_portfolio_value(prices, positions, cash):
    """Plot portfolio value over time."""
    portfolio_value = pd.Series(cash, index=prices.index)
    
    for ticker, qty in positions.items():
        if ticker in prices.columns:
            portfolio_value += prices[ticker] * qty
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_returns_distribution(returns):
    """Plot returns distribution."""
    portfolio_returns = returns.sum(axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=50,
        name='Returns Distribution',
        marker_color='#1f77b4',
        opacity=0.7
    ))
    
    # Add normal distribution overlay
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
    y = stats.norm.pdf(x, mean, std) * len(portfolio_returns) * (portfolio_returns.max() - portfolio_returns.min()) / 50
    
    from scipy import stats
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Returns Distribution',
        xaxis_title='Daily Return',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_drawdown(prices, positions, cash):
    """Plot drawdown chart."""
    portfolio_value = pd.Series(cash, index=prices.index)
    
    for ticker, qty in positions.items():
        if ticker in prices.columns:
            portfolio_value += prices[ticker] * qty
    
    cumulative = portfolio_value / portfolio_value.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_correlation_matrix(returns):
    """Plot correlation heatmap."""
    corr = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        template='plotly_white',
        height=500
    )
    
    return fig


def main():
    # Header
    st.title("📊 Retail Risk Kit Dashboard")
    st.markdown("### Interactive Portfolio Risk Analysis")
    
    # Load data
    data = load_data()
    
    # Sidebar
    st.sidebar.header("Portfolio Configuration")
    
    # Check if data exists
    if data['risk_summary'] is None:
        st.warning("⚠️ No analysis results found. Please run the pipeline first:")
        st.code("python run_all.py --portfolio sample_data/example_portfolio.json")
        st.stop()
    
    # Load portfolio config
    portfolio_path = Path('sample_data/example_portfolio.json')
    if portfolio_path.exists():
        with open(portfolio_path, 'r') as f:
            portfolio_config = json.load(f)
    else:
        st.error("Portfolio file not found!")
        st.stop()
    
    # Sidebar - Portfolio Info
    st.sidebar.subheader("Holdings")
    positions = {pos['ticker']: pos['quantity'] for pos in portfolio_config['positions']}
    
    for ticker, qty in positions.items():
        st.sidebar.text(f"{ticker}: {qty:.4f}")
    st.sidebar.text(f"Cash: ${portfolio_config.get('cash', 0):,.2f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Risk Metrics", "🎲 Simulation", "⚙️ Settings"])
    
    # Tab 1: Overview
    with tab1:
        st.header("Portfolio Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        risk = data['risk_summary']
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${risk['portfolio_value_end']:,.2f}",
                f"{risk['total_return']:.2%}"
            )
        
        with col2:
            st.metric(
                "Annualized Return",
                f"{risk['annualized_return']:.2%}",
                f"Vol: {risk['annualized_volatility']:.2%}"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{risk['sharpe_ratio']:.2f}",
                f"Sortino: {risk['sortino_ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{risk['max_drawdown']:.2%}",
                delta=None,
                delta_color="inverse"
            )
        
        # Charts
        if data['prices'] is not None:
            st.plotly_chart(
                plot_portfolio_value(data['prices'], positions, portfolio_config.get('cash', 0)),
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_drawdown(data['prices'], positions, portfolio_config.get('cash', 0)),
                    use_container_width=True
                )
            
            with col2:
                if data['returns'] is not None:
                    st.plotly_chart(
                        plot_returns_distribution(data['returns']),
                        use_container_width=True
                    )
    
    # Tab 2: Risk Metrics
    with tab2:
        st.header("Detailed Risk Metrics")
        
        risk = data['risk_summary']
        
        # VaR and CVaR
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Value at Risk (VaR)")
            st.markdown(f"""
            **Daily VaR (95%)**: ${risk['var_95_daily_dollar']:,.2f} ({risk['var_95_daily_pct']:.2%})
            
            *This means there's only a 5% chance you'll lose more than this amount in a single day.*
            """)
            
            if 'var_95_monthly_dollar' in risk:
                st.markdown(f"""
                **Monthly VaR (95%)**: ${risk['var_95_monthly_dollar']:,.2f} ({risk['var_95_monthly_pct']:.2%})
                """)
        
        with col2:
            st.subheader("Conditional VaR (CVaR)")
            st.markdown(f"""
            **Daily CVaR (95%)**: ${risk['cvar_95_daily_dollar']:,.2f} ({risk['cvar_95_daily_pct']:.2%})
            
            *When you do hit that worst 5% of days, this is your average loss.*
            """)
        
        st.divider()
        
        # Other metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Omega Ratio", f"{risk['omega_ratio']:.2f}")
            st.caption("Ratio of gains to losses. Higher is better.")
        
        with col2:
            st.metric("Skewness", f"{risk['skewness']:.2f}")
            st.caption("Asymmetry of returns. Negative = more downside.")
        
        with col3:
            st.metric("Kurtosis", f"{risk['kurtosis']:.2f}")
            st.caption("Tail thickness. Higher = more extreme events.")
        
        # Benchmark comparison
        if 'beta_vs_benchmark' in risk:
            st.divider()
            st.subheader(f"Benchmark Comparison ({risk['benchmark_ticker']})")
            st.metric("Beta", f"{risk['beta_vs_benchmark']:.2f}")
            st.caption("How much your portfolio moves with the market. 1.0 = same as market.")
        
        # Correlation matrix
        if data['returns'] is not None:
            st.divider()
            st.plotly_chart(
                plot_correlation_matrix(data['returns']),
                use_container_width=True
            )
    
    # Tab 3: Simulation
    with tab3:
        st.header("Monte Carlo Simulation Results")
        
        if data['simulation'] is not None:
            sim = data['simulation']
            stats = sim['statistics']
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Expected Final Value",
                    f"${stats['mean_final_value']:,.2f}",
                    f"{(stats['mean_final_value'] / sim['initial_value'] - 1):.2%}"
                )
            
            with col2:
                st.metric(
                    "5th Percentile",
                    f"${stats['percentile_5']:,.2f}",
                    f"{(stats['percentile_5'] / sim['initial_value'] - 1):.2%}"
                )
            
            with col3:
                st.metric(
                    "95th Percentile",
                    f"${stats['percentile_95']:,.2f}",
                    f"{(stats['percentile_95'] / sim['initial_value'] - 1):.2%}"
                )
            
            # Probabilities
            st.divider()
            st.subheader("Loss Probabilities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Any Loss", f"{stats['probability_of_loss']:.1%}")
            
            with col2:
                st.metric("Loss > 10%", f"{stats['probability_loss_gt_10pct']:.1%}")
            
            with col3:
                st.metric("Loss > 20%", f"{stats['probability_loss_gt_20pct']:.1%}")
            
            # Distribution plot
            st.divider()
            st.subheader("Simulated Final Value Distribution")
            
            # Create histogram
            values = np.random.normal(
                stats['mean_final_value'],
                stats['std_final_value'],
                10000
            )
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=values,
                nbinsx=50,
                name='Simulated Values',
                marker_color='#1f77b4',
                opacity=0.7
            ))
            
            # Add percentile lines
            fig.add_vline(x=stats['percentile_5'], line_dash="dash", line_color="red",
                         annotation_text="5th %ile")
            fig.add_vline(x=stats['percentile_95'], line_dash="dash", line_color="green",
                         annotation_text="95th %ile")
            
            fig.update_layout(
                xaxis_title='Final Portfolio Value ($)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No simulation results available. Run simulation first.")
    
    # Tab 4: Settings
    with tab4:
        st.header("Settings & Actions")
        
        st.subheader("Re-run Analysis")
        
        if st.button("🔄 Refresh Data & Recompute", type="primary"):
            st.info("Running analysis pipeline...")
            import subprocess
            result = subprocess.run(
                ["python", "run_all.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Analysis complete! Refresh the page to see new results.")
            else:
                st.error(f"Error running analysis: {result.stderr}")
        
        st.divider()
        
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Risk Summary JSON"):
                if data['risk_summary']:
                    st.download_button(
                        "Download",
                        json.dumps(data['risk_summary'], indent=2),
                        "risk_summary.json",
                        "application/json"
                    )
        
        with col2:
            if st.button("📥 Download Simulation Results"):
                if data['simulation']:
                    st.download_button(
                        "Download",
                        json.dumps(data['simulation'], indent=2),
                        "simulation_results.json",
                        "application/json"
                    )
        
        st.divider()
        
        st.subheader("About")
        st.markdown("""
        **Retail Risk Kit** v1.0
        
        A comprehensive portfolio risk analysis system for small retail investors.
        
        - 📊 Real-time data from Yahoo Finance
        - 📈 VaR, CVaR, and drawdown analysis
        - 🎲 Monte Carlo simulation
        - ⚡ Fast, reproducible, open-source
        
        [Documentation](https://github.com/your-repo) | [Report Issues](https://github.com/your-repo/issues)
        """)


if __name__ == '__main__':
    main()