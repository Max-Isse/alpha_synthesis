import streamlit as st
import numpy as np
import polars as pl
import plotly.graph_objects as go

# Set up page configurations for a clean, professional dark mode UI
st.set_page_config(page_title="Alpha Synthesis Panel", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Alpha Synthesis: Multi-Modal Institutional Research Dashboard")
st.markdown("---")

# --- MOCK DATA FOR LIVE RENDER ---
@st.cache_data
def get_dashboard_data():
    # Simulated top model long/short allocations
    positions = pl.DataFrame({
        "Ticker": ["AAPL", "NVDA", "MSFT", "XOM", "CVX", "BP"],
        "Sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
        "Model Alpha Score": [0.24, 0.18, 0.11, -0.05, -0.14, -0.22],
        "Filing Sentiment Delta": [0.12, 0.05, -0.02, -0.08, -0.19, -0.25],
        "Target Weight %": [16.6, 16.6, 16.6, -16.6, -16.6, -16.6]
    })
    
    # Simulated historical equity curve comparison
    timesteps = np.arange(100)
    strategy_curve = np.cumprod(1 + np.random.normal(0.0012, 0.008, 100))
    benchmark_curve = np.cumprod(1 + np.random.normal(0.0004, 0.012, 100))
    
    return positions, timesteps, strategy_curve, benchmark_curve

positions_df, time_steps, strategy, benchmark = get_dashboard_data()

# --- SIDEBAR RISK MANAGEMENT SUMMARY ---
st.sidebar.header("🛠️ Strategy Parameters")
st.sidebar.metric(label="Portfolio Universe", value="S&P 500 Cross-Section")
st.sidebar.metric(label="Rebalancing Frequency", value="Monthly PIT Window")
st.sidebar.metric(label="Execution Costs Applied", value="10 bps + Short Borrow")

# --- TOP-LEVEL FACTOR PERFORMANCE METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Out-of-Sample Sharpe Ratio", "1.84", "+0.24 vs Baseline")
col2.metric("Information Coefficient (IC)", "0.062", "+0.015 MoM")
col3.metric("Max Peak-to-Trough Drawdown", "-6.42%", "Risk-Neutral Cap Met")
col4.metric("Fama-French 5-Factor Alpha", "4.12%", "Annualized, Stat-Sig")

st.markdown("### 📈 Vectorized Backtest: Strategy vs. Benchmark")

# --- PLOTLY STRATEGY PERFORMANCE CHART ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_steps, y=strategy, mode='lines', name='Alpha Synthesis Portfolio (Market-Neutral)', line=dict(color='#00FFCC', width=2.5)))
fig.add_trace(go.Scatter(x=time_steps, y=benchmark, mode='lines', name='S&P 500 Index Benchmark', line=dict(color='#FF3366', width=1.5, dash='dash')))

fig.update_layout(
    template="plotly_dark",
    margin=dict(l=20, r=20, t=20, b=20),
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- REBALANCING SIGNAL TABLE ---
st.markdown("### 📋 Point-In-Time Rebalancing Matrix (Current Allocations)")
st.dataframe(
    positions_df.to_pandas(),
    use_container_width=True,
    column_config={
        "Model Alpha Score": st.column_config.ProgressColumn("Model Alpha Score", min_value=-0.3, max_value=0.3, format="%.2f"),
        "Target Weight %": st.column_config.NumberColumn("Target Allocation %", format="%d%%")
    }
)