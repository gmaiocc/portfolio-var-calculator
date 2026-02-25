import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys

# Append source directory
sys.path.append('src')

# Import custom quantitative modules
from data import download_data, calculate_returns
from portfolio import create_portfolio, portfolio_stats, efficient_frontier
from var_models import historical_var, parametric_var, monte_carlo_var, backtest_var, cvar

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantitative Risk Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced CSS Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

/* Global Reset and Typography */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

/* Custom Metric Cards */
.metric-container {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 24px;
    transition: all 0.2s ease-in-out;
    margin-bottom: 20px;
}

.metric-container:hover {
    transform: translateY(-4px);
    border-color: #58a6ff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 12px;
    font-weight: 600;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1;
    margin-bottom: 8px;
}

.metric-sub {
    font-size: 0.8rem;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub.negative { color: #f85149; }
.metric-sub.positive { color: #3fb950; }

/* Section Headers */
.section-header {
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-top: 32px;
    margin-bottom: 24px;
    font-size: 1.25rem;
    font-weight: 500;
    color: #e6edf3;
}

/* Status Badges */
.status-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.status-pass { background: rgba(46, 160, 67, 0.1); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
.status-fail { background: rgba(248, 81, 73, 0.1); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
.status-warn { background: rgba(210, 153, 34, 0.1); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.4); }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #010409 !important;
    border-right: 1px solid #30363d !important;
}
</style>
""", unsafe_allow_html=True)

# --- UI Helper Functions ---
def metric_card(title, value, subtitle, sub_class=""):
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub {sub_class}">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Data Processing Logic ---
@st.cache_data(show_spinner=False)
def load_and_process_data(tickers, start_date, end_date, weights_decimal):
    prices = download_data(tickers, str(start_date), str(end_date))
    returns = calculate_returns(prices)
    port_ret = create_portfolio(returns, weights_decimal)
    return prices, returns, port_ret

def calculate_drawdowns(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return drawdowns

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<h2 style='color: #e6edf3; font-size: 1.1rem; font-weight: 600; margin-bottom: 20px;'>PORTFOLIO SETUP</h2>", unsafe_allow_html=True)
    
    tickers_input = st.text_input("Asset Identifiers (CSV)", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    weights_input = st.text_input("Target Allocation (%)", "40, 30, 30")
    try:
        weights = [float(w.strip()) for w in weights_input.split(",")]
    except ValueError:
        st.error("Format error: Numeric values required.")
        st.stop()
        
    weights_decimal = [w / 100 for w in weights]
    
    st.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #e6edf3; font-size: 1.1rem; font-weight: 600; margin-bottom: 20px;'>PARAMETERS</h2>", unsafe_allow_html=True)
    
    start_date = st.date_input("Inception Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Valuation Date", pd.to_datetime("2024-12-31"))
    
    portfolio_value = st.number_input("Notional Value (USD)", min_value=1000, value=1000000, step=50000)
    n_simulations = st.selectbox("Monte Carlo Iterations", [1000, 5000, 10000, 25000], index=2)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_analysis = st.button("Compute Risk Metrics", type="primary", use_container_width=True)

# --- Validation Engine ---
if round(sum(weights_decimal), 5) != 1.0:
    st.error(f"Allocation Error: Sum of weights equals {sum(weights)}%. Must equal 100%.")
    st.stop()

if len(tickers) != len(weights):
    st.error("Dimension Error: Asset count must match weight count.")
    st.stop()

if not run_analysis:
    st.markdown("""
        <div style="height: 80vh; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <h1 style="color: #30363d; font-size: 3rem; font-weight: 700; letter-spacing: -0.05em;">QUANTITATIVE TERMINAL</h1>
            <p style="color: #8b949e; font-family: 'JetBrains Mono', monospace;">Awaiting parameter configuration and execution trigger.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Execution Engine ---
with st.spinner("Aggregating timeseries and calculating risk parameters..."):
    prices, returns, port_ret = load_and_process_data(tickers, start_date, end_date, weights_decimal)
    stats = portfolio_stats(port_ret)
    drawdowns = calculate_drawdowns(port_ret)
    max_dd = drawdowns.min()
    
    h_var = historical_var(port_ret)
    p_var = parametric_var(port_ret)
    mc_var, sim_ret = monte_carlo_var(port_ret, n_simulations=n_simulations)
    bt = backtest_var(port_ret)
    cv = cvar(port_ret)
    
    ef_df, min_v, max_s = efficient_frontier(returns)

# --- Header ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 20px; border-bottom: 1px solid #30363d; margin-bottom: 30px;">
        <div>
            <h1 style="margin: 0; font-size: 2rem; color: #f0f6fc;">Portfolio Risk & Optimization</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- KPI Tier ---
col1, col2, col3, col4 = st.columns(4)
with col1: metric_card("Annualized Return", f"{stats['Retorno anualizado']*100:.2f}%", "Geometrically linked", "positive" if stats['Retorno anualizado'] > 0 else "negative")
with col2: metric_card("Annualized Volatility", f"{stats['Volatilidade anualizada']*100:.2f}%", "Standard Deviation", "")
with col3: metric_card("Sharpe Ratio", f"{stats['Retorno anualizado']/stats['Volatilidade anualizada']:.3f}", "Risk-Adjusted Premium", "positive")
with col4: metric_card("Maximum Drawdown", f"{max_dd*100:.2f}%", "Peak-to-Trough Decline", "negative")

# --- Interface Tabs ---
tab_risk, tab_perf, tab_ef = st.tabs(["Value at Risk & Shortfall", "Performance & Backtesting", "Markowitz Optimization"])

# --- TAB 1: Risk Models ---
with tab_risk:
    st.markdown("<div class='section-header'>Tail Risk Distribution (VaR & CVaR)</div>", unsafe_allow_html=True)
    
    var_matrix = {
        "Model Type": ["Historical Simulation", "Parametric (Gaussian)", f"Monte Carlo ({n_simulations:,})"],
        "95% Confidence Threshold": [f"{h_var['VaR 95%']*100:.3f}%", f"{p_var['VaR 95%']*100:.3f}%", f"{mc_var['VaR 95%']*100:.3f}%"],
        "99% Confidence Threshold": [f"{h_var['VaR 99%']*100:.3f}%", f"{p_var['VaR 99%']*100:.3f}%", f"{mc_var['VaR 99%']*100:.3f}%"],
        "1-Day 99% Exposure (USD)": [
            f"${abs(h_var['VaR 99%'])*portfolio_value:,.0f}",
            f"${abs(p_var['VaR 99%'])*portfolio_value:,.0f}",
            f"${abs(mc_var['VaR 99%'])*portfolio_value:,.0f}"
        ]
    }
    
    st.dataframe(
        pd.DataFrame(var_matrix), 
        use_container_width=True, 
        hide_index=True
    )
    
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Expected Shortfall (CVaR 95%)", f"{cv['CVaR 95%']['cvar']*100:.2f}%", f"Average conditional loss over {cv['CVaR 95%']['n_days_beyond']} events", "negative")
    with c2:
        metric_card("Expected Shortfall (CVaR 99%)", f"{cv['CVaR 99%']['cvar']*100:.2f}%", f"Average conditional loss over {cv['CVaR 99%']['n_days_beyond']} events", "negative")

# --- TAB 2: Performance & Backtesting ---
with tab_perf:
    st.markdown("<div class='section-header'>Historical Equity & Drawdown Profile</div>", unsafe_allow_html=True)
    
    cumulative_returns = (1 + port_ret).cumprod() * portfolio_value
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, name="NAV", line=dict(color="#58a6ff", width=2)))
    fig_equity.update_layout(
        title="Net Asset Value (NAV)",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="USD", xaxis_title=""
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdowns.index, y=drawdowns.values * 100,
        mode='lines', name="Drawdown",
        line=dict(color='#f85149', width=1.5),
        fill='tozeroy', fillcolor='rgba(248,81,73,0.08)'
    ))
    fig_dd.update_layout(
        title="Drawdown Profile",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Drawdown (%)", xaxis_title=""
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("<div class='section-header'>Regulatory Backtesting (Basel III & Kupiec)</div>", unsafe_allow_html=True)
    
    for conf, res in bt.items():
        parts = res['basel_zone'].split()
        b_zone = next((p for p in parts if p.isalpha()), 'N/A').upper()        
        b_class = "status-pass" if "GREEN" in b_zone else ("status-warn" if "YELLOW" in b_zone else "status-fail")
        
        k_pass = res['kupiec_pass']
        k_class = "status-pass" if k_pass else "status-fail"
        k_text = "ACCEPTED" if k_pass else "REJECTED"
        
        st.markdown(f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <span style="font-weight: 600; font-size: 1.1rem;">Interval: {conf}</span>
                    <div>
                        <span class="status-badge {k_class}" style="margin-right: 10px;">KUPIEC: {k_text}</span>
                        <span class="status-badge {b_class}">BASEL: {b_zone}</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
                    <div><span style="color: #8b949e;">Calibrated VaR:</span> <br>{res['var']*100:.3f}%</div>
                    <div><span style="color: #8b949e;">Test Horizon:</span> <br>{res['n_test_days']} days</div>
                    <div><span style="color: #8b949e;">Realized Exceptions:</span> <br>{res['n_exceptions']}</div>
                    <div><span style="color: #8b949e;">Expected Exceptions:</span> <br>{res['expected_exceptions']:.1f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 3: Markowitz Optimization ---
with tab_ef:
    st.markdown("<div class='section-header'>Efficient Frontier & Capital Allocation Line</div>", unsafe_allow_html=True)
    
    fig_ef = go.Figure()
    
    fig_ef.add_trace(go.Scatter(
        x=ef_df['Volatility']*100, y=ef_df['Return']*100, mode='markers',
        marker=dict(color=ef_df['Sharpe'], colorscale='Blues', showscale=True, size=5,
                    colorbar=dict(title="Sharpe", tickfont=dict(color="#8b949e"))),
        name='Stochastic Portfolios', hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%"
    ))
    
    fig_ef.add_trace(go.Scatter(
        x=[stats['Volatilidade anualizada']*100], y=[stats['Retorno anualizado']*100],
        mode='markers', marker=dict(color='#f85149', size=14, symbol='cross'),
        name='Target Allocation'
    ))
    
    fig_ef.add_trace(go.Scatter(
        x=[min_v['Volatility']*100], y=[min_v['Return']*100],
        mode='markers+text',
        marker=dict(color='#3fb950', size=12, symbol='diamond'),
        text=["Min Variance"], textposition="top right",
        textfont=dict(color='#3fb950', size=11),
        name='Min Variance'
    ))

    fig_ef.add_trace(go.Scatter(
        x=[max_s['Volatility']*100], y=[max_s['Return']*100],
        mode='markers+text',
        marker=dict(color='#d29922', size=12, symbol='star'),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color='#d29922', size=11),
        name='Max Sharpe'
    ))

    fig_ef.update_layout(
        xaxis_title="Volatility (Risk) %", yaxis_title="Expected Return %",
        template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=20, b=0), hovermode="closest"
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    st.markdown("<div class='section-header'>Asset Cross-Correlation Matrix</div>", unsafe_allow_html=True)
    corr = returns.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_dark"
    )
    fig_corr.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)