# Portfolio VaR Calculator

A quantitative risk management dashboard built in Python, implementing the three standard Value at Risk methodologies used in professional market risk environments.

---

## Overview

This project calculates and compares Value at Risk (VaR) across three methods, alongside Expected Shortfall (CVaR), regulatory backtesting, and Markowitz portfolio optimization — all visualized in an interactive Streamlit dashboard.

Built as a learning project to develop practical skills in market risk, quantitative finance, and data visualization.

---

## Features

- **Historical VaR** — percentile-based approach using actual return distribution
- **Parametric VaR** — Gaussian assumption with z-score methodology
- **Monte Carlo VaR** — 10,000 simulated scenarios
- **CVaR / Expected Shortfall** — average loss beyond the VaR threshold
- **Backtesting** — Kupiec POF test and Basel III traffic light zones
- **Efficient Frontier** — Markowitz optimization with 10,000 random portfolios
- **Correlation Matrix** — pairwise asset correlation heatmap and 60-day rolling correlation
- **Drawdown Profile** — peak-to-trough decline over time

---

## Tech Stack

- `pandas` / `numpy` / `scipy` — data processing and statistics
- `yfinance` — market data via Yahoo Finance API
- `plotly` — interactive charts
- `streamlit` — web dashboard

---

## Project Structure
```
portfolio-var-calculator/
│
├── src/
│   ├── data.py              # data download and return calculation
│   ├── portfolio.py         # portfolio construction and efficient frontier
│   └── var_models.py        # VaR, CVaR, and backtesting models
│
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/portfolio-var-calculator.git
cd portfolio-var-calculator
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run app.py
```

---

## Methodology

### Value at Risk (VaR)
VaR answers the question: *"What is the maximum loss I can expect on a bad day, with X% confidence?"*

| Method | Approach | Assumption |
|---|---|---|
| Historical | Uses actual past returns directly | None |
| Parametric | Uses mean + z-score × std deviation | Normal distribution |
| Monte Carlo | Simulates 10,000 random scenarios | Normal distribution |

### CVaR (Expected Shortfall)
Extends VaR by answering: *"Given that we are in the worst X% of days, what is the average loss?"* Required under Basel IV / FRTB.

### Backtesting
Verifies model accuracy by splitting data into train/test periods and counting VaR exceptions. Uses the Kupiec POF test for statistical validity and Basel III traffic light zones (Green / Yellow / Red).

### Efficient Frontier
Simulates 10,000 random portfolio weight combinations to map the risk-return space, identifying the Minimum Variance and Maximum Sharpe Ratio portfolios.

---

## Example Results (AAPL 40% / MSFT 30% / GOOGL 30% — 2020 to 2024)

| Method | VaR 95% | VaR 99% |
|---|---|---|
| Historical | -2.74% | -4.75% |
| Parametric | -2.83% | -4.05% |
| Monte Carlo | -2.85% | -4.04% |

CVaR 99%: **-6.44%** — on the worst 1% of days, the average loss was 6.44%.

Backtesting (99%): **Kupiec Pass · Basel Green Zone**

---

## License

MIT License — free to use and modify.