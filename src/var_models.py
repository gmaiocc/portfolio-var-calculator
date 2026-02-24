import numpy as np
from scipy import stats

def historical_var(portfolio_returns, confidence_levels=[0.95, 0.99]):
    results = {}

    for confidence in confidence_levels:
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        results[f"VaR {int(confidence*100)}%"] = var

    return results


def parametric_var(portfolio_returns, confidence_levels=[0.95, 0.99]):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    
    results = {}
    
    for confidence in confidence_levels:
        z = stats.norm.ppf(1 - confidence)
        var = mean + (z * std)
        results[f"VaR {int(confidence*100)}%"] = var
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from data import download_data, calculate_returns
    from portfolio import create_portfolio

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    weights = [0.4, 0.3, 0.3]

    prices = download_data(tickers, start='2020-01-01', end='2024-12-31')
    returns = calculate_returns(prices)
    portfolio_returns = create_portfolio(returns, weights)

    var_results = historical_var(portfolio_returns)
    param_results = parametric_var(portfolio_returns)


    print("VaR HISTÓRICO")
    for k, v in var_results.items():
        print(f"{k}: {v:.4f} ({v*100:.2f}%)")
        
    print("\nVaR PARAMÉTRICO")
    for k, v in param_results.items():
        print(f"{k}: {v:.4f} ({v*100:.2f}%)")