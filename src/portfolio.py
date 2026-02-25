import numpy as np
import pandas as pd

def create_portfolio(returns, weights):
    weights = np.array(weights)
    
    assert round(sum(weights), 5) == 1, "Os pesos têm de somar 1"
    
    portfolio_returns = returns.dot(weights)
    portfolio_returns.name = "Portfolio"
    return portfolio_returns
    
def portfolio_stats(portfolio_returns):
    stats = {
        "Retorno Médio Diário": portfolio_returns.mean(),
        "Volatilidade diária": portfolio_returns.std(),
        "Retorno anualizado": portfolio_returns.mean() * 252,
        "Volatilidade anualizada": portfolio_returns.std() * np.sqrt(252),
        "Pior dia": portfolio_returns.min(),
        "Melhor dia": portfolio_returns.max(),
    }
    return stats

def efficient_frontier(returns, n_portfolios=10000):
    n_assets = len(returns.columns)
    results = {
        "weights": [],
        "returns": [],
        "volatility": [],
        "sharpe": []
    }

    np.random.seed(42)

    for _ in range(n_portfolios):
        w = np.random.random(n_assets)
        w = w / w.sum()

        port_return = np.dot(w, returns.mean()) * 252
        port_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        sharpe = port_return / port_vol

        results["weights"].append(w)
        results["returns"].append(port_return)
        results["volatility"].append(port_vol)
        results["sharpe"].append(sharpe)

    results_df = pd.DataFrame({
        "Return": results["returns"],
        "Volatility": results["volatility"],
        "Sharpe": results["sharpe"],
    })
    results_df["Weights"] = results["weights"]

    min_var_idx = results_df["Volatility"].idxmin()
    min_var = results_df.loc[min_var_idx]

    max_sharpe_idx = results_df["Sharpe"].idxmax()
    max_sharpe = results_df.loc[max_sharpe_idx]

    return results_df, min_var, max_sharpe


if __name__ == "__main__":
    from data import download_data, calculate_returns

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    weights = [0.4, 0.3, 0.3]

    prices = download_data(tickers, start='2020-01-01', end='2024-12-31')
    returns = calculate_returns(prices)
    
    portfolio_returns = create_portfolio(returns, weights)

    print("RETORNOS DO PORTFOLIO")
    print(portfolio_returns.tail())

    print("\nESTATÍSTICAS DO PORTFOLIO")
    stats = portfolio_stats(portfolio_returns)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
        
        
    ef_df, min_var, max_sharpe = efficient_frontier(returns)

    print("\nEFFICIENT FRONTIER")
    print(f"Portfolios simulados: {len(ef_df)}")
    print(f"\nMínima Variância:")
    print(f"  Retorno:     {min_var['Return']*100:.2f}%")
    print(f"  Volatilidade:{min_var['Volatility']*100:.2f}%")
    print(f"  Sharpe:      {min_var['Sharpe']:.3f}")
    print(f"\nMáximo Sharpe:")
    print(f"  Retorno:     {max_sharpe['Return']*100:.2f}%")
    print(f"  Volatilidade:{max_sharpe['Volatility']*100:.2f}%")
    print(f"  Sharpe:      {max_sharpe['Sharpe']:.3f}")