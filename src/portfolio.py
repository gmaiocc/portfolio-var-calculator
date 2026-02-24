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