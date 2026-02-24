import yfinance as yf
import pandas as pd

def download_data(tickers, start, end):
    """
    Descarrega preços históricos do Yahoo Finance.
    tickers: lista de ações ex: ['AAPL', 'MSFT', 'GOOGL']
    start/end: datas no formato 'YYYY-MM-DD'
    """
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = data ['Close']
    return prices

def calculate_returns(prices):
    """
    Calcula os retornos diários.
    Formula: (preco_hoje - preco_ontem) / preco_ontem
    """
    returns = prices.pct_change().dropna()
    return returns

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = download_data(tickers, start='2020-01-01', end='2024-12-31')
    returns = calculate_returns(prices)
    
    print("=== PREÇOS (últimas 5 linhas) ===")
    print(prices.tail())
    print("\n=== RETORNOS DIÁRIOS (últimas 5 linhas) ===")
    print(returns.tail())
    
    