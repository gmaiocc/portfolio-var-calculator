import yfinance as yf
import pandas as pd

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = data ['Close']
    return prices

def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = download_data(tickers, start='2020-01-01', end='2024-12-31')
    returns = calculate_returns(prices)
    
    print("PREÇOS CLOSE")
    print(prices.tail())
    print("\nRETORNOS DIÁRIOS")
    print(returns.tail())
    
    