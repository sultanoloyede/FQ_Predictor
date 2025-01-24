import os
import pandas as pd
import yfinance as yf

def main():
    # Define the ticker and fetch parameters
    ticker = "AAPL"
    period = "2y"       # valid examples: "1d", "5d", "1mo", "6mo", "1y", "2y", "5y"
    interval = "1h"     # valid intervals: "1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"

    print(f"Fetching {ticker} data for the last {period} at interval {interval}...\n")
    
    # Download the data
    stock_data = yf.download(ticker, period=period, interval=interval)

    # Display a brief summary
    print("Data Downloaded:")
    print(stock_data.head(), "\n")   # Show first 5 rows
    print(stock_data.tail(), "\n")   # Show last 5 rows
    print(f"Total rows fetched: {len(stock_data)}\n")

    # Optionally, save the data to CSV in the same directory as the script
    csv_filename = f"{ticker}_{period}_{interval}.csv"
    stock_data.to_csv(csv_filename)
    print(f"Data for {ticker} saved to {csv_filename}")

if __name__ == "__main__":
    main()
