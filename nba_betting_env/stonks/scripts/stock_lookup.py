import os
import pandas as pd
import yfinance as yf
import requests

def get_sp500_tickers():
    """Scrape the S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)
    df = tables[0]
    
    tickers = df[df['GICS Sector'] == 'Information Technology']['Symbol'].tolist()
    
    return tickers

def fetch_data(tickers):
    market_cap = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cap = stock.info.get("marketCap", None)
            if cap is not None:
                market_cap[ticker] = cap
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    sorted_market_cap = sorted(market_cap.items(), key=lambda x: x[1], reverse=True)
    top_stocks = sorted_market_cap[:40]
    return top_stocks

def main():
    print("Fetching stocks...")
    tickers = get_sp500_tickers()

    print("Fetching data...")
    top_volume_stocks = fetch_data(tickers)

    print("\nTop 40 Highest Stocks by Market Cap:")
    for stock, cap in top_volume_stocks:
        print(f"{stock}: ${cap:,.2f}")

    # Create path to the data folder outside (one level above) the current script folder
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)

    # Convert only the ticker symbols to a DataFrame
    df = pd.DataFrame([stock for stock, cap in top_volume_stocks], columns=['Ticker'])

    # Save to CSV in the data folder
    csv_path = os.path.join(data_dir, 'top_40_tech_stocks.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV of top 40 tickers saved to: {csv_path}")

if __name__ == "__main__":
    main()
