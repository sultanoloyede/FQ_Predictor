import os
import pandas as pd
import yfinance as yf
import datetime

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

    # Path to the CSV file with the top 20 tech tickers
    csv_path = os.path.join(data_dir, 'top_20_tech_stocks.csv')
    df_tickers = pd.read_csv(csv_path)

    # Create the 'stocks' folder if it doesn't exist
    stocks_dir = os.path.join(data_dir, 'stocks')
    os.makedirs(stocks_dir, exist_ok=True)

    for ticker in df_tickers['Ticker']:
        print(f"Processing {ticker}...")
        try:
            # 1) Download 5 years of weekly-like data (interval="5d")
            df_weekly = yf.download(ticker, period="5y", interval="5d")
            if df_weekly.empty:
                print(f"No data returned for {ticker}. Skipping.")
                continue

            # 2) Reset index so 'Date' becomes a normal column
            df_weekly.reset_index(inplace=True)

            # 3) Create Year, Month, Day columns
            df_weekly['Year'] = df_weekly['Date'].dt.year
            df_weekly['Month'] = df_weekly['Date'].dt.month
            df_weekly['Day'] = df_weekly['Date'].dt.day

            # 4) Remove 'Volume' column if it exists
            if 'Volume' in df_weekly.columns:
                df_weekly.drop(columns=['Volume'], inplace=True, errors='ignore')

            # 5) Add a Ticker column (repeated for each row)
            df_weekly['Ticker'] = ticker

            # 6) Create columns for 1%, 2%, 3% increase relative to previous row's Close
            #    We'll use pct_change() for the Close column.
            pct_change = df_weekly['Close'].pct_change()  # e.g., 0.01 = +1%
            
            # Fill the first row (which is NaN for pct_change) with 0
            pct_change = pct_change.fillna(0)

            df_weekly['Above1Pct'] = (pct_change >= 0.01).astype(int)
            df_weekly['Above2Pct'] = (pct_change >= 0.02).astype(int)
            df_weekly['Above3Pct'] = (pct_change >= 0.03).astype(int)

            # 7) Save final DataFrame to CSV
            out_path = os.path.join(stocks_dir, f"{ticker}.csv")
            df_weekly.to_csv(out_path, index=False)
            print(f"Weekly data for {ticker} saved to {out_path}\n")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
