import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker="SPY", start_date="2005-01-01", end_date="2025-01-01"):
    """
    Connects to Yahoo Finance and downloads daily OHLCV data.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df = df[['Close', 'Volume']].copy()
    df.dropna(inplace=True)
    return df

def calculate_hurst_rs(price_series):
    """
    Calculates the Hurst Exponent using R/S Analysis for a 1D array of prices.
    This function will be applied to a rolling window.
    """
    # 1. Convert prices to logarithmic returns
    returns = np.diff(np.log(price_series))
    
    # We need a minimum number of data points to do meaningful chunking
    if len(returns) < 20:
        return np.nan

    # Define the sizes of the chunks (lags) we will divide the window into
    max_lag = len(returns) // 2
    lags = range(2, max_lag)
    rs_values = []

    for lag in lags:
        # Split returns into non-overlapping chunks of size 'lag'
        chunks = [returns[i:i+lag] for i in range(0, len(returns), lag) if len(returns[i:i+lag]) == lag]
        
        rs_chunk = []
        for chunk in chunks:
            # R/S Calculation step-by-step
            mean = np.mean(chunk)
            centered = chunk - mean
            cum_sum = np.cumsum(centered)
            R = np.max(cum_sum) - np.min(cum_sum) # Range
            S = np.std(chunk)                     # Standard Deviation
            
            if S == 0: # Avoid division by zero
                continue
            rs_chunk.append(R / S)
            
        # Store the average R/S value for this specific lag size
        if len(rs_chunk) > 0:
            rs_values.append(np.mean(rs_chunk))
        else:
            rs_values.append(np.nan)

    # Clean NaNs before regression
    valid_idx = ~np.isnan(rs_values)
    if sum(valid_idx) < 3:
        return np.nan

    valid_lags = np.array(list(lags))[valid_idx]
    valid_rs = np.array(rs_values)[valid_idx]

    # 2. Linear regression on Log-Log scale (y = mx + b)
    # np.polyfit returns [slope, intercept]. The slope is the Hurst Exponent.
    poly = np.polyfit(np.log(valid_lags), np.log(valid_rs), 1)
    hurst_exponent = poly[0]
    
    return hurst_exponent

def calculate_all_features(df):
    """
    Transforms raw price data into stationary features, including Hurst.
    """
    print("Calculating fast basic features...")
    df['Log_Ret_1d'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    df['Vol_20d'] = df['Log_Ret_1d'].rolling(window=20).std()
    df['Mom_10d'] = df['Log_Ret_1d'].rolling(window=10).sum()
    df['Rel_Vol'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    print("Calculating rolling Hurst Exponent (this might take 10-20 seconds)...")
    # rolling().apply() takes our custom function and applies it to every 60-day window.
    # We pass raw='False' because our function expects a pandas Series.
    df['Hurst_60d'] = df['Close'].rolling(window=60).apply(calculate_hurst_rs, raw=True)
    
    # Target Variable (y): 1 if tomorrow's return is positive, 0 if negative or flat
    # We shift(-1) to look at TOMORROW's return.
    df['Target'] = (df['Log_Ret_1d'].shift(-1) > 0).astype(int)
    
    # Drop rows with NaNs (the first 60 days will be NaN because of the rolling window)
    # Also drops the very last row because it has no 'Target' (we don't know tomorrow yet)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # 1. Fetch
    raw_data = fetch_data(ticker="SPY", start_date="2005-01-01", end_date="2025-01-01")
    
    # 2. Cook
    final_data = calculate_all_features(raw_data)
    
    print("\n--- FINAL FEATURE MATRIX ---")
    print(f"Total rows ready for Machine Learning: {len(final_data)}")
    # Set pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    print(final_data.tail(10))