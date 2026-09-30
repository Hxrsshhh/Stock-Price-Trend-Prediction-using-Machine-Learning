import os
import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker="RELIANCE.NS",
    start="2015-01-01",
    end="2026-01-01"
):
    """
    Download historical stock data from Yahoo Finance.
    """

    print(f"Downloading data for {ticker}...")

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("No stock data was downloaded.")

    # Handle MultiIndex columns returned by some yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Keep only useful columns
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data[required_columns]

    # Remove missing values
    data = data.dropna()

    return data


def save_stock_data(data, filepath="data/stock_data.csv"):
    """
    Save stock data to CSV.
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data.to_csv(filepath, index=False)

    print(f"Data saved to {filepath}")


def load_stock_data(filepath="data/stock_data.csv"):
    """
    Load previously saved stock data.
    """

    data = pd.read_csv(filepath)

    data["Date"] = pd.to_datetime(data["Date"])

    return data