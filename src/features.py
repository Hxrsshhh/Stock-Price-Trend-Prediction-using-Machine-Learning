import numpy as np
import pandas as pd


# =================================================
# CREATE FEATURES
# =================================================

def create_features(data):
    """
    Create historical technical indicators,
    lagged features, and the prediction target.

    Target:
        1 = Next day's closing price is higher
        0 = Next day's closing price is lower or equal
    """

    data = data.copy()

    # ==========================================
    # 1. Daily Return
    # ==========================================

    data["Return"] = (
        data["Close"].pct_change()
    )

    # ==========================================
    # 2. Simple Moving Averages
    # ==========================================

    data["SMA_10"] = (
        data["Close"]
        .rolling(window=10)
        .mean()
    )

    data["SMA_20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["SMA_50"] = (
        data["Close"]
        .rolling(window=50)
        .mean()
    )

    # ==========================================
    # 3. Exponential Moving Averages
    # ==========================================

    data["EMA_10"] = (
        data["Close"]
        .ewm(
            span=10,
            adjust=False
        )
        .mean()
    )

    data["EMA_20"] = (
        data["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # ==========================================
    # 4. Momentum
    # ==========================================

    data["Momentum_10"] = (
        data["Close"]
        / data["Close"].shift(10)
        - 1
    )

    # ==========================================
    # 5. Volatility
    # ==========================================

    data["Volatility_10"] = (
        data["Return"]
        .rolling(window=10)
        .std()
    )

    # ==========================================
    # 6. Volume Change
    # ==========================================

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    # ==========================================
    # 7. RSI
    # ==========================================

    data["RSI"] = calculate_rsi(
        data["Close"]
    )

    # ==========================================
    # 8. MACD
    # ==========================================

    data["MACD"] = calculate_macd(
        data["Close"]
    )

    # ==========================================
    # 9. LAGGED FEATURES
    # ==========================================

    data["Return_Lag1"] = (
        data["Return"].shift(1)
    )

    data["Return_Lag2"] = (
        data["Return"].shift(2)
    )

    data["Return_Lag3"] = (
        data["Return"].shift(3)
    )

    data["Return_Lag5"] = (
        data["Return"].shift(5)
    )

    data["Return_Lag10"] = (
        data["Return"].shift(10)
    )

    data["Volume_Change_Lag1"] = (
        data["Volume_Change"].shift(1)
    )

    data["Volume_Change_Lag2"] = (
        data["Volume_Change"].shift(2)
    )

    data["RSI_Lag1"] = (
        data["RSI"].shift(1)
    )

    data["MACD_Lag1"] = (
        data["MACD"].shift(1)
    )

    # ==========================================
    # 10. TARGET VARIABLE
    # ==========================================

    future_close = (
        data["Close"].shift(-1)
    )

    data["Target"] = np.where(
        future_close.notna(),
        (
            future_close > data["Close"]
        ).astype(int),
        np.nan
    )

    # ==========================================
    # 11. CLEAN DATASET
    # ==========================================

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    data = data.reset_index(
        drop=True
    )

    return data


# =================================================
# CALCULATE RSI
# =================================================

def calculate_rsi(
    close,
    period=14
):
    """
    Calculate Relative Strength Index (RSI).
    """

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .rolling(window=period)
        .mean()
    )

    average_loss = (
        loss
        .rolling(window=period)
        .mean()
    )

    rs = (
        average_gain
        / average_loss
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    return rsi


# =================================================
# CALCULATE MACD
# =================================================

def calculate_macd(close):
    """
    Calculate MACD indicator.

    MACD = EMA(12) - EMA(26)
    """

    ema_12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema_26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    macd = (
        ema_12 - ema_26
    )

    return macd


# =================================================
# FEATURE SET 1
# BASELINE FEATURES
# =================================================

BASELINE_FEATURES = [
    "Return",
    "Volume_Change"
]


# =================================================
# FEATURE SET 2
# HISTORICAL INDICATOR FEATURES
# =================================================

INDICATOR_FEATURES = [
    "Return",

    "SMA_10",
    "SMA_20",
    "SMA_50",

    "EMA_10",
    "EMA_20",

    "Momentum_10",
    "Volatility_10",

    "Volume_Change",

    "RSI",
    "MACD"
]


# =================================================
# FEATURE SET 3
# INDICATORS + LAGGED FEATURES
# =================================================

LAGGED_FEATURES = [
    "Return",

    "SMA_10",
    "SMA_20",
    "SMA_50",

    "EMA_10",
    "EMA_20",

    "Momentum_10",
    "Volatility_10",

    "Volume_Change",

    "RSI",
    "MACD",

    # Return history
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Return_Lag10",

    # Volume history
    "Volume_Change_Lag1",
    "Volume_Change_Lag2",

    # Indicator history
    "RSI_Lag1",
    "MACD_Lag1"
]
