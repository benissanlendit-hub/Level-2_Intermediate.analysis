"""
Task 2: Time Series Analysis
Objective: Analyze a time-series dataset (stock prices) to detect trends
and seasonality.
Tools: Python, pandas, matplotlib, statsmodels

Dataset: Daily stock prices for 505 companies (2014-01-02 to 2017-12-29).
We analyze a single stock (AAPL by default) using its closing price.
Change STOCK_SYMBOL below to analyze a different company.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------
FILE_PATH = "2__Stock_Prices_Data_Set.csv"
STOCK_SYMBOL = "AAPL"          # change to any symbol present in the dataset
SEASONAL_PERIOD = 252          # ~252 trading days per year (annual seasonality)

# ---------------------------------------------------------------------
# 1. Load and prepare the dataset
# ---------------------------------------------------------------------
df = pd.read_csv(FILE_PATH)
print("Full dataset shape:", df.shape)
print("Number of unique stock symbols:", df["symbol"].nunique())

# Filter for a single stock symbol
stock = df[df["symbol"] == STOCK_SYMBOL].copy()
stock["date"] = pd.to_datetime(stock["date"])
stock = stock.sort_values("date").reset_index(drop=True)
stock = stock.set_index("date")

print(f"\n{STOCK_SYMBOL} data shape: {stock.shape}")
print(f"Date range: {stock.index.min().date()} to {stock.index.max().date()}")
print("Missing values:\n", stock.isnull().sum())

# Use the closing price as the series to analyze
series = stock["close"]

# ---------------------------------------------------------------------
# 2. Plot the raw time series and identify patterns
# ---------------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(series.index, series.values, color="steelblue", linewidth=1.2)
plt.title(f"{STOCK_SYMBOL} Closing Price (2014-2017)")
plt.xlabel("Date")
plt.ylabel("Closing Price ($)")
plt.tight_layout()
plt.savefig("01_raw_time_series.png", dpi=150)
plt.show()

print(
    "\nVisual inspection: look for an overall upward/downward trend, "
    "periods of high volatility, and any repeating yearly patterns."
)

# ---------------------------------------------------------------------
# 3. Decompose the series into trend, seasonality, and residuals
# ---------------------------------------------------------------------
# statsmodels needs a fixed-frequency index; business-day frequency ('B')
# matches stock market trading days. Any gaps (holidays) are filled by
# interpolation so the decomposition has no missing values to break on.
series_daily = series.asfreq("B")
series_daily = series_daily.interpolate(method="linear")

decomposition = seasonal_decompose(
    series_daily, model="additive", period=SEASONAL_PERIOD
)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
fig.suptitle(f"{STOCK_SYMBOL} Time Series Decomposition (Additive)", y=1.02)
plt.tight_layout()
plt.savefig("02_decomposition.png", dpi=150)
plt.show()

print("\n--- Decomposition summary ---")
print("Trend component (non-null values):", decomposition.trend.notna().sum())
print("Seasonal component range:",
      round(decomposition.seasonal.min(), 3), "to",
      round(decomposition.seasonal.max(), 3))
print("Residual component std deviation:", round(decomposition.resid.std(), 3))
print(
    "\nInterpretation: the trend component captures the long-term "
    "direction of the stock price. The seasonal component shows a "
    "small repeating annual pattern (its amplitude is usually tiny "
    "compared to price levels for stocks, since prices behave closer "
    "to a random walk than a seasonal process). The residuals capture "
    "day-to-day noise not explained by trend or seasonality."
)

# ---------------------------------------------------------------------
# 4. Moving average smoothing
# ---------------------------------------------------------------------
ma_short = series.rolling(window=30).mean()   # ~1 trading month
ma_long = series.rolling(window=90).mean()    # ~1 trading quarter

plt.figure(figsize=(12, 5))
plt.plot(series.index, series.values, color="lightgray", linewidth=1, label="Daily Close Price")
plt.plot(ma_short.index, ma_short.values, color="darkorange", linewidth=1.6, label="30-Day Moving Average")
plt.plot(ma_long.index, ma_long.values, color="crimson", linewidth=1.6, label="90-Day Moving Average")
plt.title(f"{STOCK_SYMBOL} Closing Price with Moving Average Smoothing")
plt.xlabel("Date")
plt.ylabel("Closing Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("03_moving_averages.png", dpi=150)
plt.show()

print(
    "\nInterpretation: the 30-day moving average smooths out short-term "
    "noise while still tracking medium-term swings; the 90-day moving "
    "average is smoother still and highlights the underlying trend, "
    "filtering out most daily volatility."
)

print(
    "\nPlots saved: 01_raw_time_series.png, 02_decomposition.png, "
    "03_moving_averages.png"
)
