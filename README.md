# Level 2 (Intermediate) — Codveda Data Analyst Internship

## Task 1: Regression Analysis
Simple linear regression predicting median house value (MEDV) from average 
room count (RM), using the Boston Housing dataset (506 records).

**Key result:** R² = 0.37, RMSE ≈ $6,790. Each additional room is associated 
with a ~$9,350 increase in predicted house value.

Run: `python task1_regression_analysis.py`

## Task 2: Time Series Analysis
Trend and seasonality analysis of AAPL stock closing price (2014-2017, 
1,007 trading days), with additive decomposition and moving average smoothing.

**Key result:** Clear long-term uptrend across the period, with minor seasonal 
signal relative to trend and daily noise.

Run: `python task2_time_series_analysis.py`

## Files
- `task1_regression_analysis.py` / `task2_time_series_analysis.py` — source code
- `4__house_Prediction_Data_Set.csv` / `2__Stock_Prices_Data_Set.csv` — datasets
- `Codveda_Level2_Reordered.pptx` — presentation deck
- `Guide_Code_Task1_Task2.pdf` — line-by-line code walkthrough

**Tools:** Python, pandas, scikit-learn, statsmodels, matplotlib
