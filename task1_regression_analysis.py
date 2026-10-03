"""
Task 1: Regression Analysis
Objective: Perform a simple linear regression analysis to predict one
variable based on another.
Tools: Python, scikit-learn, pandas

Dataset: Boston Housing dataset (506 rows, 14 columns, no header row).
We predict MEDV (median house value, in $1000s) from RM (average
number of rooms per dwelling).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------
FILE_PATH = "4__house_Prediction_Data_Set.csv"

column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]

df = pd.read_csv(FILE_PATH, sep=r"\s+", header=None, names=column_names)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------------
# 2. Select variables for simple linear regression
#    Predictor (X): RM  -> average number of rooms per dwelling
#    Target (y):    MEDV -> median value of owner-occupied homes ($1000s)
# ---------------------------------------------------------------------
X = df[["RM"]]
y = df["MEDV"]

# ---------------------------------------------------------------------
# 3. Split the dataset into training and testing sets
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows")

# ---------------------------------------------------------------------
# 4. Fit a linear regression model
# ---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 5. Interpret the coefficients
# ---------------------------------------------------------------------
intercept = model.intercept_
coefficient = model.coef_[0]

print("\n--- Model Coefficients ---")
print(f"Intercept (b0): {intercept:.4f}")
print(f"Coefficient for RM (b1): {coefficient:.4f}")
print(f"\nRegression equation: MEDV = {intercept:.4f} + {coefficient:.4f} * RM")
print(
    f"\nInterpretation: each additional room (RM) is associated with an "
    f"increase of approximately {coefficient:.2f} (in $1000s) in the "
    f"predicted median house value, holding all else constant."
)

# ---------------------------------------------------------------------
# 6. Evaluate the model on the test set
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Model Evaluation on Test Set ---")
print(f"R-squared (R2):        {r2:.4f}")
print(f"Mean Squared Error:    {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(
    f"\nInterpretation: the model explains about {r2*100:.1f}% of the "
    f"variance in house prices using RM alone. On average, predictions "
    f"deviate from the actual price by about {rmse:.2f} (in $1000s)."
)

# ---------------------------------------------------------------------
# 7. (Optional) Preview predictions vs actual values
# ---------------------------------------------------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Residual": y_test.values - y_pred,
})
print("\nSample of predictions vs actual values:")
print(results.head(10).round(2))

# ---------------------------------------------------------------------
# 8. Visualizations
# ---------------------------------------------------------------------

# 8a. Scatter plot with regression line
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, color="steelblue", alpha=0.6, label="Actual data (test set)")

# Sort X_test for a clean line
X_line = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color="crimson", linewidth=2, label="Regression line")

plt.xlabel("RM (average number of rooms per dwelling)")
plt.ylabel("MEDV (median house value, $1000s)")
plt.title("Simple Linear Regression: RM vs MEDV")
plt.legend()
plt.tight_layout()
plt.savefig("regression_line_plot.png", dpi=150)
plt.show()

# 8b. Residuals plot
residuals = y_test.values - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, color="darkorange", alpha=0.6)
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Predicted MEDV")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals Plot")
plt.tight_layout()
plt.savefig("residuals_plot.png", dpi=150)
plt.show()

print("\nPlots saved as 'regression_line_plot.png' and 'residuals_plot.png'.")
