import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lstm_example import lstm_example
from sklearn.model_selection import TimeSeriesSplit
from models import rf_hyperparameter_tuning

# --- Step 1: Load and Preprocess Data ---
# Read train and test data
train_df = pd.read_csv("train_df.csv", index_col=0)
test_df = pd.read_csv("test_df.csv", index_col=0)

# Extract target (CPI) and drop from features
train_target = train_df['CPIAUCSL']
test_target = test_df['CPIAUCSL']
train_df.drop(columns=['CPIAUCSL'], inplace=True)
test_df.drop(columns=['CPIAUCSL'], inplace=True)

# Standardize the feature data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)


# --- Step 2: Define Helper Function for RMSE ---
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Step 3: Benchmark Models ---
# Benchmark 1: AR(p) Model
lag_p = 12  # You can adjust or determine based on PACF
model_ar = AutoReg(train_target, lags=lag_p).fit()
predictions_ar = model_ar.predict(start=len(train_target), end=len(train_target) + len(test_target) - 1)
predictions_ar.index = test_target.index  # Align indices for comparison
rmse_ar = compute_rmse(test_target, predictions_ar)

# Benchmark 2: Random Walk Model
# Naive forecasting: Predict next value as the last observed value
predictions_rw = test_target.shift(1).fillna(method='bfill')
predictions_rw.index = test_target.index  # Align indices
rmse_rw = compute_rmse(test_target, predictions_rw)

# --- Step 4: Random Forest Model ---
rmse_rf = rf_hyperparameter_tuning(train_scaled, train_target, test_scaled, test_target)

#LSTM Model
#rmse_lstm = lstm_example()

# --- Step 5: Compare Results ---
print("\nBenchmark Model Comparison:")
print(f"AR({lag_p}) RMSE: {rmse_ar}")
print(f"Random Walk RMSE: {rmse_rw}")
print(f"Random Forest RMSE: {rmse_rf}")
#print(f"LSTM RMSE: {rmse_lstm}")


# The following functions will implement the following:
# 1. A Random Forest model with cross-validation for time series forecasting using cross-val-score with RMSE and timeseriessplit
# 2. A function to perform hyperparameter tuning for the Random Forest model using GridSearchCV
# 3. A function to plot the feature importance of the Random Forest model
# 4. A function to calculate the RMSE of the LSTM model with cross-validation
# 5. A function to plot the learning curve of the LSTM model
# 6. A function to plot the predictions of the LSTM model
# 7. A function to calculate the RMSE of the AR model with cross-validation
# 8. A function to plot the predictions of the AR model
# 9. A function to calculate the RMSE of the Random Walk model
# 10. A function to plot the predictions of the Random Walk model
# 11. A function to calculate the RMSE of the Random Forest model with cross-validation
# 12. A function to plot the predictions of the Random Forest model
# 13. A function to gather all RMSE values and plot them for comparison


