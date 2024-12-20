import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

def lstm_example():
    """
    This function demonstrates how to use LSTM for time series forecasting with cross-validation.
    Returns the average RMSE from cross-validation.

    """
    # Read the train and test data
    train_df = pd.read_csv("train_df.csv", index_col=0)
    test_df = pd.read_csv("test_df.csv", index_col=0)

    # Extract the target (CPI)
    train_target = train_df['CPIAUCSL']
    test_target = test_df['CPIAUCSL']

    # Drop the target from the datasets
    train_df.drop(columns=['CPIAUCSL'], inplace=True)
    test_df.drop(columns=['CPIAUCSL'], inplace=True)

    # Standardize the data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Reshaping the input data to 3D (samples, time_steps, features)
    train_reshaped = np.reshape(train_scaled, (train_scaled.shape[0], 1, train_scaled.shape[1]))
    test_reshaped = np.reshape(test_scaled, (test_scaled.shape[0], 1, test_scaled.shape[1]))

    # Define the LSTM model
    def create_model():
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(1, train_scaled.shape[1])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for train_index, val_index in tscv.split(train_reshaped):
        X_train, X_val = train_reshaped[train_index], train_reshaped[val_index]
        y_train, y_val = train_target.iloc[train_index], train_target.iloc[val_index]

        model = create_model()
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        rmse_scores.append(rmse)

    avg_rmse = np.mean(rmse_scores)

    # Train the model on the entire training set and evaluate on the test set
    final_model = create_model()
    final_model.fit(train_reshaped, train_target, epochs=100, batch_size=32, verbose=0)
    test_predictions = final_model.predict(test_reshaped)
    test_rmse = np.sqrt(mean_squared_error(test_target, test_predictions))

    print(f"Cross-Validation RMSE: {avg_rmse}")
    print(f"Test RMSE: {test_rmse}")

    return test_rmse


def plot_predictions(index, target, predictions):
    """
    This function plots the predictions against the actual target variable.
    :param index: the time index
    :param target: the actual target variable
    :param predictions: the predicted target variable

    """
    # Plotting
    plt.figure(figsize=(12, 6))  # Set the figure size
    plt.plot(index, target, label='Actual', color='blue', linewidth=2)
    plt.plot(index, predictions, label='Predictions', color='red', linestyle='--', linewidth=2)

    # Adding titles and labels
    plt.title('Predictions vs Actual Target', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Target Variable (e.g., Inflation)', fontsize=12)
    plt.legend(loc='best')

    # Adding grid for better readability
    plt.grid(True)

    # Rotate x-ticks for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()  # Ensures everything fits without overlap
    plt.show()

