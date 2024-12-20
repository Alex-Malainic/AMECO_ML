# Inflation Forecasting with Machine Learning

This repository demonstrates a comprehensive analysis of inflation forecasting using traditional and machine learning models, with a focus on comparing model performances. The project combines statistical techniques and economic insights to deliver meaningful results and showcase advanced forecasting methodologies.

---

## 📌 **Project Overview**

Inflation forecasting is critical for policymakers, businesses, and investors to make informed decisions. This project leverages:

- **Traditional Time-Series Models**: AR(12) and Random Walk.
- **Machine Learning**: Random Forest and LSTM.
- **Economic Interpretability**: Analysis of influential predictors for inflation.

The goal is to compare these methods, identify the best-performing model, and draw actionable insights.

---

## 🚀 **Key Features**

1. **Data Preprocessing**:
   - Standardized feature scaling.
   - Stationarity checks for time-series modeling.
   - Creation of lagged features for machine learning models.

2. **Model Benchmarks**:
   - **AR(12)**: A traditional autoregressive model with lag 12.
   - **Random Walk**: A naive forecasting benchmark.
   - **Random Forest**: Capturing nonlinear relationships with feature importance analysis.
   - **LSTM**: Long-Short Term Memory neural networks for sequence learning.

3. **Evaluation**:
   - Models are evaluated using **Root Mean Squared Error (RMSE)**.
   - Statistical tests, including the **Diebold-Mariano test**, to compare model performances.

4. **Economic Insights**:
   - Feature importance analysis for economic interpretability.
   - Scenario-based analysis to assess model robustness during stable and volatile periods.