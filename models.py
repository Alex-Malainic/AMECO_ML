from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np


def rf_hyperparameter_tuning(X_train, y_train, X_test, y_test, n_splits=5):
    """
    Random Forest model hyperparameter tuning using GridSearchCV.
    Returns the final RMSE on the test data.

    :param X_train: scaled feature train dataset
    :param y_train: train target
    :param X_test: scaled feature test dataset
    :param y_test: test target
    :param n_splits: number of splits for cross-validation

    """
    # Define RMSE as the scoring metric
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize Random Forest model
    rf = RandomForestRegressor(random_state=42)

    # Hyperparameter Tuning with GridSearch
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring=rmse_scorer,
        verbose=0,
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit GridSearchCV on the training data and get the best parameters
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Final Model Training with Best Hyperparameters
    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    # Evaluate on Test Data
    test_predictions = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    return test_rmse