#https://fg-research.com/blog/general/posts/fred-md-overview.html

import os
import pandas as pd
import numpy as np
import statsmodels.imputation.mice as mice
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

# Prepare seed for reproducibility
seed = 1042

def transform_series(x: pd.Series, tcode:int):
    """
    Transform the time series.

    :param x: pandas Series to apply the transformation on
    :param tcode: transformation code (difference, log, etc)

    """

    if tcode == 1:
        return x
    elif tcode == 2: # first order difference
        return x.diff()
    elif tcode == 3: # second order difference
        return x.diff().diff()
    elif tcode == 4: # logarithm
        return np.log(x)
    elif tcode == 5: # first order logarithmic difference
        return np.log(x).diff()
    elif tcode == 6: # second order logarithmic difference
        return np.log(x).diff().diff()
    elif tcode == 7: # percentage change
        return x.pct_change()
    else:
        raise ValueError(f"unknown `tcode` {tcode}")

def download_data(year: int, month: int, transform: bool=True):
    """
    Download and (optionally) transform the time series.

    :param year: the year of the dataset vintage.
    :param month: the month of the dataset vintage
    :param transform: whether the time series should be transformed or not.

    """

    # Get the dataset URL
    file = f"https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{year}-{format(month, '02d')}.csv"

    # Read the dataset
    df = pd.read_csv(file, skiprows=[1], index_col=0)
    df.columns = [c.upper() for c in df.columns]

    # Process the dates
    df = df.loc[pd.notna(df.index), :]
    df.index = pd.date_range(start="1959-01-01", freq="MS", periods=len(df))

    # Transform the columns, if necessary
    if transform:

        # Get the transformation codes
        tcodes = pd.read_csv(file, nrows=1, index_col=0)
        tcodes.columns = [c.upper() for c in tcodes.columns]

        # Transform CPI and aggregate PCE in log differences instead
        tcodes['CPIAUCSL'] = 5
        tcodes['PCEPI'] = 5

        # Transform the time series
        data = df.apply(lambda x: transform_series(x, tcodes[x.name].item()))

    return data

# Step 1 : download and transform the data
dataset = download_data(year=2019, month=12, transform=True)

# Step 2: fill missing data with the EM-based IterativeImputer algorithm
imputer = IterativeImputer(max_iter = 20, random_state= seed)
imputed_data = imputer.fit_transform(dataset)

# Reconstruct the filled dataframe
imputed_df = pd.DataFrame(imputed_data, columns=dataset.columns, index=dataset.index)

# Split data in train-test, 70-30%
train, test = train_test_split(imputed_df, test_size=0.3, shuffle=False)

# Save both dataframes
train.to_csv("train_df.csv")
test.to_csv("test_df.csv")