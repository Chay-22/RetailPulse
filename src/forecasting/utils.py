"""
utils.py
----------------------------------------
Utility functions for RetailPulse Forecasting
Supports:
- Prophet preprocessing
- LSTM preprocessing
- Scaling
- Sequence generation
- Train/Test split
- Model persistence
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

SCALER_PATH = "models/scaler.pkl"


# ---------------------------------------------------
# Aggregate transactional data into daily sales
# ---------------------------------------------------

def aggregate_daily_sales(
    df,
    date_col="InvoiceDate",
    value_col="TotalPrice"
):
    """
    Convert transactional data into daily sales.
    """

    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col],errors="coerce")
    data = data.dropna(
        subset=[date_col]
    )

    daily = (
        data.groupby(data[date_col].dt.date)[value_col]
        .sum()
        .reset_index()
    )

    daily.columns = ["ds", "y"]

    daily["ds"] = pd.to_datetime(daily["ds"])

    daily = daily.sort_values("ds")

    return daily


# ---------------------------------------------------
# Fill Missing Dates
# ---------------------------------------------------

def fill_missing_dates(df):

    full_dates = pd.date_range(
        df["ds"].min(),
        df["ds"].max(),
        freq="D"
    )

    df = (
        df.set_index("ds")
        .reindex(full_dates)
        .fillna(0)
        .rename_axis("ds")
        .reset_index()
    )

    df.columns = ["ds", "y"]

    return df


# ---------------------------------------------------
# Scaling
# ---------------------------------------------------

def scale_series(
    series,
    retrain=False,
    scaler_path=SCALER_PATH,
):
    """
    Scale a series using a persisted scaler.

    Parameters
    ----------
    series : pandas.Series

    retrain : bool
        If True, fit a new scaler.

    scaler_path : str

    Returns
    -------
    scaler
    scaled
    """

    if len(series) == 0:

        raise ValueError(
            "Series is empty."
        )
    series = series.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    if (
        not retrain
        and
        os.path.exists(scaler_path)
    ):

        scaler = load_object(
            scaler_path
        )

        if scaler is not None:

            scaled = scaler.transform(
                series.values.reshape(-1, 1)
            )

            return scaler, scaled

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        series.values.reshape(-1, 1)
    )

    save_object(
        scaler,
        scaler_path,
    )

    return scaler, scaled

# ---------------------------------------------------
# Create LSTM Sequences
# ---------------------------------------------------

def create_sequences(
    data,
    sequence_length=30
):

    X = []
    y = []

    for i in range(len(data) - sequence_length):

        X.append(
            data[i:i + sequence_length]
        )

        y.append(
            data[i + sequence_length]
        )

    return np.array(X), np.array(y)


# ---------------------------------------------------
# Train/Test Split
# ---------------------------------------------------

def train_test_split_sequences(
    X,
    y,
    test_ratio=0.2
):

    split = int(len(X) * (1 - test_ratio))

    return (
        X[:split],
        X[split:],
        y[:split],
        y[split:]
    )


# ---------------------------------------------------
# Save Object
# ---------------------------------------------------

def save_object(
    obj,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    joblib.dump(obj, path)


# ---------------------------------------------------
# Load Object
# ---------------------------------------------------

def load_object(path):

    if not os.path.exists(path):

        return None

    try:

        return joblib.load(path)

    except Exception:

        return None


# ---------------------------------------------------
# Prepare Complete Dataset
# ---------------------------------------------------

def prepare_dataset(df):

    required = {

        "InvoiceDate",

        "TotalPrice"

    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(

            f"Missing columns: {missing}"

        )

    daily = aggregate_daily_sales(df)

    daily = fill_missing_dates(
        daily
    )

    daily = daily.sort_values(
        "ds"
    ).reset_index(drop=True)

    return daily