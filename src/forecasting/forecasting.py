"""
=========================================================
RetailPulse Production Forecasting Pipeline
---------------------------------------------------------
Production Orchestration Layer

Features
--------
✓ Production-grade orchestration
✓ Training / Inference separation
✓ Intelligent model reuse
✓ Retrain only when necessary
✓ Input validation
✓ Structured logging
✓ Robust exception handling
✓ Consistent metrics
✓ Backward compatible public API
=========================================================
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecasting.utils import (
    prepare_dataset,
    scale_series,
)

from src.forecasting.metrics import ForecastMetrics

try:
    from src.forecasting.prophet_model import ProphetForecast

    PROPHET_AVAILABLE = True
    PROPHET_IMPORT_ERROR: Optional[BaseException] = None
except ModuleNotFoundError as exc:
    if exc.name != "prophet":
        raise

    ProphetForecast = None
    PROPHET_AVAILABLE = False
    PROPHET_IMPORT_ERROR = exc

try:
    from src.forecasting.trainer import LSTMTrainer
    from src.forecasting.predictor import LSTMPredictor
    from src.forecasting.ensemble import ForecastEnsemble

    LSTM_AVAILABLE = True
    LSTM_IMPORT_ERROR: Optional[BaseException] = None
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    LSTMTrainer = None
    LSTMPredictor = None
    ForecastEnsemble = None
    LSTM_AVAILABLE = False
    LSTM_IMPORT_ERROR = exc


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_HORIZON = 30

LSTM_MODEL_PATH = "models/lstm_model.pth"


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def _validate_inputs(
    df: pd.DataFrame,
    horizon: int,
) -> None:
    """
    Validate forecast inputs.

    Raises
    ------
    ValueError
        When dataset or parameters are invalid.
    """

    if df is None:
        raise ValueError("Input dataframe cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _prepare(df: pd.DataFrame):
    """
    Prepare forecasting dataset.
    """

    daily = prepare_dataset(df)

    if len(daily) < 35:
        raise ValueError(
            "Insufficient observations for forecasting."
        )

    scaler, scaled = scale_series(
        daily["y"]
    )

    return daily, scaler, scaled


# ---------------------------------------------------------
# LSTM Model Availability
# ---------------------------------------------------------

def _lstm_checkpoint_exists() -> bool:
    return os.path.exists(LSTM_MODEL_PATH)


def _prophet_available() -> bool:
    """
    Return whether optional Prophet dependencies are importable.
    """

    if PROPHET_AVAILABLE:
        return True

    logger.warning(
        "Prophet is not installed; using statistical fallback forecasting. "
        "Original import error: %s",
        PROPHET_IMPORT_ERROR,
    )

    return False


def _lstm_available() -> bool:
    """
    Return whether optional PyTorch/LSTM dependencies are importable.
    """

    if LSTM_AVAILABLE:
        return True

    logger.warning(
        "PyTorch is not installed; using Prophet-only forecasting. "
        "Original import error: %s",
        LSTM_IMPORT_ERROR,
    )

    return False


# ---------------------------------------------------------
# Prophet Stage
# ---------------------------------------------------------

def _run_prophet(
    daily: pd.DataFrame,
    horizon: int,
    retrain: bool,
):
    """
    Execute Prophet stage.
    """

    logger.info("Running Prophet forecasting...")

    if not _prophet_available():
        return _run_statistical_forecast(
            daily=daily,
            horizon=horizon,
        )

    prophet = ProphetForecast()

    return prophet.run(
        train_df=daily,
        periods=horizon,
        retrain=retrain,
    )


# ---------------------------------------------------------
# LSTM Training Stage
# ---------------------------------------------------------

def _train_lstm_if_required(
    trainer,
    scaled,
    retrain: bool,
):
    """
    Train only when necessary.

    Existing checkpoint is reused automatically.
    """

    should_train = (
        retrain
        or
        not _lstm_checkpoint_exists()
    )

    if should_train:

        logger.info(
            "Training LSTM model..."
        )

        trainer.fit(
            scaled,
            resume=False,
        )

    else:

        logger.info(
            "Using existing LSTM checkpoint."
        )
# ---------------------------------------------------------
# LSTM Inference Stage
# ---------------------------------------------------------

def _run_lstm_prediction(
    trainer,
    scaler,
    scaled,
    horizon: int,
):
    """
    Execute inference using the trained (or reused) LSTM model.
    """

    logger.info("Running LSTM inference...")

    predictor = LSTMPredictor()

    sequence = scaled[-trainer.sequence_length:]

    prediction = predictor.predict(
        sequence,
        scaler,
        steps=horizon,
    )

    return predictor, prediction


# ---------------------------------------------------------
# Ensemble Stage
# ---------------------------------------------------------

def _run_ensemble(
    prophet_forecast,
    lstm_forecast,
    horizon,
):
    """
    Combine Prophet and LSTM predictions.
    """

    logger.info("Combining forecasts...")

    ensemble = ForecastEnsemble()

    combined = ensemble.combine(
        prophet_forecast.tail(horizon)["yhat"].values,
        lstm_forecast,
    )

    return ensemble, combined


def _confidence_interval(values):
    """
    Build a lightweight fallback confidence interval for Prophet-only mode.
    """

    values = np.asarray(values, dtype=float)
    std = float(np.nanstd(values))
    margin = 0.0 if std == 0 else 1.96 * std

    return values - margin, values + margin


def _run_prophet_only_forecast(
    daily: pd.DataFrame,
    prophet_forecast: pd.DataFrame,
    horizon: int,
):
    """
    Build a full forecast payload when the optional LSTM stack is unavailable.
    """

    prophet_future = prophet_forecast.tail(horizon)["yhat"].to_numpy(dtype=float)

    if {"yhat_lower", "yhat_upper"}.issubset(prophet_forecast.columns):
        lower = prophet_forecast.tail(horizon)["yhat_lower"].to_numpy(dtype=float)
        upper = prophet_forecast.tail(horizon)["yhat_upper"].to_numpy(dtype=float)
    else:
        lower, upper = _confidence_interval(prophet_future)

    return _build_forecast_dataframe(
        daily=daily,
        prophet_forecast=prophet_forecast,
        lstm_forecast=prophet_future,
        final_forecast=prophet_future,
        lower=lower,
        upper=upper,
        horizon=horizon,
    )


def _run_statistical_forecast(
    daily: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Produce a dependency-free baseline forecast with Prophet-compatible columns.
    """

    history = daily[["ds", "y"]].copy()
    y = pd.to_numeric(history["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if len(y) == 0:
        raise ValueError("Cannot forecast an empty prepared dataset.")

    window = min(7, len(y))
    baseline = float(np.mean(y[-window:]))

    if len(y) >= 2:
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    steps = np.arange(1, horizon + 1, dtype=float)
    future_yhat = np.maximum(baseline + slope * steps, 0.0)

    residuals = y[-window:] - baseline
    residual_std = float(np.std(residuals))
    margin = 1.96 * residual_std

    future_dates = pd.date_range(
        start=history["ds"].max(),
        periods=horizon + 1,
        freq="D",
    )[1:]

    fitted = pd.DataFrame(
        {
            "ds": history["ds"],
            "yhat": y,
            "yhat_lower": y,
            "yhat_upper": y,
        }
    )

    future = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": future_yhat,
            "yhat_lower": np.maximum(future_yhat - margin, 0.0),
            "yhat_upper": future_yhat + margin,
        }
    )

    logger.info(
        "Statistical fallback forecast generated for %s period(s).",
        horizon,
    )

    return pd.concat([fitted, future], ignore_index=True)


# ---------------------------------------------------------
# Forecast DataFrame Builder
# ---------------------------------------------------------

def _build_forecast_dataframe(
    daily,
    prophet_forecast,
    lstm_forecast,
    final_forecast,
    lower,
    upper,
    horizon,
):
    """
    Build standardized forecast dataframe.
    """

    future_dates = pd.date_range(
        start=daily["ds"].max(),
        periods=horizon + 1,
        freq="D",
    )[1:]

    return pd.DataFrame(
        {
            "ds": future_dates,
            "Forecast": final_forecast,
            "Lower": lower,
            "Upper": upper,
            "Prophet": prophet_forecast.tail(
                horizon
            )["yhat"].values,
            "LSTM": lstm_forecast,
        }
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def _calculate_metrics(
    daily,
    prophet_forecast,
    horizon,
):
    """
    Produce consistent metrics across the pipeline.
    """

    logger.info("Calculating forecast metrics...")

    actual = daily["y"].tail(horizon)

    predicted = prophet_forecast.tail(
        horizon
    )["yhat"]

    return ForecastMetrics.evaluate(
        actual.values,
        predicted.values,
    )


# ---------------------------------------------------------
# Public Forecast API
# ---------------------------------------------------------

def forecast(
    df,
    horizon=DEFAULT_HORIZON,
    retrain=False,
):
    """
    Production Forecast Pipeline.

    Parameters
    ----------
    df : pandas.DataFrame

    horizon : int
        Number of future periods.

    retrain : bool
        Force retraining.

    Returns
    -------
    forecast_df : pandas.DataFrame

    metrics : dict
    """

    logger.info(
        "Forecast pipeline started."
    )

    _validate_inputs(
        df=df,
        horizon=horizon,
    )

    try:

        # -------------------------------------
        # Data Preparation
        # -------------------------------------

        daily, scaler, scaled = _prepare(df)

        logger.info(
            "Dataset prepared successfully."
        )

        # -------------------------------------
        # Prophet
        # -------------------------------------

        prophet_forecast = _run_prophet(
            daily,
            horizon,
            retrain,
        )

        if not _lstm_available():
            forecast_df = _run_prophet_only_forecast(
                daily=daily,
                prophet_forecast=prophet_forecast,
                horizon=horizon,
            )

            metrics = _calculate_metrics(
                daily=daily,
                prophet_forecast=prophet_forecast,
                horizon=horizon,
            )

            logger.info(
                "Forecast pipeline completed successfully without LSTM."
            )

            return forecast_df, metrics

        trainer = LSTMTrainer()

        # -------------------------------------
        # LSTM Training
        # -------------------------------------

        _train_lstm_if_required(
            trainer,
            scaled,
            retrain,
        )

        # -------------------------------------
        # LSTM Prediction
        # -------------------------------------

        predictor, lstm_forecast = _run_lstm_prediction(
            trainer,
            scaler,
            scaled,
            horizon,
        )
            # -------------------------------------
        # Ensemble
        # -------------------------------------

        _, final_forecast = _run_ensemble(
            prophet_forecast=prophet_forecast,
            lstm_forecast=lstm_forecast,
            horizon=horizon,
        )

        # -------------------------------------
        # Confidence Interval
        # -------------------------------------

        lower, upper = predictor.confidence_interval(
            final_forecast
        )

        # -------------------------------------
        # Forecast DataFrame
        # -------------------------------------

        forecast_df = _build_forecast_dataframe(
            daily=daily,
            prophet_forecast=prophet_forecast,
            lstm_forecast=lstm_forecast,
            final_forecast=final_forecast,
            lower=lower,
            upper=upper,
            horizon=horizon,
        )

        # -------------------------------------
        # Metrics
        # -------------------------------------

        metrics = _calculate_metrics(
            daily=daily,
            prophet_forecast=prophet_forecast,
            horizon=horizon,
        )

        logger.info(
            "Forecast pipeline completed successfully."
        )

        return forecast_df, metrics

    except KeyboardInterrupt:

        logger.warning(
            "Forecasting interrupted by user."
        )

        raise

    except ValueError:

        logger.exception(
            "Validation failed during forecasting."
        )

        raise

    except FileNotFoundError:

        logger.exception(
            "Required model checkpoint not found."
        )

        raise

    except Exception as exc:

        logger.exception(
            "Unexpected forecasting error: %s",
            exc,
        )

        raise RuntimeError(
            "Forecast pipeline failed."
        ) from exc
