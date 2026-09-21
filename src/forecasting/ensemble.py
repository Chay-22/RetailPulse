"""
=========================================================
RetailPulse Forecast Ensemble
---------------------------------------------------------
Combines Prophet + LSTM Forecasts
Evaluates performance
Produces weighted ensemble prediction
=========================================================
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.forecasting.prophet_model import ProphetForecast
from src.forecasting.predictor import LSTMPredictor


class ForecastEnsemble:

    def __init__(self):

        self.prophet_weight = 0.70
        self.lstm_weight = 0.30

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    def _validate_predictions(
        self,
        prophet_forecast,
        lstm_forecast,
    ):

        if prophet_forecast is None:

            raise ValueError(
                "Prophet forecast cannot be None."
            )

        if lstm_forecast is None:

            raise ValueError(
                "LSTM forecast cannot be None."
            )

        if len(prophet_forecast) == 0:

            raise ValueError(
                "Empty Prophet forecast."
            )

        if len(lstm_forecast) == 0:

            raise ValueError(
                "Empty LSTM forecast."
            )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    def calculate_metrics(self, y_true, y_pred):

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        mape = np.mean(
            np.abs((y_true - y_pred) / (y_true + 1e-8))
        ) * 100

        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "MAPE": round(mape, 2),
            "R2": round(r2, 3)
        }

    # -----------------------------------------------------
    # Dynamic Weight Calculation
    # -----------------------------------------------------

    def update_weights(
        self,
        prophet_rmse,
        lstm_rmse
    ):

        if prophet_rmse <= 0:

            prophet_rmse = 1e-8

        if lstm_rmse <= 0:

            lstm_rmse = 1e-8

        inverse = np.array([
            1 / prophet_rmse,
            1 / lstm_rmse,
        ])

        weights = inverse / inverse.sum()

        self.prophet_weight = float(weights[0])

        self.lstm_weight = float(weights[1])
    # -----------------------------------------------------
    # Ensemble Forecast
    # -----------------------------------------------------

    def combine(
        self,
        prophet_forecast,
        lstm_forecast
    ):
        total = (
            self.prophet_weight
            +
            self.lstm_weight
        )

        if total == 0:

            self.prophet_weight = 0.5
            self.lstm_weight = 0.5

        else:

            self.prophet_weight /= total
            self.lstm_weight /= total

        self._validate_predictions(
            prophet_forecast,
            lstm_forecast
        )

        prophet = np.asarray(
            prophet_forecast,
            dtype=float
        )

        lstm = np.asarray(
            lstm_forecast,
            dtype=float
        )

        prophet = np.nan_to_num(prophet)

        lstm = np.nan_to_num(lstm)

        length = min(
            len(prophet),
            len(lstm)
        )

        prophet = prophet[:length]
        lstm = lstm[:length]

        try:

            ensemble = (

                self.prophet_weight * prophet

                +

                self.lstm_weight * lstm

            )

        except Exception:

            if len(prophet):

                ensemble = prophet

            else:

                ensemble = lstm

        if not np.isfinite(ensemble).all():

            raise RuntimeError(
                "Invalid ensemble forecast."
            )

        return ensemble

    # -----------------------------------------------------
    # Complete Pipeline
    # -----------------------------------------------------

    def run(
        self,
        prophet_df,
        lstm_sequence,
        scaler,
        periods=30
    ):

        # ---------------- Prophet ----------------

        prophet_model = ProphetForecast()

        prophet = prophet_model.run(
            prophet_df,
            periods=periods,
            retrain=False
        )

        prophet_future = prophet.tail(periods)["yhat"].values

        # ---------------- LSTM ----------------

        predictor = LSTMPredictor()

        lstm_future = predictor.predict(
            lstm_sequence,
            scaler,
            steps=periods
        )

        lower, upper = predictor.confidence_interval(
            final
        )

        # ---------------- Ensemble ----------------

        final = self.combine(
            prophet_future,
            lstm_future
        )

        forecast = pd.DataFrame({

            "Forecast": final,

            "Prophet": prophet_future,

            "LSTM": lstm_future,

            "Lower": lower,

            "Upper": upper

        })

        return forecast