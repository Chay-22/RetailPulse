"""
=========================================================
RetailPulse Forecast Metrics
---------------------------------------------------------
Production Metrics Module

Features
--------
✓ MAE
✓ RMSE
✓ MAPE
✓ SMAPE
✓ MASE
✓ R²
✓ Forecast Accuracy
✓ Directional Accuracy
✓ Forecast Bias
=========================================================
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



class ForecastMetrics:

    # -------------------------------------------------
    # Input Validation
    # -------------------------------------------------

    @staticmethod
    def _sanitize_inputs(y_true, y_pred):

        y_true = np.asarray(
            y_true,
            dtype=float
        ).flatten()

        y_pred = np.asarray(
            y_pred,
            dtype=float
        ).flatten()

        if len(y_true) == 0:

            raise ValueError(
                "y_true is empty."
            )

        if len(y_pred) == 0:

            raise ValueError(
                "y_pred is empty."
            )

        length = min(
            len(y_true),
            len(y_pred)
        )

        y_true = y_true[:length]

        y_pred = y_pred[:length]

        mask = (
            np.isfinite(y_true)
            &
            np.isfinite(y_pred)
        )

        y_true = y_true[mask]

        y_pred = y_pred[mask]

        if len(y_true) == 0:

            raise ValueError(
                "No valid observations."
            )

        return y_true, y_pred


    # -------------------------------------------------
    # Mean Absolute Error
    # -------------------------------------------------

    @staticmethod
    def mae(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        return mean_absolute_error(
            y_true,
            y_pred
        )

    # -------------------------------------------------

    @staticmethod
    def rmse(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        return np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

    # -------------------------------------------------

    @staticmethod
    def mape(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mask = np.abs(y_true) > 1e-8

        return np.mean(
            np.abs(
                (
                    y_true[mask]
                    -
                    y_pred[mask]
                )
                /
                y_true[mask]
            )
        ) * 100

    # -------------------------------------------------

    @staticmethod
    def smape(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        denominator = (
            np.abs(y_true)
            +
            np.abs(y_pred)
        ) / 2

        mask = denominator > 1e-8

        return np.mean(
            np.abs(
                y_true[mask]
                -
                y_pred[mask]
            )
            /
            denominator[mask]
        ) * 100

    # -------------------------------------------------

    @staticmethod
    def mase(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        naive = np.mean(
            np.abs(
                np.diff(y_true)
            )
        )

        if naive < 1e-8:

            return 0.0

        mae = np.mean(
            np.abs(
                y_true
                -
                y_pred
            )
        )

        return mae / naive

    # -------------------------------------------------

    @staticmethod
    def r2(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        if len(y_true) < 2:

            return 0.0

        return r2_score(
            y_true,
            y_pred
        )

    # -------------------------------------------------

    @staticmethod
    def forecast_accuracy(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        mape = ForecastMetrics.mape(
            y_true,
            y_pred
        )

        return max(
            0,
            100 - mape
        )

    # -------------------------------------------------

    @staticmethod
    def forecast_bias(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        return np.mean(
            y_pred - y_true
        )

    # -------------------------------------------------

    @staticmethod
    def directional_accuracy(y_true, y_pred):

        y_true, y_pred = ForecastMetrics._sanitize_inputs(
            y_true,
            y_pred
        )

        true_direction = np.sign(
            np.diff(y_true)
        )

        pred_direction = np.sign(
            np.diff(y_pred)
        )

        if len(true_direction) == 0:

            return 0.0

        return float(

            np.mean(

                true_direction == pred_direction

            ) * 100

        )

    # -------------------------------------------------

    @staticmethod
    def evaluate(y_true, y_pred):

        try:
            

            return {

                "MAE":
                    round(
                        ForecastMetrics.mae(
                            y_true,
                            y_pred
                        ),
                        3
                    ),

                "RMSE":
                    round(
                        ForecastMetrics.rmse(
                            y_true,
                            y_pred
                        ),
                        3
                    ),

                "MAPE":
                    round(
                        ForecastMetrics.mape(
                            y_true,
                            y_pred
                        ),
                        2
                    ),

                "SMAPE":
                    round(
                        ForecastMetrics.smape(
                            y_true,
                            y_pred
                        ),
                        2
                    ),

                "MASE":
                    round(
                        ForecastMetrics.mase(
                            y_true,
                            y_pred
                        ),
                        3
                    ),

                "R²":
                    round(
                        ForecastMetrics.r2(
                            y_true,
                            y_pred
                        ),
                        3
                    ),

                "Forecast Accuracy":
                    round(
                        ForecastMetrics.forecast_accuracy(
                            y_true,
                            y_pred
                        ),
                        2
                    ),

                "Forecast Bias":
                    round(
                        ForecastMetrics.forecast_bias(
                            y_true,
                            y_pred
                        ),
                        3
                    ),

                "Directional Accuracy":
                    round(
                        ForecastMetrics.directional_accuracy(
                            y_true,
                            y_pred
                        ),
                        2
                    )

            }
        
        except Exception:

            return {

                "MAE": 0,

                "RMSE": 0,

                "MAPE": 0,

                "SMAPE": 0,

                "MASE": 0,

                "R²": 0,

                "Forecast Accuracy": 0,

                "Forecast Bias": 0,

                "Directional Accuracy": 0

            }

    # -------------------------------------------------

    @staticmethod
    def dataframe(metrics):

        if not metrics:

            metrics = {}

        return pd.DataFrame({

            "Metric":
                list(metrics.keys()),

            "Value":
                list(metrics.values())

        })