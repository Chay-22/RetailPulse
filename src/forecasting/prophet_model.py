"""
=========================================================
RetailPulse - Prophet Forecasting Engine
---------------------------------------------------------
Author  : RetailPulse Team
Version : 2.0

Features
--------
✔ Automatic training
✔ Model persistence
✔ Reload existing model
✔ Future forecasting
✔ Confidence interval
✔ Hyperparameter configuration
✔ Logging support
✔ Production-ready
=========================================================
"""

import logging
import os
import joblib
import pandas as pd

from prophet import Prophet

# -----------------------------------------------------
# Logging
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Prophet Forecast Model
# -----------------------------------------------------

class ProphetForecast:

    def __init__(

        self,

        changepoint_prior_scale=0.05,

        seasonality_prior_scale=10,

        yearly_seasonality=True,

        weekly_seasonality=True,

        daily_seasonality=False,

        interval_width=0.95,

        model_path="models/prophet.pkl"

    ):

        self.model = None

        self.model_path = model_path

        self.params = {

            "changepoint_prior_scale":
                changepoint_prior_scale,

            "seasonality_prior_scale":
                seasonality_prior_scale,

            "yearly_seasonality":
                yearly_seasonality,

            "weekly_seasonality":
                weekly_seasonality,

            "daily_seasonality":
                daily_seasonality,

            "interval_width":
                interval_width

        }


    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    def train(self, train_df):

        required_columns = {"ds", "y"}

        if not required_columns.issubset(train_df.columns):

            raise ValueError(
                "Prophet requires columns: ds, y"
            )

        if len(train_df) < 10:

            raise ValueError(
                "Too few observations for Prophet."
            )

        logger.info(

            f"Training Prophet model "

            f"({len(train_df)} samples)"

        )

        model = Prophet(

            changepoint_prior_scale=
                self.params["changepoint_prior_scale"],

            seasonality_prior_scale=
                self.params["seasonality_prior_scale"],

            yearly_seasonality=
                self.params["yearly_seasonality"],

            weekly_seasonality=
                self.params["weekly_seasonality"],

            daily_seasonality=
                self.params["daily_seasonality"],

            interval_width=
                self.params["interval_width"]

        )

        try:

            model.fit(train_df)

        except Exception as e:

            logger.exception(

                "Prophet training failed."

            )

            raise

        self.model = model

        logger.info("Training Complete")

        return model


    # -------------------------------------------------
    # Forecast
    # -------------------------------------------------

    def forecast(

        self,

        periods=30,

        frequency="D"

    ):
        if periods <= 0:

            raise ValueError(

                "Forecast periods must be positive."

            )

        if self.model is None:

            raise Exception(
                "Train model first."
            )

        future = self.model.make_future_dataframe(

            periods=periods,

            freq=frequency

        )

        forecast = self.model.predict(future)
        required = {

            "ds",

            "yhat"

        }

        missing = required - set(forecast.columns)

        if missing:

            raise RuntimeError(

                f"Forecast missing columns: {missing}"

            )

        return forecast


    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    def save(self):

        if self.model is None:
            raise Exception("No trained model")

        os.makedirs(
            os.path.dirname(self.model_path),
            exist_ok=True
        )

        payload = {

            "model": self.model,

            "metadata": {

                "params": self.params,

                "saved_at": pd.Timestamp.utcnow(),

                "version": "2.0"

            }

        }

        joblib.dump(
            payload,
            self.model_path
        )

        logger.info(
            f"Model saved -> {self.model_path}"
        )


    # -------------------------------------------------
    # Load Model
    # -------------------------------------------------

    def load(self):

        if os.path.exists(self.model_path):

            payload = joblib.load(
                self.model_path
            )

            if isinstance(payload, dict):

                self.model = payload["model"]

            else:

                # old checkpoints

                self.model = payload

            logger.info("Saved model loaded.")

            return True

        logger.info("No saved model found.")

        return False


    # -------------------------------------------------
    # Train or Load
    # -------------------------------------------------

    def train_or_load(

        self,

        train_df,

        retrain=False

    ):

        if not retrain:

            if self.load():

                logger.info(

                    "Using existing Prophet model."

                )

                return self.model

        self.train(train_df)

        self.save()

        return self.model


    # -------------------------------------------------
    # Forecast Pipeline
    # -------------------------------------------------

    def run(

        self,

        train_df,

        periods=30,

        retrain=False

    ):

        self.train_or_load(

            train_df,

            retrain

        )

        forecast = self.forecast(periods)

        return forecast