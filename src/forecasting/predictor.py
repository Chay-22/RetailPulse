"""
=========================================================
RetailPulse LSTM Predictor
---------------------------------------------------------
Features
--------
✓ Load trained model
✓ Recursive Forecasting
✓ GPU Compatible
✓ Inverse Scaling
✓ Multi-step Forecast
✓ Confidence Estimation
=========================================================
"""


import os
import logging
import numpy as np
import torch

from src.forecasting.lstm_model import RetailLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

class LSTMPredictor:

    def __init__(

        self,

        model_path="models/lstm_model.pth",

        hidden_size=128,

        num_layers=2,

        dropout=0.2

    ):

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        self.model = RetailLSTM(

            hidden_size=hidden_size,

            num_layers=num_layers,

            dropout=dropout

        ).to(self.device)

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"LSTM checkpoint not found: {model_path}"
            )

        checkpoint = None

        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
            )

            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    "Invalid LSTM checkpoint format."
                )

            if "model_state_dict" not in checkpoint:
                raise RuntimeError(
                    "Invalid checkpoint: model_state_dict is missing."
                )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        finally:
            # Release the full checkpoint after loading model weights.
            del checkpoint

        self.model.eval()

        logger.info(f"LSTM model loaded from {model_path}")


    # -----------------------------------------------------

    def predict(

        self,

        last_sequence,

        scaler,

        steps=30

    ):

        """
        Recursive multi-step prediction.

        Parameters
        ----------
        last_sequence : numpy array
            Last scaled sequence.

        scaler : MinMaxScaler

        steps : int

        Returns
        -------
        DataFrame-ready forecast
        """
        if len(last_sequence) == 0:

            raise ValueError(
                "Input sequence is empty."
            )

        if steps <= 0:

            raise ValueError(
                "steps must be greater than zero."
            )

        predictions = []

        sequence = np.asarray(
            last_sequence,
            dtype=np.float32
        ).flatten()

        for _ in range(steps):

            x = torch.tensor(

                sequence,

                dtype=torch.float32

            ).unsqueeze(0).unsqueeze(-1)

            x = x.to(self.device)

            with torch.inference_mode():

                pred = self.model(x)

            value = float(pred.squeeze().cpu().numpy())

            predictions.append(value)

            sequence = np.append(

                sequence[1:],

                value

            )

        predictions = np.array(
            predictions
        ).reshape(-1, 1)

        # -----------------------------
        # Validate Scaler
        # -----------------------------

        if scaler is None:

            raise ValueError(
                "Scaler cannot be None."
            )

        predictions = scaler.inverse_transform(
            predictions
        )

        predictions = predictions.flatten()

        # -----------------------------
        # Validate Output
        # -----------------------------

        if np.isnan(predictions).any():

            raise RuntimeError(
                "NaN values detected in prediction."
            )

        return predictions


    # -----------------------------------------------------

    def confidence_interval(

        self,

        predictions,

        confidence=0.95

    ):

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        std = np.std(predictions)

        if std == 0:

            margin = 0

        else:

            margin = 1.96 * std

        lower = predictions - margin

        upper = predictions + margin

        return lower, upper