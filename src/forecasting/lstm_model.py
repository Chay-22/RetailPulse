"""
=========================================================
RetailPulse Production LSTM Forecasting Model
---------------------------------------------------------
Features
--------
✔ Multi-layer LSTM
✔ Dropout
✔ Xavier Initialization
✔ GPU Compatible
✔ Batch First
✔ Regression Head
=========================================================
"""

import torch
import torch.nn as nn


class RetailLSTM(nn.Module):
    """
    Production LSTM model for demand forecasting.
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_size=1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Fully Connected Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

        self._initialize_weights()

    # -------------------------------------------------

    def _initialize_weights(self):
        """
        Xavier Initialization
        """

        for name, param in self.named_parameters():

            if "weight" in name:
                nn.init.xavier_uniform_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

    # -------------------------------------------------

    def forward(self, x):

        # x
        # (batch, sequence_length, input_size)

        output, (hidden, cell) = self.lstm(x)

        # Last Hidden State
        last_hidden = output[:, -1, :]

        prediction = self.regressor(last_hidden)

        return prediction


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def create_model(device=None):

    model = RetailLSTM()

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)

    return model