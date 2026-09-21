"""
=========================================================
RetailPulse - LSTM Dataset
---------------------------------------------------------
Creates sliding window sequences for time-series forecasting.
Compatible with PyTorch DataLoader.
=========================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset for LSTM Time-Series Forecasting.
    """

    def __init__(self, series, sequence_length=30):
        """
        Parameters
        ----------
        series : numpy.ndarray
            Scaled time series values.
        sequence_length : int
            Number of previous time steps used
            to predict the next value.
        """

        self.sequence_length = sequence_length

        self.X = []
        self.y = []

        for i in range(len(series) - sequence_length):

            self.X.append(
                series[i:i + sequence_length]
            )

            self.y.append(
                series[i + sequence_length]
            )

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])

        return x, y