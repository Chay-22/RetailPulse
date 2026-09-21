"""
=========================================================
RetailPulse Production LSTM Trainer
---------------------------------------------------------
Features
--------
✓ GPU / CPU Training
✓ Mixed Precision Support
✓ Early Stopping
✓ Checkpoint Saving
✓ Resume Training
✓ Learning Rate Scheduler
✓ Training History
=========================================================
"""

import os
import time
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.forecasting.dataset import TimeSeriesDataset
from src.forecasting.lstm_model import RetailLSTM


# ---------------------------------------------------
# Logger
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Trainer
# ---------------------------------------------------

class LSTMTrainer:

    def __init__(

        self,

        sequence_length=30,

        batch_size=64,

        learning_rate=0.001,

        epochs=100,

        patience=15,

        hidden_size=128,

        num_layers=2,

        dropout=0.2,

        checkpoint_path="models/lstm_model.pth"

    ):

        self.sequence_length = sequence_length

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.epochs = epochs

        self.patience = patience

        self.checkpoint_path = checkpoint_path

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        logger.info(f"Using Device : {self.device}")

        # -----------------------------
        # Model
        # -----------------------------

        self.model = RetailLSTM(

            hidden_size=hidden_size,

            num_layers=num_layers,

            dropout=dropout

        ).to(self.device)

        # -----------------------------
        # Loss
        # -----------------------------

        self.criterion = nn.MSELoss()

        # -----------------------------
        # Optimizer
        # -----------------------------

        self.optimizer = Adam(

            self.model.parameters(),

            lr=self.learning_rate

        )

        # -----------------------------
        # Scheduler
        # -----------------------------

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            self.optimizer,

            factor=0.5,

            patience=5

        )

        # -----------------------------
        # Mixed Precision

        self.scaler = torch.cuda.amp.GradScaler(

            enabled=torch.cuda.is_available()

        )

        # -----------------------------
        # History

        self.train_losses = []

        self.val_losses = []

        self.best_loss = float("inf")

        self.early_counter = 0

        # ---------------------------------------------------
    # Prepare Dataset
    # ---------------------------------------------------

    def prepare_data(self, scaled_series):

        dataset = TimeSeriesDataset(
            scaled_series,
            sequence_length=self.sequence_length
        )

        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        logger.info(
            f"Training Samples : {len(train_dataset)} | "
            f"Validation Samples : {len(val_dataset)}"
        )

        return train_loader, val_loader
    
        # ---------------------------------------------------
    # ---------------------------------------------------
    # Dataset Fingerprint
    # ---------------------------------------------------

    def dataset_hash(self, scaled_series):
        """
        Create a deterministic fingerprint of the training data.
        """

        try:

            return hashlib.sha256(
                scaled_series.tobytes()
            ).hexdigest()

        except Exception:

            return None

    def _prepare_features(self, X_batch):
        """
        Ensure LSTM inputs have shape:
        (batch, sequence_length, input_size)
        """

        if X_batch.dim() == 2:

            X_batch = X_batch.unsqueeze(-1)

        elif X_batch.dim() == 4 and X_batch.size(-1) == 1:

            X_batch = X_batch.squeeze(-1)

        if X_batch.dim() != 3:

            raise ValueError(
                f"LSTM: Expected input to be 2D or 3D, got {X_batch.dim()}D instead"
            )

        return X_batch.to(self.device)
    
    # Train One Epoch
    # ---------------------------------------------------

    def train_epoch(self, train_loader):

        self.model.train()

        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = self._prepare_features(X_batch)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

                predictions = self.model(X_batch)

                loss = self.criterion(
                    predictions.squeeze(),
                    y_batch.squeeze()
                )

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        self.train_losses.append(epoch_loss)

        return epoch_loss
    
        # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------

    def validate(self, val_loader):

        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = self._prepare_features(X_batch)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)

                loss = self.criterion(
                    predictions.squeeze(),
                    y_batch.squeeze()
                )

                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)

        self.val_losses.append(val_loss)

        return val_loss
    
        # ---------------------------------------------------
    # Save Checkpoint
    # ---------------------------------------------------

    def save_checkpoint(self):

        os.makedirs(
            os.path.dirname(self.checkpoint_path),
            exist_ok=True
        )

        checkpoint = {

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "train_loss":
                self.train_losses,

            "val_loss":
                self.val_losses,

            "best_loss":
                self.best_loss,

            # -------------------------
            # Metadata
            # -------------------------

            "metadata": {

                "sequence_length":
                    self.sequence_length,

                "batch_size":
                    self.batch_size,

                "learning_rate":
                    self.learning_rate,

                "epochs":
                    self.epochs,

                "device":
                    str(self.device),

                "trainer_version":
                    "2.0",

                "dataset_hash":
                    getattr(
                        self,
                        "_dataset_hash",
                        None
                    )

            }

        }

        torch.save(
            checkpoint,
            self.checkpoint_path
        )

        logger.info(
            f"Checkpoint Saved -> {self.checkpoint_path}"
        )

    # ---------------------------------------------------
    # Load Checkpoint
    # ---------------------------------------------------

    def load_checkpoint(self):

        if not os.path.exists(self.checkpoint_path):

            logger.info("No checkpoint found.")

            return False

        try:

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )

            required_keys = [
                "model_state_dict"
            ]

            for key in required_keys:

                if key not in checkpoint:

                    logger.error(
                        f"Checkpoint missing key: {key}"
                    )

                    return False

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            if "optimizer_state_dict" in checkpoint:

                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )

            self.train_losses = checkpoint.get(
                "train_loss",
                []
            )

            self.val_losses = checkpoint.get(
                "val_loss",
                []
            )

            self.best_loss = checkpoint.get(
                "best_loss",
                float("inf")
            )

            logger.info(
                "Checkpoint loaded successfully."
            )

            return True

        except RuntimeError as e:

            logger.error(
                f"Model architecture mismatch: {e}"
            )

            return False

        except Exception as e:

            logger.exception(
                f"Unable to load checkpoint: {e}"
            )

            return False
    
    # ---------------------------------------------------
    # Check Retraining Requirement
    # ---------------------------------------------------

    def requires_training(
        self,
        scaled_series,
        retrain=False,
    ):
        """
        Decide whether training is required.
        """

        if retrain:
            return True

        if not os.path.exists(
            self.checkpoint_path
        ):
            return True

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )

        metadata = checkpoint.get(
            "metadata",
            {}
        )

        old_hash = metadata.get(
            "dataset_hash"
        )

        new_hash = self.dataset_hash(
            scaled_series
        )

        if old_hash != new_hash:

            logger.info(
                "Dataset changed. Retraining required."
            )

            return True

        logger.info(
            "Existing model matches dataset."
        )

        return False


    # ---------------------------------------------------
    # Training Loop
    # ---------------------------------------------------

    def fit(

        self,

        scaled_series,

        resume=False

    ):
        if len(scaled_series) <= self.sequence_length:

            raise ValueError(

                "Dataset length must be greater than sequence_length."

            )

        train_loader, val_loader = self.prepare_data(
            scaled_series
        )


        self._dataset_hash = self.dataset_hash(
            scaled_series
        )

        if not self.requires_training(
            scaled_series,
            retrain=not resume,
        ):

            logger.info(
                "Skipping training. Using existing checkpoint."
            )

            self.load_checkpoint()

            return self.model

        if resume:

            self.load_checkpoint()

        logger.info("Training Started")

        start = time.time()

        for epoch in range(self.epochs):

            train_loss = self.train_epoch(train_loader)

            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            logger.info(

                f"Epoch {epoch+1:03d}/{self.epochs}"

                f" | Train={train_loss:.6f}"

                f" | Val={val_loss:.6f}"

                f" | Best={self.best_loss:.6f}"

                f" | LR={self.optimizer.param_groups[0]['lr']:.7f}"

            )

            # ----------------------------
            # Early Stopping
            # ----------------------------

            if val_loss < self.best_loss:

                self.best_loss = val_loss

                self.early_counter = 0

                logger.info(
                    "Validation improved. Saving best checkpoint."
                )

                self.save_checkpoint()

            else:

                self.early_counter += 1

            if self.early_counter >= self.patience:

                logger.info("Early Stopping Triggered")

                break

        elapsed = time.time() - start

        logger.info(

            f"Training Finished"

            f" ({elapsed:.2f} sec)"

        )

        self.load_checkpoint()

        return self.model


    # ---------------------------------------------------
    # Return Training History
    # ---------------------------------------------------

    def history(self):

        return {

            "train_loss":
                self.train_losses,

            "validation_loss":
                self.val_losses

        }
