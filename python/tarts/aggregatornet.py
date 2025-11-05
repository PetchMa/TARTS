"""Aggregator Network for combining multiple donut predictions into a single estimate.

This module implements a transformer-based neural network that aggregates
predictions from multiple donut images to produce a single, improved estimate
of Zernike coefficients for the LSST Active Optics System.
"""

# Third-party imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local/application imports
from .utils import convert_zernikes_deploy


class AggregatorNet(pl.LightningModule):
    """Aggregator Network.

    Implements a transformer encoder network to aggregate the
    values of multiple donuts and performs a single point estimations

    Attributes
    ----------
    transformer_encoder : obj
        model architecture
    fc : obj
        fully connected linear layer

    Methods
    -------
    forward()
        forward prop model
    training_step()
        single train step update
    validation_step()
        single validation update
    loss_fn()
        loss function (MSE)
    configure_optimizers()
        setup optimizers
        reduce on plateu by 1/2 until a limit
        uses AdaM optimizer with tuned learning rate
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_length: int,
        lr=0.002507905395321983,
        num_zernikes=17,
    ):
        """Initialize the AggregatorNet model.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input
            (embedding dimension).
        nhead : int
            The number of attention heads in the multi-head
            attention mechanism.
        num_layers : int
            The number of transformer encoder layers.
        dim_feedforward : int
            The dimension of the feedforward network model
            inside the transformer encoder.
        max_seq_length : int
            The maximum sequence length of input data.
        lr : float, optional
            The learning rate for model training (default is 1e-3).
        num_zernikes : int, optional
            The number of Zernike polynomial coefficients to
            predict (default is 19).

        Notes
        -----
        - The transformer encoder consists of `num_layers` encoder layers.
        - The model outputs a linear transformation of size `num_zernikes`.

        """
        super().__init__()
        self.save_hyperparameters()  # Save model hyperparameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # final layer to transform to the shape of number of zernikes
        self.fc = nn.Linear(d_model, num_zernikes)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the AggregatorNet model.

        Parameters
        ----------
        x : tuple of (torch.Tensor, torch.Tensor)
            A tuple where:
            - x[0] (torch.Tensor): The input sequence tensor of shape
                (batch_size, seq_length, d_model).
            - x[1] (torch.Tensor): The mean tensor used for output adjustment.

        Returns
        -------
        torch.Tensor
            The transformed output tensor of shape (batch_size, num_zernikes).

        Notes
        -----
        - The transformer encoder processes the first element of the tuple.
        - The last token's output is extracted and passed through a linear
            layer.
        - The mean correction (second element) is added to the final output.

        """
        x_input, mean = x
        x_tensor = self.transformer_encoder(x_input)
        x_tensor = x_tensor[:, -1, :]  # Take the last token's output
        x_tensor = self.fc(x_tensor)  # Predict the next token
        x_tensor += mean
        return x_tensor

    def training_step(self, batch: tuple, batch_idx: int):
        """Perform a single training step.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, torch.Tensor)
            A tuple containing:
            - x (torch.Tensor): The input tensor of shape
                (batch_size, seq_length, d_model).
            - y (torch.Tensor): The target tensor of shape
                (batch_size, num_zernikes).
        batch_idx : int
            The index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The computed loss for the batch.

        Notes
        -----
        - The model processes `x` using the `forward`
            method to obtain predictions.
        - The loss is computed between the predicted
            logits and target `y`.
        - The training loss is logged for monitoring.

        """
        x, y = batch  # y is the target token
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Parameters
        ----------
        batch : tuple of (torch.Tensor, torch.Tensor)
            A tuple containing:
            - x (torch.Tensor): The input tensor of shape
                (batch_size, seq_length, d_model).
            - y (torch.Tensor): The target tensor of shape
            (batch_size, num_zernikes).
        batch_idx : int
            The index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The computed validation loss for the batch.

        Notes
        -----
        - The model processes `x` using the `forward`
            method to obtain predictions.
        - The loss is computed between the predicted
            logits and target `y`.
        - The validation loss is logged for monitoring.

        """
        x, y = batch  # y is the target token
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mRSSE", loss, prog_bar=True)  # mRSSE is the same as loss for this model
        return loss

    def loss_fn(self, x, y):
        """Compute the loss using the Root Sum of Squared Errors (mRSSe).

        Parameters
        ----------
        x : torch.Tensor
            The predicted tensor of shape (batch_size, num_zernikes).
        y : torch.Tensor
            The target tensor of shape (batch_size, num_zernikes).

        Returns
        -------
        torch.Tensor
            The computed mean Root Sum of Squared Errors (mRSSe).

        Notes
        -----
        - The loss is calculated as the mean of the square root of
            the sum of squared errors.
        - Mean squared error (MSE) is computed first, followed by
            summation along the last dimension.
        - The final value is the mean of the root sum of squared
            errors across the batch.

        """
        x = convert_zernikes_deploy(x)
        y = convert_zernikes_deploy(y)
        sse = F_loss.mse_loss(x, y, reduction="none").sum(dim=-1)
        mRSSe = torch.sqrt(sse).mean()
        return mRSSe

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
