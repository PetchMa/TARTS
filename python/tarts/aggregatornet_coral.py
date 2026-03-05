"""Aggregator Network with DARE-GRAM domain adaptation for coral data.

This module implements a transformer-based aggregator network with DARE-GRAM loss
for unsupervised domain adaptation. The method aligns inverse Gram matrices between
source (simulation) and target (coral/real) domains without requiring target labels.
"""

# Standard library imports
import logging
from typing import Any, Tuple

# Third-party imports
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local/application imports
from .utils import convert_zernikes_deploy

logger = logging.getLogger(__name__)


class AggregatorNet_Coral(pl.LightningModule):
    """Aggregator Network with DARE-GRAM domain adaptation.

    Implements a transformer encoder network to aggregate the
    values of multiple donuts and performs single point estimations,
    with DARE-GRAM loss for domain adaptation to real data.

    Attributes
    ----------
    input_proj : nn.Linear
        Projects input features to d_model dimensions
    transformer_encoder : nn.TransformerEncoder
        Transformer encoder architecture
    fc : nn.Linear
        Fully connected output layer
    transformer_features : torch.Tensor
        Cached transformer features for domain adaptation

    Methods
    -------
    forward()
        Forward propagation through the model
    training_step()
        Single train step with domain adaptation
    validation_step()
        Single validation step
    loss_fn()
        Regression loss function (mRSSE)
    dare_gram_loss()
        Domain adaptation loss between source and target features
    configure_optimizers()
        Setup optimizers with learning rate scheduling
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_length: int,
        lr: float = 0.002507905395321983,
        num_zernikes: int = 17,
        tradeoff_angle: float = 0.05,
        tradeoff_scale: float = 0.001,
        threshold: float = 0.9,
        dare_gram_weight: float = 1.0,
        exp_rise_x1: float = 0.15,
        exp_rise_x2: float = 0.16,
        weight_decay: float = 0.0,
        coral_mode: bool = False,
        rotate: bool = False,
    ):
        """Initialize the AggregatorNet_Coral model.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input (embedding dimension).
        nhead : int
            The number of attention heads in the multi-head attention mechanism.
        num_layers : int
            The number of transformer encoder layers.
        dim_feedforward : int
            The dimension of the feedforward network model inside the transformer encoder.
        max_seq_length : int
            The maximum sequence length of input data.
        lr : float, optional
            The learning rate for model training (default is 0.002507905395321983).
        num_zernikes : int, optional
            The number of Zernike polynomial coefficients to predict (default is 17).
        tradeoff_angle : float, optional, default=0.05
            Weight for the DARE-GRAM angle alignment loss.
        tradeoff_scale : float, optional, default=0.001
            Weight for the DARE-GRAM scale alignment loss.
        threshold : float, optional, default=0.9
            Cumulative variance threshold for low-rank approximation.
        dare_gram_weight : float, optional, default=1.0
            Overall weight to scale the DARE-GRAM loss relative to regression loss.
            Higher values prioritize domain adaptation over regression accuracy.
        exp_rise_x1 : float, optional, default=0.15
            Lower threshold for exponential rise function in DARE-GRAM scaling.
            When mRSSE <= x1, scale_loss = 1.0 (full DARE-GRAM influence).
        exp_rise_x2 : float, optional, default=0.16
            Upper threshold for exponential rise function in DARE-GRAM scaling.
            When mRSSE >= x2, scale_loss = 0.0 (no DARE-GRAM influence).
            Exponential transition occurs between x1 and x2.
        weight_decay : float, optional, default=0.0
            Weight decay (L2 regularization) coefficient for the optimizer.
            Helps prevent overfitting by penalizing large weights.
        coral_mode : bool, optional, default=False
            Whether coral/real data sampling is enabled for domain adaptation.
            When True, the dataset samples real observational data alongside simulation data.
        rotate : bool, optional, default=False
            Whether to use rotated Zernikes (zk_true_camera) in camera frame.
            When True, uses camera frame coordinates instead of telescope frame.

        Notes
        -----
        - The transformer encoder consists of `num_layers` encoder layers.
        - The model outputs a linear transformation of size `num_zernikes`.
        - Domain adaptation is performed by aligning features from the transformer.
        """
        super().__init__()
        self.save_hyperparameters()  # Save model hyperparameters

        # Input projection layer: (num_zernikes + 3) -> d_model
        # The +3 accounts for field_x, field_y, and snr features
        input_dim = num_zernikes + 3
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,  # 50% dropout for regularization
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # final layer to transform to the shape of number of zernikes
        self.fc = nn.Linear(d_model, num_zernikes)

        # Cache for transformer features (for domain adaptation)
        self.transformer_features = None
        self.val_mRSSE: torch.Tensor | None = None
        # Accumulators for epoch-average validation mRSSE
        self._val_mRSSE_sum: torch.Tensor | None = None
        self._val_batch_count: int = 0

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], cache_features: bool = False) -> torch.Tensor:
        """Forward pass of the AggregatorNet_Coral model.

        Parameters
        ----------
        x : tuple of (torch.Tensor, torch.Tensor)
            A tuple where:
            - x[0] (torch.Tensor): The input sequence tensor of shape
                (batch_size, seq_length, num_zernikes + 3).
            - x[1] (torch.Tensor): The mean tensor used for output adjustment.
        cache_features : bool, optional, default=False
            Whether to cache transformer features for domain adaptation.

        Returns
        -------
        torch.Tensor
            The transformed output tensor of shape (batch_size, num_zernikes).

        Notes
        -----
        - Input features are first projected from (num_zernikes + 3) to d_model dimensions.
        - The transformer encoder processes the projected features.
        - The last token's output is extracted and passed through a linear layer.
        - The mean correction (second element) is added to the final output.
        - If cache_features=True, transformer features are stored in self.transformer_features.
        """
        x_input, mean = x
        # Project input features to d_model dimensions
        x_projected = self.input_proj(x_input)
        # Pass through transformer
        x_tensor = self.transformer_encoder(x_projected)

        # Cache features for domain adaptation if requested
        if cache_features:
            # Use the last token's features before FC layer
            # Convert to float32 for SVD operations in DARE-GRAM loss
            # IMPORTANT: Do NOT detach here - we need gradients for DARE-GRAM loss training
            self.transformer_features = x_tensor[:, -1, :].float()

        x_tensor = x_tensor[:, -1, :]  # Take the last token's output
        x_tensor = self.fc(x_tensor)  # Predict the next token
        x_tensor += mean
        return x_tensor

    def dare_gram_loss(self, features_source: torch.Tensor, features_target: torch.Tensor) -> torch.Tensor:
        """Compute DARE-GRAM loss between source and target features.

        Parameters
        ----------
        features_source: torch.Tensor
            Source domain features of shape (batch_size, n_features).
        features_target: torch.Tensor
            Target domain features of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            DARE-GRAM alignment loss.
        """
        # Convert to float32 for SVD operations (not supported in half precision)
        features_source = features_source.float()
        features_target = features_target.float()

        batch_size, n_features = features_source.shape

        # Check for NaN or Inf values
        if torch.any(torch.isnan(features_source)) or torch.any(torch.isnan(features_target)):
            return torch.tensor(0.0, device=self.device)
        if torch.any(torch.isinf(features_source)) or torch.any(torch.isinf(features_target)):
            return torch.tensor(0.0, device=self.device)

        # Add bias term (ones column) to features
        A = torch.cat((torch.ones(batch_size, 1, dtype=torch.float32).to(self.device), features_source), 1)
        B = torch.cat((torch.ones(batch_size, 1, dtype=torch.float32).to(self.device), features_target), 1)

        # Compute covariance matrices
        cov_A = A.t() @ A
        cov_B = B.t() @ B

        # SVD to get eigenvalues
        _, L_A, _ = torch.linalg.svd(cov_A)
        _, L_B, _ = torch.linalg.svd(cov_B)

        # Normalize eigenvalues to get cumulative variance
        # Temporarily disable deterministic algorithms for cumsum (not supported on CUDA)
        is_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            if is_deterministic:
                torch.use_deterministic_algorithms(False, warn_only=False)
            eigen_A = torch.cumsum(L_A, dim=0) / L_A.sum()
            eigen_B = torch.cumsum(L_B, dim=0) / L_B.sum()
        finally:
            if is_deterministic:
                torch.use_deterministic_algorithms(True)

        # Determine rank k based on threshold
        # Use at least eigenvalue at index 1, but prefer threshold if reached
        T = self.hparams.threshold

        # Find index where cumulative variance reaches threshold
        if eigen_A[1] > T:
            T_A = eigen_A[1]
        else:
            T_A = T

        index_A = torch.argwhere(eigen_A <= T_A)
        if len(index_A) > 0:
            index_A_val = int(index_A[-1][0].item())
        else:
            index_A_val = 1

        if eigen_B[1] > T:
            T_B = eigen_B[1]
        else:
            T_B = T

        index_B = torch.argwhere(eigen_B <= T_B)
        if len(index_B) > 0:
            index_B_val = int(index_B[-1][0].item())
        else:
            index_B_val = 1

        k = max(index_A_val, index_B_val)

        # Ensure k is within valid range (avoid numerical issues)
        n_eigen = min(len(L_A), len(L_B))
        k = min(k, n_eigen - 1)  # Ensure k < n_eigen

        # Add safety check for numerical stability
        if L_A[0] < 1e-10 or L_B[0] < 1e-10:
            # Near-singular matrix, return small loss
            return torch.tensor(0.0, device=self.device)

        # Compute pseudo-inverse with low-rank regularization
        # Add minimum rtol to prevent numerical instability
        rtol_A = max((L_A[k] / L_A[0]).item(), 1e-6)
        rtol_B = max((L_B[k] / L_B[0]).item(), 1e-6)
        A_pinv = torch.linalg.pinv(cov_A, rtol=rtol_A)
        B_pinv = torch.linalg.pinv(cov_B, rtol=rtol_B)

        # Compute cosine similarity for angle alignment
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_distance = torch.dist(
            torch.ones(n_features + 1, dtype=torch.float32).to(self.device), cos_sim(A_pinv, B_pinv), p=1
        ) / (n_features + 1)

        # Compute scale alignment loss
        scale_loss = torch.dist(L_A[:k], L_B[:k], p=1) / k

        # Clamp losses to prevent extreme values
        cos_distance = torch.clamp(cos_distance, min=0.0, max=10.0)
        scale_loss = torch.clamp(scale_loss, min=0.0, max=100.0)

        # Combined DARE-GRAM loss
        dare_gram_loss = self.hparams.tradeoff_angle * cos_distance + self.hparams.tradeoff_scale * scale_loss

        # Final clamp to prevent explosion
        dare_gram_loss = torch.clamp(dare_gram_loss, min=0.0, max=100.0)

        return dare_gram_loss

    def exp_rise_flipped(self, loss, a=6.0):
        """Exponential rise function for DARE-GRAM scaling, flipped to give higher scale at lower loss.

        Uses exp_rise_x1 and exp_rise_x2 hyperparameters to control transition region.
        When loss <= x1: scale = 1.0 (full DARE-GRAM)
        When loss >= x2: scale = 0.0 (no DARE-GRAM)
        Between x1 and x2: exponential transition
        """
        # Convert loss to scalar
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        # Get hyperparameters (must be provided, will error if not)
        x1 = float(self.hparams.exp_rise_x1)
        x2 = float(self.hparams.exp_rise_x2)
        a = float(a)

        # Region 1: loss <= x1 → f = 0
        if loss_val <= x1:
            f = 0.0
        # Region 3: loss >= x2 → f = 1
        elif loss_val >= x2:
            f = 1.0
        # Region 2: x1 < loss < x2 → exponential rise
        else:
            t = (loss_val - x1) / (x2 - x1)
            f = (1 - np.exp(-a * t)) / (1 - np.exp(-a))

        # Flip: high f (when loss is high) becomes low scale (when loss is high)
        # We want: low loss (good) → high scale (more DARE-GRAM), high loss (bad) → low scale (less DARE-GRAM)
        f = -f + 1

        # Convert back to tensor on correct device
        return torch.tensor(f, dtype=torch.float32, device=self.device)

    def calc_losses(
        self,
        batch: tuple,
        batch_idx: int,
        use_coral: bool = False,
        add_dare_gram_to_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate losses with optional DARE-GRAM domain adaptation.

        Parameters
        ----------
        batch: tuple
            Batch of training data. Format depends on coral mode:
            - Without coral: (x, y) where x=(x_input, x_mean, filter_name, chipid)
            - With coral: (x, y) where x=(x_input, x_mean, filter_name, chipid,
                          coral_x_total, coral_x_mean, coral_filter, coral_chipid)
        batch_idx: int
            Batch index.
        use_coral: bool, default=False
            Whether to compute DARE-GRAM loss with coral/target data.
        add_dare_gram_to_loss: bool, default=True
            Whether to add DARE-GRAM loss to the total loss. If False, DARE-GRAM is computed
            for logging but not included in the total loss.

        Returns
        -------
        tuple
            (total_loss, mRSSE, dare_gram_loss, dare_gram_scale)
            dare_gram_loss is 0 if use_coral=False or coral data not available.
            dare_gram_scale is the scaling factor from exp_rise_flipped function.
        """
        x, y = batch  # y is the target token

        # Check if coral data is present (8 elements in x vs 4 elements)
        has_coral = len(x) == 8
        if has_coral:
            x_input, x_mean, filter_name, chipid, coral_x_total, coral_x_mean, coral_filter, coral_chipid = x
        else:
            x_input, x_mean, filter_name, chipid = x

        # Forward pass on source data
        logits = self.forward((x_input, x_mean), cache_features=True)
        source_features = self.transformer_features

        # Calculate regression loss
        regression_loss = self.loss_fn(logits, y)

        # Extract mRSSE for monitoring
        logits_converted = convert_zernikes_deploy(logits)
        y_converted = convert_zernikes_deploy(y)
        sse = F_loss.mse_loss(logits_converted, y_converted, reduction="none").sum(dim=-1)
        mRSSE = torch.sqrt(sse).mean()

        # DARE-GRAM loss if coral data is available
        # Skip computation entirely if dare_gram_weight is 0.0 for faster training
        dare_gram_loss = torch.tensor(0.0, device=self.device)
        if use_coral and has_coral and self.hparams.dare_gram_weight > 0.0:
            try:
                # Forward pass on target/coral data
                # Use eval mode to prevent BN updates (keep statistics source-domain only)
                # NOTE: eval() doesn't disable gradients, only affects batch norm/dropout
                was_training = self.training
                try:
                    self.eval()
                    _ = self.forward((coral_x_total, coral_x_mean), cache_features=True)
                    target_features = self.transformer_features
                finally:
                    if was_training:
                        self.train()

                # Compute DARE-GRAM loss (now has gradients since we removed .detach() from features)
                dare_gram_loss = self.dare_gram_loss(source_features, target_features)
            except (RuntimeError, ValueError, IndexError) as e:
                logger.warning(f"DARE-GRAM loss computation failed: {e}")
                # Use zero tensor (no gradients needed if loss computation failed)
                dare_gram_loss = torch.tensor(0.0, device=self.device)

        # Add DARE-GRAM to loss only if requested
        if add_dare_gram_to_loss:
            # Use epoch-averaged val_mRSSE if available, otherwise use current batch mRSSE
            # (for first training batch before first validation completes)
            mRSSE_for_scaling = self.val_mRSSE if self.val_mRSSE is not None else mRSSE
            scale_loss = self.exp_rise_flipped(mRSSE_for_scaling)
            total_loss = regression_loss + self.hparams.dare_gram_weight * scale_loss * dare_gram_loss
        else:
            total_loss = regression_loss
            # Still compute scale_loss for logging even if not added to loss
            mRSSE_for_scaling = self.val_mRSSE if self.val_mRSSE is not None else mRSSE
            scale_loss = self.exp_rise_flipped(mRSSE_for_scaling)

        return total_loss, mRSSE, dare_gram_loss, scale_loss

    def training_step(self, batch: tuple, batch_idx: int):
        """Perform a single training step with domain adaptation.

        Parameters
        ----------
        batch : tuple
            A tuple containing input data and targets.
        batch_idx : int
            The index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The computed loss for the batch.

        Notes
        -----
        - The model processes the batch using calc_losses with coral data.
        - DARE-GRAM loss is added to regression loss if coral data is available.
        - Training loss, mRSSE, and DARE-GRAM loss are logged for monitoring.
        """
        loss, mRSSE, dare_gram_loss, dare_gram_scale = self.calc_losses(batch, batch_idx, use_coral=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)
        self.log("train_dare_gram_loss", dare_gram_loss, sync_dist=True)
        self.log("train_dare_gram_scale", dare_gram_scale, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Parameters
        ----------
        batch : tuple
            A tuple containing input data and targets.
        batch_idx : int
            The index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The computed validation loss for the batch.

        Notes
        -----
        - DARE-GRAM is computed for logging but not added to validation loss.
        - This allows monitoring domain adaptation without affecting validation metrics.
        - Validation loss, mRSSE, and DARE-GRAM loss are logged for monitoring.
        - mRSSE is accumulated across batches to compute epoch average.
        """
        # Compute DARE-GRAM for logging but don't add it to validation loss
        loss, mRSSE, dare_gram_loss, dare_gram_scale = self.calc_losses(
            batch, batch_idx, use_coral=True, add_dare_gram_to_loss=False
        )
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mRSSE", mRSSE, prog_bar=True, sync_dist=True)
        self.log("val_dare_gram_loss", dare_gram_loss, sync_dist=True)
        self.log("val_dare_gram_scale", dare_gram_scale, sync_dist=True)

        # Accumulate mRSSE for epoch-average calculation
        mRSSE_detached = mRSSE.detach()
        if self._val_mRSSE_sum is None:
            self._val_mRSSE_sum = torch.zeros_like(mRSSE_detached)
            self._val_batch_count = 0
        self._val_mRSSE_sum += mRSSE_detached
        self._val_batch_count += 1

        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to compute epoch-average mRSSE.

        Computes the average validation mRSSE across all batches in the epoch
        and stores it for use in the next training epoch's DARE-GRAM scaling.
        """
        if self._val_mRSSE_sum is not None and self._val_batch_count > 0:
            # Store epoch-averaged mRSSE for use in next training epoch
            self.val_mRSSE = (self._val_mRSSE_sum / self._val_batch_count).clone().detach()
            # Reset accumulators for next epoch
            self._val_mRSSE_sum = None
            self._val_batch_count = 0

    def loss_fn(self, x, y):
        """Compute the loss using the Root Sum of Squared Errors (mRSSE).

        Parameters
        ----------
        x : torch.Tensor
            The predicted tensor of shape (batch_size, num_zernikes).
        y : torch.Tensor
            The target tensor of shape (batch_size, num_zernikes).

        Returns
        -------
        torch.Tensor
            The computed mean Root Sum of Squared Errors (mRSSE).

        Notes
        -----
        - The loss is calculated as the mean of the square root of
            the sum of squared errors.
        - Zernikes are converted to deployment format before computing loss.
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

    def configure_optimizers(self) -> Any:
        """Configure the optimizer with learning rate scheduling."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,  # Reduce LR by half when plateau
                    patience=5,  # Wait 5 validation checks (epochs) before reducing
                    threshold=1e-3,  # Minimum change to qualify as an improvement
                    threshold_mode="abs",  # Use absolute threshold
                    min_lr=1e-7,  # Minimum learning rate
                    verbose=True,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
