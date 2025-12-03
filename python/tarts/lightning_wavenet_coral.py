"""WaveNet with DARE-GRAM domain adaptation for coral data.

This module implements a WaveNet system with DARE-GRAM loss for unsupervised domain adaptation.
The method aligns inverse Gram matrices between source and target domains without requiring target labels.
"""

# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Third-party imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local/application imports
from .constants import (
    BAND_MEAN,
    BAND_STD,
    BAND_VALUES_TENSOR,
    CAMERA_TYPE,
    DEFAULT_INPUT_SHAPE,
    DEG_TO_RAD,
    FIELD_MEAN,
    FIELD_STD,
    INTRA_MEAN,
    INTRA_STD,
    ZERNIKE_SCALE_FACTOR,
)
from .utils import convert_zernikes_deploy
from .wavenet import WaveNet

logger = logging.getLogger(__name__)


class WaveNetSystem_Coral(pl.LightningModule):
    """WaveNet with DARE-GRAM domain adaptation.

    This system applies DARE-GRAM loss to align source and target (coral) features
    by aligning their inverse Gram matrices in a low-rank subspace.
    """

    def __init__(
        self,
        cnn_model: str = "resnet34",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
        n_zernikes: int = 25,
        alpha: float = 0,
        lr: float = 1e-3,
        lr_schedule: bool = False,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        pretrained: bool = False,
        tradeoff_angle: float = 0.05,
        tradeoff_scale: float = 0.001,
        threshold: float = 0.9,
        dare_gram_weight: float = 1.0,
    ) -> None:
        """Create the WaveNet with DARE-GRAM domain adaptation.

        Parameters
        ----------
        cnn_model: str, default="resnet34"
            The name of the pre-trained CNN model from torchvision or timm.
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256,)
            Number of nodes in the hidden layers of the Zernike predictor network.
        n_zernikes: int, default=25
            Number of Zernike coefficients to predict.
        alpha: float, default=0
            Weight for the L2 penalty.
        lr: float, default=1e-3
            The initial learning rate for AdamW.
        lr_schedule: bool, default=False
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        weight_decay: float, default=1e-4
            The weight decay (L2 penalty) coefficient for the AdamW optimizer.
        device: str, default='cuda'
            The device to use for computation ('cuda' or 'cpu').
        pretrained: bool, default=False
            Whether to use pre-trained CNN weights.
        tradeoff_angle: float, default=0.05
            Weight for the DARE-GRAM angle alignment loss.
        tradeoff_scale: float, default=0.001
            Weight for the DARE-GRAM scale alignment loss.
        threshold: float, default=0.9
            Cumulative variance threshold for low-rank approximation.
        dare_gram_weight: float, default=1.0
            Overall weight to scale the DARE-GRAM loss relative to regression loss.
            Higher values prioritize domain adaptation over regression accuracy.
        """
        super().__init__()
        self.save_hyperparameters()
        self.device_val = torch.device(device if torch.cuda.is_available() else "cpu")
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            n_predictor_layers=n_predictor_layers,
            n_zernikes=n_zernikes,
            device=str(self.device_val),
            pretrained=pretrained,
        )

        self.camType = CAMERA_TYPE
        self.inputShape = DEFAULT_INPUT_SHAPE
        self.val_mRSSE: torch.Tensor | None = None

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
        batch_size, n_features = features_source.shape

        # Check for NaN or Inf values
        if torch.any(torch.isnan(features_source)) or torch.any(torch.isnan(features_target)):
            return torch.tensor(0.0, device=self.device_val)
        if torch.any(torch.isinf(features_source)) or torch.any(torch.isinf(features_target)):
            return torch.tensor(0.0, device=self.device_val)

        # Add bias term (ones column) to features
        A = torch.cat((torch.ones(batch_size, 1).to(self.device_val), features_source), 1)
        B = torch.cat((torch.ones(batch_size, 1).to(self.device_val), features_target), 1)

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
            return torch.tensor(0.0, device=self.device_val)

        # Compute pseudo-inverse with low-rank regularization
        # Add minimum rtol to prevent numerical instability
        rtol_A = max((L_A[k] / L_A[0]).item(), 1e-6)
        rtol_B = max((L_B[k] / L_B[0]).item(), 1e-6)
        A_pinv = torch.linalg.pinv(cov_A, rtol=rtol_A)
        B_pinv = torch.linalg.pinv(cov_B, rtol=rtol_B)

        # Compute cosine similarity for angle alignment
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_distance = torch.dist(
            torch.ones(n_features + 1).to(self.device_val), cos_sim(A_pinv, B_pinv), p=1
        ) / (n_features + 1)

        # Compute scale alignment loss
        scale_loss = torch.dist(L_A[:k], L_B[:k], p=1) / k

        # Clamp losses to prevent extreme values (helps with train/val differences)
        cos_distance = torch.clamp(cos_distance, min=0.0, max=10.0)
        scale_loss = torch.clamp(scale_loss, min=0.0, max=100.0)

        # Combined DARE-GRAM loss
        dare_gram_loss = self.hparams.tradeoff_angle * cos_distance + self.hparams.tradeoff_scale * scale_loss

        # Final clamp to prevent explosion
        dare_gram_loss = torch.clamp(dare_gram_loss, min=0.0, max=100.0)

        return dare_gram_loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Zernikes and return with truth."""
        img = batch["image"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        band = batch["band"]
        zk_true = batch["zernikes"].to(self.device_val)

        zk_pred = self.wavenet(img, fx, fy, intra, band)

        return zk_pred, zk_true

    def exp_rise_flipped(self, loss, a=6.0):
        """Exponentially rises from 0 at loss=0.13 to 1 at loss=0.16.

        Then flattens at 1 above 0.16 and 0 below 0.13.

        Parameters
        ----------
        loss: torch.Tensor or float
            The loss value to compute the exponential rise function for.
        a: float
            Controls steepness (larger = sharper rise).
        """
        # Convert loss to tensor and ensure float32 dtype
        loss = torch.as_tensor(loss, dtype=torch.float32, device=self.device_val)

        # Ensure scalar values are float32 tensors
        x1 = torch.tensor(0.13, dtype=torch.float32, device=loss.device)
        x2 = torch.tensor(0.16, dtype=torch.float32, device=loss.device)
        a = torch.tensor(a, dtype=torch.float32, device=loss.device)

        f = torch.zeros_like(loss, dtype=torch.float32)

        # Region 1: loss <= 0.13 → f = 0
        f[loss <= x1] = 0.0

        # Region 2: 0.13 < loss < 0.16 → exponential rise
        mask = (loss > x1) & (loss < x2)
        t = (loss[mask] - x1) / (x2 - x1)
        f[mask] = (1 - torch.exp(-a * t)) / (1 - torch.exp(-a))

        # Region 3: loss >= 0.16 → f = 1
        f[loss >= x2] = 1.0
        f = -f + 1
        return f

    def calc_losses(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        use_coral: bool = False,
        add_dare_gram_to_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict Zernikes and calculate losses with optional DARE-GRAM.

        Parameters
        ----------
        batch: Dict[str, Any]
            Batch of training data. If use_coral=True, should contain coral data.
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
            (total_loss, mRSSE, dare_gram_loss)
            dare_gram_loss is 0 if use_coral=False.
        """
        zk_pred, zk_true = self.predict_step(batch, batch_idx)

        # Convert to FWHM contributions
        zk_pred = convert_zernikes_deploy(zk_pred)
        zk_true = convert_zernikes_deploy(zk_true)

        # Pull out the weights from the final linear layer
        *_, A, _ = self.wavenet.predictor.parameters()

        # Calculate regression loss
        sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
        regression_loss = sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

        # DARE-GRAM loss if coral data is available
        dare_gram_loss = torch.tensor(0.0, device=self.device_val)
        if use_coral and "coral_image" in batch:
            try:
                # Forward pass on source data to get features
                zk_pred_s, _ = self.predict_step(batch, batch_idx)

                # Forward pass on target/coral data
                coral_img = batch["coral_image"]
                coral_fx = batch["coral_field_x"]
                coral_fy = batch["coral_field_y"]
                coral_intra = batch["coral_intrafocal"]
                coral_band = batch["coral_band"]

                # Always use eval mode for coral forward to prevent BN updates from coral data
                # This keeps BN statistics source-domain only regardless of dare_gram_weight
                # Features are kept attached - dare_gram_weight multiplication handles gradient contribution
                source_features = self.wavenet.predictor_features
                was_training = self.wavenet.training
                try:
                    self.wavenet.eval()  # BN uses running stats, not batch stats
                    _ = self.wavenet(coral_img, coral_fx, coral_fy, coral_intra, coral_band)
                    target_features = self.wavenet.predictor_features
                finally:
                    if was_training:
                        self.wavenet.train()

                # Compute DARE-GRAM loss for logging
                dare_gram_loss = self.dare_gram_loss(source_features, target_features)
            except (RuntimeError, ValueError, IndexError) as e:
                logger.warning(f"DARE-GRAM loss computation failed: {e}")
                dare_gram_loss = torch.tensor(0.0, device=self.device_val)

        # Add DARE-GRAM to loss only if requested
        if add_dare_gram_to_loss:
            scale_loss = self.exp_rise_flipped(self.val_mRSSE if self.val_mRSSE is not None else mRSSE)
            total_loss = regression_loss + self.hparams.dare_gram_weight * scale_loss * dare_gram_loss
        else:
            total_loss = regression_loss

        return total_loss, mRSSE, dare_gram_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        loss, mRSSE, dare_gram_loss = self.calc_losses(batch, batch_idx, use_coral=True)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)
        self.log("train_dare_gram_loss", dare_gram_loss, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        # Compute DARE-GRAM for logging but don't add it to validation loss
        # This allows monitoring domain adaptation without affecting validation metrics
        loss, mRSSE, dare_gram_loss = self.calc_losses(
            batch, batch_idx, use_coral=True, add_dare_gram_to_loss=False
        )
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True)
        self.log("val_dare_gram_loss", dare_gram_loss, sync_dist=True)
        self.val_mRSSE = mRSSE.clone().detach()  # Store a copy to avoid tensor reference issues
        return loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        if self.hparams.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer),
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def get_band_values(self, bands: torch.Tensor) -> torch.Tensor:
        """Retrieve band values for a batch of indices."""
        return BAND_VALUES_TENSOR.to(self.device_val)[bands]

    def rescale_image_batched(self, data):
        """Rescale batched image data with normalization."""
        shape = data.shape
        B = shape[0]
        batch = data.view(B, -1)
        min_vals = batch.min(dim=1, keepdim=True).values
        max_vals = batch.max(dim=1, keepdim=True).values
        min_vals = min_vals.view([B] + [1] * (data.dim() - 1))
        max_vals = max_vals.view([B] + [1] * (data.dim() - 1))
        data = data - min_vals
        data = data / (max_vals + 1e-8)
        means = data.view(data.shape[0], -1).mean(dim=1)
        stds = data.view(data.shape[0], -1).std(dim=1)
        means = means.view(-1, 1, 1)
        stds = stds.view(-1, 1, 1)
        data = (data - means) / (stds + 1e-8)
        return data

    def forward(
        self,
        img: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        focalFlag: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Predict zernikes for production."""
        img = self.rescale_image_batched(img)

        fx *= DEG_TO_RAD
        fy *= DEG_TO_RAD

        fx = (fx - FIELD_MEAN) / FIELD_STD
        fy = (fy - FIELD_MEAN) / FIELD_STD

        focalFlag = (focalFlag - INTRA_MEAN) / INTRA_STD

        band = self.get_band_values(band)[:, 0]
        band = (band - BAND_MEAN) / BAND_STD

        zk_pred = self.wavenet(img, fx, fy, focalFlag, band)

        zk_pred *= ZERNIKE_SCALE_FACTOR

        return zk_pred
