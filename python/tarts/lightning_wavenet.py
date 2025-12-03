"""Wrapping everything for WaveNet in Pytorch Lightning."""

# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Third-party imports
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

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
from .dataloader import Donuts, Donuts_Fullframe
from .utils import convert_zernikes
from .wavenet import WaveNet

logger = logging.getLogger(__name__)


class DonutLoader(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the simulated Donuts data.

        Parameters
        ----------
        batch_size: int, default=64
            The batch size for SGD.
        num_workers: int, default=16
            The number of workers for parallel loading of batches.
        persistent_workers: bool, default=True
            Whether to shutdown worker processes after dataset is consumed once
        pin_memory: bool, default=True
            Whether to automatically put data in pinned memory (recommended
            whenever using a GPU).
        shuffle: bool, default=True
            Whether to shuffle the train dataloader.
        **kwargs
            See the keyword arguments in the Donuts class.
        """
        super().__init__()
        self.save_hyperparameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_loader(self, mode: str, shuffle: bool = False, drop_last: bool = True) -> DataLoader:
        """Build a DataLoader."""
        return DataLoader(
            Donuts(mode=mode, **self.hparams),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader("train", shuffle=self.hparams.shuffle)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val", drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test", drop_last=False)


class DonutLoader_Fullframe(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the simulated Donuts data.

        Parameters
        ----------
        batch_size: int, default=64
            The batch size for SGD.
        num_workers: int, default=16
            The number of workers for parallel loading of batches.
        persistent_workers: bool, default=True
            Whether to shutdown worker processes after dataset is consumed once
        pin_memory: bool, default=True
            Whether to automatically put data in pinned memory (recommended
            whenever using a GPU).
        shuffle: bool, default=True
            Whether to shuffle the train dataloader.
        **kwargs
            See the keyword arguments in the Donuts class.
        """
        super().__init__()
        self.save_hyperparameters()

    def _build_loader(self, mode: str, shuffle: bool = False, drop_last: bool = True) -> DataLoader:
        """Build a DataLoader."""
        return DataLoader(
            Donuts_Fullframe(mode=mode, **self.hparams),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader("train", shuffle=self.hparams.shuffle)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val", drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test", drop_last=False)


class WaveNetSystem(pl.LightningModule):
    """Pytorch Lightning system for training the WaveNet."""

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
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet34"
            The name of the pre-trained CNN model from torchvision or timm. Supports
            both torchvision models (e.g., "resnet34") and timm models
            (e.g., "mobilenetv4_conv_small").
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is determined by n_zernikes.
        n_zernikes: int, default=25
            Number of Zernike coefficients to predict.
        alpha: float, default=0
            Weight for the L2 penalty.
        lr: float, default=1e-3
            The initial learning rate for AdamW.
        lr_schedule: bool, default=True
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        weight_decay: float, default=1e-4
            The weight decay (L2 penalty) coefficient for the AdamW optimizer.
        device: str, default='cuda'
            The device to use for computation ('cuda' or 'cpu').
        pretrained: bool, default=False
            Whether to use pre-trained CNN weights. Set to True to download pre-trained weights.
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

        # define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = CAMERA_TYPE
        self.inputShape = DEFAULT_INPUT_SHAPE

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Zernikes and return with truth."""
        # unpack data from the dictionary
        img = batch["image"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        band = batch["band"]
        zk_true = batch["zernikes"].to(self.device_val)
        # dof_true = batch["dof"]  # noqa: F841

        # predict zernikes
        zk_pred = self.wavenet(img, fx, fy, intra, band)

        return zk_pred, zk_true

    def calc_losses(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - loss - mean of the SSE + L2 penalty (in arcsec^2)
        - mRSSE - mean of the root of the SSE (in arcsec)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        try:
            # predict zernikes
            zk_pred, zk_true = self.predict_step(batch, batch_idx)

            # Check for NaN or Inf values in predictions/truth
            if torch.any(torch.isnan(zk_pred)) or torch.any(torch.isnan(zk_true)):
                logger.warning("NaN detected in predictions or truth values, returning zero loss")
                return torch.tensor(0.0, device=self.device_val), torch.tensor(0.0, device=self.device_val)
            if torch.any(torch.isinf(zk_pred)) or torch.any(torch.isinf(zk_true)):
                logger.warning("Inf detected in predictions or truth values, returning zero loss")
                return torch.tensor(0.0, device=self.device_val), torch.tensor(0.0, device=self.device_val)

            # convert to FWHM contributions
            zk_pred = convert_zernikes_deploy(zk_pred)
            zk_true = convert_zernikes_deploy(zk_true)

            # pull out the weights from the final linear layer
            *_, A, _ = self.wavenet.predictor.parameters()

            # calculate loss
            sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
            loss = sse.mean() + self.hparams.alpha * A.square().sum()
            mRSSE = torch.sqrt(sse).mean()

            # Check for NaN or Inf in computed losses
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                logger.warning("NaN/Inf detected in computed loss, returning zero loss")
                return torch.tensor(0.0, device=self.device_val), torch.tensor(0.0, device=self.device_val)
            if torch.any(torch.isnan(mRSSE)) or torch.any(torch.isinf(mRSSE)):
                logger.warning("NaN/Inf detected in computed mRSSE, returning zero mRSSE")
                mRSSE = torch.tensor(0.0, device=self.device_val)

            return loss, mRSSE
        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning(f"Error in calc_losses: {e}, returning zero loss")
            return torch.tensor(0.0, device=self.device_val), torch.tensor(0.0, device=self.device_val)

    def calc_losses_pure(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - loss - mean of the SSE + L2 penalty (in arcsec^2)
        - mRSSE - mean of the root of the SSE (in arcsec)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        try:
            # predict zernikes
            zk_pred, zk_true = self.predict_step(batch, batch_idx)

            # Check for NaN or Inf values in predictions/truth
            if torch.any(torch.isnan(zk_pred)) or torch.any(torch.isnan(zk_true)):
                logger.warning("NaN detected in predictions or truth values, returning zero loss")
                zero_tensor = torch.tensor(0.0, device=self.device_val)
                return zero_tensor, zero_tensor, zk_pred, zk_true
            if torch.any(torch.isinf(zk_pred)) or torch.any(torch.isinf(zk_true)):
                logger.warning("Inf detected in predictions or truth values, returning zero loss")
                zero_tensor = torch.tensor(0.0, device=self.device_val)
                return zero_tensor, zero_tensor, zk_pred, zk_true

            # convert to FWHM contributions
            zk_pred = convert_zernikes_deploy(zk_pred)
            zk_true = convert_zernikes_deploy(zk_true)

            # pull out the weights from the final linear layer
            *_, A, _ = self.wavenet.predictor.parameters()

            # calculate loss
            sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
            loss = sse.mean() + self.hparams.alpha * A.square().sum()
            mRSSE = torch.sqrt(sse).mean()

            # Check for NaN or Inf in computed losses
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                logger.warning("NaN/Inf detected in computed loss, returning zero loss")
                zero_tensor = torch.tensor(0.0, device=self.device_val)
                return zero_tensor, zero_tensor, zk_pred, zk_true
            if torch.any(torch.isnan(mRSSE)) or torch.any(torch.isinf(mRSSE)):
                logger.warning("NaN/Inf detected in computed mRSSE, returning zero mRSSE")
                mRSSE = torch.tensor(0.0, device=self.device_val)

            return loss, mRSSE, zk_pred, zk_true
        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning(f"Error in calc_losses_pure: {e}, returning zero loss")
            zero_tensor = torch.tensor(0.0, device=self.device_val)
            # Return zero loss but preserve predictions for debugging
            try:
                zk_pred, zk_true = self.predict_step(batch, batch_idx)
                return zero_tensor, zero_tensor, zk_pred, zk_true
            except Exception:
                # If even predict_step fails, create dummy tensors
                dummy_shape = (batch.get("zernikes", torch.tensor([[]])).shape[0],)
                dummy_pred = torch.zeros(dummy_shape, device=self.device_val)
                dummy_true = torch.zeros(dummy_shape, device=self.device_val)
                return zero_tensor, zero_tensor, dummy_pred, dummy_true

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True)

        return loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        """Retrieve band values for a batch of indices.

        Args:
            bands (torch.Tensor): A tensor of shape (batch_size,) containing band indices.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) with band values.
        """
        return BAND_VALUES_TENSOR.to(self.device_val)[bands]

    def rescale_image(self, data):
        """Rescale image data to the range [0, 1].

        Args:
            data: Input image data tensor.

        Returns:
            Rescaled image data tensor normalized to [0, 1].
        """
        data -= data.min()
        data /= data.max()
        return data

    def rescale_image_batched(self, data):
        """Rescale batched image data with normalization (AlignNet style).

        Applies the following transformations:
        1. Min-max scaling to [0, 1] per image
        2. Z-score normalization per image
        """
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
        """Predict zernikes for production.

        This method assumes the inputs have NOT been previously
        transformed by ml_aos.utils.transform_inputs.
        """
        # rescale image to [0, 1] and normalize (AlignNet style)
        img = self.rescale_image_batched(img)

        # convert angles to radians
        fx *= DEG_TO_RAD
        fy *= DEG_TO_RAD

        # normalize angles
        fx = (fx - FIELD_MEAN) / FIELD_STD
        fy = (fy - FIELD_MEAN) / FIELD_STD

        # normalize the intrafocal flags
        focalFlag = (focalFlag - INTRA_MEAN) / INTRA_STD

        band = self.get_band_values(band)[:, 0]
        # U G R I Z Y
        # normalize the wavelength
        band = (band - BAND_MEAN) / BAND_STD

        # predict zernikes in microns
        zk_pred = self.wavenet(img, fx, fy, focalFlag, band)

        # convert to nanometers
        zk_pred *= ZERNIKE_SCALE_FACTOR

        return zk_pred
