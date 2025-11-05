"""Wrapping everything for WaveNet in Pytorch Lightning."""

# Standard library imports
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
)
from .alignnet import AlignNet
from .dataloader import Donuts


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


class AlignNetSystem(pl.LightningModule):
    """Pytorch Lightning system for training the AlignNet."""

    def __init__(
        self,
        cnn_model: str = "mobilenetv4_conv_small",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
        alpha: float = 0,
        lr: float = 1e-3,
        lr_schedule: bool = False,
        device="cuda",
        pretrained: bool = False,
    ) -> None:
        """Initialize the AlignNet model.

        This initializes the AlignNet architecture, including loading a pre-trained
        CNN model as a feature extractor and setting up the predictor network for donut offsets.


        Parameters
        ----------
        cnn_model : str, optional, default="mobilenetv4_conv_small"
            The name of the pre-trained CNN model from torchvision or timm to be used
            as the feature extractor. Common options include "resnet18", "resnet34",
            "mobilenetv4_conv_small", etc.

        freeze_cnn : bool, optional, default=False
            If True, the CNN weights will be frozen during training, meaning they will not be updated.

        n_predictor_layers : tuple of int, optional, default=(256,)
            The number of nodes in each hidden layer of the Zernike predictor network.
            The output layer is fixed to have 19 nodes (the number of Zernike coefficients).

        alpha : float, optional, default=0
            The weight for the L2 penalty (regularization term) applied during training.
            A value of 0 disables the regularization.

        lr : float, optional, default=1e-3
            The initial learning rate used by the Adam optimizer.

        lr_schedule : bool, optional, default=False
            If True, a learning rate scheduler (ReduceLROnPlateau) will be used to adjust the learning rate
            based on the validation loss during training.

        device : str, optional, default='cuda'
            The device to use for computation ('cuda' or 'cpu').

        pretrained : bool, optional, default=False
            Whether to use pre-trained CNN weights. Set to True to download pre-trained weights.

        """
        super().__init__()
        self.save_hyperparameters()
        self.device_val = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alignnet = AlignNet(
            cnn_model=cnn_model,
            n_predictor_layers=n_predictor_layers,
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
        true_offset = batch["offset_vec"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        band = batch["band"]
        # predict zernikes
        pred_offset = self.alignnet(img, fx, fy, intra, band)

        return pred_offset, true_offset

    def calc_losses(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - loss - mean of the SSE + L2 penalty (in pixel^2)
        - mRSSE - mean of the root of the SSE (in pixel)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        # predict zernikes
        pred_offset, true_offset = self.predict_step(batch, batch_idx)

        # pull out the weights from the final linear layer
        *_, A, _ = self.alignnet.predictor.parameters()

        # calculate loss
        sse = F.mse_loss(pred_offset, true_offset, reduction="none").sum(dim=-1)
        loss = sse.mean() + self.hparams.alpha * A.square().sum()
        # Clamp SSE to prevent sqrt of negative values due to numerical errors
        sse_clamped = torch.clamp(sse, min=0.0)
        mRSSE = torch.sqrt(sse_clamped).mean()

        return loss, mRSSE

    def calc_losses_pure(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - loss - mean of the SSE + L2 penalty (in pixel^2)
        - mRSSE - mean of the root of the SSE (in pixel)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        # predict zernikes
        pred_offset, true_offset = self.predict_step(batch, batch_idx)

        # pull out the weights from the final linear layer
        *_, A, _ = self.alignnet.predictor.parameters()

        # calculate loss
        sse = F.mse_loss(pred_offset, true_offset, reduction="none").sum(dim=-1)
        loss = sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

        return loss, mRSSE, pred_offset, true_offset

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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

    def rescale_image_batched(self, data: torch.Tensor) -> torch.Tensor:
        """Rescale batched image data with normalization.

        Parameters
        ----------
        data : torch.Tensor
            Input batch of images to rescale.

        Returns
        -------
        torch.Tensor
            Rescaled and normalized image batch.

        Notes
        -----
        Applies the following transformations:
        1. Min-max scaling to [0, 1] per image
        2. Z-score normalization with fixed mean=0.347, std=0.226
        """
        # mean = 0.347
        # std = 0.226

        shape = data.shape
        B = shape[0]

        batch = data.view(B, -1)
        min_vals = batch.min(dim=1, keepdim=True).values
        max_vals = batch.max(dim=1, keepdim=True).values

        min_vals = min_vals.view([B] + [1] * (data.dim() - 1))
        max_vals = max_vals.view([B] + [1] * (data.dim() - 1))

        data = data - min_vals
        data = data / (max_vals + 1e-8)
        means = data.view(data.shape[0], -1).mean(dim=1)  # shape: [n]
        stds = data.view(data.shape[0], -1).std(dim=1)  # shape: [n]

        # Reshape for proper broadcasting with 4D tensor
        means = means.view(-1, 1, 1)  # shape: [n, 1, 1, 1]
        stds = stds.view(-1, 1, 1)  # shape: [n, 1, 1, 1]

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
        # rescale image to [0, 1]
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
        # normalize the wavelength
        band = (band - BAND_MEAN) / BAND_STD

        # predict zernikes in microns
        offset = self.alignnet(img, fx, fy, focalFlag, band)
        return offset
