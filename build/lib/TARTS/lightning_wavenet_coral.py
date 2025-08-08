"""Wrapping everything for WaveNet in Pytorch Lightning."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import vmap
from .dataloader import Donuts, Donuts_Fullframe
from .utils import convert_zernikes_deploy, CORALLoss
from .wavenet import WaveNet


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

    def _build_loader(
        self, mode: str, shuffle: bool = False, drop_last: bool = True
    ) -> DataLoader:
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
        coral_mode: bool = False,
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
        coral_mode: bool, default=False
            Whether to enable coral mode for domain adaptation.
        **kwargs
            See the keyword arguments in the Donuts class.
        """
        super().__init__()
        self.save_hyperparameters()

    def _build_loader(
        self, mode: str, shuffle: bool = False, drop_last: bool = True
    ) -> DataLoader:
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
        alpha: float = 0,
        coral_lambda: float = 0,
        lr: float = 1e-3,
        lr_schedule: bool = False,
        coral_mode: bool = False,
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet18"
            The name of the pre-trained CNN model from torchvision.
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is fixed to 19.
        alpha: float, default=0
            Weight for the L2 penalty.
        coral_lambda: float, default=0
            Weight for the CORAL loss component.
        lr: float, default=1e-3
            The initial learning rate for Adam.
        lr_schedule: bool, default=True
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        coral_mode: bool, default=False
            Whether to enable coral mode for domain adaptation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            n_predictor_layers=n_predictor_layers,
        )

        # define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"
        self.inputShape = (160, 160)
        self.coral_mode = coral_mode
        self.coral_lambda = coral_lambda
        if self.coral_mode:
            self.coral_loss = CORALLoss()
        self.device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_step(
        self, batch: dict, batch_idx: int,
        val: bool = False,
    ) -> tuple:
        """Predict Zernikes and return with truth."""
        # unpack data from the dictionary
        img = batch["image"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        band = batch["band"]
        zk_true = batch["zernikes"].cuda()
        # dof_true = batch["dof"]  # noqa: F841

        # predict zernikes
        zk_pred = self.wavenet(img, fx, fy, intra, band)

        if self.coral_mode:
            img_coral = batch["coral_image"]
            img_coral_transform = self.wavenet._reshape_image(img_coral)
            ft = self.wavenet.cnn(img_coral_transform)
            img_transform = self.wavenet._reshape_image(img)
            fs = self.wavenet.cnn(img_transform)
            if val and self.coral_mode:
                img = batch["coral_image"]
                fx = batch["coral_field_x"]
                fy = batch["coral_field_y"]
                intra = batch["coral_intrafocal"]
                band = batch["coral_band"]
                zk_true_coral = batch["coral_zernikes"].cuda()
                zk_pred_coral = self.wavenet(img, fx, fy, intra, band)
                return zk_pred, zk_true, zk_pred_coral, zk_true_coral, ft, fs
            else:
                return zk_pred, zk_true, ft, fs
        else:
            return zk_pred, zk_true

    def get_coral_lambda(self, loss: float) -> float:
        """Calculate the current coral lambda based on warmup schedule.

        Parameters
        ----------
        loss: float
            The current loss value.

        Returns
        -------
        float
            The current coral lambda value.
        """
        if loss < 0.14:
            return self.coral_lambda
        else:
            return 0

    def calc_losses(self, batch: dict, batch_idx: int, val: bool = False) -> tuple:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - loss - mean of the SSE + L2 penalty (in arcsec^2)
        - mRSSE - mean of the root of the SSE (in arcsec)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        # predict zernikes
        if self.coral_mode:

            if val:
                zk_pred, zk_true, zk_pred_coral, zk_true_coral, ft, fs = self.predict_step(batch, batch_idx, val)
            else:
                zk_pred, zk_true, ft, fs = self.predict_step(batch, batch_idx, val)
        else:
            zk_pred, zk_true = self.predict_step(batch, batch_idx, val)

        # convert to FWHM contributions
        zk_pred = convert_zernikes_deploy(zk_pred)
        zk_true = convert_zernikes_deploy(zk_true)

        if val and self.coral_mode:
            zk_pred_coral = convert_zernikes_deploy(zk_pred_coral)
            zk_true_coral = convert_zernikes_deploy(zk_true_coral)
            sse_coral = F.mse_loss(zk_pred_coral, zk_true_coral, reduction="none").sum(dim=-1)
            sse_coral = sse_coral.mean()
            mRSSE_coral = torch.sqrt(sse_coral).mean()
        # pull out the weights from the final linear layer
        *_, A, _ = self.wavenet.predictor.parameters()

        # calculate loss
        sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
        # loss = sse.mean() + self.hparams.alpha * A.square().sum()
        loss = 0 * sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

        # coral_lambda = self.get_coral_lambda_multi_stage(mRSSE)
        # self.log("lambda", coral_lambda, sync_dist=True)
        if self.coral_mode:
            # Use feature-based coral loss for training, zernike-based for validation
            if val:
                coral_loss = self.coral_loss(zk_pred_coral, zk_true_coral)
                # loss += coral_lambda * coral_loss
                loss += sse_coral
                return loss, mRSSE, coral_loss, sse_coral, mRSSE_coral
            else:
                # For training, use feature-based coral loss
                coral_loss = self.coral_loss(zk_pred_coral, zk_true_coral)
                # loss += coral_lambda * coral_loss
                loss += sse_coral
                return loss, mRSSE, coral_loss
        else:
            return loss, mRSSE

    def calc_losses_pure(self, batch: dict, batch_idx: int) -> tuple:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - loss - mean of the SSE + L2 penalty (in arcsec^2)
        - mRSSE - mean of the root of the SSE (in arcsec)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        # predict zernikes
        zk_pred, zk_true = self.predict_step(batch, batch_idx)

        # convert to FWHM contributions
        zk_pred = convert_zernikes_deploy(zk_pred, device=self.device)
        zk_true = convert_zernikes_deploy(zk_true, device=self.device)

        # pull out the weights from the final linear layer
        *_, A, _ = self.wavenet.predictor.parameters()

        # calculate loss
        sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
        loss = sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

        return loss, mRSSE, zk_pred, zk_true

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        # Fix: Use val=False for training step
        if self.coral_mode:
            loss, mRSSE, coral_loss, sse_coral, mRSSE_coral = self.calc_losses(batch, batch_idx, val=True)
            self.log("train_loss", loss, sync_dist=True, prog_bar=True)
            self.log("train_coral_loss", coral_loss, sync_dist=True)
            self.log("train_mRSSE", mRSSE, sync_dist=True)
            self.log("train_sse_coral", sse_coral, sync_dist=True)
            self.log("train_mRSSE_coral", mRSSE_coral, sync_dist=True)
        else:
            loss, mRSSE = self.calc_losses(batch, batch_idx, val=False)
            self.log("train_loss", loss, sync_dist=True, prog_bar=True)
            self.log("train_mRSSE", mRSSE, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss, mRSSE, coral_loss, sse_coral, mRSSE_coral = self.calc_losses(batch, batch_idx, val=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_coral_loss", coral_loss, sync_dist=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True)
        self.log("val_sse_coral", sse_coral, sync_dist=True)
        self.log("val_mRSSE_coral", mRSSE_coral, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
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
        # Create a tensor with band values
        band_values = torch.tensor([
            [0.3671],
            [0.4827],
            [0.6223],
            [0.7546],
            [0.8691],
            [0.9712]
        ]).to(self.device_val)

        return band_values[bands]

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

    def get_coral_lambda_multi_stage(self, loss) -> float:
        """Multi-stage lambda schedule with different behaviors in different loss ranges.

        Stage 1 (loss > 0.3): No CORAL (focus on main task)
        Stage 2 (0.15 < loss <= 0.3): Gradual CORAL introduction
        Stage 3 (0.1 < loss <= 0.15): Full CORAL
        Stage 4 (loss <= 0.1): Reduced CORAL (fine-tuning)

        Parameters
        ----------
        loss: float or torch.Tensor
            Current mRSSE loss
        """
        # Convert tensor to float if needed
        if hasattr(loss, 'item'):
            loss = loss.item()

        if loss > 0.3:
            # Stage 1: No domain adaptation
            return 0.0
        elif loss > 0.15:
            # Stage 2: Linear ramp-up
            progress = (0.3 - loss) / (0.3 - 0.15)
            return self.coral_lambda * progress * 0.5  # Start with half strength
        elif loss > 0.1:
            # Stage 3: Full strength
            return self.coral_lambda
        else:
            # Stage 4: Gradual reduction for fine-tuning
            reduction_factor = max(0.2, (loss - 0.05) / (0.1 - 0.05))
            return self.coral_lambda * reduction_factor

    def rescale_image_batched(self, data):
        """Apply rescale_image to a batch of images using vectorized mapping.

        Args:
            data: Batch of image data tensors.

        Returns:
            Batch of rescaled image data tensors.
        """
        return vmap(self.rescale_image)(data)

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

        # normalize image
        image_mean = 0.347
        image_std = 0.226
        img = (img - image_mean) / image_std

        # convert angles to radians
        fx *= torch.pi / 180
        fy *= torch.pi / 180

        # normalize angles
        field_mean = 0.000
        field_std = 0.021
        fx = (fx - field_mean) / field_std
        fy = (fy - field_mean) / field_std

        # normalize the intrafocal flags
        intra_mean = 0.5
        intra_std = 0.5
        focalFlag = (focalFlag - intra_mean) / intra_std

        band = self.get_band_values(band)[:, 0]
        # U G R I Z Y
        # normalize the wavelength
        band_mean = 0.710
        band_std = 0.174
        band = (band - band_mean) / band_std

        # predict zernikes in microns
        zk_pred = self.wavenet(img, fx, fy, focalFlag, band)

        # convert to nanometers
        zk_pred *= 1_000

        return zk_pred
