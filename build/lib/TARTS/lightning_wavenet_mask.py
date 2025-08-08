"""Wrapping everything for WaveNet in Pytorch Lightning."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from .dataloader import Donuts, Donuts_Fullframe
from .utils import convert_zernikes_deploy
from .wavenet import WaveNet
from NeuralAOS.KERNEL import CUTOUT as DONUT
import torch.nn as nn
from NeuralAOS.unet_mask import UNetMask


class DynamicDonutKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_generator = UNetMask(n_channels=1, temperature=0.5)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        x = x[:, None, :, :]
        # Handle multi-channel input by taking the first channel
        if x.shape[1] > 1:
            print(f"Warning: Input has {x.shape[1]} channels, using first channel only")
            x = x[:, 0:1, :, :]  # Take only the first channel
        
        # Generate dynamic mask
        mask = self.mask_generator(x)
        
        # Check for NaN values in mask and replace with zeros
        if torch.isnan(mask).any():
            print("Warning: NaN detected in mask, replacing with zeros")
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        # output = mask[:,0,:,:]
        # # Apply convolution
       
        # conv_output = self.conv(x)
        
        # # Check for NaN values in conv_output and replace with zeros
        # if torch.isnan(conv_output).any():
        #     print("Warning: NaN detected in conv_output, replacing with zeros")
        #     conv_output = torch.where(torch.isnan(conv_output), torch.zeros_like(conv_output), conv_output)
        
        # # Apply dynamic mask
        # output = (x - mask)[:, 0, :, :]
        output = x[:,0,:,:]
        # background = x - x * mask
        
        # # Final output
        # output = (masked_output + background)[:, 0, :, :]
        
        # # Final NaN check
        # if torch.isnan(output).any():
        #     print("Warning: NaN detected in final output, replacing with zeros")
        #     output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        return output
    
    def get_learned_weights(self):
        """Get the learned convolution kernel weights.
        
        Returns
        -------
        torch.Tensor
            The learned convolution kernel weights.
        """
        return self.conv.weight.detach().cpu()
    
    def get_masked_weights(self):
        """Get the convolution kernel weights (same as learned weights for conv).
        
        Returns
        -------
        torch.Tensor
            The convolution kernel weights.
        """
        return self.conv.weight.detach().cpu()
    
    def get_mask(self):
        """Get the current mask (placeholder for compatibility).
        
        Returns
        -------
        torch.Tensor
            A placeholder mask tensor.
        """
        # Return a placeholder mask since dynamic masks are generated per image
        return torch.ones(160, 160).detach().cpu()


# class DonutKernel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         mask = DONUT[40:200, 40:200].float()
#         # Register the binary mask as a buffer (non-trainable)
#         self.register_buffer('mask', mask)
#         # Create a learnable 21x21 convolution kernel
#         self.conv = nn.Conv2d(
#             in_channels=1,
#             out_channels=1, 
#             kernel_size=3, 
#             padding=1,  # padding to maintain spatial dimensions
#             bias=False
#         )
        
#         # Initialize convolution weights
#         nn.init.ones_(self.conv.weight)

#     def forward(self, x):
#         # Apply convolution to the entire input image
#         x = x[:,None,:,:]
#         # x_clone = x.clone()
#         # background = x_clone - x_clone * self.mask # background
#         # cutout = self.conv(x) * self.mask 
#         # vals = (cutout + background)[:,0,:,:]
#         vals = self.conv(x)[:,0,:,:]
#         return vals

#     def extra_repr(self):
#         return f'DonutKernel(conv_kernel=21x21)'
    
#     def get_learned_weights(self):
#         """Get the learned convolution kernel weights.
        
#         Returns
#         -------
#         torch.Tensor
#             The learned convolution kernel weights.
#         """
#         return self.conv.weight.detach().cpu()
    
#     def get_masked_weights(self):
#         """Get the convolution kernel weights (same as learned weights for conv).
        
#         Returns
#         -------
#         torch.Tensor
#             The convolution kernel weights.
#         """
#         return self.conv.weight.detach().cpu()
        
# class DonutKernel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         mask = DONUT[40:200, 40:200].float()

#         # Register full weight as parameter
#         self.weight = nn.Parameter(torch.ones_like(mask))
        
#         # Register the binary mask as a buffer (non-trainable)
#         self.register_buffer('mask', mask)

#     def forward(self, x):
#         # Ensure mask is on the same device as weight
#         if self.mask.device != self.weight.device:
#             self.mask = self.mask.to(self.weight.device)
        
#         # Apply the mask to freeze certain weights
#         masked_weight = self.weight * self.mask
        
#         cutout =  x * masked_weight
#         background = x - x * self.mask
#         return cutout + background

#     def extra_repr(self):
#         return f'DonutKernel(kernel={self.weight.shape[2:]})'
    
#     def get_learned_weights(self):
#         """Get the learned weights of the donut kernel.
        
#         Returns
#         -------
#         torch.Tensor
#             The learned weights of the donut kernel.
#         """
#         return self.weight.detach().cpu()
    
#     def get_masked_weights(self):
#         """Get the masked weights (weight * mask) of the donut kernel.
        
#         Returns
#         -------
#         torch.Tensor
#             The masked weights of the donut kernel.
#         """
#         return (self.weight * self.mask).detach().cpu()


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
        lr: float = 1e-3,
        lr_schedule: bool = False,
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
        lr: float, default=1e-3
            The initial learning rate for Adam.
        lr_schedule: bool, default=True
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters()
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            n_predictor_layers=n_predictor_layers,
        )
        
        # Initialize the donut kernel
        self.donut_kernel = DynamicDonutKernel()

        # define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"
        self.inputShape = (160, 160)
        self.device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_step(
        self, batch: dict, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        img = self.donut_kernel(img) # apply donut kernel!!!
        img = self.rescale_image_batched(img)
        zk_pred = self.wavenet(img, fx, fy, intra, band)

        return zk_pred, zk_true

    def calc_losses(self, batch: dict, batch_idx: int) -> tuple:
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
        zk_pred = convert_zernikes_deploy(zk_pred)
        zk_true = convert_zernikes_deploy(zk_true)

        # pull out the weights from the final linear layer
        *_, A, _ = self.wavenet.predictor.parameters()

        # calculate loss
        sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
        loss = sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

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
        loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        if self.hparams.lr_schedule:
            return {
                "optimizer": optimizer,
                "gradient_clip_val": 1.0,  # Add gradient clipping
            }
        else:
            return {
                "optimizer": optimizer,
                "gradient_clip_val": 1.0,  # Add gradient clipping
            }

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
        img = self.donut_kernel(img) # apply donut kernel!!!
        zk_pred = self.wavenet(img, fx, fy, focalFlag, band)

        # convert to nanometers
        zk_pred *= 1_000

        return zk_pred
    
    def get_donut_kernel_weights(self):
        """Get the learned weights of the donut kernel.
        
        Returns
        -------
        torch.Tensor
            The learned weights of the donut kernel.
        """
        return self.donut_kernel.get_learned_weights()
    
    def get_donut_kernel_masked_weights(self):
        """Get the masked weights of the donut kernel.
        
        Returns
        -------
        torch.Tensor
            The masked weights of the donut kernel.
        """
        return self.donut_kernel.get_masked_weights()
    
    def get_donut_kernel_mask(self):
        """Get the binary mask of the donut kernel.
        
        Returns
        -------
        torch.Tensor
            The binary mask of the donut kernel.
        """
        return self.donut_kernel.get_mask()
    
    def process_image_with_donut_kernel(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process an image with the donut kernel and return both before and after.
        
        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (image_before, image_after) donut kernel processing.
        """
        # Ensure image is on the correct device
        if img.device != self.donut_kernel.conv.weight.device:
            img = img.to(self.donut_kernel.conv.weight.device)
        
        # Get the image before donut kernel processing
        img_before = img.clone()
        
        # Apply donut kernel to get the processed image
        img_after = self.donut_kernel(img)
        img_after = self.rescale_image_batched(img_after)
        return img_before, img_after
