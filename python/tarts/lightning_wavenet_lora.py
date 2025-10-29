"""Wrapping everything for WaveNet in Pytorch Lightning with LoRA support."""

from typing import Any, Tuple, Optional
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from .dataloader import Donuts, Donuts_Fullframe
from .utils import convert_zernikes_deploy
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
    """Pytorch Lightning system for training the WaveNet with LoRA support.

    This version supports both training from scratch and finetuning with LoRA
    for domain adaptation. Toggle between modes using the `use_lora` parameter.

    Examples
    --------
    Train from scratch:
    >>> model = WaveNetSystem(cnn_model="resnet34", use_lora=False)
    >>> trainer.fit(model, datamodule)

    Finetune with LoRA:
    >>> # Load pretrained model and add LoRA adapters
    >>> model = WaveNetSystem.load_from_checkpoint(
    ...     checkpoint_path="checkpoints/best-model.ckpt",
    ...     use_lora=True,
    ...     lora_r=8,
    ...     lora_alpha=16,
    ...     lora_target_modules="cnn",
    ...     lr=1e-4
    ... )
    >>> trainer.fit(model, datamodule_new_domain)
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
        pretrained: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: str = "cnn",
        lora_modules_to_save: Optional[list] = None,
    ) -> None:
        """Create the WaveNet with optional LoRA support.

        Parameters
        ----------
        cnn_model: str, default="resnet34"
            The name of the pre-trained CNN model from torchvision.
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights (when not using LoRA).
        n_predictor_layers: tuple, default=(256,)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is determined by n_zernikes.
        n_zernikes: int, default=25
            Number of Zernike coefficients to predict.
        alpha: float, default=0
            Weight for the L2 penalty.
        lr: float, default=1e-3
            The initial learning rate for Adam.
        lr_schedule: bool, default=False
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        pretrained: bool, default=False
            Whether to use pre-trained CNN weights. Set to True to download pre-trained weights.
        use_lora: bool, default=False
            Whether to use LoRA (Low-Rank Adaptation) for efficient finetuning.
            When True, only LoRA adapter weights are trained, making finetuning
            much more parameter-efficient.
        lora_r: int, default=8
            LoRA rank (dimension of the low-rank matrices). Higher values allow
            more adaptation capacity but use more parameters. Common values: 4, 8, 16, 32.
        lora_alpha: int, default=16
            LoRA scaling factor. Controls the magnitude of the LoRA updates.
            Typically set to 2*lora_r.
        lora_dropout: float, default=0.1
            Dropout probability for LoRA layers.
        lora_target_modules: str, default="cnn"
            Which modules to apply LoRA to. Options:
            - "cnn": Apply LoRA only to CNN backbone (recommended for domain adaptation)
            - "predictor": Apply LoRA only to predictor MLP
            - "both": Apply LoRA to both CNN and predictor
        lora_modules_to_save: Optional[list], default=None
            List of module names to save in addition to LoRA adapters.
            Useful if you want to keep some layers fully trainable.
        """
        super().__init__()
        self.save_hyperparameters()

        # Create base WaveNet model
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            n_predictor_layers=n_predictor_layers,
            n_zernikes=n_zernikes,
            pretrained=pretrained,
        )

        # Apply LoRA if requested
        # This happens AFTER base model creation, so checkpoint weights
        # can be loaded before LoRA adapters are added
        if use_lora:
            self._apply_lora()

        # Define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"
        self.inputShape = (160, 160)
        self.device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model.

        This method is called during initialization if use_lora=True.
        It wraps the model with LoRA adapters using the peft library.

        Raises
        ------
        ImportError
            If the peft library is not installed.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "The 'peft' library is required for LoRA support. "
                "Install it with: pip install peft"
            )

        # Determine which modules to target based on lora_target_modules
        target_modules = self._get_lora_target_modules()

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=target_modules,
            bias="none",  # Don't adapt biases
            task_type=TaskType.FEATURE_EXTRACTION,  # We're doing regression
            modules_to_save=self.hparams.lora_modules_to_save,
        )

        # Apply LoRA to the WaveNet
        self.wavenet = get_peft_model(self.wavenet, lora_config)

        # Print trainable parameters info
        self.wavenet.print_trainable_parameters()

    def _get_lora_target_modules(self) -> list:
        """Get the list of module names to apply LoRA to.

        Returns
        -------
        list
            List of module name patterns to target with LoRA.

        Raises
        ------
        ValueError
            If lora_target_modules is not one of "cnn", "predictor", or "both".
        """
        target = self.hparams.lora_target_modules

        # For ResNet architectures, target these patterns
        cnn_patterns = [
            "cnn.layer1",
            "cnn.layer2",
            "cnn.layer3",
            "cnn.layer4",
        ]

        # For the predictor MLP
        predictor_patterns = ["predictor"]

        if target == "cnn":
            return cnn_patterns
        elif target == "predictor":
            return predictor_patterns
        elif target == "both":
            return cnn_patterns + predictor_patterns
        else:
            raise ValueError(
                f"lora_target_modules must be 'cnn', 'predictor', or 'both', "
                f"got '{target}'"
            )

    def predict_step(
        self, batch: dict, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        When using LoRA, only LoRA adapter parameters will be optimized.
        The base model weights remain frozen.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
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

    def merge_and_unload_lora(self) -> None:
        """Merge LoRA weights into the base model and remove adapters.

        This is useful for deployment - it creates a single model with
        merged weights, removing the overhead of LoRA adapters.

        Warning: This operation is irreversible. After calling this,
        you cannot separate the LoRA weights from the base weights.

        Raises
        ------
        RuntimeError
            If the model is not using LoRA.
        """
        if not self.hparams.use_lora:
            raise RuntimeError(
                "Cannot merge LoRA weights: model is not using LoRA"
            )

        self.wavenet = self.wavenet.merge_and_unload()
        print("LoRA weights merged into base model")

    def save_lora_adapters(self, save_path: str) -> None:
        """Save only the LoRA adapter weights.

        This creates a very small checkpoint file containing only the
        LoRA adapters, not the full model weights.

        Parameters
        ----------
        save_path: str
            Path to save the LoRA adapter weights.

        Raises
        ------
        RuntimeError
            If the model is not using LoRA.
        """
        if not self.hparams.use_lora:
            raise RuntimeError(
                "Cannot save LoRA adapters: model is not using LoRA"
            )

        self.wavenet.save_pretrained(save_path)
        print(f"LoRA adapters saved to {save_path}")

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
        zk_pred = self.wavenet(img, fx, fy, focalFlag, band)

        # convert to nanometers
        zk_pred *= 1_000

        return zk_pred
