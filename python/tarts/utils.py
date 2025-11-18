"""Utility functions."""

# Standard library imports
import logging
import os
import warnings
from pathlib import Path
from random import randint
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from numpy.linalg import svd

from lsst.ts.ofc import SensitivityMatrix
from lsst.ts.ofc.utils.ofc_data_helpers import get_intrinsic_zernikes

# Third-party imports
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn

# Optional imports
try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    git = None

# Local/application imports
from .constants import (
    BAND_MAP,
    BAND_STR_INT,
    BAND_MEAN,
    BAND_STD,
    DONUT_TEMPLATE,
    FIELD_MEAN,
    FIELD_POSITIONS,
    FIELD_STD,
    INTRA_MEAN,
    INTRA_STD,
    LSST_AVAILABLE,
)

# Alias for backward compatibility
DONUT = DONUT_TEMPLATE

logger = logging.getLogger(__name__)


def safe_yaml_load(file_path: str) -> Dict[str, Any]:
    """Safely load YAML files that may contain Python objects like tuples.

    Parameters
    ----------
    file_path : str
        Path to the YAML file to load

    Returns
    -------
    Dict[str, Any]
        Loaded YAML content as dictionary
    """
    try:
        # First try standard loader
        with open(file_path, "r") as f:
            result: Any = yaml.safe_load(f)
            if result is not None and isinstance(result, dict):
                return cast(Dict[str, Any], result)
            return {}
    except yaml.YAMLError as e:
        # If that fails, try with FullLoader which can handle more Python objects
        try:
            with open(file_path, "r") as f:
                result2: Any = yaml.load(f, Loader=yaml.FullLoader)
                if result2 is not None and isinstance(result2, dict):
                    return cast(Dict[str, Any], result2)
                return {}
        except yaml.YAMLError:
            # If both fail, try to load and fix common issues
            with open(file_path, "r") as f:
                content = f.read()

            # Replace problematic tuple tags with list format
            content = content.replace("!!python/tuple", "")
            content = content.replace("tag:yaml.org,2002:python/tuple", "")

            # Try to load the cleaned content
            try:
                result3: Any = yaml.safe_load(content)
                if result3 is not None and isinstance(result3, dict):
                    return cast(Dict[str, Any], result3)
                return {}
            except yaml.YAMLError:
                # Last resort: return empty dict and warn
                warnings.warn(f"Could not parse YAML file {file_path}: {e}")
                return {}


class LearningRateThresholdCallback(pl.Callback):
    """Stop training when learning rate drops below threshold."""

    def __init__(self, threshold=1e-6):
        """Initialize the learning rate threshold callback.

        Parameters
        ----------
        threshold : float, optional, default=1e-6
            The learning rate threshold below which training will be stopped.
        """
        self.threshold = threshold

    def on_train_epoch_end(self, trainer, pl_module):
        """Check learning rate at end of epoch and stop training if below threshold.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The PyTorch Lightning module being trained.
        """
        # Get current learning rate
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]

        if current_lr < self.threshold:
            logger.info(
                f"Learning rate {current_lr:.2e} below threshold {self.threshold:.2e}. Stopping training."
            )
            trainer.should_stop = True


# Alias for backward compatibility
BAND_str_int = BAND_STR_INT
FIELD = FIELD_POSITIONS


def convert_zernikes(zernikes: torch.Tensor) -> torch.Tensor:
    """Convert zernike units from microns to quadrature contribution to FWHM.

    Parameters
    ----------
    zernikes: torch.Tensor
        Tensor of zernike coefficients in microns.

    Returns
    -------
    torch.Tensor
        Zernike coefficients in units of quadrature contribution to FWHM.
    """
    # these conversion factors depend on telescope radius and obscuration
    # the numbers below are for the Rubin telescope; different numbers
    # are needed for Auxtel. For calculating these factors, see ts_phosim
    arcsec_per_micron = torch.tensor(
        [
            0.751,  # Z4
            0.271,  # Z5
            0.271,  # Z6
            0.819,  # Z7
            0.819,  # Z8
            0.396,  # Z9
            0.396,  # Z10
            1.679,  # Z11
            0.937,  # Z12
            0.937,  # Z13
            0.517,  # Z14
            0.517,  # Z15
            1.755,  # Z16
            1.755,  # Z17
            1.089,  # Z18
            1.089,  # Z19
            0.635,  # Z20
            0.635,  # Z21
            2.810,  # Z22
        ]
    ).to(zernikes.device)

    return zernikes * arcsec_per_micron


def convert_zernikes_deploy(zernikes: torch.Tensor, device="cpu") -> torch.Tensor:
    """Convert zernike units from microns to quadrature contribution to FWHM.

    Parameters
    ----------
    zernikes: torch.Tensor
        Tensor of zernike coefficients in microns.
    device: str, optional, default='cpu'
        Device to place the conversion tensor on ('cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        Zernike coefficients in units of quadrature contribution to FWHM.
    """
    # these conversion factors depend on telescope radius and obscuration
    # the numbers below are for the Rubin telescope; different numbers
    # are needed for Auxtel. For calculating these factors, see ts_phosim
    # ZK4-ZK28
    arcsec_per_micron = torch.tensor(
        [
            0.75453564,
            0.2711315,
            0.2711315,
            0.82237164,
            0.82237164,
            0.39572461,
            0.39572461,
            1.68719298,
            0.93956571,
            0.93956571,
            0.51643078,
            0.51643078,
            1.76255706,
            1.76255706,
            1.09112012,
            1.09112012,
            0.63498144,
            0.63498144,
            2.82321385,
            1.87588235,
            1.87588235,
            1.26437424,
            1.26437424,
            0.75241174,
            0.75241174,
        ]
    ).to(zernikes.device)

    return zernikes * arcsec_per_micron


def plot_zernikes(z_pred: torch.Tensor, z_true: torch.Tensor) -> plt.Figure:
    """Plot true and predicted zernikes (up to 8).

    Parameters
    ----------
    z_pred: torch.Tensor
        2D Array of predicted Zernike coefficients
    z_true: torch.Tensor
        2D Array of true Zernike coefficients

    Returns
    -------
    plt.Figure
        Figure containing the 8 axes with the true and predicted Zernike
        coefficients plotted together.
    """
    # create the figure
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 5),
        constrained_layout=True,
        dpi=150,
        sharex=True,
        sharey=True,
    )

    # loop through the axes/zernikes
    for ax, zt, zp in zip(axes.flatten(), z_true, z_pred):
        ax.plot(np.arange(4, 23), convert_zernikes(zt), label="True")
        ax.plot(np.arange(4, 23), convert_zernikes(zp), label="Predicted")
        ax.set(xticks=np.arange(4, 23, 2))

    axes[0, 0].legend()  # add legend to first panel

    # set axis labels
    for ax in axes[:, 0]:
        ax.set_ylabel("Arcsec FWHM")
    for ax in axes[1, :]:
        ax.set_xlabel("Zernike number (Noll)")

    return fig


def count_parameters(model: torch.nn.Module, trainable: bool = True) -> int:
    """Count the number of parameters in the model.

    Parameters
    ----------
    model: torch.nn.Module
        The Pytorch model to count parameters for
    trainable: bool, default=True
        If True, only counts trainable parameters

    Returns
    -------
    int
        The number of trainable parameters
    """
    if trainable:
        return sum(params.numel() for params in model.parameters() if params.requires_grad)
    else:
        return sum(params.numel() for params in model.parameters())


def printOnce(msg: str, header: bool = False) -> None:
    """Log message once to avoid duplicate messages in distributed settings.

    This avoids the problem where statements get logged multiple times in
    a distributed setting.

    Parameters
    ----------
    msg: str
        Message to log
    header: bool, default=False
        Whether to add extra space and underline for the message
    """
    rank = os.environ.get("LOCAL_RANK", None)
    if rank is None or rank == "0":
        if header:
            formatted_msg = f"\n{msg}\n{'-'*len(msg)}\n"
            logger.info(formatted_msg)
        else:
            logger.info(msg)


def transform_inputs(
    image: np.ndarray,
    fx: float,
    fy: float,
    intra: bool,
    band: int,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Transform inputs to the neural network.

    Parameters
    ----------
    image: np.ndarray
        The donut image
    fx: float
        X angle of source with respect to optic axis (radians)
    fy: float
        Y angle of source with respect to optic axis (radians)
    intra: bool
        Boolean indicating whether the donut is intra or extra focal
    band: int
        Band index in the string "ugrizy". I.e., 0="u", ..., 5="y".

    Returns
    -------
        same as above, with transformations applied
    """
    # rescale image to [0, 1]
    image -= image.min()
    image /= image.max()

    # normalize image
    # image_mean = 0.347
    # image_std = 0.226
    image_mean = image.mean()
    image_std = image.std()
    image = (image - image_mean) / image_std

    # normalize angles
    fx = (fx - FIELD_MEAN) / FIELD_STD
    fy = (fy - FIELD_MEAN) / FIELD_STD

    # normalize the intrafocal flags
    intra = (intra - INTRA_MEAN) / INTRA_STD  # type: ignore

    # get the effective wavelength in microns
    band = BAND_MAP[band]  # type: ignore

    # normalize the wavelength
    band = (band - BAND_MEAN) / BAND_STD  # type: ignore

    return image, fx, fy, intra, band


def get_root() -> Path:
    """Get the root directory of the git repository.

    Returns
    -------
    pathlib.PosixPath
    """
    if not GIT_AVAILABLE:
        # If git is not available, return current working directory
        return Path.cwd()
    assert git is not None  # Type narrowing for mypy
    repo_root = git.Repo(".", search_parent_directories=True).working_tree_dir
    if repo_root is None:
        return Path.cwd()
    root = Path(repo_root)
    return root


def noise_est(frame):
    """Estimate the noise standard deviation and mean from the corners of a frame.

    Noise estimation is used to inpaint the shifted donut for augmentation.

    This function calculates the standard deviation and mean of pixel values in the four corners
    of the input frame. It then returns the standard deviation and mean of the corner with the
    lowest standard deviation, along with the index of that corner. This is done to estimate
    the noise characteristics in a region of the frame that is likely to be relatively uniform
    and free of strong signal.

    Args:
        frame (torch.Tensor): The input frame as a PyTorch tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The estimated noise standard deviation (torch.Tensor).
            - The estimated noise mean (torch.Tensor).
            - The index of the corner with the lowest standard deviation (torch.Tensor).
              (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right).
    """
    corner_1 = frame[:30, :30].flatten()  # top left
    corner_2 = frame[:30, 130:].flatten()  # top right
    corner_3 = frame[130:, :30].flatten()  # bottom left
    corner_4 = frame[130:, 130:].flatten()  # bottom right
    std_1, mean_1 = torch.std_mean(corner_1)
    std_2, mean_2 = torch.std_mean(corner_2)
    std_3, mean_3 = torch.std_mean(corner_3)
    std_4, mean_4 = torch.std_mean(corner_4)
    stds = torch.stack([std_1, std_2, std_3, std_4])
    means = torch.stack([mean_1, mean_2, mean_3, mean_4])
    ind = torch.argmin(stds)
    # Use gather for vmap-friendly indexing
    selected_std = torch.gather(stds, 0, ind.unsqueeze(0)).squeeze(0)
    selected_mean = torch.gather(means, 0, ind.unsqueeze(0)).squeeze(0)
    return selected_std, selected_mean, ind


def detect_direction(frame):
    """Detect the dominant direction to shift based on the statistical analysis of the image borders.

    This function computes the standard deviation and mean of pixel values along the four borders
    (up, left, down, right) of a given image frame. It then identifies which border has the highest
    mean pixel value and returns the corresponding direction.
    (To try and avoid unrealistic donuts i.e shift towards a direction that has no donuts creating
    cropped blended donuts)


    Parameters
    ----------
    frame : torch.Tensor
        A 2D tensor representing an image frame. The frame is expected to have a height and width
        greater than or equal to 140 pixels.

    Returns
    -------
    str
        The direction ('up', 'left', 'down', or 'right') corresponding to the border with the highest
        mean pixel value.

    Notes
    -----
    - The borders are defined as follows:
        - Up: The top 20 rows of the image.
        - Left: The left 20 columns of the image.
        - Down: The bottom 140 rows of the image.
        - Right: The right 140 columns of the image.
    - The function returns the direction based on the highest mean value,
        where the direction is chosen
      from the four image borders.
    - If there is an issue in the code where both `ind == 1`
        checks are encountered (as in the case of
      'left' and 'down'), the 'down' direction will not be returned correctly.
      The second `elif` statement
      should be adjusted to handle 'down' properly.
    """
    border_1 = frame[:20, :].flatten()  # up
    border_2 = frame[:, :20].flatten()  # left
    border_3 = frame[140:, :].flatten()  # down
    border_4 = frame[:, 140:].flatten()  # right

    std_1, mean_1 = torch.std_mean(border_1)
    std_2, mean_2 = torch.std_mean(border_2)
    std_3, mean_3 = torch.std_mean(border_3)
    std_4, mean_4 = torch.std_mean(border_4)

    means = torch.tensor([mean_1, mean_2, mean_3, mean_4])
    ind = torch.argmax(means)
    if ind == 0:
        return "up"
    elif ind == 1:
        return "left"
    elif ind == 2:
        return "down"
    else:
        return "right"


def shift_offcenter(frame, adjust=0, return_offset=True):
    """Shift the input image off-center based on its detected direction and apply random displacement.

    This function detects the dominant direction of the given image frame (up, down, left, or right)
    and applies a random shift within a specified range (`adjust`) along the x and y axes. It then
    generates a synthetic background based on the noise statistics of the image and shifts the image
    onto the background.

    Parameters
    ----------
    frame : torch.Tensor
        The input image frame, expected to be a 2D tensor (e.g., shape (160, 160)).
    adjust : int, default=0
        The maximum amount of shift applied to the image in the x and y directions.
        The shift is chosen randomly within the range (-adjust, adjust) based on the image direction.
    return_offset : bool, default=True
        Whether to return the random offset applied to the image in addition to the shifted image.
        If False, only the shifted image is returned.

    Returns
    -------
    torch.Tensor
        The shifted image, which is cropped to the center region (160x160) of the 480x480 background.
    list
        The list of random x and y shifts applied (only returned if `return_offset=True`).

    Notes
    -----
    - The function uses the `detect_direction` function to determine the dominant border of the image
      and then randomly shifts the image within that direction.
    - The background is generated with noise based on the standard deviation (`std`) and mean (`mean`)
      of the input frame.
    - If `return_offset` is set to `False`, only the shifted image will be returned, and the random offset
      used for the shift will be discarded.
    """
    direction = detect_direction(frame)
    if direction == "left":
        random_x = randint(-adjust, 0)
        random_y = randint(-adjust, adjust)
    elif direction == "right":
        random_x = randint(0, adjust)
        random_y = randint(-adjust, adjust)
    elif direction == "down":
        random_x = randint(-adjust, adjust)
        random_y = randint(-adjust, 0)
    else:
        random_x = randint(-adjust, adjust)
        random_y = randint(0, adjust)
    std, mean, _ = noise_est(frame)
    backplate = torch.empty(160 * 3, 160 * 3).normal_(mean=mean, std=std)
    backplate[160 + random_y : 160 * 2 + random_y, 160 + random_x : 160 * 2 + random_x] = frame
    if return_offset:
        return backplate[160 : 160 * 2, 160 : 160 * 2], [random_x, random_y]
    else:
        return backplate[160 : 160 * 2, 160 : 160 * 2]


def compute_2d_fft_torch(image, pixel_scale=1.0):
    """Compute 2D FFT of an image.

    Parameters
    ----------
    image : torch.Tensor
        Input image as a PyTorch tensor. Can be 2D or 3D (will squeeze if 3D).
    pixel_scale : float, default=1.0
        Pixel scale for frequency computation.

    Returns
    -------
    tuple
        (fshift, kx, ky, magnitude_spectrum, phase_spectrum)
        - fshift: Shifted FFT
        - kx, ky: Frequency grids
        - magnitude_spectrum: Log-magnitude spectrum
        - phase_spectrum: Phase spectrum
    """
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.dim() == 3:
        image = image.squeeze(0)

    f = torch.fft.fft2(image)
    fshift = torch.fft.fftshift(f)

    ny, nx = image.shape
    kx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=pixel_scale))
    ky = torch.fft.fftshift(torch.fft.fftfreq(ny, d=pixel_scale))

    magnitude_spectrum = torch.log1p(torch.abs(fshift))
    phase_spectrum = torch.angle(fshift)

    return fshift, kx, ky, magnitude_spectrum, phase_spectrum


def inverse_2d_fft_torch(fshift):
    """Inverse 2D FFT.

    Parameters
    ----------
    fshift : torch.Tensor
        Shifted FFT coefficients.

    Returns
    -------
    torch.Tensor
        Reconstructed real-space image.
    """
    f_ishift = torch.fft.ifftshift(fshift)
    reconstructed = torch.real(torch.fft.ifft2(f_ishift))
    return reconstructed


def apply_k_filter_torch(fshift, kx, ky, kmin=0.0, kmax=float("inf"), sigma_frac=0.1):
    """Apply smooth circular band-pass filter in Fourier space.

    Parameters
    ----------
    fshift : torch.Tensor
        Shifted FFT coefficients.
    kx : torch.Tensor
        Frequency grid in x direction.
    ky : torch.Tensor
        Frequency grid in y direction.
    kmin : float, default=0.0
        Minimum frequency magnitude to keep.
    kmax : float, default=float("inf")
        Maximum frequency magnitude to keep.
    sigma_frac : float, default=0.1
        Fractional Gaussian width for smooth filter edges.

    Returns
    -------
    tuple
        (f_filtered, weight)
        - f_filtered: Filtered FFT coefficients
        - weight: Filter weight mask
    """
    device = fshift.device
    KX, KY = torch.meshgrid(kx.to(device), ky.to(device), indexing="xy")
    K_mag = torch.sqrt(KX**2 + KY**2)
    kmax_valid = torch.max(K_mag)
    kmax = min(kmax, kmax_valid.item())

    sigma = sigma_frac * (kmax - kmin)
    sigma = max(sigma, 1e-6)

    low_edge = 1 / (1 + torch.exp(-(K_mag - kmin) / sigma))
    high_edge = 1 / (1 + torch.exp((K_mag - kmax) / sigma))
    weight = low_edge * high_edge

    f_filtered = fshift * weight
    return f_filtered, weight


def augment_data_torch(sim, real, scale=0.1, kmin=0.1):
    """Augment simulation image by injecting randomized high-frequency structure from real image.

    This function extracts high-frequency components from a real (coral) image, randomizes
    their phase, and injects them into the simulation image in regions where the simulation
    has signal (above noise threshold).

    Parameters
    ----------
    sim : torch.Tensor
        Simulation image (2D tensor).
    real : torch.Tensor
        Real/coral image (2D tensor) to extract high-frequency structure from.
        Should have the same shape as sim.
    scale : float, default=0.1
        Scaling factor for the injected high-frequency structure.
    kmin : float, default=0.1
        Minimum frequency magnitude to extract from real image.

    Returns
    -------
    torch.Tensor
        Augmented simulation image with injected high-frequency structure.
    """
    # Ensure images are 2D and on the same device
    sim = torch.as_tensor(sim, dtype=torch.float32)
    real = torch.as_tensor(real, dtype=torch.float32)

    # Remove extra dimensions if present
    while sim.dim() > 2:
        sim = sim.squeeze(0)
    while real.dim() > 2:
        real = real.squeeze(0)

    # Ensure real image has same shape as sim (crop or pad if needed)
    if real.shape != sim.shape:
        # Crop or pad to match sim shape
        if real.shape[0] > sim.shape[0]:
            real = real[: sim.shape[0], :]
        elif real.shape[0] < sim.shape[0]:
            pad_h = sim.shape[0] - real.shape[0]
            # F.pad format for 2D: (pad_left, pad_right, pad_top, pad_bottom)
            # Add batch and channel dims for F.pad, then remove them
            real = (
                F.pad(real.unsqueeze(0).unsqueeze(0), (0, 0, 0, pad_h), mode="reflect").squeeze(0).squeeze(0)
            )
        if real.shape[1] > sim.shape[1]:
            real = real[:, : sim.shape[1]]
        elif real.shape[1] < sim.shape[1]:
            pad_w = sim.shape[1] - real.shape[1]
            # F.pad format for 2D: (pad_left, pad_right, pad_top, pad_bottom)
            real = (
                F.pad(real.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, 0), mode="reflect").squeeze(0).squeeze(0)
            )

    device = sim.device
    real = real.to(device)

    # --- FFT of real image ---
    fshift_real, kx, ky, mag_real, _ = compute_2d_fft_torch(real)
    f_high, _ = apply_k_filter_torch(fshift_real, kx, ky, kmin=kmin, kmax=float("inf"))

    # --- Randomize phase of high-frequency component ---
    random_phase = torch.exp(1j * 2 * torch.pi * torch.rand(f_high.shape, device=device, dtype=torch.float32))
    f_high_randomized = torch.abs(f_high) * random_phase

    # --- Background reconstruction ---
    # Avoid division by zero in magnitude spectrum
    mag_real_safe = mag_real + 1e-8
    background = inverse_2d_fft_torch(f_high_randomized / mag_real_safe)
    background = background.to(device)

    # Normalize background
    bg_max = background.abs().max()
    if bg_max > 1e-8:
        background_norm = background / bg_max
    else:
        background_norm = background

    # --- Threshold mask on simulation ---
    std, mean, ind = noise_est(sim)
    mask = sim > (mean + 3 * std)
    # --- Combine ---
    sim_max = sim.abs().max()
    augmented_data = sim + mask.float() * sim_max * background_norm * scale
    return augmented_data


def add_random_hot_pixel(image, sigma=1.0, prob=0.5, min_scale=5.0, max_scale=20):
    """Randomly adds a hot pixel (with small Gaussian spread) to a 2D image tensor.

    Parameters
    ----------
    image : torch.Tensor
        2D tensor (H, W) representing the image.
    sigma : float
        Standard deviation (in pixels) of the Gaussian spread.
    prob : float
        Probability of applying the augmentation.
    min_scale : float
        Minimum multiple of the image max value for the hot pixel intensity.
    max_scale : float
        Maximum multiple of the image max value for the hot pixel intensity.

    Returns
    -------
    torch.Tensor
        Image with (possibly) one hot pixel added.
    """
    if torch.rand(1).item() > prob:
        return image  # skip augmentation randomly

    # Ensure image is 2D
    original_shape = image.shape
    while image.dim() > 2:
        image = image.squeeze(0)

    H, W = image.shape

    # --- Pick random location ---
    y = torch.randint(0, H, (1,), device=image.device).item()
    x = torch.randint(0, W, (1,), device=image.device).item()

    # --- Determine random intensity ---
    image_max = image.max().item() if image.numel() > 0 else 1.0
    intensity_scale = torch.empty(1, device=image.device).uniform_(min_scale, max_scale).item()
    intensity = image_max * intensity_scale

    # --- Build Gaussian hot pixel ---
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device), torch.arange(W, device=image.device), indexing="ij"
    )
    gauss = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    gauss = gauss / gauss.max() * intensity

    # --- Add to image ---
    augmented = image + gauss

    # Restore original shape if needed
    if len(original_shape) > 2:
        while len(augmented.shape) < len(original_shape):
            augmented = augmented.unsqueeze(0)

    return augmented


def batched_crop(image_tensor: torch.Tensor, centers: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Crop square patches from an image tensor given a list of center points.

    Args:
        image_tensor (Tensor): Image tensor of shape (C, H, W).
        centers (Tensor): Tensor of shape (N, 2) with (x, y) coordinates.
        crop_size (int): Size of square crops.

    Returns:
        Tensor: Batched cropped images of shape (N, C, crop_size, crop_size).
    """
    _, H, W = image_tensor.shape  # Get image dimensions

    # Compute top-left coordinates
    left = torch.clamp(centers[:, 0] - crop_size // 2, 0, W - crop_size)
    top = torch.clamp(centers[:, 1] - crop_size // 2, 0, H - crop_size)

    # Use advanced indexing to extract crops
    crops = torch.stack([image_tensor[:, t : t + crop_size, l : l + crop_size] for t, l in zip(top, left)])

    return crops


def get_centers(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Generate grid of center coordinates for cropping patches from an image.

    This function generates centers using two methods and combines them:
    1. Regular grid-based centers at fixed intervals
    2. Refined centers based on brightest regions using pooling

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Input image tensor of shape (H, W).
    crop_size : int
        Size of square crops to generate centers for.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2) containing (x, y) center coordinates combining both
        grid-based and refined bright-region centers.
    """
    # Convert to torch tensor if numpy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    H, W = image.shape

    # --- Part 1: Regular grid centers ---
    x_coords = torch.arange(crop_size // 2, W, crop_size)
    y_coords = torch.arange(crop_size // 2, H, crop_size)

    # Create a grid of center coordinates
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # Shape: [N, 2]

    # --- Part 2: Refined centers from bright regions ---
    img = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Mean pooling to find bright regions
    stride = 80
    pooled = F.avg_pool2d(img, kernel_size=160, stride=stride)

    pooled2d = pooled.squeeze(0).squeeze(0)  # (H', W')
    H_p, W_p = pooled2d.shape

    # Flatten and get top-30 indices
    flat = pooled2d.flatten()
    values, indices = torch.topk(flat, k=30)

    # Convert flattened indices into pooled coords
    coords = torch.stack([indices // W_p, indices % W_p], dim=1)  # y in pooled  # x in pooled

    # Convert pooled coords -> original coords
    refined_coords = coords * stride  # stride size

    # Filter out points within 160px of any border
    y = refined_coords[:, 0]
    x = refined_coords[:, 1]

    margin = 160
    mask = (y >= margin) & (y < H - margin) & (x >= margin) & (x < W - margin)

    refined_centers = refined_coords[mask]

    # Swap x and y to match grid_centers format (x, y)
    refined_centers = torch.stack([refined_centers[:, 1], refined_centers[:, 0]], dim=1)

    # Combine both sets of centers
    all_centers = torch.cat([grid_centers, refined_centers], dim=0)
    all_centers = grid_centers
    return all_centers


def single_conv(image, device="cuda"):
    """Compute signal-to-noise ratio for donut detection using template matching.

    Parameters
    ----------
    image : torch.Tensor
        Input image patch to analyze.
    device : str, optional, default='cuda'
        Device to perform computation on ('cuda' or 'cpu').

    Returns
    -------
    torch.Tensor
        Signal-to-noise ratio indicating donut quality/detection confidence.

    Notes
    -----
    Uses a predefined donut template to compute the ratio of signal in donut regions
    to noise in background regions, providing a quality metric for donut detection.
    """
    device = image.device
    # donut = np.loadtxt("/home/peterma/research/Rubin_LSST/Rubin_AO_ML/training/extra_template-R22_S11.txt")
    donut = DONUT_TEMPLATE
    donut = donut[40:200, 40:200].float().to(device)
    not_donut = (1 - donut).bool().float().to(device)
    donut_mean = torch.mean(image * donut)
    not_donut_mean = torch.mean(image * not_donut)
    not_donut_std = torch.std(image * not_donut)
    sigma_dev = abs(donut_mean - not_donut_mean) / not_donut_std
    return sigma_dev


def filter_SNR(images, alpha):
    """Filter images based on signal-to-noise ratio (SNR) threshold.

    Args:
        images (torch.Tensor): Batch of images with shape (N, C, H, W).
        alpha (float): SNR threshold for filtering (default: 0.05).

    Returns:
        Tuple of (filtered_images, indices, snr_values) for images that pass the SNR threshold.
    """
    N, C, H, W = images.shape  # Batch size, Channels, Height, Width

    keep_images = []
    img_index = []
    snr_list = []
    for i in range(N):
        image = images[i]  # Shape (C, H, W)

        ratio = single_conv(image)
        if ratio > alpha:
            keep_images.append(image)
            img_index.append(i)
            snr_list.append(ratio)

    if not keep_images:
        return np.array([[]]), np.array([[]]), np.array([[]])
    else:
        keep_images_tensor = torch.concatenate(keep_images)
        img_index_tensor = torch.tensor(img_index)
        return keep_images_tensor, img_index_tensor, snr_list


class CORALLoss(nn.Module):
    """CORAL (Correlation Alignment) loss for domain adaptation.

    This loss function aligns the second-order statistics of source and target
    feature distributions to reduce domain shift.
    """

    def __init__(self):
        """Initialize the CORAL loss."""
        super().__init__()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of CORAL loss.

        Parameters
        ----------
        source : torch.Tensor
            Source domain features.
        target : torch.Tensor
            Target domain features.

        Returns
        -------
        torch.Tensor
            CORAL loss value.
        """
        # d = source.size(1)  # Unused variable

        # Optional: normalize features to prevent scale issues
        # source = F.normalize(source, dim=1)
        # target = F.normalize(target, dim=1)

        # Check for NaNs or infs
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            return torch.tensor(0.0, device=source.device, requires_grad=True)

        source_covar = self._compute_covariance(source)
        target_covar = self._compute_covariance(target)

        loss = torch.mean((source_covar - target_covar) ** 2)
        # loss = loss / (4 * d * d)
        return loss

    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if n <= 1:
            return torch.zeros((x.size(1), x.size(1)), device=x.device)

        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (n - 1)
        return cov


def compute_error(pred, true):
    """Compute mean Root Sum of Squared Errors between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values tensor.
    true : torch.Tensor
        Ground truth values tensor.

    Returns
    -------
    torch.Tensor
        Mean Root Sum of Squared Errors (mRSSE) loss value.
    """
    sse = F.mse_loss(pred, true, reduction="none").sum(dim=-1)
    mRSSE = torch.sqrt(sse).mean()
    return mRSSE


def getTable(day):
    """Get table from consdb."""
    os.environ["no_proxy"] += ",.consdb"
    # URL = "http://consdb-pq.consdb:8080/consdb"  # Unused variable

    # Remove unused variable 'headers'
    # cdb_client = ConsDbClient(URL)  # Undefined name

    # seq_min = 1  # Unused variable
    # seq_max = 700  # Unused variable
    # cdb_table['zernikes_fwhm'] = cdb_table['zernikes'].apply(convertZernikesToPsfWidth)
    # zk in microns
    # query = f"""
    # SELECT
    #         e.air_temp AS air_temp,
    #         e.airmass AS airmass,
    #         e.dimm_seeing AS dimm,
    #         e.altitude AS elevation,
    #         e.azimuth AS azimuth,
    #         e.exposure_id AS visit_id,
    #         e.physical_filter as band,
    #         e.day_obs AS day,
    #         e.exp_midpt AS time,
    #         e.dimm_seeing AS seeing,
    #         e.humidity AS humidity,
    #         e.pressure AS pressure,
    #         e.seq_num AS seq,
    #         e.sky_rotation AS sky_rotation,
    #         e.wind_dir AS wind_dir,
    #         e.wind_speed AS wind_speed,
    #         e.s_ra,
    #         e.s_dec,
    #         ccdvisit1_quicklook.psf_sigma,
    #         ccdvisit1_quicklook.z4,
    #         ccdvisit1_quicklook.z5,
    #         ccdvisit1_quicklook.z6,
    #         ccdvisit1_quicklook.z7,
    #         ccdvisit1_quicklook.z8,
    #         ccdvisit1_quicklook.z9,
    #         ccdvisit1_quicklook.z10,
    #         ccdvisit1_quicklook.z11,
    #         ccdvisit1_quicklook.z12,
    #         ccdvisit1_quicklook.z13,
    #         ccdvisit1_quicklook.z14,
    #         ccdvisit1_quicklook.z15,
    #         ccdvisit1_quicklook.z16,
    #         ccdvisit1_quicklook.z17,
    #         ccdvisit1_quicklook.z18,
    #         ccdvisit1_quicklook.z19,
    #         ccdvisit1_quicklook.z20,
    #         ccdvisit1_quicklook.z21,
    #         ccdvisit1_quicklook.z22,
    #         ccdvisit1_quicklook.z23,
    #         ccdvisit1_quicklook.z24,
    #         ccdvisit1_quicklook.z25,
    #         ccdvisit1_quicklook.z26,
    #         ccdvisit1_quicklook.z27,
    #         ccdvisit1_quicklook.z28,
    #         ccdvisit1.detector as detector,
    #         q.sky_noise_median AS sky_noise,
    #         q.sky_noise_max AS sky_noise_max,
    #         q.sky_noise_min AS sky_noise_min,
    #         q.sky_bg_median AS sky_bg,
    #         q.sky_bg_max AS sky_bg_max,
    #         q.sky_bg_min AS sky_bg_min,
    #         q.psf_sigma_median AS psf_fwhm,
    #         q.psf_sigma_min AS psf_fwhm_min,
    #         q.psf_sigma_max AS psf_fwhm_max,
    #         q.psf_ixx_median AS psf_ixx_median,
    #         q.psf_ixx_max AS psf_ixx_max,
    #         q.psf_ixx_min AS psf_ixx_min,
    #         q.psf_iyy_median AS psf_iyy_median,
    #         q.psf_iyy_max AS psf_iyy_max,
    #         q.psf_iyy_min AS psf_iyy_min,
    #         q.psf_ixy_median AS psf_ixy_median,
    #         q.psf_ixy_max AS psf_ixy_max,
    #         q.psf_ixy_min AS psf_ixy_min,
    #         q.psf_area_max AS psf_area_max,
    #         q.psf_area_median AS psf_area_median,
    #         q.psf_area_min AS psf_area_min,
    #         e.obs_end,
    #         e.obs_start
    #         FROM
    #         cdb_lsstcam.ccdvisit1_quicklook AS ccdvisit1_quicklook,
    #         cdb_lsstcam.ccdvisit1 AS ccdvisit1,
    #         cdb_lsstcam.visit1 AS visit1,
    #         cdb_lsstcam.visit1_quicklook AS q,
    #         cdb_lsstcam.exposure AS e
    #         WHERE
    #         ccdvisit1.detector IN (191, 192, 195, 196, 199, 200, 203, 204)
    #         AND ccdvisit1.ccdvisit_id = ccdvisit1_quicklook.ccdvisit_id
    #         AND ccdvisit1.visit_id = visit1.visit_id
    #         AND ccdvisit1.visit_id = q.visit_id
    #         AND ccdvisit1.visit_id = e.exposure_id
    #         AND (e.img_type = 'science' or e.img_type = 'acq')
    #         AND e.day_obs = {day}
    #         AND (e.seq_num BETWEEN {seq_min} AND {seq_max})
    #         AND e.airmass > 0
    #         AND e.band != 'none'
    # """
    # cdb_client = ConsDbClient("http://consdb-pq.consdb:8080/consdb")
    # cdb_table = cdb_client.query(query).to_pandas()  # Undefined name
    # return cdb_table  # Undefined name


# Block comment should start with '# '
# This is a proper block comment


def getzk(row):
    """Get Zernike coefficients."""
    zk4 = row["z4"].tolist()
    zk5 = row["z5"].tolist()
    zk6 = row["z6"].tolist()
    zk7 = row["z7"].tolist()
    zk8 = row["z8"].tolist()
    zk9 = row["z9"].tolist()
    zk10 = row["z10"].tolist()
    zk11 = row["z11"].tolist()
    zk12 = row["z12"].tolist()
    zk13 = row["z13"].tolist()
    zk14 = row["z14"].tolist()
    zk15 = row["z15"].tolist()
    zk20 = row["z20"].tolist()
    zk21 = row["z21"].tolist()
    zk22 = row["z22"].tolist()
    zk27 = row["z27"].tolist()
    zk28 = row["z28"].tolist()
    table_val = np.stack(
        [
            zk4,
            zk5,
            zk6,
            zk7,
            zk8,
            zk9,
            zk10,
            zk11,
            zk12,
            zk13,
            zk14,
            zk15,
            zk20,
            zk21,
            zk22,
            zk27,
            zk28,
        ]
    )
    return table_val


def getRealData(butler, cdb_table, ind):
    """Get real data from source."""
    if not LSST_AVAILABLE:
        raise ImportError("LSST dependencies not available. This function requires LSST installation.")

    exposure_id = cdb_table["visit_id"][ind]
    detector_name = cdb_table["detector"][ind]
    data_id1 = {
        "instrument": "LSSTCam",  # Replace with your instrument
        "exposure": int(exposure_id),  # Replace with your exposure ID
        "detector": detector_name,
    }
    data_id2 = {
        "instrument": "LSSTCam",  # Replace with your instrument
        "exposure": int(exposure_id),  # Replace with your exposure ID
        "detector": detector_name + 1,
    }
    data_1 = butler.get("raw", dataId=data_id1, collections="LSSTCam/raw/all")
    data_2 = butler.get("raw", dataId=data_id2, collections="LSSTCam/raw/all")
    row = cdb_table[(cdb_table["visit_id"] == exposure_id) & (cdb_table["detector"] == detector_name)]
    zk = getzk(row)
    return (data_1, detector_name, zk), (data_2, detector_name + 1, zk)


def zernikes_to_dof(
    filter_name: str,
    measured_zk: np.ndarray,
    sensor_names: list[str],
    rotation_angle: float,
    ofc_data,
    trunc_index: int | None = 12,
    verbose: bool = True,
):
    """Estimate DOF state from measured Zernikes.

    Solves y = A * (W * x_dof), where W is normalization_weights.

    Parameters
    ----------
    filter_name : str
        Optical filter name.
    measured_zk : ndarray
        Measured Zernike coefficients [#sensors, #Zernikes].
    sensor_names : list[str]
        List of sensors to use.
    rotation_angle : float
        Rotation angle in degrees.
    ofc_data : OfcData
        OFC configuration.
    trunc_index : int | None
        If set, number of singular values to keep in SVD (truncated SVD).
        If None, all nonzero singular values are used.
    verbose : bool
        Print debug diagnostics.
    """
    # --- Adjust Zernike range ---
    n_zk_meas = measured_zk.shape[1]
    zn_idx = np.arange(n_zk_meas)

    # --- Field rotation ---
    field_angles = np.array([ofc_data.sample_points[s] for s in sensor_names])
    rot_rad = np.deg2rad(-rotation_angle)
    rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
    field_angles = field_angles @ rot_mat

    # --- Sensitivity matrix (Zernike × DOF) ---
    dz_matrix = SensitivityMatrix(ofc_data)
    sens = dz_matrix.evaluate(field_angles, 0.0)
    sens = sens[:, zn_idx, :].reshape((-1, sens.shape[-1]))
    valid_idx = [i for i in ofc_data.dof_idx if i < sens.shape[-1]]
    sens = sens[:, valid_idx]

    # --- Apply normalization once (LSST convention) ---
    norm_mat = np.diag(ofc_data.normalization_weights[valid_idx])
    sens = sens @ norm_mat

    # --- Build target vector (measured - intrinsic - static) ---
    intrinsic = get_intrinsic_zernikes(ofc_data, filter_name, sensor_names, rotation_angle)[:, zn_idx]
    y2_corr = np.array([ofc_data.y2_correction[s] for s in sensor_names])[:, zn_idx]
    y = (measured_zk - intrinsic - y2_corr).reshape(-1, 1)

    # --- SVD pseudo-inverse with truncation index ---
    U, S, Vh = svd(sens, full_matrices=False)
    if trunc_index is None:
        trunc_index = len(S)  # keep all
    trunc_index = min(trunc_index, len(S))  # safety

    if verbose:
        print(f"Using {trunc_index}/{len(S)} singular values")

    # Zero-out smaller singular values
    S_inv = np.zeros_like(S)
    S_inv[:trunc_index] = 1.0 / S[:trunc_index]

    # Reconstruct pseudo-inverse
    A_pinv = Vh.T @ np.diag(S_inv) @ U.T

    # --- Solve for DOFs ---
    x_dof = A_pinv @ y

    if verbose:
        print(f"[Z→DOF] sens: {sens.shape}, y: {y.shape}, x_dof: {x_dof.shape}")
        print(f"||y||={np.linalg.norm(y):.3f}, ||A@x||={np.linalg.norm(sens @ x_dof):.3f}")
        print(f"x_dof range: {x_dof.min():.3f} → {x_dof.max():.3f}")

    return x_dof.ravel()


def dof_to_zernikes(
    filter_name: str,
    x_dof: np.ndarray,
    sensor_names: list[str],
    rotation_angle: float,
    ofc_data,
    n_zk_target: int | None = None,
    measured_zk: np.ndarray | None = None,
    verbose: bool = True,
):
    """Predict Zernikes from DOF state (forward model).

    Returns total wavefront = intrinsic + static + misalignment term.
    """
    # --- Field rotation ---
    field_angles = np.array([ofc_data.sample_points[s] for s in sensor_names])
    rot_rad = np.deg2rad(-rotation_angle)
    rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
    field_angles = field_angles @ rot_mat

    # --- Sensitivity matrix ---
    dz_matrix = SensitivityMatrix(ofc_data)
    sens = dz_matrix.evaluate(field_angles, 0.0)
    n_zk_total = sens.shape[1]
    if n_zk_target is None or n_zk_target > n_zk_total:
        n_zk_target = n_zk_total
    zn_idx = np.arange(n_zk_target)
    sens = sens[:, zn_idx, :].reshape((-1, sens.shape[-1]))
    sens = sens[:, ofc_data.dof_idx]

    # --- Normalize ---
    norm_mat = np.diag(ofc_data.normalization_weights[ofc_data.dof_idx])
    x_dof = x_dof.reshape(-1, 1)
    zk_pred = (sens @ norm_mat @ x_dof).reshape(len(sensor_names), n_zk_target)

    # --- Add intrinsic + static ---
    intrinsic = get_intrinsic_zernikes(ofc_data, filter_name, sensor_names, rotation_angle)
    y2_corr = np.array([ofc_data.y2_correction[s] for s in sensor_names])
    zk_total = intrinsic[:, zn_idx] + y2_corr[:, zn_idx] + zk_pred

    # --- Diagnostics ---
    if verbose:
        print(f"[DOF→Z] sens: {sens.shape}, x_dof: {x_dof.shape}, zk_total: {zk_total.shape}")
        if measured_zk is not None:
            n_meas = min(measured_zk.shape[1], n_zk_target)
            diff = measured_zk[:, :n_meas] - zk_total[:, :n_meas]
            rms_nm = np.sqrt(np.mean(diff**2)) * 1e3  # assuming µm input
            print(f"RMS difference (meas vs recon): {rms_nm:.3f} nm")
            if rms_nm > 100:
                print("⚠️  Warning: Possible unit or normalization mismatch.")

    return zk_total


# ============================================================================
# PyTorch DOF Conversion Functions
# ============================================================================

# Global cache for precomputed PyTorch matrices
_PRECOMPUTED_PYTORCH_MATRICES: Optional[Dict[str, Any]] = None


def precompute_pytorch_dof_matrices(
    ofc_data,
    sensor_names: Optional[list[str]] = None,
    filter_names: Optional[list[str]] = None,
    n_zk_max: int = 28,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Precompute PyTorch matrices for DOF conversion at rotation_angle=0.

    This function precomputes sensitivity matrices, intrinsic Zernikes, and
    corrections for each sensor individually at rotation_angle=0. These matrices
    can then be used by the PyTorch conversion functions.

    Parameters
    ----------
    ofc_data : OFCData
        OFC configuration data.
    sensor_names : list[str], optional
        List of sensor names to precompute. Defaults to ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"].
    filter_names : list[str], optional
        List of filter names to precompute intrinsic Zernikes for.
        Defaults to ["U", "G", "R", "I", "Z", "Y"].
    n_zk_max : int, default=28
        Maximum number of Zernikes to precompute.
    device : str, default="cpu"
        Device to store tensors on.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing precomputed matrices:
        - 'sensor_sensitivity_matrices': Dict[str, torch.Tensor] - Per-sensor sensitivity matrices
        - 'intrinsic_zernikes': Dict[str, Dict[str, torch.Tensor]] - Per-filter, per-sensor intrinsic Zernikes
        - 'y2_corrections': Dict[str, torch.Tensor] - Per-sensor y2 corrections
        - 'normalization_weights': torch.Tensor - DOF normalization weights
        - 'dof_idx': torch.Tensor - DOF indices
        - 'n_zk_total': int - Total number of Zernikes
        - 'n_dof': int - Number of DOFs
    """
    global _PRECOMPUTED_PYTORCH_MATRICES

    if sensor_names is None:
        sensor_names = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
    if filter_names is None:
        filter_names = ["U", "G", "R", "I", "Z", "Y"]

    logger.info(f"Precomputing PyTorch DOF matrices for {len(sensor_names)} sensors at rotation_angle=0")

    # Initialize sensitivity matrix evaluator
    dz_matrix = SensitivityMatrix(ofc_data)

    # Get DOF indices and normalization weights
    dof_idx_np = np.array(ofc_data.dof_idx)
    dof_idx = torch.tensor(ofc_data.dof_idx, dtype=torch.long, device=device)
    normalization_weights = torch.tensor(ofc_data.normalization_weights, dtype=torch.float64, device=device)

    # Determine valid DOF indices by checking first sensor
    # (all sensors should have same DOF structure)
    first_sensor_field = np.array([ofc_data.sample_points[sensor_names[0]]])
    first_sens = dz_matrix.evaluate(first_sensor_field, 0.0)
    n_dof_total = first_sens.shape[-1]
    valid_dof_mask_np = dof_idx_np < n_dof_total
    valid_dof_idx_np = dof_idx_np[valid_dof_mask_np]
    valid_dof_mask = torch.tensor(valid_dof_mask_np, dtype=torch.bool, device=device)
    valid_dof_idx = dof_idx[valid_dof_mask]

    # Precompute per-sensor sensitivity matrices at rotation_angle=0
    sensor_sensitivity_matrices: Dict[str, torch.Tensor] = {}
    sensor_intrinsic_zernikes: Dict[str, Dict[str, torch.Tensor]] = {filt: {} for filt in filter_names}
    sensor_y2_corrections: Dict[str, torch.Tensor] = {}

    for sensor_name in sensor_names:
        # Get field angle for this sensor (no rotation)
        field_angle = np.array([ofc_data.sample_points[sensor_name]])

        # Evaluate sensitivity matrix for this sensor
        sens = dz_matrix.evaluate(field_angle, 0.0)  # Shape: (1, n_zk, n_dof)
        sens = sens[0, :n_zk_max, :]  # Shape: (n_zk_max, n_dof)

        # Select valid DOF indices (use precomputed mask)
        sens = sens[:, valid_dof_idx_np]  # Shape: (n_zk_max, n_valid_dof)

        # Apply normalization
        norm_weights = normalization_weights[valid_dof_mask]
        norm_mat = torch.diag(norm_weights)
        sens_normalized = torch.tensor(sens, dtype=torch.float64, device=device) @ norm_mat

        # Store per-sensor matrix: (n_zk_max, n_valid_dof)
        sensor_sensitivity_matrices[sensor_name] = sens_normalized

        # Precompute intrinsic Zernikes for each filter (at rotation_angle=0)
        for filter_name in filter_names:
            intrinsic = get_intrinsic_zernikes(ofc_data, filter_name, [sensor_name], 0.0)
            sensor_intrinsic_zernikes[filter_name][sensor_name] = torch.tensor(
                intrinsic[0, :n_zk_max], dtype=torch.float64, device=device
            )

        # Store y2 correction for this sensor
        y2_corr = ofc_data.y2_correction[sensor_name]
        sensor_y2_corrections[sensor_name] = torch.tensor(
            y2_corr[:n_zk_max], dtype=torch.float64, device=device
        )

    n_zk_total = n_zk_max
    n_dof = len(valid_dof_idx)

    result = {
        "sensor_sensitivity_matrices": sensor_sensitivity_matrices,
        "intrinsic_zernikes": sensor_intrinsic_zernikes,
        "y2_corrections": sensor_y2_corrections,
        "normalization_weights": normalization_weights,
        "dof_idx": dof_idx,
        "valid_dof_mask": valid_dof_mask,
        "valid_dof_idx": valid_dof_idx,
        "n_zk_total": n_zk_total,
        "n_dof": n_dof,
        "sensor_names": sensor_names,
        "filter_names": filter_names,
    }

    _PRECOMPUTED_PYTORCH_MATRICES = result
    logger.info(f"Precomputation complete: {n_zk_total} Zernikes, {n_dof} DOFs")

    return result


def _get_precomputed_matrices() -> Dict[str, Any]:
    """Get precomputed matrices, raising error if not initialized."""
    if _PRECOMPUTED_PYTORCH_MATRICES is None:
        raise RuntimeError(
            "Precomputed matrices not initialized. Call precompute_pytorch_dof_matrices() first."
        )
    return _PRECOMPUTED_PYTORCH_MATRICES


def _build_rotated_sensitivity_matrix(
    sensor_names: list[str],
    rotation_angle: float,
    ofc_data,
    n_zk: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Build sensitivity matrix for given sensors and rotation angle.

    This function rotates field angles and evaluates the sensitivity matrix.
    Used when rotation is needed at runtime.
    """
    # Rotate field angles
    field_angles = np.array([ofc_data.sample_points[s] for s in sensor_names])
    rot_rad = np.deg2rad(-rotation_angle)
    rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
    field_angles_rotated = field_angles @ rot_mat

    # Evaluate sensitivity matrix
    dz_matrix = SensitivityMatrix(ofc_data)
    sens = dz_matrix.evaluate(field_angles_rotated, 0.0)  # Shape: (n_sensors, n_zk, n_dof)
    sens = sens[:, :n_zk, :]  # Shape: (n_sensors, n_zk, n_dof)

    # Reshape to (n_sensors * n_zk, n_dof)
    n_sensors, n_zk, n_dof_total = sens.shape
    sens = sens.reshape(-1, n_dof_total)

    # Select valid DOF indices
    valid_idx = [i for i in ofc_data.dof_idx if i < sens.shape[-1]]
    sens = sens[:, valid_idx]

    # Apply normalization
    norm_mat = np.diag(ofc_data.normalization_weights[valid_idx])
    sens_normalized = sens @ norm_mat

    return torch.tensor(sens_normalized, dtype=torch.float64, device=device)


def zernikes_to_dof_torch(
    filter_name: str,
    measured_zk: torch.Tensor,
    sensor_names: list[str],
    rotation_angle: float = 0.0,
    ofc_data=None,
    trunc_index: int | None = 50,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """Estimate DOF state from measured Zernikes using Pytorch.

    This function uses precomputed matrices when rotation_angle=0, otherwise
    it requires ofc_data to compute rotated matrices.

    Parameters
    ----------
    filter_name : str
        Optical filter name (e.g., "R", "Z", "I").
    measured_zk : torch.Tensor
        Measured Zernike coefficients [#sensors, #Zernikes].
    sensor_names : list[str]
        List of sensors to use. Must be a subset of precomputed sensors.
    rotation_angle : float, default=0.0
        Rotation angle in degrees. If 0.0, uses precomputed matrices.
    ofc_data : OFCData, optional
        OFC configuration. Required if rotation_angle != 0.0.
    trunc_index : int | None, default=12
        If set, number of singular values to keep in SVD (truncated SVD).
        If None, all nonzero singular values are used.
    device : str, default="cpu"
        Device for computation.
    verbose : bool, default=True
        Print debug diagnostics.

    Returns
    -------
    torch.Tensor
        DOF state vector [n_dof].
    """
    # Ensure measured_zk is on correct device and dtype
    measured_zk = measured_zk.to(device=device, dtype=torch.float64)
    n_sensors, n_zk_meas = measured_zk.shape

    # Handle rotation: if rotation_angle != 0, we need ofc_data to recompute
    if abs(rotation_angle) > 1e-6:
        if ofc_data is None:
            raise ValueError(
                "ofc_data is required when rotation_angle != 0. "
                "Either set rotation_angle=0 or provide ofc_data."
            )
        # Build rotated sensitivity matrix
        sens = _build_rotated_sensitivity_matrix(sensor_names, rotation_angle, ofc_data, n_zk_meas, device)
        # Get intrinsic and y2 corrections with rotation
        intrinsic_list = []
        y2_corr_list = []
        for sensor_name in sensor_names:
            intrinsic = get_intrinsic_zernikes(ofc_data, filter_name, [sensor_name], rotation_angle)
            intrinsic_list.append(torch.tensor(intrinsic[0, :n_zk_meas], dtype=torch.float64, device=device))
            y2_corr = ofc_data.y2_correction[sensor_name]
            y2_corr_list.append(torch.tensor(y2_corr[:n_zk_meas], dtype=torch.float64, device=device))
        intrinsic = torch.stack(intrinsic_list)
        y2_corr = torch.stack(y2_corr_list)
    else:
        # Use precomputed matrices
        matrices = _get_precomputed_matrices()

        # Verify sensor_names are in precomputed set
        precomputed_sensors = matrices["sensor_names"]
        if not all(s in precomputed_sensors for s in sensor_names):
            raise ValueError(
                f"All sensor_names must be in precomputed set {precomputed_sensors}. " f"Got: {sensor_names}"
            )

        # Build combined sensitivity matrix from precomputed per-sensor matrices
        sens_list = []
        intrinsic_list = []
        y2_corr_list = []

        for sensor_name in sensor_names:
            # Get precomputed sensitivity matrix for this sensor
            sens_sensor = matrices["sensor_sensitivity_matrices"][sensor_name]
            # Select only the Zernikes we need
            sens_sensor = sens_sensor[:n_zk_meas, :]
            sens_list.append(sens_sensor)

            # Get intrinsic Zernikes for this filter and sensor
            intrinsic_sensor = matrices["intrinsic_zernikes"][filter_name.upper()][sensor_name]
            intrinsic_list.append(intrinsic_sensor[:n_zk_meas])

            # Get y2 correction for this sensor
            y2_sensor = matrices["y2_corrections"][sensor_name]
            y2_corr_list.append(y2_sensor[:n_zk_meas])

        # Stack: (n_sensors * n_zk_meas, n_dof)
        sens = torch.cat(sens_list, dim=0)
        intrinsic = torch.stack(intrinsic_list)
        y2_corr = torch.stack(y2_corr_list)

    # Build target vector: measured - intrinsic - static
    y = (measured_zk - intrinsic - y2_corr).reshape(-1, 1)

    # SVD pseudo-inverse with truncation index
    U, S, Vh = torch.linalg.svd(sens, full_matrices=False)
    if trunc_index is None:
        trunc_index = len(S)  # keep all
    trunc_index = min(trunc_index, len(S))  # safety

    if verbose:
        print(f"Using {trunc_index}/{len(S)} singular values")

    # Zero-out smaller singular values
    S_inv = torch.zeros_like(S)
    S_inv[:trunc_index] = 1.0 / S[:trunc_index]

    # Reconstruct pseudo-inverse
    A_pinv = Vh.T @ torch.diag(S_inv) @ U.T

    # Solve for DOFs
    x_dof = A_pinv @ y

    if verbose:
        print(f"[Z→DOF] sens: {sens.shape}, y: {y.shape}, x_dof: {x_dof.shape}")
        y_norm = torch.linalg.norm(y).item()
        Ax_norm = torch.linalg.norm(sens @ x_dof).item()
        print(f"||y||={y_norm:.3f}, ||A@x||={Ax_norm:.3f}")
        print(f"x_dof range: {x_dof.min().item():.3f} → {x_dof.max().item():.3f}")

    return x_dof.ravel()


def dof_to_zernikes_torch(
    filter_name: str,
    x_dof: torch.Tensor,
    sensor_names: list[str],
    rotation_angle: float = 0.0,
    ofc_data=None,
    n_zk_target: int | None = None,
    measured_zk: torch.Tensor | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """Predict Zernikes from DOF state using Pytorch (forward model).

    This function uses precomputed matrices when rotation_angle=0, otherwise
    it requires ofc_data to compute rotated matrices.

    Parameters
    ----------
    filter_name : str
        Optical filter name (e.g., "R", "Z", "I").
    x_dof : torch.Tensor
        DOF state vector [n_dof] or [n_dof, 1].
    sensor_names : list[str]
        List of sensors to use. Must be a subset of precomputed sensors.
    rotation_angle : float, default=0.0
        Rotation angle in degrees. If 0.0, uses precomputed matrices.
    ofc_data : OFCData, optional
        OFC configuration. Required if rotation_angle != 0.0.
    n_zk_target : int, optional
        Number of Zernikes to predict. Defaults to n_zk_meas or precomputed max.
    measured_zk : torch.Tensor, optional
        Measured Zernikes for comparison/validation.
    device : str, default="cpu"
        Device for computation.
    verbose : bool, default=True
        Print debug diagnostics.

    Returns
    -------
    torch.Tensor
        Predicted Zernike coefficients [#sensors, #Zernikes].
    """
    # Ensure x_dof is on correct device and dtype
    x_dof = x_dof.to(device=device, dtype=torch.float64)
    if x_dof.dim() == 1:
        x_dof = x_dof.unsqueeze(1)

    # Determine number of Zernikes
    # If not specified, infer from measured_zk if provided, otherwise use precomputed max
    if n_zk_target is None:
        if measured_zk is not None:
            n_zk_target = measured_zk.shape[1]
        else:
            # Default to precomputed max, but this might need adjustment based on actual usage
            matrices = _get_precomputed_matrices()
            n_zk_target = matrices["n_zk_total"]
    # Note: n_zk_target may be adjusted later based on actual sensitivity matrix shape

    # Handle rotation: if rotation_angle != 0, we need ofc_data to recompute
    if abs(rotation_angle) > 1e-6:
        if ofc_data is None:
            raise ValueError(
                "ofc_data is required when rotation_angle != 0. "
                "Either set rotation_angle=0 or provide ofc_data."
            )
        # Build rotated sensitivity matrix
        sens = _build_rotated_sensitivity_matrix(sensor_names, rotation_angle, ofc_data, n_zk_target, device)
        # Get intrinsic and y2 corrections with rotation
        intrinsic_list = []
        y2_corr_list = []
        for sensor_name in sensor_names:
            intrinsic = get_intrinsic_zernikes(ofc_data, filter_name, [sensor_name], rotation_angle)
            intrinsic_list.append(
                torch.tensor(intrinsic[0, :n_zk_target], dtype=torch.float64, device=device)
            )
            y2_corr = ofc_data.y2_correction[sensor_name]
            y2_corr_list.append(torch.tensor(y2_corr[:n_zk_target], dtype=torch.float64, device=device))
        intrinsic = torch.stack(intrinsic_list)
        y2_corr = torch.stack(y2_corr_list)
    else:
        # Use precomputed matrices
        matrices = _get_precomputed_matrices()

        # Verify sensor_names are in precomputed set
        precomputed_sensors = matrices["sensor_names"]
        if not all(s in precomputed_sensors for s in sensor_names):
            raise ValueError(
                f"All sensor_names must be in precomputed set {precomputed_sensors}. " f"Got: {sensor_names}"
            )

        # Build combined sensitivity matrix from precomputed per-sensor matrices
        sens_list = []
        intrinsic_list = []
        y2_corr_list = []

        for sensor_name in sensor_names:
            # Get precomputed sensitivity matrix for this sensor
            sens_sensor = matrices["sensor_sensitivity_matrices"][sensor_name]
            # Select only the Zernikes we need
            sens_sensor = sens_sensor[:n_zk_target, :]
            sens_list.append(sens_sensor)

            # Get intrinsic Zernikes for this filter and sensor
            intrinsic_sensor = matrices["intrinsic_zernikes"][filter_name.upper()][sensor_name]
            intrinsic_list.append(intrinsic_sensor[:n_zk_target])

            # Get y2 correction for this sensor
            y2_sensor = matrices["y2_corrections"][sensor_name]
            y2_corr_list.append(y2_sensor[:n_zk_target])

        # Stack: (n_sensors * n_zk_target, n_dof)
        sens = torch.cat(sens_list, dim=0)
        intrinsic = torch.stack(intrinsic_list)
        y2_corr = torch.stack(y2_corr_list)

    # Predict Zernikes from DOF: sens @ x_dof
    # Infer n_zk from sensitivity matrix shape (more reliable than n_zk_target parameter)
    # This handles cases where n_zk_target doesn't match the actual matrix dimensions
    n_sensors_actual = len(sensor_names)
    n_zk_from_sens = sens.shape[0] // n_sensors_actual

    # Verify the shape makes sense
    if sens.shape[0] % n_sensors_actual != 0:
        raise ValueError(
            f"Sensitivity matrix shape {sens.shape} is not compatible with "
            f"{n_sensors_actual} sensors. Expected n_rows to be divisible by n_sensors."
        )
    zk_pred_flat = sens @ x_dof  # Shape: (n_sensors * n_zk_from_sens, 1)
    zk_pred = zk_pred_flat.reshape(n_sensors_actual, n_zk_from_sens)

    # Trim intrinsic and y2_corr to match if needed
    if intrinsic.shape[1] > n_zk_from_sens:
        intrinsic = intrinsic[:, :n_zk_from_sens]
    if y2_corr.shape[1] > n_zk_from_sens:
        y2_corr = y2_corr[:, :n_zk_from_sens]

    # Add intrinsic + static corrections
    zk_total = intrinsic + y2_corr + zk_pred

    # Diagnostics
    if verbose:
        print(f"[DOF→Z] sens: {sens.shape}, x_dof: {x_dof.shape}, zk_total: {zk_total.shape}")
        print(f"Using {n_sensors_actual} sensors, {n_zk_from_sens} Zernikes per sensor")
        if measured_zk is not None:
            measured_zk = measured_zk.to(device=device, dtype=torch.float64)
            n_meas = min(measured_zk.shape[1], n_zk_from_sens)
            n_meas_sensors = min(measured_zk.shape[0], n_sensors_actual)
            diff = measured_zk[:n_meas_sensors, :n_meas] - zk_total[:n_meas_sensors, :n_meas]
            rms_nm = torch.sqrt(torch.mean(diff**2)).item() * 1e3  # assuming µm input
            print(f"RMS difference (meas vs recon): {rms_nm:.3f} nm")
            if rms_nm > 100:
                print("⚠️  Warning: Possible unit or normalization mismatch.")

    return zk_total
