"""Utility functions."""
import os
from pathlib import Path
from random import randint
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from astropy.io import fits
from torch import nn
# from lsst.summit.utils import ConsDbClient

from .KERNEL import CUTOUT as DONUT

import pytorch_lightning as pl
from torch.ao.quantization.qconfig import default_qat_qconfig
from typing import Optional
import yaml
import warnings

# Optional imports
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    git = None

from lsst.daf.butler import Butler
from lsst.ip.isr import AssembleCcdTask
from lsst.meas.algorithms import subtractBackground
from lsst.ts.imsim.imsim_cmpt import ImsimCmpt
LSST_AVAILABLE = True




MAP_DETECTOR_TO_NUMBER = {
    'R00_SW0': 191,
    'R00_SW1': 192,
    'R04_SW0': 195,
    'R04_SW1': 196,
    'R40_SW0': 197,
    'R40_SW1': 198,
    'R44_SW0': 201,
    'R44_SW1': 202,
}

def safe_yaml_load(file_path: str):
    """Safely load YAML files that may contain Python objects like tuples.

    Parameters
    ----------
    file_path : str
        Path to the YAML file to load

    Returns
    -------
    dict
        Loaded YAML content as dictionary
    """
    try:
        # First try standard loader
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        # If that fails, try with FullLoader which can handle more Python objects
        try:
            with open(file_path, 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError:
            # If both fail, try to load and fix common issues
            with open(file_path, 'r') as f:
                content = f.read()

            # Replace problematic tuple tags with list format
            content = content.replace('!!python/tuple', '')
            content = content.replace('tag:yaml.org,2002:python/tuple', '')

            # Try to load the cleaned content
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError:
                # Last resort: return empty dict and warn
                warnings.warn(f"Could not parse YAML file {file_path}: {e}")
                return {}


class QuantizationAwareTrainingCallback(pl.Callback):
    """Generic Callback for Quantization Aware Training (QAT) with PyTorch Lightning.

    Allows specifying the model attribute name (e.g., 'alignnet', 'wavenet').
    """
    def __init__(
        self,
        model_attr: str = "alignnet",
        start_epoch: int = 5,
        quantization_backend: str = "fbgemm",
        qconfig_dict: Optional[dict] = None
    ):
        """Initialize the QuantizationAwareTrainingCallback.

        Parameters
        ----------
        model_attr : str, default="alignnet"
            Name of the model attribute to quantize.
        start_epoch : int, default=5
            Epoch to start quantization.
        quantization_backend : str, default="fbgemm"
            Backend for quantization.
        qconfig_dict : Optional[dict], default=None
            Quantization configuration dictionary.
        """
        super().__init__()
        self.model_attr = model_attr
        self.start_epoch = start_epoch
        self.quantization_backend = quantization_backend
        self.qconfig_dict = qconfig_dict
        self.qat_enabled = False

    def on_train_epoch_start(self, trainer, pl_module):
        """Enable QAT at the start of training epoch."""
        if not self.qat_enabled and trainer.current_epoch >= self.start_epoch:
            self._enable_qat(pl_module)
            self.qat_enabled = True

    def _enable_qat(self, pl_module):
        print(f"🔥 Enabling Quantization Aware Training at epoch {self.start_epoch}")
        torch.backends.quantized.engine = self.quantization_backend

        model = getattr(pl_module, self.model_attr)
        model.qconfig = default_qat_qconfig

        if self.qconfig_dict:
            for module_name, qconfig in self.qconfig_dict.items():
                module = model
                for attr in module_name.split('.'):
                    module = getattr(module, attr)
                module.qconfig = qconfig

        torch.quantization.prepare_qat(model, inplace=True)
        print("✅ Model prepared for Quantization Aware Training")

    def on_train_end(self, trainer, pl_module):
        """Disable QAT at the end of training."""
        if self.qat_enabled:
            print("🔧 Converting QAT model to quantized model...")
            model = getattr(pl_module, self.model_attr)
            model.eval()
            quantized_model = torch.quantization.convert(model)
            quantized_path = f"{trainer.default_root_dir}/quantized_{self.model_attr}_model.pth"
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'model_architecture': type(quantized_model).__name__,
                'quantization_config': self.qconfig_dict,
                'backend': self.quantization_backend,
            }, quantized_path)
            print(f"💾 Quantized model saved to: {quantized_path}")
            original_size = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
            print(f"📊 Original model parameters: {original_size:.2f}MB equivalent")
            print("📊 Quantized model should be ~4x smaller for int8 quantization")


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
        current_lr = trainer.optimizers[0].param_groups[0]['lr']

        if current_lr < self.threshold:
            print(f"Learning rate {current_lr:.2e} below threshold {self.threshold:.2e}. Stopping training.")
            trainer.should_stop = True


BAND_MAP = {  # type: ignore
        0: 0.3671,
        1: 0.4827,
        2: 0.6223,
        3: 0.7546,
        4: 0.8691,
        5: 0.9712,
    }
BAND_str_int = {  # type: ignore
    'u': 0,
    'g': 1,
    'r': 2,
    'i': 3,
    'z': 4,
    'y': 5,
}
FIELD = {
    'R00': {'fieldx': -1.1897, 'fieldy': -1.1897},
    'R04': {'fieldx': -1.1897, 'fieldy': 1.1897},
    'R40': {'fieldx': 1.1897, 'fieldy': -1.1897},
    'R44': {'fieldx': 1.1897, 'fieldy': 1.1897}
}


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


def convert_zernikes_deploy(
    zernikes: torch.Tensor, device='cpu'
) -> torch.Tensor:
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
    arcsec_per_micron = torch.tensor(
        [
            0.75453564,  # Z4
            0.2711315,   # Z5
            0.2711315,   # Z6
            0.82237164,  # Z7
            0.82237164,  # Z8
            0.39572461,  # Z9
            0.39572461,  # Z10
            1.68719298,  # Z11
            0.93956571,  # Z12
            0.93956571,  # Z13
            0.51643078,  # Z14
            0.51643078,  # Z15
            0.63498144,  # Z20
            0.63498144,  # Z21
            2.82321385,  # Z22
            0.75241174,  # Z27
            0.75241174   # Z28
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
        return sum(
            params.numel() for params in model.parameters() if params.requires_grad
        )
    else:
        return sum(params.numel() for params in model.parameters())


def printOnce(msg: str, header: bool = False) -> None:
    """Print message once to the terminal.

    This avoids the problem where statements get printed multiple times in
    a distributed setting.

    Parameters
    ----------
    msg: str
        Message to print
    header: bool, default=False
        Whether to add extra space and underline for the message
    """
    rank = os.environ.get("LOCAL_RANK", None)
    if rank is None or rank == "0":
        if header:
            msg = f"\n{msg}\n{'-'*len(msg)}\n"
        print(msg)


def transform_inputs(
    image: np.ndarray,
    fx: float,
    fy: float,
    intra: bool,
    band: int,
) -> tuple[np.ndarray, float, float, float, float]:
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
    field_mean = 0.000
    field_std = 0.021
    fx = (fx - field_mean) / field_std
    fy = (fy - field_mean) / field_std

    # normalize the intrafocal flags
    intra_mean = 0.5
    intra_std = 0.5
    intra = (intra - intra_mean) / intra_std  # type: ignore

    # get the effective wavelength in microns
    band = BAND_MAP[band]  # type: ignore

    # normalize the wavelength
    band_mean = 0.710
    band_std = 0.174
    band = (band - band_mean) / band_std  # type: ignore

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
    root = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)
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
    stds = torch.tensor([std_1, std_2, std_3, std_4])
    means = torch.tensor([mean_1, mean_2, mean_3, mean_4])
    ind = torch.argmin(stds)
    return stds[ind], means[ind], ind


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
        return 'up'
    elif ind == 1:
        return 'left'
    elif ind == 2:
        return 'down'
    else:
        return 'right'


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
    if direction == 'left':
        random_x = randint(-adjust, 0)
        random_y = randint(-adjust, adjust)
    elif direction == 'right':
        random_x = randint(0, adjust)
        random_y = randint(-adjust, adjust)
    elif direction == 'down':
        random_x = randint(-adjust, adjust)
        random_y = randint(-adjust, 0)
    else:
        random_x = randint(-adjust, adjust)
        random_y = randint(0, adjust)
    std, mean, _ = noise_est(frame)
    backplate = torch.empty(160*3, 160*3).normal_(mean=mean, std=std)
    backplate[160 + random_y:160*2 + random_y, 160+random_x:160*2+random_x] = frame
    if return_offset:
        return backplate[160:160*2, 160:160*2], [random_x, random_y]
    else:
        return backplate[160:160*2, 160:160*2]


def filepath_to_numpy(filename, opd_filename, field='2',
                      repo_path='../temp', return_data=False,
                      return_angle=False,
                      noll_indices=None):
    """Convert FITS file to numpy array with associated metadata and Zernike coefficients.

    Parameters
    ----------
    filename : str
        Path to the FITS file containing the exposure data.
    opd_filename : str
        Path to the OPD (Optical Path Difference) file containing Zernike truth data.
    field : str, optional, default='2'
        Field identifier for the exposure.
    repo_path : str, optional, default='../temp'
        Path to the Butler repository for data access.
    return_data : bool, optional, default=False
        Whether to return the raw exposure data along with processed arrays.
    return_angle : bool, optional, default=False
        Whether to return rotation angle information from dataset file.
    noll_indices : array-like, optional
        Specific Noll indices to extract from Zernike coefficients.

    Returns
    -------
    tuple
        Contains processed image array, header information, and true Zernike coefficients.
        If return_data=True, also includes raw exposure data.
        If return_angle=True, also includes rotation angle.
    """
    if not LSST_AVAILABLE:
        raise ImportError("LSST dependencies not available. This function requires LSST installation.")

    with fits.open(filename) as hdulist:
        # Extract data and header from the primary HDU
        header = hdulist[0].header

    butler = Butler(repo_path, collections=["LSSTCam/raw/all", "LSSTCam/calib"], writeable=True)
    try:
        seqnum = ''
        for i in range(4-len(str(header['SEQNUM']))):
            seqnum += '0'
        seqnum += str(header['SEQNUM'])
        exposure_id = header['DAYOBS'] + '0' + str(seqnum)
        exposure_id = exposure_id[1:]
        exposure_id = field + exposure_id

        detector = int(header['OUTFILE'].split('-')[-1].split('.')[0].replace('det', ''))

        data_id = {
            "instrument": "LSSTCam",  # Replace with your instrument
            "exposure": int(exposure_id),  # Replace with your exposure ID
            "detector": detector
        }
        print(data_id)
        data = butler.get('raw', dataId=data_id, collections="LSSTCam/raw/all")
    except Exception:
        os.system(f'butler ingest-raws {repo_path} {filename} > /dev/null 2>&1')

        seqnum = ''
        for i in range(4-len(str(header['SEQNUM']))):
            seqnum += '0'
        seqnum += str(header['SEQNUM'])
        exposure_id = header['DAYOBS'] + '0' + str(seqnum)
        exposure_id = exposure_id[1:]
        exposure_id = field + exposure_id

        detector = int(header['OUTFILE'].split('-')[-1].split('.')[0].replace('det', ''))

        data_id = {
            "instrument": "LSSTCam",  # Replace with your instrument
            "exposure": int(exposure_id),  # Replace with your exposure ID
            "detector": detector
        }
        data = butler.get('raw', dataId=data_id, collections="LSSTCam/raw/all")

    assembleCcdTask = AssembleCcdTask()
    new = assembleCcdTask.assembleCcd(data)
    SubtractBackground = subtractBackground.SubtractBackgroundTask()
    SubtractBackground.run(new)
    image = new.getImage().array

    obj = ImsimCmpt(25)
    obj.opd_file_path = opd_filename
    print(opd_filename)
    opd_data = obj._map_opd_to_zk(0, 4)

    true_zk = {}
    for sensor_id, opd_zk in zip([191, 195, 199, 203], opd_data):
        true_zk[sensor_id] = opd_zk[:25]

    detector = detector - int(header['CCDSLOT'].replace('SW', ''))
    zk_true = torch.tensor(np.array([true_zk[detector]]))/1000

    if return_data:
        if return_angle:
            print(header['SEQNUM'])
            df = pd.read_csv("../dataset.txt", sep=r'\s+')
            row = df[df["SEQID"] == int(header['SEQNUM'])]
            rotation_angle = row['RTP'].tolist()[0]  # degrees
            rotation_angle = np.radians(rotation_angle)
            return image, header, zk_true, data, rotation_angle
        else:
            return image, header, zk_true, data
    else:
        if return_angle:
            print(header['SEQNUM'])
            df = pd.read_csv("../dataset.txt", sep=r'\s+')
            row = df[df["SEQID"] == int(header['SEQNUM'])]
            rotation_angle = row['RTP'].tolist()[0]  # degrees
            rotation_angle = np.radians(rotation_angle)
            return image, header, zk_true, rotation_angle
        else:
            return image, header, zk_true


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
    crops = torch.stack([
        image_tensor[:, t:t + crop_size, l:l + crop_size]
        for t, l in zip(top, left)
    ])

    return crops


def get_centers(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Generate grid of center coordinates for cropping patches from an image.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (H, W).
    crop_size : int
        Size of square crops to generate centers for.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2) containing (x, y) center coordinates for non-overlapping crops.
    """
    H, W = image.shape
    x_coords = torch.arange(crop_size // 2, W, crop_size)
    y_coords = torch.arange(crop_size // 2, H, crop_size)

    # Create a grid of center coordinates
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # Shape: [N, 2]

    return centers


def single_conv(image, device='cuda'):
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
    donut = DONUT
    donut = donut[40:200, 40:200].float().to(device)
    not_donut = (1-donut).bool().float().to(device)
    donut_mean = torch.mean(image * donut)
    not_donut_mean = torch.mean(image * not_donut)
    not_donut_std = torch.std(image * not_donut)
    sigma_dev = abs(donut_mean-not_donut_mean)/not_donut_std
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

    if len(keep_images) == 0:
        return np.array([[]]), np.array([[]]), np.array([[]])
    else:
        keep_images = torch.concatenate(keep_images)
        img_index = torch.tensor(img_index)
        return keep_images, img_index, snr_list


class CORALLoss(nn.Module):
    """CORAL (Correlation Alignment) loss for domain adaptation.

    This loss function aligns the second-order statistics of source and target
    feature distributions to reduce domain shift.
    """
    def __init__(self):
        """Initialize the CORAL loss."""
        super(CORALLoss, self).__init__()

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
    zk4 = row['z4'].tolist()
    zk5 = row['z5'].tolist()
    zk6 = row['z6'].tolist()
    zk7 = row['z7'].tolist()
    zk8 = row['z8'].tolist()
    zk9 = row['z9'].tolist()
    zk10 = row['z10'].tolist()
    zk11 = row['z11'].tolist()
    zk12 = row['z12'].tolist()
    zk13 = row['z13'].tolist()
    zk14 = row['z14'].tolist()
    zk15 = row['z15'].tolist()
    zk20 = row['z20'].tolist()
    zk21 = row['z21'].tolist()
    zk22 = row['z22'].tolist()
    zk27 = row['z27'].tolist()
    zk28 = row['z28'].tolist()
    table_val = np.stack([zk4, zk5, zk6, zk7, zk8, zk9, zk10,
                          zk11, zk12, zk13, zk14, zk15, zk20, zk21,
                          zk22, zk27, zk28])
    return table_val


def getRealData(butler, cdb_table, ind):
    """Get real data from source."""
    if not LSST_AVAILABLE:
        raise ImportError("LSST dependencies not available. This function requires LSST installation.")

    exposure_id = cdb_table['visit_id'][ind]
    detector_name = cdb_table['detector'][ind]
    data_id1 = {
            "instrument": "LSSTCam",  # Replace with your instrument
            "exposure": int(exposure_id),        # Replace with your exposure ID
            "detector": detector_name
            }
    data_id2 = {
            "instrument": "LSSTCam",  # Replace with your instrument
            "exposure": int(exposure_id),        # Replace with your exposure ID
            "detector": detector_name + 1
            }
    data_1 = butler.get('raw', dataId=data_id1, collections="LSSTCam/raw/all")
    data_2 = butler.get('raw', dataId=data_id2, collections="LSSTCam/raw/all")
    row = cdb_table[(cdb_table['visit_id'] == exposure_id) & (cdb_table['detector'] == detector_name)]
    zk = getzk(row)
    return (data_1, detector_name, zk), (data_2, detector_name + 1, zk)
