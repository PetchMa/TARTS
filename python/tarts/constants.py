"""Constants used throughout the TARTS package.

This module centralizes all constants, configuration values, and magic numbers
used across the codebase for better maintainability and consistency.
"""

# Third-party imports
import torch

# Local/application imports
from .KERNEL import CUTOUT as DONUT_KERNEL

# ============================================================================
# System Configuration
# ============================================================================

# LSST system availability flag
LSST_AVAILABLE = True

# Camera type identifier
CAMERA_TYPE = "LsstCam"

# Input image shape (height, width)
DEFAULT_INPUT_SHAPE = (160, 160)

# Default crop size for donut images
DEFAULT_CROP_SIZE = 160

# ============================================================================
# Detector Mappings
# ============================================================================

MAP_DETECTOR_TO_NUMBER = {
    "R00_SW0": 191,
    "R00_SW1": 192,
    "R04_SW0": 195,
    "R04_SW1": 196,
    "R40_SW0": 199,
    "R40_SW1": 200,
    "R44_SW0": 203,
    "R44_SW1": 204,
}

# ============================================================================
# Band Mappings and Values
# ============================================================================

# Band wavelength values (microns)
BAND_MAP = {
    0: 0.3671,  # u-band
    1: 0.4827,  # g-band
    2: 0.6223,  # r-band
    3: 0.7546,  # i-band
    4: 0.8691,  # z-band
    5: 0.9712,  # y-band
}

# Band string to integer mapping
BAND_STR_INT = {
    "u": 0,
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
    "y": 5,
}

# Band values as tensor (for efficient batch processing)
BAND_VALUES_TENSOR = torch.tensor([[0.3671], [0.4827], [0.6223], [0.7546], [0.8691], [0.9712]])

# ============================================================================
# Field Position Mappings
# ============================================================================

FIELD_POSITIONS = {
    "R00": {"fieldx": -1.1897, "fieldy": -1.1897},
    "R04": {"fieldx": -1.1897, "fieldy": 1.1897},
    "R40": {"fieldx": 1.1897, "fieldy": -1.1897},
    "R44": {"fieldx": 1.1897, "fieldy": 1.1897},
}

# ============================================================================
# Normalization Constants
# ============================================================================

# Field position normalization (degrees)
FIELD_MEAN = 0.000
FIELD_STD = 0.021

# Intrafocal flag normalization (0 or 1)
INTRA_MEAN = 0.5
INTRA_STD = 0.5

# Band wavelength normalization (microns)
BAND_MEAN = 0.710
BAND_STD = 0.174

# ============================================================================
# Zernike Defaults
# ============================================================================

# Default Noll Zernike indices (1-indexed, converted to 0-indexed in code)
# Matches the default in dataset_params.yaml
DEFAULT_NOLL_ZK = [
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
]

# ============================================================================
# Model Defaults
# ============================================================================

# Default CNN model
DEFAULT_CNN_MODEL = "resnet18"

# Default number of Zernike coefficients
DEFAULT_N_ZERNIKES = 25

# Default predictor layer sizes
DEFAULT_PREDICTOR_LAYERS = (256,)

# Default learning rate
DEFAULT_LEARNING_RATE = 1e-3

# Default L2 penalty weight
DEFAULT_ALPHA = 0.0

# ============================================================================
# Dataset Defaults
# ============================================================================

# Default adjustment factor for image shifting
DEFAULT_ADJUSTMENT_FACTOR = 0

# Default SNR threshold for filtering
DEFAULT_SNR_THRESHOLD = 1

# Training data fraction (for debugging/development)
DEFAULT_TRAIN_FRACTION = 0.5

# ============================================================================
# Processing Constants
# ============================================================================

# Conversion factor: degrees to radians
DEG_TO_RAD = 3.141592653589793 / 180.0  # π / 180

# Zernike scaling factor (microns to arcsec)
ZERNIKE_SCALE_FACTOR = 1000

# ============================================================================
# Kernel Templates
# ============================================================================

# Donut cutout template (imported from KERNEL module)
DONUT_TEMPLATE = DONUT_KERNEL
