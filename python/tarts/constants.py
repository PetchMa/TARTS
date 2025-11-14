"""Constants used throughout the TARTS package.

This module centralizes all constants, configuration values, and magic numbers
used across the codebase for better maintainability and consistency.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Dict, Optional

# Third-party imports
import numpy as np
import torch

# Local/application imports
from .KERNEL import CUTOUT as DONUT_KERNEL

logger = logging.getLogger(__name__)

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

# ============================================================================
# OFC DOF Conversion Matrices (Pre-computed)
# ============================================================================

# Default sensor names for OFC operations
OFC_SENSOR_NAMES = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]

# Default filter names (uppercase for OFC system)
OFC_FILTER_NAMES = ["U", "G", "R", "I", "Z", "Y"]

# Filter name mapping (lowercase to uppercase)
OFC_FILTER_NAME_MAP = {
    "u": "U",
    "g": "G",
    "r": "R",
    "i": "I",
    "z": "Z",
    "y": "Y",
    "U": "U",
    "G": "G",
    "R": "R",
    "I": "I",
    "Z": "Z",
    "Y": "Y",  # Allow uppercase too
}

# OFC matrix cache (lazy loaded)
_OFC_MATRICES_CACHE: Optional[Dict[str, torch.Tensor]] = None
_OFC_MATRICES_PATH: Optional[Path] = None


def set_ofc_matrices_path(path: str) -> None:
    """Set the path to pre-computed OFC matrices.

    Parameters
    ----------
    path : str
        Directory containing the pre-computed OFC matrices.
    """
    global _OFC_MATRICES_PATH, _OFC_MATRICES_CACHE
    _OFC_MATRICES_PATH = Path(path)
    _OFC_MATRICES_CACHE = None  # Clear cache to force reload


def load_ofc_matrices(matrices_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Load pre-computed OFC matrices for DOF conversion.

    This function lazy-loads matrices generated by precompute_ofc_matrices.py.
    Matrices are cached after first load for efficiency.

    Parameters
    ----------
    matrices_dir : str, optional
        Directory containing the OFC matrices. If None, looks in package directory.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing:
        - 'sensitivity_matrix': (n_sensors*n_zk, n_dof) sensitivity matrix
        - 'intrinsic_zernikes': dict of {filter: (n_sensors, n_zk)} arrays
        - 'y2_correction': dict of {filter: (n_sensors, n_zk)} arrays
        - 'metadata': dict with configuration info
    """
    global _OFC_MATRICES_CACHE, _OFC_MATRICES_PATH

    # Return cached matrices if available
    if _OFC_MATRICES_CACHE is not None:
        return _OFC_MATRICES_CACHE

    # Determine path
    if matrices_dir is not None:
        matrix_path = Path(matrices_dir)
    elif _OFC_MATRICES_PATH is not None:
        matrix_path = _OFC_MATRICES_PATH
    else:
        # Default: look in package directory
        matrix_path = Path(__file__).parent / "ofc_matrices"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"OFC matrices not found at {matrix_path}. "
            f"Please run precompute_ofc_matrices.py first or set the path using set_ofc_matrices_path()."
        )

    try:
        # Load matrices
        logger.info(f"Loading OFC matrices from {matrix_path}")

        sens_matrix = np.load(matrix_path / "sensitivity_matrix.npy")
        intrinsic_dict = np.load(matrix_path / "intrinsic_zernikes.npy", allow_pickle=True).item()
        y2_dict = np.load(matrix_path / "y2_correction.npy", allow_pickle=True).item()
        metadata = np.load(matrix_path / "metadata.npy", allow_pickle=True).item()

        # Convert to tensors and cache - preserve float64 precision
        _OFC_MATRICES_CACHE = {
            "sensitivity_matrix": torch.from_numpy(sens_matrix).double(),
            "intrinsic_zernikes": {k: torch.from_numpy(v).double() for k, v in intrinsic_dict.items()},
            "y2_correction": {k: torch.from_numpy(v).double() for k, v in y2_dict.items()},
            "metadata": metadata,
        }

        # Load per-sensor data if available (for flexible sensor order support)
        sample_points_path = matrix_path / "sample_points.npy"
        intrinsic_per_sensor_path = matrix_path / "intrinsic_per_sensor.npy"
        y2_correction_per_sensor_path = matrix_path / "y2_correction_per_sensor.npy"
        norm_weights_path = matrix_path / "normalization_weights.npy"
        dof_indices_path = matrix_path / "dof_indices.npy"

        if sample_points_path.exists():
            sample_points_dict = np.load(sample_points_path, allow_pickle=True).item()
            # Keep as numpy arrays (not torch) since they're used for field angle computation
            _OFC_MATRICES_CACHE["sample_points"] = sample_points_dict
            logger.info(f"Loaded sample_points for {len(sample_points_dict)} sensors")

        if intrinsic_per_sensor_path.exists():
            intrinsic_per_sensor = np.load(intrinsic_per_sensor_path, allow_pickle=True).item()
            # Convert nested dict to torch tensors
            intrinsic_per_sensor_torch = {}
            for filter_name, sensor_dict in intrinsic_per_sensor.items():
                intrinsic_per_sensor_torch[filter_name] = {
                    sensor: torch.from_numpy(arr).double() for sensor, arr in sensor_dict.items()
                }
            _OFC_MATRICES_CACHE["intrinsic_per_sensor"] = intrinsic_per_sensor_torch
            logger.info(f"Loaded per-sensor intrinsic zernikes for {len(intrinsic_per_sensor)} filters")

        if y2_correction_per_sensor_path.exists():
            y2_correction_per_sensor = np.load(y2_correction_per_sensor_path, allow_pickle=True).item()
            # Convert nested dict to torch tensors
            y2_correction_per_sensor_torch = {}
            for filter_name, sensor_dict in y2_correction_per_sensor.items():
                y2_correction_per_sensor_torch[filter_name] = {
                    sensor: torch.from_numpy(arr).double() for sensor, arr in sensor_dict.items()
                }
            _OFC_MATRICES_CACHE["y2_correction_per_sensor"] = y2_correction_per_sensor_torch
            logger.info(f"Loaded per-sensor y2_correction for {len(y2_correction_per_sensor)} filters")

        if norm_weights_path.exists():
            norm_weights = np.load(norm_weights_path)
            _OFC_MATRICES_CACHE["normalization_weights"] = torch.from_numpy(norm_weights).double()

        if dof_indices_path.exists():
            dof_indices = np.load(dof_indices_path)
            # Store as list for easier indexing
            _OFC_MATRICES_CACHE["dof_indices"] = dof_indices.tolist()

        logger.info(f"Loaded OFC matrices: {metadata}")
        return _OFC_MATRICES_CACHE

    except Exception as e:
        logger.error(f"Failed to load OFC matrices: {e}")
        raise RuntimeError(
            f"Could not load OFC matrices from {matrix_path}. "
            f"Please ensure precompute_ofc_matrices.py has been run successfully."
        ) from e
