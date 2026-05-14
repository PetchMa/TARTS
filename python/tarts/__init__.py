"""Neural Active Optics System (TARTS) package.

This package implements a deep learning-based active optics system for the LSST telescope.
It provides neural network models for processing out-of-focus donut images to predict
wavefront aberrations in the form of Zernike coefficients, enabling real-time telescope
optics correction.

The package includes:
- AlignNet: For donut alignment and centroiding
- WaveNet: For Zernike coefficient prediction from donut images
- AggregatorNet: For combining predictions from multiple donuts
- Utilities for data processing, model training, and deployment

Deployment (Summit / USDF / RA)
-------------------------------
- Prefer loading ``NeuralActiveOpticsSys`` with WaveNet/AlignNet checkpoint paths so no
  backbone weights are fetched at runtime.
- By default ``pretrained=False`` on ``NeuralActiveOpticsSys`` to avoid Hugging Face Hub
  downloads and cache writes under ``HOME``.
- To redirect the Hub cache, set ``HF_HOME`` or ``TARTS_HF_HOME`` (alias used if ``HF_HOME``
  is unset) before ``import tarts``.
"""

import logging

# Configure logging for the package
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Create a handler if none exists
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

from . import _runtime_env  # noqa: F401  # configures HF cache before timm/hub imports

from .NeuralActiveOpticsSys import *  # noqa: F403, F401
from .utils import *  # noqa: F403, F401
from .dataloader import *  # noqa: F403, F401
