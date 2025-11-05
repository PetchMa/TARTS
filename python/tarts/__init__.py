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
"""

from .NeuralActiveOpticsSys import *  # noqa: F403, F401
from .utils import *  # noqa: F403, F401
from .dataloader import *  # noqa: F403, F401
