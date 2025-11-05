"""Donut Kernel used (typically) for classical donut alignment.

This module generates the donut kernel as a PyTorch tensor at runtime,
eliminating the need for a large hardcoded array.
"""

import numpy as np
import torch


def paint_circle(arr, center, radius, value=1, fill=True):
    """Paint a circle onto an existing 2D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        2D array to paint onto (modified in place).
    center : tuple of floats
        (y, x) coordinates of the circle center.
    radius : float
        Radius of the circle.
    value : float or int, optional
        Value to paint (default = 1).
    fill : bool, optional
        If True, fill the circle. If False, draw only the outline.

    Returns
    -------
    arr : np.ndarray
        The same array, modified.
    """
    y, x = np.ogrid[: arr.shape[0], : arr.shape[1]]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    if fill:
        mask = dist <= radius
    else:
        mask = (dist >= radius - 0.5) & (dist <= radius + 0.5)

    arr[mask] = value
    return arr


def generate_donut_kernel(size=(240, 240), outer_radius=68, inner_radius=37, center=None):
    """Generate a donut kernel pattern.

    Parameters
    ----------
    size : tuple of int, optional
        Size of the kernel array (height, width). Default is (240, 240).
    outer_radius : float, optional
        Radius of the outer circle. Default is 68.
    inner_radius : float, optional
        Radius of the inner circle (hole). Default is 37.
    center : tuple of floats, optional
        (y, x) coordinates of the center. If None, uses the center of the array.
        Default is None.

    Returns
    -------
    torch.Tensor
        Donut kernel as a PyTorch tensor of shape (height, width).
    """
    if center is None:
        center = (size[0] // 2, size[1] // 2)

    # Create base array
    cutout = np.zeros(size, dtype=np.float32)

    # Paint outer circle (filled)
    cutout = paint_circle(cutout, center=center, radius=outer_radius, value=1, fill=True)

    # Paint inner circle (hole) to create donut shape
    cutout = paint_circle(cutout, center=center, radius=inner_radius, value=0, fill=True)

    # Convert to PyTorch tensor
    return torch.tensor(cutout, dtype=torch.float32)


# Generate the donut kernel at module import time
# This matches the original hardcoded kernel dimensions and pattern
CUTOUT = generate_donut_kernel(size=(240, 240), outer_radius=68, inner_radius=37, center=(121, 121))
