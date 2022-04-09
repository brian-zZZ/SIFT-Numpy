"""
This file contains misc functions for the others files.
"""

import os
import sys
import numpy as np



def get_init_sigma() -> float:
    """ Calculate the initial sigma for gaussian convolution.
    """
    # Standard config values from Lowe paper
    min_delta = 0.5
    orig_sigma = 0.5
    min_sigma = 0.8

    init_sigma = 1 / min_delta * np.sqrt(min_sigma ** 2 - orig_sigma ** 2)
    return init_sigma


def gaussian_kernel(sigma: float) -> np.ndarray:
    """ Generate Gaussian kernel with specific sigma.
    """
    size = 2 * np.ceil(3 * sigma) + 1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) / (2 * np.pi * sigma**2)
    return g / g.sum()


def convolve(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    """ Compute 2D convolution with zero padding.
    """
    (Xh, Xw), (Kh, Kw)= X.shape, K.shape
    Y = np.zeros(X.shape)
    # zero padding
    X_pad = np.pad(X, (Kh//2, Kw//2), 'constant', constant_values=0)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X_pad[i:i + Kh, j:j + Kw] * K).sum()
    
    return Y


def shift(array: np.ndarray, shift_spec: list or tuple) -> np.ndarray:
    """ Shift in a specified direction for approximate derivatives calculation or neighbors subtraction.
    """
    padded = np.pad(array, 1, mode='edge')
    s, y, x = shift_spec
    shifted = padded[1 + s: -1 + s if s != 1 else None,
                    1 + y: -1 + y if y != 1 else None,
                    1 + x: -1 + x if x != 1 else None]
    return shifted


def get_coarse_mag_thresh() -> float:
    """ Calculate coarse magnitude threshold determining whether interpolates a DoG extrema.
    """
    num_scales_per_octave = 3
    magnitude_thresh = 0.015 * ((2 ** (1 / num_scales_per_octave) - 1) / (2 ** (1 / 3) - 1))
    coarse_mag_thresh = 0.85 * magnitude_thresh
    return coarse_mag_thresh


def get_smooth_kernel() -> np.ndarray:
    """ Generate the smooth kernel defined as multiple convolutions 
        with a three-tap box filter [1, 1, 1] / 3.
    """
    smooth_kernel = np.array([1, 1, 1]) / 3
    for i in range(6 - 1):
        # len = 3 + 2 * 5 = 13
        smooth_kernel = np.convolve(np.array([1, 1, 1]) / 3, smooth_kernel)
    return smooth_kernel


def patch_in_frame(coord: np.array, half_width: float, shape: tuple) -> bool:
    """ Checks whether a square patch falls within the borders of a tensor.
    """
    s, y, x = coord.round()
    s_lim, y_lim, x_lim = shape

    valid = (y - half_width > 0
             and y + half_width < y_lim
             and x - half_width > 0
             and x + half_width < x_lim
             and 0 <= s < s_lim)

    return valid


def hist_centers() -> np.ndarray:
    """ Calculates relative coordinates of histogram centers within a descriptor patch.
    """
    xs, ys = [], []

    bin_width = 3
    hist_center_offset = bin_width / 2
    start_coord = -6 + hist_center_offset

    for row_idx in range(4):
        for col_idx in range(4):
            y = start_coord + bin_width * row_idx
            x = start_coord + bin_width * col_idx
            ys.append(y)
            xs.append(x)

    centers = np.array([xs, ys]).T # shape: (16, 2)
    
    return centers


def histogram_per_row(data: np.ndarray,
                      bins: int,
                      range_: tuple,
                      weights: np.ndarray) -> np.ndarray:
    """ Calculates histograms for each row of a 2D matrix efficiently.
    """
    n_rows = data.shape[0]
    bin_edges = np.linspace(range_[0], range_[1], bins + 1)
    idx = np.searchsorted(bin_edges, data, 'right') - 1
    bad_mask = idx == bins
    idx[bad_mask] = bins - 1
    scaled_idx = idx + bins * np.arange(n_rows)[:, None]
    limit = bins * n_rows
    histograms = np.bincount(scaled_idx.ravel(), minlength=limit, weights=weights.ravel())
    histograms.shape = (n_rows, bins)
    
    return histograms


def get_resource_path(relative_path) -> str:
    """ Generate abstract path for images loading in all conditions.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
