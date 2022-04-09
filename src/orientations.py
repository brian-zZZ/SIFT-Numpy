"""
This file contains functions related to assigning a reference orientation to a keypoint. 
"""

import numpy as np
from typing import Tuple, List

from src.keypoints import Keypoint
from src.octaves import scale_sigma
from src.miscs import shift, get_smooth_kernel, patch_in_frame



def gradients(octave: np.array) -> Tuple[np.array, np.array]:
    """ Finds the magnitude and orientation of image gradients.
    """
    o = octave
    dy = (shift(o, [0, 1, 0]) - shift(o, [0, -1, 0])) / 2
    dx = (shift(o, [0, 0, 1]) - shift(o, [0, 0, -1])) / 2

    magnitudes = np.sqrt(dy ** 2 + dx ** 2)
    orientations = np.arctan2(dy, dx) % (2 * np.pi)

    return magnitudes, orientations


def get_weighting_matrix(center_offset: np.array, patch_shape: tuple, octave_idx: int,
                     sigma: float, locality: float) -> np.array:
    """ Calculates the Gaussian weighting matrix determining the weight that gradients
        in a keypoint's neighborhood have when contributing to the keypoint's orientation hist
    """
    center = np.array(patch_shape) / 2 + center_offset
    xs, ys = np.meshgrid(np.arange(patch_shape[0]), np.arange(patch_shape[1]))
    rel_dists = np.sqrt((xs - center[1]) ** 2 + (ys - center[0]) ** 2)
    abs_dists = rel_dists * (0.5 * (2 ** octave_idx)) # abs = rel * pixel_dist
    weights = np.exp(-((abs_dists ** 2) / (2 * ((locality * sigma) ** 2))))
    
    return weights


# Convolution operations are associative, thus the smoothing filter
# is calculated beforehand and treated as a constant.
smooth_kernel = get_smooth_kernel()
def smoothen_histogram(hist: np.array) -> np.array:
    """ Smoothen a histogram with an average filter defined as multiple convolutions
        with a three-tap box filter [1, 1, 1] / 3.
    """
    pad_amount = round(len(smooth_kernel) / 2)
    hist_pad = np.pad(hist, pad_width=pad_amount, mode='wrap')
    hist_smoothed = np.convolve(hist_pad, smooth_kernel, mode='valid')
    
    return hist_smoothed


def find_histogram_peaks(hist: np.array) -> List[float]:
    """ Finds peaks in the binned gradient orientations histogram,
        and returns the corresponding orientations in radians.
    """
    orientations = []
    global_max = None
    hist_masked = hist.copy()
    num_bins = 36

    for i in range(2): # 2 peaks at most
        max_idx = np.argmax(hist_masked)
        max_ = hist[max_idx]

        if global_max is None:
            global_max = max_

        if i == 0 or max_ > (0.8 * global_max):
            k_left, k_right = (max_idx - 1) % num_bins, (max_idx + 1) % num_bins
            left, right = hist[k_left], hist[k_right]
            # Extract ref orientations with interpolated values
            interpol_max_radians = (2 * np.pi * max_idx) / num_bins \
                                    + (np.pi / num_bins) \
                                    * ((left - right) / (left - 2 * max_ + right))
            # Taking account for the fact that the first and last bin are neighbors
            interpol_max_radians = interpol_max_radians % (2 * np.pi)
            orientations.append(interpol_max_radians)

            # After a peak is found, it and its surrounding bins are masked
            # to enable other peaks to be found with `argmax`.
            for j in range(4 + 1):
                hist_masked[(max_idx - j) % num_bins] = 0
                hist_masked[(max_idx + j) % num_bins] = 0

    return orientations


def assign_reference_orientations(keypoint_coords: np.array,
                                  gauss_octave: np.array,
                                  octave_idx: int) -> list[Keypoint]:
    """ Assigns dominant local neighborhood gradient orientations to keypoints.
        These dominant orientations are also known as reference orientations.
        Returns a list of keypoints that have been assigned an orientation.
    """
    keypoints = []
    magnitudes, orientations = gradients(gauss_octave)
    num_bins = 36
    orientation_bins = np.round((orientations / (2 * np.pi)) * num_bins) # orientation to bin idx

    for coord in keypoint_coords:
        s, y, x = coord.round().astype(int)
        sigma = scale_sigma(s, octave_idx)
        patch_width_half = round((sigma * 9) / \
                                 (0.5 * (2 ** octave_idx)) / 2)

        if patch_in_frame(coord, patch_width_half, gauss_octave.shape):
            center_offset = [coord[1] - y, coord[2] - x]
            slices = (s, slice(y - patch_width_half, y + patch_width_half),
                         slice(x - patch_width_half, x + patch_width_half))
            orientation_patch = orientation_bins[slices]
            magnitude_patch = magnitudes[slices]
            weights = get_weighting_matrix(center_offset, magnitude_patch.shape, octave_idx,
                                            sigma, 1.5)
            contribution = weights * magnitude_patch
            hist, bin_edges = np.histogram(orientation_patch,  bins=num_bins,
                                           range=(0, num_bins), weights=contribution)
            hist = smoothen_histogram(hist)
            ref_orientations = find_histogram_peaks(hist)

            for orientation in ref_orientations:
                keypoint = Keypoint(coord=coord, octave_idx=octave_idx, orientation=orientation)
                keypoints.append(keypoint)

    return keypoints
