"""
This file contains functions related to assigning a descriptor to a keypoint.
"""

import numpy as np

from src.keypoints import Keypoint
from src.orientations import gradients, patch_in_frame, get_weighting_matrix
from src.miscs import patch_in_frame, hist_centers, histogram_per_row



def relative_patch_coordinates(center_offset: list,
                               patch_shape: tuple,
                               pixel_dist: float,
                               sigma: float,
                               keypoint_orientation: float) -> np.ndarray:
    """ Calculates the coordinates of pixels in a descriptor patch.
    """
    center = np.array(patch_shape) / 2 + center_offset
    xs, ys = np.meshgrid(np.arange(patch_shape[0]), np.arange(patch_shape[1]))

    # Coordinates are rotated to align with the keypoint's orientation.
    rel_xs = ((xs - center[1]) * np.cos(keypoint_orientation)
              + (ys - center[0]) * np.sin(keypoint_orientation)) / (sigma / pixel_dist)
    rel_ys = (-(xs - center[1]) * np.sin(keypoint_orientation)
              + (ys - center[0]) * np.cos(keypoint_orientation)) / (sigma / pixel_dist)

    return np.array([rel_xs, rel_ys])


def mask_outliers(magnitude_patch: np.ndarray,
                  rel_patch_coords: np.ndarray,
                  threshold: float,
                  axis: int = 0) -> np.ndarray:
    """ Masks outliers in a patch. Here, an outlier has a distance
        from the patch's center keypoint along the y or x axis that
        is larger than the threshold.
    """
    mask = np.max(np.abs(rel_patch_coords), axis=axis) <= threshold
    magnitude_patch = magnitude_patch * mask
    return magnitude_patch


def interpolate_2d_grid_contribution(magnitude_patch: np.ndarray,
                                     coords_rel_to_hist: np.ndarray) -> np.ndarray:
    """ Interpolates gradient contributions to surrounding histograms.
        In other words: Calculates to what extent gradients in a descriptor
        patch contribute to a histogram, based on the gradient's pixel's
        y & x distance to that histogram's location.
    Args:
        magnitude_patch: array of shape (2, 32, 32) with semantics (y_or_x, patch_row, patch_col).
        coords_rel_to_hist: array of shape (2, 16, 32, 32) after axes swap,
            with semantics (y_or_x, hist_idx, patch_row, patch_col).
    """
    coords_rel_to_hist = np.swapaxes(coords_rel_to_hist, 0, 1)
    xs, ys = np.abs(coords_rel_to_hist)
    y_contrib = 1 - (ys / 3)
    x_contrib = 1 - (xs / 3)
    contrib = y_contrib * x_contrib
    magnitude_patch = magnitude_patch * contrib
    
    return magnitude_patch


def interpolate_1d_hist_contribution(magnitude_patch: np.ndarray,
                                     orientation_patch: np.ndarray) -> np.ndarray:
    """ Interpolates an orientation's contribution between two orientation bins.
        When creating an orientation histogram, rather than adding an orientation's
        contribution to a single bin, it contributes mass to 2 bins, its left and
        right neighbor. This contribution is linear interpolated given the distance
        to each of these bins.
    """
    nr_hists = magnitude_patch.shape[0] # 16 in standard configuration
    orientation_patch = np.repeat(orientation_patch[None, ...], nr_hists, axis=0)

    descriptor_bin_width = 8 / (2 * np.pi)
    hist_bin_width = descriptor_bin_width
    dist_to_next_bin = (orientation_patch % hist_bin_width)
    norm_dist_to_next_bin = dist_to_next_bin / hist_bin_width
    norm_dist_current_bin = 1 - norm_dist_to_next_bin

    current_bin_orients = orientation_patch
    next_bin_orients = (orientation_patch + hist_bin_width) % (2 * np.pi)

    hist_current = histogram_per_row(current_bin_orients.reshape((nr_hists, -1)),
                                     bins=8,
                                     range_=(0, 2 * np.pi),
                                     weights=norm_dist_current_bin * magnitude_patch)
    hist_next = histogram_per_row(next_bin_orients.reshape((nr_hists, -1)),
                                  bins=8,
                                  range_=(0, 2 * np.pi),
                                  weights=norm_dist_to_next_bin * magnitude_patch)
    interpol_hist = hist_current + hist_next
    
    return interpol_hist


def normalize_sift_feature(hists: np.ndarray) -> np.ndarray:
    """ Normalizes a keypoint's descriptor histograms to a unit length vector.
    """
    hists = hists / np.linalg.norm(hists)
    hists = np.clip(hists, a_min=None, a_max=.2)
    hists = hists / np.linalg.norm(hists)
    return hists


# All patches have the same relative hist centers, so calculate beforehand and treat as constants
histogram_centers = hist_centers()
def assign_descriptor(keypoints: list[Keypoint],
                      gauss_octave: np.array,
                      octave_idx: int) -> list[Keypoint]:
    """ Assigns a descriptor to each keypoint.
        A descriptor is a collection of histograms that capture the distribution of
        gradients orientations in an oriented keypoint's local neighborhood.
    """
    magnitudes, orientations = gradients(gauss_octave)

    described_keypoints = []
    for keypoint in keypoints:
        coord, sigma = keypoint.coordinate, keypoint.sigma
        s, y, x = coord.round().astype(int)

        pixel_dist = 0.5 * (2 ** octave_idx)
        max_width = (np.sqrt(2) * 6 * sigma) / pixel_dist
        max_width = max_width.round().astype(int)

        if patch_in_frame(coord, max_width, gauss_octave.shape):
            slices = (s, slice(y - max_width, y + max_width),
                         slice(x - max_width, x + max_width))
            orientation_patch = orientations[slices]
            magnitude_patch = magnitudes[slices]
            center_offset = [coord[1] - y, coord[2] - x]

            orientation_patch = (orientation_patch - keypoint.orientation) % (2 * np.pi)
            rel_patch_coords = relative_patch_coordinates(center_offset, magnitude_patch.shape, pixel_dist, sigma,
                                                          keypoint.orientation)
            magnitude_patch = mask_outliers(magnitude_patch, rel_patch_coords, 6)
            weights = get_weighting_matrix(center_offset, magnitude_patch.shape, octave_idx,
                                           sigma, 6)
            magnitude_patch = magnitude_patch * weights
            
            coords_rel_to_hists = rel_patch_coords[None] - histogram_centers[..., None, None]
            hists_magnitude_patch = mask_outliers(magnitude_patch[None], coords_rel_to_hists, 1.5, 1)
            hists_magnitude_patch = interpolate_2d_grid_contribution(hists_magnitude_patch, coords_rel_to_hists)
            hists = interpolate_1d_hist_contribution(hists_magnitude_patch, orientation_patch).ravel()
            keypoint.descriptor = normalize_sift_feature(hists)
            described_keypoints.append(keypoint)

    return described_keypoints
