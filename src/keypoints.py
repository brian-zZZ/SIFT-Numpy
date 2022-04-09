"""
This file contains functions related to finding candidate keypoints for SIFT features in an image
"""

import numpy as np
from typing import Tuple

from src.octaves import scale_sigma, derivatives
from src.miscs import get_coarse_mag_thresh


class Keypoint:
    """ A keypoint object that is created when a reference orientation is found.
        A descriptor is assigned to the keypoint at a later time.
    """
    def __init__(self, coord: np.ndarray, octave_idx: int, orientation: float):
        self.octave_idx = octave_idx
        self.coordinate = coord
        self.orientation = orientation
        self.sigma = scale_sigma(coord[0], self.octave_idx)
        self.descriptor = None

    @property
    def absolute_coordinate(self):
        """ Calculates the keypoint's coordinates relative to the input image.
        """
        s, y, x = self.coordinate
        pixel_dist = 0.5 * (2 ** self.octave_idx)
        return np.array([self.sigma, y * pixel_dist, x * pixel_dist])


def interpolation_offsets(deriv: np.ndarray, second_deriv: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Calculates the coordinate offset and value offset of an extremum,
        relative to the non-interpolated extremum.
    """
    second_deriv = second_deriv.reshape((3, 3))
    second_deriv_inv = np.linalg.pinv(second_deriv)
    offset = -np.dot(second_deriv_inv, deriv) # calculate accurate offset
    val_change = (1 / 2) * np.dot(deriv, offset) # calculate half of the offset value
    return offset, val_change


def contrastive(Dx_hat: float) -> bool:
    """ Reject unstable extrema with low contrast. """
    Dx_hat = abs(Dx_hat)
    magnitude_thresh = 0.015
    return Dx_hat >= magnitude_thresh


def without_edge_response(second_deriv: np.ndarray) -> bool:
    """ Eliminates keypoints along edges for improved detection repeatability. 
    """
    Hxy = second_deriv.reshape((3, 3))[1:, 1:].copy()
    trace = np.trace(Hxy)
    det = np.linalg.det(Hxy) + np.finfo(dtype=float).eps
    Tr2_Det = (trace ** 2) / (det + np.finfo(dtype=float).eps)
    r = 10 # the ratio between the largest magnitude eigenvalue and the smaller one
    threshold = ((r + 1) ** 2) / r
    return Tr2_Det < threshold


def interpolate(extremum_coord: np.ndarray,
                dog_octave: np.ndarray,
                derivs: np.ndarray,
                second_derivs: np.ndarray) -> Tuple[bool, np.ndarray, float]:
    """ Interpolates the coordinate and value of an extremum in a DoG octave.
        This enables more precise sub-pixel keypoint locations, which improves descriptor quality.
    """
    interpol_coord = extremum_coord
    interpol_val = dog_octave[tuple(interpol_coord)]
    shape = np.array(dog_octave.shape)
    success = False

    for _ in range(3): # attempt to interpolate an extrema three times
        s, y, x = interpol_coord.round().astype(int)
        deriv = derivs[:, s, y, x]
        second_deriv = second_derivs[:, s, y, x] # hessian at point [s, y, x]
        offset, val_change = interpolation_offsets(deriv, second_deriv)
        interpol_coord = interpol_coord + offset
        interpol_val = interpol_val + val_change

        within = (interpol_coord >= 0).all() and (interpol_coord <= shape - 1).all() # within frame
        # Stop the loop when offset of any args (s, y, x) less than 0.5
        if (abs(offset) < 0.5).all() and within:
            success = True
            break
        elif not within:
            break
 
    return success, interpol_coord, interpol_val


def find_keypoints(extrema: np.ndarray, dog_octave: np.ndarray) -> np.ndarray:
    """ Finds valid keypoint coordinates among candidate DoG extrema. A candidate keypoint coordinate
        must be interpolated, pass a magnitude threshold, and pass the edge test.
    """
    keypoint_coords = list()
    derivs, second_derivs = derivatives(dog_octave)

    for extremum_coord in extrema.T:
        # Attempt only to interpolate a DoG extrema's value larger than a magnitude threshold
        if abs(dog_octave[tuple(extremum_coord)]) > get_coarse_mag_thresh():
            success, extremum_coord, extremum_val = interpolate(extremum_coord, dog_octave, derivs, second_derivs)
            # Reject unstable extrema with low contrast (0.015)
            if success and contrastive(extremum_val):
                s, y, x = extremum_coord.round().astype(int)
                # Eliminate unstable extrema with edge response
                if without_edge_response(second_derivs[:, s, y, x]):
                    keypoint_coords.append(extremum_coord)

    return np.array(keypoint_coords)
