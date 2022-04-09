"""
This file contains functions related to generating and manipulating
the Gaussian octaves and Difference of Gaussian octaves of an image.
"""

import math
import itertools
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

from src.miscs import get_init_sigma, gaussian_kernel, shift, convolve
# from scipy.ndimage import convolve # higher efficient convolution from scipy



def scale_sigma(layer_idx: int, octave_idx: Optional[int] = None) -> float:
    """ Following paper: Anatomy of the SIFT Method, section 2.2 - Digital Gaussian scale-space.
        Compute the corresponding sigma of the digital Gaussian scale-space.
        Progression sigma: for calculating the Gaussian blur's standard deviation required
                            to move from (layer_idx - 1) -> layer_idx in scale-space.
        Absolute sigma: for calculating the Gaussian blur's standard deviation required
                            to move from the original image to current layer of blurring in scale-space.
    """
    min_sigma, min_delta, num_scales_per_octave = .8, .5, 3 # follow Lowe SIFT
    
    # Compute progression sigma
    if octave_idx is None:
        sigma = (min_sigma / min_delta) * math.sqrt(
                            2 ** (2 * layer_idx / num_scales_per_octave)
                            - 2 ** (2 * (layer_idx - 1) / num_scales_per_octave))
    # Compute absolute sigma
    else:
        sigma = ((min_delta * (2 ** octave_idx)) / min_delta) * min_sigma \
                            * 2 ** (layer_idx / num_scales_per_octave)
    return sigma



def generate_gaussian_octaves(img: np.ndarray) -> List[np.ndarray]:
    """ Generate Gaussian octaves, consisting of an image repeatedly convolved with a Gaussian kernel.
        Return a list of octaves of Gaussian convolved images (with shape [s, y, x]).
    """
    octaves = []

    # Start the first octave with the 2x upsampled input image.
    # All other octaves start with the 2x downsampled previous octave's second from last layer.
    for octave_idx in range(8): # 8 octaves
        if octave_idx == 0:
            img = Image.fromarray(img).resize((int(img.shape[1]*2), int(img.shape[0]*2)),
                                            resample=Image.Resampling.BILINEAR)
            img = np.array(img)
            kernel = gaussian_kernel(get_init_sigma())
            img = convolve(img, kernel)
        else:
            img = Image.fromarray(previous_octave[-2])
            img = img.resize((int(img.size[1]*0.5), int(img.size[0]*0.5)), resample=Image.Resampling.BILINEAR)
            img = np.array(img)
        octave = [img]

        # Convolve layers with gaussians to generate successive layers.
        for layer_idx in range(1, 3 + 3): # double to generate 3 DoG layers per octave
            kernel = gaussian_kernel(scale_sigma(layer_idx))
            img = convolve(img, kernel)
            octave.append(img)
        octaves.append(np.array(octave))
        
        previous_octave = octave

    return octaves


def generate_dog_octave(gauss_octave: np.ndarray) -> np.ndarray:
    """ Builds a Difference of Gaussian octave.
    """
    dog_octave = []
    # Calculate difference when index >= 1
    for layer_idx in range(1, len(gauss_octave)):
        previous_layer = gauss_octave[layer_idx - 1]
        dog = gauss_octave[layer_idx] - previous_layer
        dog_octave.append(dog)

    return np.array(dog_octave)


def search_dog_extrema(dog_octave: np.ndarray) -> np.ndarray:
    """ Finds extrema in a 3d DoG octave scale space.
        Achieved by subtracting a cell by all it's direct (including diagonal) neighbors,
        and confirming all differences have the same sign.
    """
    # Generate all possible shift directions (3 x 3 x 3 - 1)
    shifts = list(itertools.product([-1, 0, 1], repeat=3))
    shifts.remove((0, 0, 0))

    diffs = []
    for shift_spec in shifts:
        shifted = shift(dog_octave, shift_spec)
        diff = dog_octave - shifted # subtract
        diffs.append(diff)

    diffs = np.array(diffs)
    maxima = np.where((diffs > 0).all(axis=0)) # confirm sign requirement
    minima = np.where((diffs < 0).all(axis=0))
    extrema_coords = np.concatenate((maxima, minima), axis=1)

    return extrema_coords


def derivatives(dog_octave: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates the first and second order s, y, x approximate derivatives for an octave.
    """
    o = dog_octave

    ds = (shift(o, [1, 0, 0]) - shift(o, [-1, 0, 0])) / 2
    dy = (shift(o, [0, 1, 0]) - shift(o, [0, -1, 0])) / 2
    dx = (shift(o, [0, 0, 1]) - shift(o, [0, 0, -1])) / 2

    dss = (shift(o, [1, 0, 0]) + shift(o, [-1, 0, 0]) - 2 * o)
    dyy = (shift(o, [0, 1, 0]) + shift(o, [0, -1, 0]) - 2 * o)
    dxx = (shift(o, [0, 0, 1]) + shift(o, [0, 0, -1]) - 2 * o)

    dsy = (shift(o, [1, 1, 0]) - shift(o, [1, -1, 0]) - shift(o, [-1, 1, 0]) + shift(o, [-1, -1, 0])) / 4
    dsx = (shift(o, [1, 0, 1]) - shift(o, [1, 0, -1]) - shift(o, [-1, 0, 1]) + shift(o, [-1, 0, -1])) / 4
    dyx = (shift(o, [0, 1, 1]) - shift(o, [0, 1, -1]) - shift(o, [0, -1, 1]) + shift(o, [0, -1, -1])) / 4

    derivs = np.array([ds, dy, dx])
    second_derivs = np.array([dss, dsy, dsx,
                              dsy, dyy, dyx,
                              dsx, dyx, dxx])
    return derivs, second_derivs

