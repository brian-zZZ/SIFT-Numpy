"""
This file contains the main function of the SIFT implementation.
"""

import os
import time
import datetime
import numpy as np
from PIL import Image

from src.octaves import generate_gaussian_octaves, generate_dog_octave, search_dog_extrema
from src.keypoints import Keypoint, find_keypoints
from src.orientations import assign_reference_orientations
from src.descriptor import assign_descriptor
from src.match import match_sift_features, visualize_matches
from src.miscs import get_resource_path



def SIFT(img: np.ndarray) -> list[Keypoint]:
    """ Detects SIFT keypoints in an image.
    """
    gauss_octaves = generate_gaussian_octaves(img)

    features = []
    for octave_idx, gauss_octave in enumerate(gauss_octaves):
        DoG_octave = generate_dog_octave(gauss_octave)
        extrema = search_dog_extrema(DoG_octave)
        keypoint_coords = find_keypoints(extrema, DoG_octave)
        keypoints = assign_reference_orientations(keypoint_coords, gauss_octave, octave_idx)
        keypoints = assign_descriptor(keypoints, gauss_octave, octave_idx)
        features += keypoints

    return features


def main():
    """ Detects and matches SIFT features of two grayscale images. """

    # Specify a images folder lies in the same path as the current file (sift.py)
    img_dir = 'imgs'
    imgs = os.listdir(img_dir)
    # Make sure proper number of images are placed
    assert len(imgs) == 2, f'Only support SIFT detection for 2 images, {len(imgs)} were given.'
    img1_dir, img2_dir = [os.path.join(img_dir, img) for img in imgs]

    img1 = np.array(Image.open(get_resource_path(img1_dir)).convert('L'))
    img2 = np.array(Image.open(get_resource_path(img2_dir)).convert('L'))
    # Min-Max normalize, [0, 1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    print(f'Size of image 1 and image 2: {img1.shape}, {img2.shape}')

    print('Start detecting SIFT features...')
    t0 = time.time()
    keypoints1 = SIFT(img1)
    keypoints2 = SIFT(img2)
    matches = match_sift_features(keypoints1, keypoints2)
    visualize_matches(matches, img1, img2)
    
    total_time = time.time() - t0
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Finish detecting and matching in {}'.format(total_time_str))
    print('Results:')
    print('len of features in image 1: ', len(keypoints1))
    print('len of features in image 2: ', len(keypoints2))
    print('len of matches: ', len(matches), '\n')

    print('Window will be colsed in 10 seconds later...')
    time.sleep(10)


if __name__ == '__main__':
    main()
