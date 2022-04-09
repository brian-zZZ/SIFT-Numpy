""" This file contains the SIFT implementation from cv2, 
    which acts as a comparison to the implementation from scratch with solely Numpy.
"""

import os
import time
import datetime
import cv2
import numpy as np

def main():
    """ Detects and matches SIFT features of two grayscale images with cv2. """

    # Specify a images folder lies in the same path as the current file (sift.py)
    img_dir = 'imgs'
    imgs = os.listdir(img_dir)
    # Make sure proper number of images are placed
    assert len(imgs) == 2, f'Only support SIFT detection for 2 images, {len(imgs)} were given.'
    img1_dir, img2_dir = [os.path.join(img_dir, img) for img in imgs]

    img1, img2 = cv2.imread(img1_dir), cv2.imread(img2_dir)
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT instance with cv2
    sift=cv2.SIFT_create()
    # Detect
    t0 = time.time()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Create matcher instance with cv2
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Select highly matching features
    highs = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            highs.append([m])

    # Plot top 20 matches
    matching_fig = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, highs[:20], None, flags=2)
    cv2.namedWindow('Matching result', 0);
    cv2.imshow('Matching result', matching_fig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('../results/cv2_matching.png', matching_fig)

    total_time = time.time() - t0
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Finish detecting and matching in {}'.format(total_time_str))
    print('Results:')
    print('len of features in image 1: ', len(keypoints1))
    print('len of features in image 2: ', len(keypoints2))
    print('len of matches: ', len(highs), '\n')


if __name__ == '__main__':
    main()
