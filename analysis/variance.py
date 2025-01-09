"""
This module computes the variance of the forces on a dune.

This script takes the images of dunes (with the forces) centered on the image
as well as the time average image (computed in another step). With this, all
images are referenced in the exact same way, and then we take the variance of
each pixel.

Global parameters:

    PATH (str): Full path to the directory containing the images of dunes with
        their force distribution (centered on the image).

    AVG_IMG (str): Full path to the image file of the time-averaged force
    distribution, needed to evaluate the standard deviation.

    FIRST_IMAGE_INDEX (int): Index to select images of fully developed dunes. Be
    careful: it needs to match the index used for time averaging!

    OUTPUT_FILENAME (str): Full path to the .mat file containing the output
    variance. This file will be used to generate the production images since it
    is faster to read the result from it than to recompute the std every time you
    make some changes on the production image.
"""

import glob
import os

import cv2
import matplotlib
import numpy as np
import scipy.io
from tqdm import tqdm

from analysis.mask_dune import apply_mask
from analysis.reverse_cmap_into_scalars import cmap2scalar

PATH = "./path/to/images/of/forces/on/dunes"
AVG_IMG = "./path/to/the/averaged/force/image.png"

OUTPUT_FILENAME = "variance.dat"

FIRST_IMAGE_INDEX = 0  # skip some images to consider only developed dunes


def main():
    image_files = sorted(glob.glob(os.path.join(PATH, "*.png")))

    image_files_only_developed_dunes = image_files[FIRST_IMAGE_INDEX:]

    progress = tqdm(image_files_only_developed_dunes)

    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    cmap = matplotlib.colormaps["jet"]

    image_average = cv2.imread(AVG_IMG)
    average = cmap2scalar(cv2.cvtColor(image_average, cv2.COLOR_BGR2RGBA), cmap, norm)

    for i, filename in enumerate(progress):
        img_centered = cv2.imread(filename)

        values = cmap2scalar(cv2.cvtColor(img_centered, cv2.COLOR_BGR2RGBA), cmap, norm)

        mask = apply_mask(img_centered)

        values[mask == 0] = 0  # region off the dune have no force

        deviation = (values - average) ** 2

        if i == 0:
            sum = deviation
        else:
            sum = np.sum([sum, deviation], axis=0)

    variance = sum / (len(image_files_only_developed_dunes) - 1)

    scipy.io.savemat(OUTPUT_FILENAME, {"variance": variance})


if __name__ == "__main__":
    main()
