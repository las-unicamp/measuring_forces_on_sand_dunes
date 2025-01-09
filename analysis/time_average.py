"""
This module evaluates the mean forces on a dune. Since the position and
morphology of the dune changes over time, we need to take steps to evaluate this
mean.

This script takes the images of dunes (with the forces) centered on the image -
so that all images are referenced in the exact same way - and then we take
the time average of each channel value pixelwise.

The idea is that the time-averaged force distribution of the experimental and
simulated dune should be close. This is how we validate that CNN-based model
is giving trustful results.

Global parameters:

    PATH (str): Full path to the directory containing the images of dunes with
        their force distribution (centered on the image).

    FIRST_IMAGE_INDEX (int): Index to select images of fully developed dunes.

    OUTPUT_FILENAME (str): Full path to the .mat file containing the output
    time average data. This file will be used to generate the production images
    since it is faster to read the result from it than to recompute the time
    average every time you make some changes on the production image. Is may also
    be used to evaluate variance or standar deviation.
"""

import glob
import os

import cv2
import matplotlib
import numpy as np
import numpy.typing
import scipy.io
from tqdm import tqdm

from analysis.mask_dune import apply_mask
from analysis.reverse_cmap_into_scalars import cmap2scalar

PATH = "./path/to/images/of/forces/on/dunes"
OUTPUT_FILENAME = "time_average.dat"
FIRST_IMAGE_INDEX = 0  # skip some images to consider only developed dunes


OpenCVMat = numpy.typing.NDArray[np.uint8]


def remove_black_bg(img: OpenCVMat) -> OpenCVMat:
    # convert to hsv
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channel_l = lab[:, :, 0]

    # threshold
    lower = 0
    upper = 10
    mask = cv2.inRange(channel_l, lower, upper)
    mask = 255 - mask

    # apply morphology opening to mask
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # # antialias mask
    # mask = cv2.GaussianBlur(
    #     mask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT
    # )
    # mask = skimage.exposure.rescale_intensity(
    #     mask, in_range=(127.5, 255), out_range=(0, 255)
    # )

    # put white where ever the mask is zero
    result = img.copy()
    result[mask == 0] = (255, 255, 255)

    return result


def time_average() -> None:
    image_files = sorted(glob.glob(os.path.join(PATH, "*.png")))

    image_files_only_developed_dunes = image_files[FIRST_IMAGE_INDEX:]

    progress = tqdm(image_files_only_developed_dunes)

    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    cmap = matplotlib.colormaps["jet"]

    for i, filename in enumerate(progress):
        img_centered = cv2.imread(filename)

        values = cmap2scalar(cv2.cvtColor(img_centered, cv2.COLOR_BGR2RGBA), cmap, norm)

        mask = apply_mask(img_centered)

        values[mask == 0] = 0  # region off the dune have no force

        if i == 0:
            sum = values
        else:
            sum = np.sum([sum, values], axis=0)

    average = sum / len(image_files_only_developed_dunes)

    scipy.io.savemat(OUTPUT_FILENAME, {"average": average})

    # cv2.imwrite("time_average.png", avg_img)

    # plt.figure(figsize=values.shape, dpi=1)
    # plt.imshow(avg_img, norm=norm, cmap=cmap)
    # plt.gca().set_aspect("equal")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig("time_average6final", dpi=1)
    # plt.close()

    # print("finished")


if __name__ == "__main__":
    time_average()
