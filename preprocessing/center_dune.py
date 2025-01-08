"""
This module evaluates the geometric centroid of a given dune and shift it to
the center of the image. It also squares the output image.

OBS 1: In case you are centering a dune from a simulation, both the image with
the grains and the image with the forces must be manipulated in the same way.
All you have to do is to specify the PATH_MASK and OUTPUT_PATH_MASK. With that
all transformations on the grains will be also applied to the mask (image with
forces).

OBS 2: If you are centering an experimental image (in which only the grains are
available), leave the PATH_MASK and OUTPUT_PATH_MASK variables as empty strings.
"""

import glob
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray

from src.time_it import time_it

PATH = "path/to/images"
OUTPUT_PATH = "path/to/save/centered/images"

PATH_MASK = ""  # or "path/to/mask"
OUTPUT_PATH_MASK = ""  # or "path/to/save/centered/mask"


def _threshold_image(img: NDArray[uint8], minimum_value: int) -> NDArray[uint8]:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img_gray, minimum_value, 255, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, np.ones((2, 2), np.uint8), iterations=1)
    return threshold


def _remove_noise(threshold: NDArray) -> NDArray[uint8]:
    height, width = threshold.shape
    cv2.floodFill(threshold, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 0)
    num_iter = 80
    for _ in range(num_iter):
        threshold = cv2.medianBlur(threshold, 7)
    return threshold


def _get_contour(threshold: NDArray) -> NDArray[uint8]:
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bigger_contour = max(contours, key=cv2.contourArea)
    return bigger_contour


def get_centroid(
    img: NDArray[uint8], is_experimental: bool = True, draw_contours: bool = False
) -> Tuple[NDArray[uint8], Tuple[int, int]]:
    # These `minimum_value`s were found to work for our dataset.
    # The idea is that the contour should fit nicely around the dune.
    # Make sure to use a `minimum_value` that works best for your dataset.
    if is_experimental:
        minimum_value = 25
    else:
        minimum_value = 0

    threshold = _threshold_image(img, minimum_value)
    noiseless_threshold = _remove_noise(threshold)

    contour = _get_contour(noiseless_threshold)

    # compute the center of the contour
    moments = cv2.moments(contour)
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    # draw the contour and center of the shape on the image
    if draw_contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cv2.circle(img, (centroid_x, centroid_y), 7, (0, 255, 0), -1)

    centroid = centroid_x, centroid_y

    return img, centroid


def crop_from_center(
    img: NDArray[uint8], desired_width: int, desired_height: int
) -> NDArray[uint8]:
    original_height, original_width, _ = img.shape

    try:
        assert desired_width <= original_width
        assert desired_height <= original_height
    except AssertionError:
        print("Desired `width` and `height` must be smaller than the original image")
        print(
            "Original image has width = {original_width} and height = {original_height}"
        )

    center_width = original_width // 2
    center_height = original_height // 2

    cropped_img = img[
        center_height - desired_height // 2 : center_height + desired_height // 2,
        center_width - desired_width // 2 : center_width + desired_width // 2,
        :,
    ]

    return cropped_img


def center_image(img: NDArray[uint8], centroid: Tuple[int, int]) -> NDArray[uint8]:
    """
    This function shifts the image such a way that the dune centroid is
    fixed at the center of the image.
    """
    height, width, _ = img.shape
    centroid_x, centroid_y = centroid

    offset_x = width / 2 - centroid_x
    offset_y = height / 2 - centroid_y

    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))

    return translated_img


def process(i: int, filename: str, filename_mask: Optional[str] = "") -> None:
    img = cv2.imread(filename)
    img, centroid = get_centroid(img, draw_contours=False)

    img_centered = center_image(img, centroid)
    centered_img_size = min(img_centered.shape[:2])

    centercropped_img = crop_from_center(
        img_centered, centered_img_size, centered_img_size
    )

    cv2.imwrite(
        os.path.join(OUTPUT_PATH, f"centered{i:04d}.png"),
        centercropped_img,
    )

    if filename_mask:
        img_mask = cv2.imread(filename_mask)
        img_centered_mask = center_image(img_mask, centroid)
        centercropped_img_mask = crop_from_center(
            img_centered_mask, centered_img_size, centered_img_size
        )
        cv2.imwrite(
            os.path.join(OUTPUT_PATH_MASK, f"centered{i:04d}.png"),
            centercropped_img_mask,
        )


@time_it
def main() -> None:
    directory = Path(OUTPUT_PATH)
    directory.mkdir(parents=True, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(PATH, "*.png")))
    num_images = len(image_files)

    indices = range(num_images)

    parallel_fn = process
    parallel_fn_args = zip(indices, image_files)

    if OUTPUT_PATH_MASK:
        directory = Path(OUTPUT_PATH_MASK)
        directory.mkdir(parents=True, exist_ok=True)

        image_files_mask = sorted(glob.glob(os.path.join(PATH_MASK, "*.png")))
        num_masks = len(image_files_mask)

        assert num_images == num_masks

        parallel_fn_args = zip(indices, image_files, image_files_mask)

    pool = Pool()
    pool.starmap(parallel_fn, parallel_fn_args)


if __name__ == "__main__":
    main()
