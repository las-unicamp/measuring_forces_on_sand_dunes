"""
Module to mask the sand dune. It expects a black background image.
"""

import cv2
import numpy as np
import numpy.typing

OpenCVMat = numpy.typing.NDArray[np.uint8]


def apply_mask(img: OpenCVMat) -> OpenCVMat:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channel_l = lab[:, :, 0]

    # threshold
    lower = 0
    upper = 10
    mask = cv2.inRange(channel_l, lower, upper)
    mask = 255 - mask

    return mask
