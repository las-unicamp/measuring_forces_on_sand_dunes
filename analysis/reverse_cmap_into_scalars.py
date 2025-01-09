"""
This module handles reversing the colormap of an image into scalars.
"""

import numpy as np
import scipy.cluster.vq as scv


def cmap2scalar(image, cmap, norm):
    """
    Turn the RGB values of the colormap into a float according to the specified
    norm. It can only handle linear norms. This function returns the scalar
    values in each pixel of the image.

    Every colormap has a gradient which is an array of RGBA 4-tuples. This
    gradient acts as a "code book" and the colors of the gradient correspond to
    values from 0 to 1.

    Here, we use scipy's vector quantization function `scipy.cluster.vq.vq` to map
    all the colors in the image onto the nearest color in gradient.

    Reference: https://stackoverflow.com/questions/3720840/how-to-reverse-a-color-map-image-to-scalar-values
    """
    if image.dtype == np.uint8:
        image = np.float32(image / 255.0)

    gradient = cmap(np.linspace(0.0, 1.0, cmap.N))

    # Reshape image to something like (240*240, 4), all the 4-tuples in a long list...
    flatten_img = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

    # Use vector quantization to shift the values in flatten_img to the nearest point in
    # the code book (gradient).
    code, dist = scv.vq(flatten_img, gradient)

    # code is an array of length flatten_img (240*240), holding the code book index for
    # each observation. (flatten_img are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype("float") / gradient.shape[0]

    # rescale values
    values = np.interp(x=values, xp=[0, 1], fp=[norm.vmin, norm.vmax])

    # Reshape values back to (240,240)
    values = values.reshape(image.shape[0], image.shape[1])

    return values
