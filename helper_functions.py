
"""
A number of image preprocessing functions for the style transfer code
Author: Christian Baumgartner (c.f.baumgartner@gmail.com)
Date: 1. April 2016
"""

import numpy as np
import skimage.transform
from lasagne.utils import floatX


def convert_to_uint8(image):
    """
    Scales an image into the [0,255] range and casts it to uint8
    """
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)


def check_image_format(image):
    """
    Checks and fixes for the following:
     - is image really uint8
     - does the image have an alpha channel
     - is the image gray scale
    """

    # pngs can sometimes not be uint8, so make sure
    image = convert_to_uint8(image)

    # delete alpha channel if there is one
    image = image[:,:,0:3]

    # add third axis to gray scale imaes
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, axis=2)

    return image


def resize_image_to_input_size(image, im_size):
    """
    Resizes an images such that the smaller dimension equals im_size and the larger dimension
    is scaled keeping the aspect ration. im_size is given by the user as global parameter.
    """
    # resize image
    h, w, _ = image.shape
    if h < w:
        image = skimage.transform.resize(image, (im_size, w * im_size / h), preserve_range=True)
    else:
        image = skimage.transform.resize(image, (h * im_size / w, im_size), preserve_range=True)

    return image


def extract_aspect_box(image, aspect_ratio):
    """
    Extract a cropped region from the input image such that it matches a given aspect_ratio. The crop is
    extracted such that one dimension is preserved and the other is centrally cropped to fit the ratio.
    """
    image_h = image.shape[0]
    image_w = image.shape[1]

    if aspect_ratio >= 1:
        box_h = round(image_w / aspect_ratio)

        image = image[image_h//2-box_h//2:image_h//2+box_h//2, ...]
    else:
        box_w = aspect_ratio * image_h

        image = image[:, image_w//2-box_w//2:image_w//2+box_w//2, ...]

    return image


def resize_image(image, size):
    """
    Just a wrapper for the skimage resize function
    """
    return skimage.transform.resize(image, size, preserve_range=True)


def get_aspect_ratio(image):
    """
    Returns the aspect ratio of an image
    """
    h, w, _ = image.shape
    return float(w) / h


def prepare_image(image, mean_values):
    """
    Get an image into the right format to feed the neural network.
    """

    # Shuffle axes to c01
    image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)

    # Convert RGB to BGR
    image = image[::-1, :, :]

    # substract train sample mean
    image = image - mean_values

    return floatX(image[np.newaxis])


def deprocess(x, mean_values):
    """
    Convert the network output back into an image
    """
    x = np.copy(x[0])
    x += mean_values

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

    x = np.clip(x, 0, 255).astype('uint8')
    return x