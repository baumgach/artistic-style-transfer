"""
Code implementing the artistic style transfer method proposed by [Gatys et al.], which allows to transfer
the style of one image or art piece onto another image which is typically a photograph. In contrast, to the original
paper, here the content loss function was modified to make it independent of the image size.

The code uses the lasagne neural network framework and is very loosely based on the following tutorial:
https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb

A reasonably fast GPU is required to run this code.

[Gatys et al.]: "A neural Algorithm of Artistic Style", arXiv preprint arXiv:1508.06576,
http://arxiv.org/abs/1508.06576

Author: Christian Baumgartner (c.f.baumgartner@gmail.com)
Date: 1. April 2016
"""

import lasagne
import numpy as np
import pickle
import scipy

import theano
import theano.tensor as T

from lasagne.utils import floatX

import matplotlib.pyplot as plt

from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

import time
import helper_functions as utils

np.random.seed(42)

# Build the VGG19 model
# Note: tweaked to use average pooling instead of maxpooling
def build_vgg19_model(image_h, image_w):
    """
    Function building the VGG-19 network with average pool layers instead of max pool layers
    and a specific image size.
    """

    net = {}
    net['input'] = InputLayer((1, 3, image_h, image_w))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net


# Implementation of the cost functions
def gram_matrix(x):
    """
    returns the gram matrix of a layer output, i.e. filter bank
    """
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    """
    Defines the content loss contribution from two images at a specific layer. Reasoning: If the content of
    two images is the same the response of a given layer for each image should be the same.
    Note that additionally there is a 1 / image size factor which was not in the original paper
    """
    p = P[layer]
    x = X[layer]

    # slight change to cost function with respect to paper, which makes it independent of image size
    M = p.shape[2] * p.shape[3]
    loss = (1./(2*M)) * ((x - p)**2).sum()

    return loss


def style_loss(A, X, layer):
    """
    Style loss contribution of two images at a specific layer. Reasoning: If the style of two images is the same
    then the relations between all the filter outputs at this layer (captured by the gram matrix) should also be
    the same.
    """
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss


def total_variation_loss(x):
    """
    A total variation loss to reduce high frequency noise
    """
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()


# Helper functions to interface with scipy.optimize
def eval_loss(x0, image_h, image_w):
    """
    Helper function to give scipy.optimize access to evaluating the loss function through Lasagne
    """
    x0 = floatX(x0.reshape((1, 3, image_h, image_w)))
    generated_image.set_value(x0)
    return f_loss().astype('float64')


def eval_grad(x0, image_h, image_w):
    """
    Helper function to give scipy.optimize access to evaluating the gradient of the loss function through Lasagne
    """
    x0 = floatX(x0.reshape((1, 3, image_h, image_w)))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')


# Main part
if __name__ == '__main__':

    # constants
    MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((3,1,1))
    CONTENT_LOSS_FACTOR = 100  # only the ratio counts, this exact number is irrelevant

    # display options
    VERBOSE_OPTIMISATION = True
    DISPLAY_INPUT_IMAGES = False

    # Method parameters with tuning instructions
    IMAGE_SIZE = 4 * 224  # 2 * 224 is faster 4*224 gives better results
    CONTENT_STYLE_RATIO = 8e-6  # increase this number if you want more content, decrease for more style
    PENALTY_FACTOR = 1e-8  # for image size 2*224, 1e-7 seems to work well, for 4*224 1e-9 is enough
    MAXFUN = 2000  # 1000 seems to be more or less the minimum, 2000 is better,
                   # running to convergence can take a very long time

    # These are the layers as described in the paper. Sometimes, removing conv1_1 and conv2_1 also gives nice results
    # paying more attention to higher level art concepts. Also taking the content from conv5_2 can lead to really cool
    # but really abstract results.
    STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    CONTENT_LAYERS = 'conv4_2'

    STYLE_LOSS_FACTOR = CONTENT_LOSS_FACTOR / CONTENT_STYLE_RATIO

    # Input and output files
    CONTENT_FILENAME = 'bernhard.jpg'
    ART_FILENAME = 'styles/munch_scream.jpg' #vegetables.jpg'
    OUTPUT_FILENAME = 'bernhard_munch2.png'

    print "loading images..."

    # read images
    photo = plt.imread(CONTENT_FILENAME)
    art = plt.imread(ART_FILENAME)

    # check images for format
    photo = utils.check_image_format(photo)
    art = utils.check_image_format(art)

    # resize and crop images to be the same size and aspect ratio
    aspect_ratio = utils.get_aspect_ratio(photo)
    art = utils.extract_aspect_box(art, aspect_ratio)

    photo = utils.resize_image_to_input_size(photo, IMAGE_SIZE)
    art = utils.resize_image(art, photo.shape[0:2])

    # Display images to user
    if DISPLAY_INPUT_IMAGES:
        plt.figure()
        plt.imshow(photo.astype('uint8'))
        plt.figure()
        plt.imshow(art.astype('uint8'))
        plt.show()

    # get images into the right format for the VGG19 model
    photo = utils.prepare_image(photo, MEAN_VALUES)
    art = utils.prepare_image(art, MEAN_VALUES)

    image_h = photo.shape[2]
    image_w = photo.shape[3]

    print "initialising model..."

    net = build_vgg19_model(image_h, image_w)
    values = pickle.load(open('vgg19_normalized.pkl'))['param values']
    lasagne.layers.set_all_param_values(net['pool5'], values)

    print "setting up loss and compiling..."

    # get all layers that are needed for the cost functions and their outputs
    layers = list(set([CONTENT_LAYERS] + STYLE_LAYERS))
    layers = {k: net[k] for k in layers}

    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                      for k, output in zip(layers.keys(), outputs)}

    art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                    for k, output in zip(layers.keys(), outputs)}

    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, image_h, image_w))))

    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    # Define loss function
    losses = []

    # content loss term
    losses.append(CONTENT_LOSS_FACTOR * content_loss(photo_features, gen_features, CONTENT_LAYERS))

    # style loss term
    for style_layer in STYLE_LAYERS:
        losses.append((1. / len(STYLE_LAYERS)) * STYLE_LOSS_FACTOR * style_loss(art_features, gen_features, style_layer))

    # total variation penalty term
    losses.append(PENALTY_FACTOR * total_variation_loss(generated_image))

    # overall cost function
    total_loss = sum(losses)

    grad = T.grad(total_loss, generated_image)

    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)

    print "generating images..."

    # Initialize with a noise image
    generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, image_h, image_w))))

    x0 = generated_image.get_value().astype('float64')
    start_time = time.time()

    scipy.optimize.fmin_l_bfgs_b(
        lambda x: eval_loss(x, image_h, image_w),
        x0.flatten(),
        fprime=lambda x: eval_grad(x, image_h, image_w),
        iprint=VERBOSE_OPTIMISATION,
        maxfun=MAXFUN
    )

    x_final = generated_image.get_value().astype('float64')

    elapsed_time = time.time() - start_time
    plt.imsave(OUTPUT_FILENAME, utils.deprocess(x_final, MEAN_VALUES))
    print "took %.2f minutes" % (elapsed_time/60)
