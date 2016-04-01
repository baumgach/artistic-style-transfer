# Introduction

Implementation of the artistic style transfer paper by Gatys et al. 
[http://arxiv.org/abs/1508.06576] using the [Lasagne](http://lasagne.readthedocs.org/en/latest/) framework. 

The code should be run on a GPU. 

# Set up environment

If you don't have Lasagne already installed you can find instructions [here](http://lasagne.readthedocs.org/en/latest/user/installation.html). 
In particular, I recommend installing the bleeding edge versions of Theano and Lasagne:

    pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
  
You will also need `matplotlib`, `scikit-image`, `scikit-learn`, and `numpy`. 

    pip install matplotlib scikit-image scikit-image scikit-image
  
# Examples

![alt tag](https://gitlab.doc.ic.ac.uk/cbaumgar/artistic-style-transfer/styles/japanese.jpg)