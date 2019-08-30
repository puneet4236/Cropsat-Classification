# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:24:40 2019

@author: Puneet Garg
"""

from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape,  Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import cv2
import numpy as np
import json


img_w = 128
img_h = 128
n_labels = 256


kernel = 3
pad = 1
pool_size = 2



encoding_layers = [
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
]




decoding_layers = [
    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]


segnet_basic = models.Sequential()
segnet_basic.add(Layer(input_shape=(128,128,7)))


segnet_basic.encoding_layers = encoding_layers
for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)


segnet_basic.decoding_layers = decoding_layers
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)
    
    
segnet_basic.add(Activation('softmax'))

print(segnet_basic.summary())

