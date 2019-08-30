# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:07:59 2019

@author: Puneet Garg
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

a=os.listdir("./input");
b=os.listdir("./target");
x=plt.imread("./input/lc8_2015245_10_11_1.tif")
print(x.size)
y=plt.imread("./input/lc8_2015245_10_11_2.tif")  
print(y.size)
pixel=[]
for filename in os.listdir("C:\\Users\\Puneet Garg\\Desktop\\bennett\\group data\\dataset\\input"):
    img=cv2.imread(os.path.join("C:\\Users\\Puneet Garg\\Desktop\\bennett\\group data\\dataset\\input",filename))
    pixel.append(img)
pixel[0].size

    from PIL import Image
    pix=[]
    j=0
    for i in os.listdir("./input/"):
        y=Image.open("./input/"+i)
        pix.append(np.array(y))  
from numpy import zeros, newaxis
for i in range(len(pix)):
    pix[i]=np.reshape(pix[i],(128,128))

'''
m=np.random.rand(3,2,2)
u=np.zeros((2,2))
v=np.zeros((2,2))
w=np.zeros((2,2))
m[0:2:2]=u
m[1:2:2]=v
m[2:2:2]=w
'''
group=[]
for i in range(1,1594):
    temp=np.random.rand(128,128,7)
    for j in range(0,7):
        x=pix[(i-1)*7+j]
        temp[:,:,j]=x
    group.append(temp)
    

res=np.array(group)



import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle




