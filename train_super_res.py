from model import *

import keras
import numpy as np
from time import time
from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec
from scipy.misc import imsave
import math
from skimage.transform import radon, rescale, resize
import cv2
from PIL import Image



h=128  # height and width of the image
w=128
total = 200000 # Number of images to train the super resolution block

# Creating and loading the trained generator from the DUG
dugan = unet(input_size = (h,w,1))
dugan.load_weights("generator_50.h5")

# Creating a super resolution resnet
super_r = srresnet(input_size = (h,w,1), feature_dim=32, resunit_num=8)

X = np.load("sino_denoise.npy")
Y = np.load("images_total.npy")

# Preparing the data for training the super resolution model
Y_pred = np.zeros((total,h,w,1))  # Initialising the Images predicted by DUG and the true data
Y_true = np.zeros((total,h,w,1))

for i in range(total):
       
    Y_pred[i] = dugan.predict(X[i].reshape(1,128,128,1))
    Y_true[i] = Y[i].reshape(1,128,128,1)

# Training the Super resolution resnet 
sample_interval = 25

for epoch in range(101):
    
    train_history = super_r.fit(Y_pred,Y_true,epochs=1,shuffle=True,validation_split=0.2,batch_size=16)
    super_r.save_weights("checkpoints/super_resnet_"+str(epoch)+".h5")
    
    if epoch % sample_interval == 0:
        
        for j in range(100):
                          
            test_d = dugan.predict(resize(X[j],(128,128)).reshape((1,128,128,1)))
            img    = super_r.predict(test_d.reshape(1,128,128,1))
            imsave("image_at_"+str(j)+".png",img.reshape(h,w))  
            imsave("/GT_at_"+str(j)+".png",Y[i].reshape(h,w))  
            imsave("/dug_at_"+str(j)+".png",test_d.reshape(h,w))  

