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



################################# Loading the data ################################
sino_real=np.load("Sinogram.npy")  # High count sinogram data
total_data=len(sino_real*3)
sino_low = np.zeros((total_data,128,128,1))
sino_high= np.zeros((total_data,128,128,1))
# Creating sinogram data woth 3 different counts 

for i in range(len(sino_real)):

    val = sino_real[i].sum()
    count1 = 250000
    sinogram    = np.random.poisson((sino_real[i] / val) * count1).astype(np.float)
    sino_low[i]    = sinogram * (val / count1)
    sino_high[i]    = sino_real[i]

    count2 = 500000
    sinogram    = np.random.poisson((sino_real[i] / val) * count1).astype(np.float)
    sino_low[i+len(sino_real)]    = sinogram * (val / count1)
    sino_high[i+len(sino_real)]    = sino_real[i]

    count1 = 600000
    sinogram    = np.random.poisson((sino_real[i] / val) * count1).astype(np.float)
    sino_low[i+2*len(sino_real)]    = sinogram * (val / count1)
    sino_high[i+2*len(sino_real)]    = sino_real[i]


################################ Creating Denoising Model #########################


model = create_full_us_unet(
        image_shape=(128,128,1),  # (H, W, C)
        channel_number=32,
        padding='same',
        residual= True
)



################################# Training Denoising Model ########################



loss = model.fit(sino_low,sino_high,epochs=100,batch_size=8,validation_split=0.2)

model.save("denoise.h5")


