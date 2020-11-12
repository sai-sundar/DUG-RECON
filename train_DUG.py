
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


h = 128  # height and width of the image
w = 128 

X_t = np.load("sino_denoise.npy") # Loading sinogram
Y_t = np.load("images_total.npy")# Loading Images

# Dividing the data into training and validation
X_train = X_t[0:40000] # We have trained on 40000 projection-image pairs
Y_train = Y_t[0:40000]

X_valid = X_t[40000:60000]
Y_valid = Y_t[40000:60000]


input_datagen = ImageDataGenerator()

input_generator = input_datagen.flow(
    X_train,
    batch_size=1,
    seed=seed,
    shuffle=False)




class GAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        #self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator2
        self.generator2 = self.build_generator2()
        
        # Build the generator1
        self.generator1 = self.build_generator1()
        
        # Generator takes  denoised projections as input and generates images
        z = Input(shape=(h,w,1,))
        img = self.generator1(z)

        # For the combined model we will only train the generator
        self.generator2.trainable = False

        # The discriminator takes generated images as input and maps to projections
        validity = self.generator2(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        

    def build_generator1(self):
        model = unet(input_size = (h,w,1))
        return model
    def build_genertator2(self):
        model = unet(input_size = (h,w,1))
        return model

    
    
    def train(self, epochs, batch_size=8, sample_interval=50):
        
        for epoch in range(epochs):


            
                      
            gen_imgs = self.generator1.predict_generator(input_generator,40000) # Generate Images
            g2_loss = self.generator2.fit(gen_imgs,X_train,epochs=1,batch_size=batch_size,shuffle=False)
            g1_loss1 = self.generator1.fit(X,Y,epochs=1,batch_size=batch_size,shuffle=False)
            g1_loss2 = self.combined.fit(X,X,epochs=1,batch_size=batch_size,shuffle=False)


    def pred(self,num=10):
        
        for i in range(num):
            imsave("pred_"+str(i)+".png",self.generator1.predict(X_valid[i]))
         
                





if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=51, batch_size=16, sample_interval=25)
    #gan.pred(num=10)



            
