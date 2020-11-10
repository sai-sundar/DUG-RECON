
import keras
import numpy as np
from scipy.misc import imsave
import pydicom
import glob
import math
from skimage.transform import radon, rescale, resize
import cv2
from PIL import Image


# Dataset used in the study was ACRIN-FLT-Breast (ACRIN 6688)


path_to_dcm = ""

n_of_patients = 200

# Converting the dicom files to npy arrays, assuming dataset organised into different patient folders (numbered:1,2.,3...)



def radon_transform(X):

    theta = np.linspace(0., 180., max(X.shape), endpoint=False)
    sinogram = radon(X.reshape(128,128), theta=theta, circle=True)

    
    return sinogram



for i in range(1,n_of_patients):
    editFiles = []
    editFiles=glob.glob(path_to_dcm+str(i)+'/*')
    
    data=[]
    sino=[]
    for files in editFiles:
        
        
        try:
            dcm  = pydicom.dcmread(files) 
        
    
        except IndexError:
            continue
    
        try:
            temp = (dcm.pixel_array).reshape(1,128,128,1)
        except ValueError:
            continue 
        
        if math.isnan(np.amax(temp)):         
            continue
        
        sinogram,temp = radon_transform(temp.reshape(512,512))
        sino.append(sinogram)
        data.append(temp)
    
    data = np.asarray(data)    
    
    sino = np.asarray(sino)    
    

    for i in range(len(data)):
        if count==20000:
            break
        temp_1 = resize(data[i],(128,128))
        Y[count]=temp_1.reshape((1,128,128,1))
        X[count]=sino[i].reshape((1,128,128,1))
        
        count = count + 1

np.save("Sinogram.npy",X)
np.save("Image.npy",Y)


