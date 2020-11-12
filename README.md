# DUG-RECON: A Framework for Tomographic Image Reconstruction with Generative Convolutional Networks
Tomographic Medical Image Reconstruction involves the mapping from projection space data collected by detectors to images that can be analysed for diagnosis. This work is an attempt to use Deep Learning to entirely perform the task of Reconstruction. 

### Requirements:
* Python 3
* Tensorflow 1.14
* Keras 2.0
* pydicom 

### Architecture Description:

This framework consists of three stages, decentralising the image reconstruction process. First stage denoises the projection data, the second stage learns the mapping from projection to image space and finally a super resolution block improves the data in the image space.

![Three-stage](https://github.com/sai-sundar/DUG-RECON/blob/main/images/three_stage.png)

1. **Denoising** 
This block is a modified version of the U-Net as depicted below.

![Denoise-stage](https://github.com/sai-sundar/DUG-RECON/blob/main/images/denoise_nn.jpg)


The DUG-RECON framework has a three stage training with Denoising, DUG and the SuperResnet block.

The code in this repo is associated to the article given below:
*DUG-RECON: A Framework for Direct Image Reconstruction using Convolutional Generative Networks*, The article can be found [here](https://doi.org/10.1109/TRPMS.2020.3033172)


