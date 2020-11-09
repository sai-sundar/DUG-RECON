 
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
import numpy as np
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model


image_shape = (256, 256, 3)
def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)

def pixel_shuffle(scale):
    return lambda x: tf.depth_to_space(x, scale)

def perceptual_loss(y_true, y_pred):
    y_pred = tf.image.grayscale_to_rgb(y_pred)
    y_true = tf.image.grayscale_to_rgb(y_true)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return (K.mean(K.abs(loss_model(y_true) - loss_model(y_pred))) + 100*K.mean(K.abs(y_true-y_pred)))


def unet(pretrained_weights = None,input_size = (256,256,1),d=0.5):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(d)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(d)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = perceptual_loss, metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def srresnet(input_size = (256,256,1), feature_dim=64, resunit_num=16):

    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(input_size)
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    
    #x = upsample(x, feature_dim * 4)
    #x = upsample(x, feature_dim * 4)
    x = Conv2D(1, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    
    
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    model.compile(optimizer=Adam(0.001), loss=perceptual_loss, metrics=[PSNR])


    return model



def create_full_us_unet(
        image_shape=(256,256,1),  # (H, W, C)
        channel_number: int = 16,
        padding='same',
        residual= True):

    # Network properties
    kernel_shape = (3, 3)
    activation_func = 'relu'
    pool_size = (2, 2)
    channel_factor = 2
    use_bias = True
    caxis = -1  # channel axis

    # Initializers
    kernel_initializer = 'glorot_uniform'
    bias_initializer = 'zeros'

    # Input layer
    inputs = Input(shape=image_shape, dtype=tf.float32)

    # Lists to track some important parts
    skip_maps = list()

    # Initial channel expansion
    fmap = Conv2D(
        filters=channel_number,
        kernel_size=(1, 1),
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        # activation=activation_func,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(inputs)

    # Downward path (i.e. encoder)#############################################
    _out_ch_nb = channel_number
    # Level 1------------------------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Prepare skip connection
    skip_maps.append(fmap)

    # Downsampling Level 1 -> Level 2------------------------------------------
    fmap = MaxPool2D(
        pool_size=pool_size,
        strides=pool_size,
        padding=padding,
    )(fmap)

    # Convolution + expansion of channel number
    _out_ch_nb *= channel_factor
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 2------------------------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Prepare skip connection
    skip_maps.append(fmap)

    # Downsampling Level 2 -> Level 3------------------------------------------
    fmap = MaxPool2D(
        pool_size=pool_size,
        strides=pool_size,
        padding=padding,
    )(fmap)

    # Convolution + expansion of channel number
    _out_ch_nb *= channel_factor
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 3------------------------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Prepare skip connection
    skip_maps.append(fmap)

    # Downsampling Level 3 -> Level 4------------------------------------------
    fmap = MaxPool2D(
        pool_size=pool_size,
        strides=pool_size,
        padding=padding,
    )(fmap)

    # Convolution + expansion of channel number
    _out_ch_nb *= channel_factor
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 4------------------------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Prepare skip connection
    skip_maps.append(fmap)

    # Downsampling Level 4 -> Level 5------------------------------------------
    fmap = MaxPool2D(
        pool_size=pool_size,
        strides=pool_size,
        padding=padding,
    )(fmap)

    # Convolution + expansion of channel number
    _out_ch_nb *= channel_factor
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 5 (bottom layer)---------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Upward path##############################################################
    # Upsampling Level 5 -> Level 4--------------------------------------------
    _out_ch_nb //= channel_factor
    # Upsample using transpose convolution
    fmap = Conv2DTranspose(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        strides=pool_size,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 4------------------------------------------------------------------
    # Concatenate skip connection
    fmap = Concatenate(axis=caxis)([fmap, skip_maps[-1]])

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Upsampling Level 4 -> Level 3--------------------------------------------
    _out_ch_nb //= channel_factor
    # Upsample using transpose convolution
    fmap = Conv2DTranspose(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        strides=pool_size,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 3------------------------------------------------------------------
    # Concatenate skip connection
    fmap = Concatenate(axis=caxis)([fmap, skip_maps[-2]])

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Upsampling Level 3 -> Level 2--------------------------------------------
    _out_ch_nb //= channel_factor
    # Upsample using transpose convolution
    fmap = Conv2DTranspose(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        strides=pool_size,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 2------------------------------------------------------------------
    # Concatenate skip connection
    fmap = Concatenate(axis=caxis)([fmap, skip_maps[-3]])

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Upsampling Level 2 -> Level 1--------------------------------------------
    _out_ch_nb //= channel_factor
    # Upsample using transpose convolution
    fmap = Conv2DTranspose(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        strides=pool_size,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 1 (Top Level)------------------------------------------------------
    # Concatenate skip connection
    fmap = Concatenate(axis=caxis)([fmap, skip_maps[-4]])

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Final channel reduction##################################################
    outputs = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding=padding,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Residual skip connection
    if residual:
        outputs = Add()([outputs, inputs])

    # Create Keras sequential model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
    return model



def create_small_us_unet(
        image_shape=(256,256,1),  # (H, W, C)
        channel_number=16,
        padding='same',
        residual= True
):

    # Network properties
    kernel_shape = (3, 3)
    activation_func = 'relu'
    pool_size = (2, 2)
    channel_factor = 2
    use_bias= True
    caxis = -1  # channel axis

    # Initializers
    kernel_initializer = 'glorot_uniform'
    bias_initializer = 'zeros'

    # Input layer
    inputs = Input(shape=image_shape, dtype=tf.float32)

    # Lists to track some important parts
    skip_maps = list()

    # Initial channel expansion
    fmap = Conv2D(
        filters=channel_number,
        kernel_size=(1, 1),
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        # activation=activation_func,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(inputs)

    # Downward path (i.e. encoder)#############################################
    _out_ch_nb = channel_number
    # Level 1------------------------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Prepare skip connection
    skip_maps.append(fmap)

    # Downsampling Level 1 -> Level 2------------------------------------------
    fmap = MaxPool2D(
        pool_size=pool_size,
        strides=pool_size,
        padding=padding,
    )(fmap)

    # Convolution + expansion of channel number
    _out_ch_nb *= channel_factor
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 2------------------------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Prepare skip connection
    skip_maps.append(fmap)

    # Downsampling Level 2 -> Level 3------------------------------------------
    fmap = MaxPool2D(
        pool_size=pool_size,
        strides=pool_size,
        padding=padding,
    )(fmap)

    # Convolution + expansion of channel number
    _out_ch_nb *= channel_factor
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        # strides=(1, 1),
        padding=padding,
        # data_format=data_format,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 3 (bottom Level)---------------------------------------------------
    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )(fmap)

    # Upward path##############################################################
    # Upsampling Level 3 -> Level 2--------------------------------------------
    _out_ch_nb //= channel_factor
    # Upsample using transpose convolution
    fmap = Conv2DTranspose(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        strides=pool_size,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 2------------------------------------------------------------------
    # Concatenate skip connection
    fmap = Concatenate(axis=caxis)([fmap, skip_maps[-1]])

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Upsampling Level 2 -> Level 1--------------------------------------------
    _out_ch_nb //= channel_factor
    # Upsample using transpose convolution
    fmap = Conv2DTranspose(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        strides=pool_size,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Level 1 (Top Level)------------------------------------------------------
    # Concatenate skip connection
    fmap = Concatenate(axis=caxis)([fmap, skip_maps[-2]])

    fmap = Conv2D(
        filters=_out_ch_nb,
        kernel_size=kernel_shape,
        padding=padding,
        activation=activation_func,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Final channel reduction##################################################
    outputs = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding=padding,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(fmap)

    # Residual skip connection
    if residual:
        outputs = Add()([outputs, inputs])

    # Create Keras sequential model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
    return model

