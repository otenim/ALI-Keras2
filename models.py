import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

def create_xgenerater():
    input = Input(shape=(1,1,64))
    x = Conv2DTranspose(256, (4,4), strides=(1,1))(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, (4,4), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, (4,4), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, (4,4), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, (5,5), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(32, (1,1), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    output = Conv2D(3, (1,1), strides=(1,1), activation='tanh')(x)
    return Model(input, output, name='xgenerater')

def create_zgenerater():
    input = Input(shape=(32,32,3))
    x = Conv2D(32, (5,5), strides=(1,1))(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (4,4), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4,4), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4,4), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (4,4), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (1,1), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    mu = Conv2D(64, (1,1), strides=(1,1))(x)
    sigma = Conv2D(64, (1,1), strides=(1,1))(x)
    concatenated = Concatenate(axis=-1)([mu, sigma])

    output = Lambda(
        function=lambda x: x[:,:,:,:64] + K.exp(x[:,:,:,64:]) * K.random_normal(shape=K.shape(x[:,:,:,64:])),
        output_shape=(1,1,64)
    )(concatenated)
    return Model(input, output, name='zgenerater')

def create_discriminater():
    input_x = Input(shape=(32,32,3))
    input_z = Input(shape=(1,1,64))

    x = Dropout(0.2)(input_x)
    x = Conv2D(32, (5,5), strides=(1,1), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (4,4), strides=(2,2), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (4,4), strides=(1,1), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (4,4), strides=(2,2), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, (4,4), strides=(1,1), activation='relu')(x)

    z = Dropout(0.2)(input_z)
    z = Conv2D(512, (1,1), strides=(1,1), activation='relu')(z)
    z = Dropout(0.5)(z)
    z = Conv2D(512, (1,1), strides=(1,1), activation='relu')(z)

    concatenated = Concatenate(axis=-1)([x, z])
    c = Dropout(0.5)(concatenated)
    c = Conv2D(1024, (1,1), strides=(1,1), activation='relu')(c)
    c = Conv2D(1024, (1,1), strides=(1,1), activation='relu')(c)
    c = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(c)
    return Model([input_x, input_z], c, name='discriminater')

def create_gan(xgenerater, zgenerater, discriminater):
    input_x = Input(shape=(32,32,3))
    input_z = Input(shape=(1,1,64))
    x_gen = xgenerater(input_z)
    z_gen = zgenerater(input_x)
    p = discriminater([x_gen, input_z])
    q = discriminater([input_x, z_gen])
    concatenated = Concatenate(axis=-1)([p,q])
    return Model([input_x, input_z], concatenated, name='gan')
