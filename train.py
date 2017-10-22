import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import Progbar
import numpy as np
import models
import utils
import os
import argparse

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)

def d_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p
    y_pred[:,1]: q
    """
    p = K.clip(y_pred[:,0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:,1], K.epsilon(), 1.0 - K.epsilon())
    return -K.mean(K.log(q) + K.log(1. - p))

def g_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p
    y_pred[:,1]: q
    """
    p = K.clip(y_pred[:,0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:,1], K.epsilon(), 1.0 - K.epsilon())
    return -K.mean(K.log(1. - q) + K.log(p))

def main(args):

    # =====================================
    # Load and preprocess dataset
    # =====================================
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = utils.preprocess_input(x_train)

    # =====================================
    # Instantiate models
    # =====================================
    generater = models.create_generater()
    encoder = models.create_encoder()
    discriminater = models.create_discriminater()

    generater.trainable = False
    encoder.trainable = False
    gan_train_d = models.create_gan(generater, encoder, discriminater)
    gan_train_d.compile(Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3), loss=d_lossfun)

    generater.trainable = True
    encoder.trainable = True
    discriminater.trainable = False
    gan_train_g = models.create_gan(generater, encoder, discriminater)
    gan_train_g.compile(Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3), loss=g_lossfun)

    # =====================================
    # Training Loop
    # =====================================
    num_train = len(x_train)
    for epoch in range(args.epochs):
        print('Epochs %d/%d' % (epoch+1, args.epochs))
        pbar = Progbar(num_train)
        for i in range(0, num_train, args.batch_size):
            x = x_train[i:i+args.batch_size]
            z = np.random.uniform(0., 1., size=(len(x), 64))

            # train discriminater
            d_loss = gan_train_d.train_on_batch([x, z], np.zeros((len(x), 2)))
            # train generaters
            g_loss = gan_train_g.train_on_batch([x, z], np.zeros((len(x), 2)))

            # update progress bar
            pbar.add(len(x), values=[
                ('d_loss', d_loss),
                ('g_loss', g_loss),
            ])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
