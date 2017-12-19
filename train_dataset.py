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
import random

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--split',type=float, default=0.8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.999)
parser.add_argument('--snap_freq', type=int, default=5)
parser.add_argument('--result', default=os.path.join(curdir, 'result'))

def save_config(path, args):
    with open(path, 'w') as f:
        f.write('Epochs: %d\n' % (args.epochs))
        f.write('Batchsize: %d\n' % (args.batch_size))
        f.write('Learning rate: %f\n' % (args.lr))
        f.write('Beta_1: %f\n' % (args.beta_1))
        f.write('Beta_2: %f\n' % (args.beta_2))

def d_lossfun(y_true, y_pred):
    """
    y_pred[:,:,:,0]: p
    y_pred[:,:,:,1]: q
    """
    p = K.clip(y_pred[:,:,:,0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:,:,:,1], K.epsilon(), 1.0 - K.epsilon())
    return -K.mean(K.log(q) + K.log(1. - p))

def g_lossfun(y_true, y_pred):
    """
    y_pred[:,:,:,0]: p
    y_pred[:,:,:,1]: q
    """
    p = K.clip(y_pred[:,:,:,0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:,:,:,1], K.epsilon(), 1.0 - K.epsilon())
    return -K.mean(K.log(1. - q) + K.log(p))

def main(args):

    # =====================================
    # Preparation (load dataset and create
    # a directory which saves results)
    # =====================================
    input_paths = utils.make_paths_from_directory(args.dataset)
    random.shuffle(input_paths)
    border = int(len(input_paths) * 0.8)
    train_paths, test_paths = input_paths[:border], input_paths[border:]

    if os.path.exists(args.result) == False:
        os.makedirs(args.result)
    save_config(os.path.join(args.result, 'config.txt'), args)

    # =====================================
    # Instantiate models
    # =====================================
    xgen = models.create_xgenerater()
    zgen = models.create_zgenerater()
    disc = models.create_discriminater()
    opt_d = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)
    opt_g = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)

    xgen.trainable = False
    zgen.trainable = False
    gan_d = models.create_gan(xgen, zgen, disc)
    gan_d.compile(optimizer=opt_d, loss=d_lossfun)

    xgen.trainable = True
    zgen.trainable = True
    disc.trainable = False
    gan_g = models.create_gan(xgen, zgen, disc)
    gan_g.compile(optimizer=opt_g, loss=g_lossfun)


    # =====================================
    # Training Loop
    # =====================================
    num_train = len(train_paths)
    for epoch in range(args.epochs):
        print('Epochs %d/%d' % (epoch+1, args.epochs))
        pbar = Progbar(num_train)
        for i in range(0, num_train, args.batch_size):
            x = utils.make_arrays_from_paths(
                train_paths[i:i+args.batch_size],
                preprocess=utils.preprocess_input,
                target_size=(32,32))
            z = np.random.normal(size=(len(x), 1, 1, 64))

            # train discriminater
            d_loss = gan_d.train_on_batch([x, z], np.zeros((len(x), 1, 1, 2)))
            # train generaters
            g_loss = gan_g.train_on_batch([x, z], np.zeros((len(x), 1, 1, 2)))

            # update progress bar
            pbar.add(len(x), values=[
                ('d_loss', d_loss),
                ('g_loss', g_loss),
            ])

        if (epoch+1) % args.snap_freq == 0:
            # ===========================================
            # Save result
            # ===========================================
            # Make a directory which stores learning results
            # at each (args.frequency)epochs
            dirname = 'epochs%d' % (epoch+1)
            path = os.path.join(args.result, dirname)
            if os.path.exists(path) == False:
                os.makedirs(path)

            # Save generaters' weights
            xgen.save_weights(os.path.join(path, 'xgen_weights.h5'))
            zgen.save_weights(os.path.join(path, 'zgen_weights.h5'))

            # Save generated images
            img = utils.generate_img(xgen)
            img.save(os.path.join(path, 'generated.png'))

            # Save reconstructed images
            x = utils.make_arrays_from_paths(
                test_paths,
                preprocess=None,
                target_size=(32,32))
            img = utils.reconstruct_img(x, xgen, zgen)
            img.save(os.path.join(path, 'reconstructed.png'))




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
