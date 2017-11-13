import keras
from keras.datasets import cifar10
import numpy as np
import models
import utils
import os
import argparse

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('weight')
parser.add_argument('--out', default=curdir)

def main(args):

    zgen = models.create_zgenerater()
    zgen.load_weights(args.weight)

    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype(np.float32)
    x_test = utils.preprocess_input(x_test)
    y_test = y_test.astype(np.float32)

    zvars = zgen.predict(x_test, batch_size=32, verbose=1)
    zvars = zvars.reshape(-1, 64)

    np.save(os.path.join(args.out, 'images.npy'), x_test)
    np.save(os.path.join(args.out, 'zvars.npy'), zvars)
    np.save(os.path.join(args.out, 'labels.npy'), y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
