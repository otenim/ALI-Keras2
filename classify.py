import keras
from keras.datasets import cifar10
import numpy as np
import os
import argparse
import utils
import pickle
import models

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('zgenerater_weights')
#parser.add_argument('images_root')
#parser.add_argument('num_classes', type=int)
#parser.add_argument('--result_root', default=os.path.join(curdir, 'result_unsuper'))
parser.add_argument('--pred_batch_size', type=int, default=1)

def main(args):

    # ===============================
    # Instantiate zgenerater(encoder)
    # ===============================
    zgenerater = models.create_zgenerater()
    zgenerater.load_weights(args.zgenerater_weights)

    # ===============================
    # Prepare dataset
    # ===============================
    # make input datasets
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train.astype(np.float32)
    x_train = utils.preprocess_input(x_train)

    # convert all images into encoded vector points
    outputs = zgenerater.predict(
        x_train,
        batch_size=args.pred_batch_size,
        verbose=1)
    print(outputs.shape)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
