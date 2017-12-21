import models
import utils
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np
from keras.datasets import cifar10

parser = argparse.ArgumentParser()
parser.add_argument('weights')
parser.add_argument('normal')
parser.add_argument('anomal')

def main(args):

    # ================================
    # Instantiate Model
    # ================================
    zgen = models.create_zgenerater()
    zgen.load_weights(args.weights)
    (_,input_h,input_w,input_c) = zgen.input_shape

    # ================================
    # Load dataset
    # ================================
    normal_input_paths = utils.make_paths_from_directory(args.normal)
    anomal_input_paths = utils.make_paths_from_directory(args.anomal)
    x_normal = utils.make_arrays_from_paths(
        paths=normal_input_paths,
        preprocess=utils.preprocess_input,
        target_size=(input_h,input_w))
    x_anomal = utils.make_arrays_from_paths(
        paths=anomal_input_paths,
        preprocess=utils.preprocess_input,
        target_size=(input_h,input_w))
    (x_cifar,_),(_,_) = cifar10.load_data()
    x_cifar = x_cifar[:100]
    x_input = np.concatenate((x_normal,x_anomal,x_cifar),axis=0)

    # ================================
    # Load dataset
    # ================================
    preds = zgen.predict(x_input)
    preds = preds.reshape(len(preds),preds.shape[-1])
    pca = decomposition.PCA(n_components=2)
    x_transformed = pca.fit_transform(preds)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(
        x_transformed[:len(x_normal),0],
        x_transformed[:len(x_normal),1],
        c='blue',label='normal')
    ax.scatter(
        x_transformed[len(x_normal):len(x_normal)+len(x_anomal),0],
        x_transformed[len(x_normal):len(x_normal)+len(x_anomal),1],
        c='red',label='anomaly')
    ax.scatter(
        x_transformed[len(x_normal)+len(x_anomal):,0],
        x_transformed[len(x_normal)+len(x_anomal):,1],
        c='yellow',label='cifar10')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('out2.png')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
