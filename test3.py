import models
import utils
import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    x_cifar = utils.preprocess_input(x_cifar[:100])
    x_input = np.concatenate((x_normal,x_anomal,x_cifar),axis=0)

    # ================================
    # Load dataset
    # ================================
    preds = zgen.predict(x_input)
    preds = preds.reshape(len(preds),preds.shape[-1])
    pca = decomposition.PCA(n_components=3)
    x_transformed = pca.fit_transform(preds)
    z_normal = x_transformed[:len(x_normal)]
    z_anomal = x_transformed[len(x_normal):len(x_normal)+len(x_anomal)]
    z_cifar = x_transformed[len(x_normal)+len(x_anomal):]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(z_normal[:,0],z_normal[:,1],z_normal[:,2],color='blue',label='normal')
    ax.scatter(z_anomal[:,0],z_anomal[:,1],z_anomal[:,2],color='red',label='anomal')
    ax.scatter(z_cifar[:,0],z_cifar[:,1],z_cifar[:,2],color='orange',label='cifar10')

    plt.legend(loc='best')
    plt.savefig('out3.png')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
