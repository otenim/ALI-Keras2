import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import argparse
import models
import utils

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('zgenerater_weights')
parser.add_argument('--samples_per_class', type=int, default=20)

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
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype(np.float32)
    x_test = utils.preprocess_input(x_test)

    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    ]
    num_classes = len(classes)
    input_data = []
    for i in range(num_classes):
        ind = (y_test == i).reshape(len(y_test))
        sample = (x_test[ind])[:args.samples_per_class]
        output = zgenerater.predict_on_batch(sample)
        output = output.reshape(len(sample), output.shape[-1])
        input_data.extend(output)
    input_data = np.array(input_data)
    pca = PCA(n_components=3)
    pca.fit(input_data)
    compressed_input_data = pca.transform(input_data)

    # ===============================
    # Make graph
    # ===============================
    colorlist = np.random.uniform(size=(num_classes, 3))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for i in range(num_classes):
        x = compressed_input_data[i*args.samples_per_class : (i+1)*args.samples_per_class, 0]
        y = compressed_input_data[i*args.samples_per_class : (i+1)*args.samples_per_class, 1]
        z = compressed_input_data[i*args.samples_per_class : (i+1)*args.samples_per_class, 2]
        ax.plot3D(x, y, z, 'o', label=classes[i])
    ax.legend(loc='best')
    #plt.savefig(os.path.join(curdir, 'test.png'))
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
