import keras
from keras.datasets import cifar10
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import os
import argparse
import utils
import pickle
import models
import shutil

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('zgenerater_weights')
parser.add_argument('--result_root', default=os.path.join(curdir, 'result'))
parser.add_argument('--pred_batch_size', type=int, default=100)

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
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train.astype(np.float32)
    x_train = utils.preprocess_input(x_train)

    # convert all images into encoded vector points
    print("converting all images into latent z variables..")
    x = zgenerater.predict(
        x_train,
        batch_size=args.pred_batch_size,
        verbose=0)
    x = x.reshape(len(x_train), 64)

    # ===============================
    # k-means clustering
    # ===============================
    # clustering
    print("clustering has started...")
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
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(x)

    # ===============================
    # Save results
    # =============================== 
    if os.path.exists(args.result_root) == False:
        os.makedirs(args.result_root)

    # save k-means result as pickle file
    with open(os.path.join(args.result_root, 'kmeans.pkl'), 'wb') as fp:
        pickle.dump(kmeans, fp)

    # classify images 
    for i in range(num_classes):
        dirname = os.path.join(args.result_root, 'class_%d' % i)
        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        ind = (kmeans.labels_ == i)
        imgs = x_train[ind]
        labels = y_train[ind]
        for j in range(len(imgs)):
            img = utils.decode_output(imgs[j])
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            label = classes[int(labels[j])]
            dst = os.path.join(dirname, '%s_%d.png' % (label, j))
            img.save(dst)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
