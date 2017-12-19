import numpy as np
import os
from PIL import Image

def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    return x

def decode_output(x):
    x = x.astype(np.float32)
    x *= 255.
    return x

def make_paths_from_directory(root):
    input_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as fp:
                magic = fp.read(8)

            if magic.startswith(b'GIF89a') or magic.startswith(b'GIF87a'):
                filetype = 'gif'
            elif magic == b'\xff\xd8\xff\xe0\x00\x10JF':
                filetype = 'jpeg'
            elif magic.startswith(b'\x89PNG'):
                filetype = 'png'
            else:
                print(' unsupported file type', repr(magic), filepath)
                continue
            input_paths.append(filepath)
    return input_paths

def make_arrays_from_paths(paths, preprocess=None, target_size=None):
    rv = []
    for path in paths:
        img = image.load_img(path, target_size=target_size)
        ar = image.img_to_array(img)
        if preprocess:
            ar = preprocess(ar)
        rv.append(ar)
    return np.array(rv)

def generate_img(xgenerater):
    num_generate_imgs = 144
    z_dim = xgenerater.input_shape[-1]
    z = np.random.normal(size=(num_generate_imgs, 1, 1, z_dim))
    x_gen = xgenerater.predict_on_batch(z)
    x_gen = decode_output(x_gen)
    x_gen = np.clip(x_gen, 0., 255.).astype(np.uint8)

    # Concatenate generated images
    grid_size = int(np.sqrt(num_generate_imgs))
    rows = []
    for i in range(0, num_generate_imgs, grid_size):
        row = np.concatenate(x_gen[i:i+grid_size], axis=1)
        rows.append(row)
    concatenated = np.concatenate(rows, axis=0)
    return Image.fromarray(concatenated)

def reconstruct_img(x, xgen, zgen):
    """
    x assumes x_train
    xgen: trained xgenerater
    zgen: trained zgenerater
    """
    # original images
    ind = np.random.permutation(len(x))
    num_generate_imgs = 144
    x = (x[ind])[:num_generate_imgs//2]
    x = x.astype(np.uint8)

    # generated images
    x_copy = np.copy(x)
    x_copy = x_copy.astype(np.float32)
    x_copy = preprocess_input(x_copy)
    z_gen = zgen.predict_on_batch(x_copy)
    x_gen = xgen.predict_on_batch(z_gen)
    x_gen = decode_output(x_gen)
    x_gen = np.clip(x_gen, 0., 255.).astype(np.uint8)

    grid_size = int(np.sqrt(num_generate_imgs))
    cols = []
    for i in range(0, num_generate_imgs//2, grid_size):
        col_orig = np.concatenate(x[i:i+grid_size], axis=0)
        col_gen = np.concatenate(x_gen[i:i+grid_size], axis=0)
        col = np.concatenate([col_orig, col_gen], axis=1)
        cols.append(col)
    concatenated = np.concatenate(cols, axis=1)
    return Image.fromarray(concatenated)
