import numpy as np
from PIL import Image

def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    return x

def decode_output(x):
    x = x.astype(np.float32)
    x *= 255.
    return x

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
