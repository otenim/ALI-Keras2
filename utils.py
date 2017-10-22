import numpy as np

def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    return x
