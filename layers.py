import keras.backend as K

def conv_maxout(x, n_piece=2):
    """
    input_shape: (n, h, w, ch)
    output_shape: (n, h, w, ch/n_piece)
    
    'ch' must be divisible by n_piece.
    """
    input_shape = x.get_shape().as_list()
    n, h, w, ch = input_shape
    x = K.reshape(x, (-1, h, w, ch//n_piece, piece))
    x = K.max(x, axis=3)
    return x
