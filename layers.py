import keras.backend as K
from keras.engine.topology import Layer

class ConvMaxout(Layer):
    def __init__(self, n_piece, **kwargs):
        self.n_piece = n_piece
        super(ConvMaxout, self).__init__(**kwargs)

    def call(self, x):
        n = K.shape(x)[0]
        h = K.shape(x)[1]
        w = K.shape(x)[2]
        ch = K.shape(x)[3]
        x = K.reshape(x, (n, h, w, ch//self.n_piece, self.n_piece))
        x = K.max(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        n, h, w, ch = input_shape
        return (n, h, w, ch//self.n_piece)
