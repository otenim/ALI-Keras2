import keras.backend as K
from keras.engine.topology import Layer

class ConvMaxout(Layer):
    def __init__(self, n_piece, **kwargs):
        self.n_piece = n_piece
        super(ConvMaxout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConvMaxout, self).build(input_shape)

    def call(self, x):
        input_shape = x.get_shape().as_list()
        n, h, w, ch = input_shape
        x = K.reshape(x, (n, h, w, ch//self.n_piece, self.n_piece))
        x = K.max(x, axis=3)
        return x

    def compute_output_shape(self, input_shape):
        n, h, w, ch = input_shape
        return (n, h, w, ch//self.n_piece)
