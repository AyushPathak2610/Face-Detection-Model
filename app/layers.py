# Custom L1 Distance Layer module

import tensorflow as tf
from tensorflow.keras.layers import Layer

# custom L1 distance from jupyter
# Siamese L1 Disatnce class: since our siamese_mode.h5 saved model needs the l1dist as custom layer
class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # similarity calculator
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)