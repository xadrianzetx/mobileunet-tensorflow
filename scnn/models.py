import tensorflow as tf


class FastSCNN:

    def __init__(self, input_size=(295, 820)):
        self.input_size = input_size

    def build(self):
        pass

    @staticmethod
    def _conv_block(inputs, mode, n_filters, kernel_size, stride, relu=True):
        if mode == 'sc':
            x = tf.keras.layers.SeparableConv2D(n_filters, kernel_size, padding='same', strides=stride)(inputs)

        else:
            x = tf.keras.layers.Conv2D(n_filters, kernel_size, padding='same', strides=stride)(inputs)

        x = tf.keras.layers.BatchNormalization()(x)

        if relu:
            x = tf.keras.activations.relu(x)

        return x

    def _res_bottleneck(self, inputs, n_filters, kernel_size, t, strides, skip=False):
        t_channel = tf.keras.backend.int_shape(inputs)[-1] * t

        x = self._conv_block(inputs, 'conv', t_channel, 1, 1)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        x = self._conv_block(x, 'conv', n_filters, 1, 1, relu=False)

        if skip:
            x = tf.keras.layers.add([x, inputs])

        return x

    def bottleneck_block(self, inputs, n_filters, kernel_size, t, strides, n_layers):
        x = self._res_bottleneck(inputs, n_filters, kernel_size, t, strides)

        for i in range(1, n_layers):
            x = self._res_bottleneck(x, n_filters, kernel_size, t, 1, skip=True)

        return x
