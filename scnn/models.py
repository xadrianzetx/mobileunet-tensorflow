import tensorflow as tf

class FastSCNN:

    def __init__(self, input_shape=(295, 820), bin_sizes=[2, 4, 6, 8]):
        self.input_shape = input_shape
        self.bin_sizes = bin_sizes

    def _build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='input')

        # learning to downsample module
        lds = self._conv_block(inputs, n_filters=32, kernel_size=3, stride=3)
        lds = self._conv_block_sc(lds, n_filters=48, kernel_size=3, stride=2)
        lds = self._conv_block_sc(lds, n_filters=64, kernel_size=3, stride=2)

        # global feature extractor
        gfe = self.bottleneck_block(lds, n_filters=64, kernel_size=3, stride=2, expansion=6, n=3)
        gfe = self.bottleneck_block(gfe, n_filters=96, kernel_size=3, stride=2, expansion=6, n=3)
        gfe = self.bottleneck_block(gfe, n_filters=128, kernel_size=3, stride=1, expansion=6, n=3)

        # pyramid pooling
        skip = [gfe]

        for bin_size in self.bin_sizes:
            # TODO get output shape from last gfe automatically
            w = 64
            h = 32
            size = (w // bin_size, h // bin_size)
            x = tf.keras.layers.AveragePooling2D(pool_size=size, strides=size)


    @staticmethod
    def _conv_block(inputs, n_filters, kernel_size, stride, relu=True):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size, stride)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if relu:
            x = tf.keras.activations.relu(x)
        
        return x

    @staticmethod
    def _conv_block_sc(inputs, n_filters, kernel_size, stride):
        x = tf.keras.layers.SeparableConv2D(n_filters, kernel_size, stride, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x
    
    @staticmethod
    def _conv_block_dw(inputs, kernel_size, stride):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, stride, depth_multiplier=1, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x

    def _bottleneck(self, inputs, n_filters, kernel_size, expansion, stride, skip=False):
        exp_channel = tf.keras.backend.int_shape(inputs)[-1] * expansion
        x = self._conv_block(inputs, exp_channel, kernel_size, 1)
        x = self._conv_block_dw(x, kernel_size, stride)
        x = self._conv_block(x, n_filters, kernel_size, 1, relu=False)

        if skip:
            tf.keras.layers.add([x, inputs])

        return x

    def bottleneck_block(self, inputs, n_filters, kernel_size, stride, expansion, n):
        x = self._bottleneck(inputs, n_filters, kernel_size, expansion, stride)

        for _ in range(n):
            x = self._bottleneck(x, n_filters, kernel_size, expansion, stride=1, skip=True)
        
        return x
