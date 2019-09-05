import tensorflow as tf

class FastSCNN:

    def __init__(self, input_shape=(720, 1080, 3), bin_sizes=[2, 4, 6, 8]):
        self.input_shape = input_shape
        self.bin_sizes = bin_sizes
        self._net = self._build()

    def _build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='input')

        # learning to downsample module
        lds = self._conv_block(inputs, n_filters=32, kernel_size=3, stride=2)
        lds = self._conv_block_sc(lds, n_filters=48, kernel_size=3, stride=2)
        lds = self._conv_block_sc(lds, n_filters=64, kernel_size=3, stride=2)

        # global feature extractor
        gfe = self._bottleneck_block(lds, n_filters=64, kernel_size=3, stride=2, expansion=6, n=3)
        gfe = self._bottleneck_block(gfe, n_filters=96, kernel_size=3, stride=2, expansion=6, n=3)
        gfe = self._bottleneck_block(gfe, n_filters=128, kernel_size=3, stride=1, expansion=6, n=3)

        # pyramid pooling
        concat = [gfe]
        gfe_shape = tf.keras.backend.int_shape(gfe)
        w = gfe_shape[1]
        h = gfe_shape[2]

        for bin_size in self.bin_sizes:
            size = (w // bin_size, h // bin_size)
            ppl = tf.keras.layers.AveragePooling2D(pool_size=size, strides=size)(gfe)
            ppl = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(ppl)
            ppl = tf.keras.layers.Lambda(lambda x: tf.image.resize(ppl, (w, h)))(ppl)
            concat.append(ppl)
        
        ppl_concat = tf.keras.layers.concatenate(concat)

        # feature fusion module
        ff_hires = self._conv_block(lds, n_filters=1, kernel_size=1, stride=1, relu=False)
        hires_shape = tf.keras.backend.int_shape(ff_hires)

        ff_lowres = tf.keras.layers.UpSampling2D((4, 4))(ppl_concat)
        ff_lowres = tf.keras.layers.DepthwiseConv2D(128, strides=1, depth_multiplier=1, padding='same')(ff_lowres)
        ff_lowres = tf.keras.layers.BatchNormalization()(ff_lowres)
        ff_lowres = tf.keras.activations.relu(ff_lowres)
        ff_lowres = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=None)(ff_lowres)
        ff_lowres = tf.keras.layers.Reshape((hires_shape[1], hires_shape[2], 128))(ff_lowres)

        ff = tf.keras.layers.add([ff_hires, ff_lowres])
        ff = tf.keras.layers.BatchNormalization()(ff)
        ff = tf.keras.activations.relu(ff)

        # classifier
        classifier = self._conv_block_sc(ff, n_filters=128, kernel_size=3, stride=1)
        classifier = self._conv_block_sc(classifier, n_filters=128, kernel_size=3, stride=1)
        classifier = self._conv_block(classifier, n_filters=19, kernel_size=1, stride=1)
        classifier = tf.keras.layers.Dropout(0.3)(classifier)
        classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)

        # per pixel binary classification
        outputs = tf.keras.activations.sigmoid(classifier)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name='fast_scnn')

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
        x = self._conv_block(inputs, exp_channel, 1, 1)
        x = self._conv_block_dw(x, kernel_size, stride)
        x = self._conv_block(x, n_filters, 1, 1, relu=False)

        if skip:
            tf.keras.layers.add([x, inputs])

        return x

    def _bottleneck_block(self, inputs, n_filters, kernel_size, stride, expansion, n):
        x = self._bottleneck(inputs, n_filters, kernel_size, expansion, stride)

        for _ in range(n):
            x = self._bottleneck(x, n_filters, kernel_size, expansion, stride=1, skip=True)
        
        return x
    
    def compile(self):
        optimizer = tf.keras.optimizers.SGD()
        self._net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self._net.summary()
