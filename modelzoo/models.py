import tensorflow as tf

class FastSCNN:

    def __init__(self, mode, input_shape=(720, 1080, 3), bin_sizes=(2, 4, 6, 8), n_classes):
        self.input_shape = input_shape
        self.bin_sizes = bin_sizes
        self._mode = mode
        self._n_classes = n_classes

    @staticmethod
    def _conv_block(inputs, n_filters, kernel_size, strides, relu=True):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size, strides, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        if relu:
            # point-wise conv does not use non-linearity
            x = tf.keras.activations.relu(x)
        
        return x

    @staticmethod
    def _conv_block_sc(inputs, n_filters, kernel_size, strides):
        x = tf.keras.layers.SeparableConv2D(n_filters, kernel_size, strides, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x
    
    @staticmethod
    def _conv_block_dc(inputs, kernel_size, strides):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, depth_multiplier=1, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x

    def _bottleneck(self, inputs, n_filters, kernel_size, strides, expansion, skip=False):
        # channles expansion factor
        exp_channel = tf.keras.backend.int_shape(inputs)[-1] * expansion

        # bottleneck residual block transfers the input from c to c' 
        # channels with expansion factor t
        x = self._conv_block(inputs, n_filters=exp_channel, kernel_size=1, strides=1)
        x = self._conv_block_dc(x, kernel_size=kernel_size, strides=strides)
        x = self._conv_block(x, n_filters=n_filters, kernel_size=1, strides=1, relu=False)

        if skip:
            tf.keras.layers.add([x, inputs])

        return x

    def _bottleneck_block(self, inputs, n_filters, kernel_size, strides, expansion, n):
        x = self._bottleneck(inputs, n_filters, kernel_size, strides=strides, expansion=expansion)

        for _ in range(n - 1):
            # add remaining n - 1 bottleneck layers with skip connections
            x = self._bottleneck(x, n_filters, kernel_size, strides=1, expansion=expansion, skip=True)
        
        return x
    
    def _pyramid_pooling_block(self, inputs):
        concat_tensors = [inputs]

        # get output shape of last layers so whole thing can scale
        inputs_shape = tf.keras.backend.int_shape(inputs)
        w = inputs_shape[1]
        h = inputs_shape[2]

        for bin_size in self.bin_sizes:
            # pyramid parsing module is applied to harvest 
            # different sub-region representations
            ppm_size = (w // bin_size, h // bin_size)
            ppl = tf.keras.layers.AveragePooling2D(pool_size=ppm_size, strides=ppm_size)(inputs)
            ppl = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(ppl)
            ppl = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(ppl)
            concat_tensors.append(ppl)
        
        return tf.keras.layers.concatenate(concat_tensors)
    
    def build(self):
        """
        Compiles Fast SCNN

        :return:    tf.keras.Model
                    Ready to compile Fast-SCNN
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='input')

        # learning to downsample module
        lds = self._conv_block(inputs, n_filters=32, kernel_size=3, strides=2)
        lds = self._conv_block_sc(lds, n_filters=48, kernel_size=3, strides=2)
        lds = self._conv_block_sc(lds, n_filters=64, kernel_size=3, strides=2)

        # global feature extractor
        gfe = self._bottleneck_block(lds, n_filters=64, kernel_size=3, strides=2, expansion=6, n=3)
        gfe = self._bottleneck_block(gfe, n_filters=96, kernel_size=3, strides=2, expansion=6, n=3)
        gfe = self._bottleneck_block(gfe, n_filters=128, kernel_size=3, strides=1, expansion=6, n=3)

        # pyramid pooling
        ppl = self._pyramid_pooling_block(gfe)

        # feature fusion module w/ low resolution reshape
        ff_hires = self._conv_block(lds, n_filters=1, kernel_size=1, strides=1, relu=False)
        hires_shape = tf.keras.backend.int_shape(ff_hires)

        ff_lowres = tf.keras.layers.UpSampling2D((4, 4))(ppl)
        ff_lowres = self._conv_block_dc(ff_lowres, kernel_size=3, strides=1)
        ff_lowres = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=None)(ff_lowres)
        ff_lowres = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (hires_shape[1], hires_shape[2])))(ff_lowres)

        ff = tf.keras.layers.add([ff_hires, ff_lowres])
        ff = tf.keras.layers.BatchNormalization()(ff)
        ff = tf.keras.activations.relu(ff)

        # classifier
        classifier = self._conv_block_sc(ff, n_filters=128, kernel_size=3, strides=1)
        classifier = self._conv_block_sc(classifier, n_filters=128, kernel_size=3, strides=1)
        classifier = self._conv_block(classifier, n_filters=self._n_classes, kernel_size=1, strides=1, relu=False)
        classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)

        if self._mode == 'binary':
            # pixel-wise binary classification
            outputs = tf.keras.activations.sigmoid(classifier)
        
        else:
            # pixel-wise multiclass classification
            outputs = tf.keras.activations.softmax(classifier)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name='fast_scnn')


class MobileUNet:

    def __init__(self, mode, input_shape, n_classes=1, train_encoder=False):
        self._input_shape = input_shape
        self._trainable = train_encoder
        self._n_classes = n_classes
        self._mode = mode
    
    @staticmethod
    def _upconv(inputs, n_filters, kernel_size, strides):
        x = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same', use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x
    
    def build(self):
        """
        Builds U-Net encoder-decoder model with
        pre trainded MobileNetV2 encoder

        Encoder is using ImageNet weights and can be
        frozen or trained with decoder

        :return:    tf.keras.models.Model
                    Ready to compile MobileUnet
        """
        base = tf.keras.applications.MobileNetV2(input_shape=self._input_shape, include_top=False, weights='imagenet')

        if not self._trainable:
            base.trainable = False
        
        # upsample and concat with block 13
        base_out = base.get_layer('block_16_project')
        skip_b13 = base.get_layer('block_13_expand_relu')
        n_filters = tf.keras.backend.int_shape(skip_b13.output)[-1]
        padding_b13 = tf.keras.layers.ZeroPadding2D()(skip_b13.output)
        x = self._upconv(base_out.output, n_filters=n_filters, kernel_size=3, strides=2)
        x = tf.keras.layers.Concatenate()([x, skip_b13.output])

        # upsample and concat with block 6
        skip_b6 = base.get_layer('block_6_expand_relu')
        n_filters = tf.keras.backend.int_shape(skip_b6.output)[-1]
        x = self._upconv(x, n_filters=n_filters, kernel_size=3, strides=2)
        x = tf.keras.layers.Concatenate()([x, skip_b6.output])

        # upsample and concat with block 3
        skip_b3 = base.get_layer('block_3_expand_relu')
        n_filters = tf.keras.backend.int_shape(skip_b3.output)[-1]
        x = self._upconv(x, n_filters=n_filters, kernel_size=3, strides=2)
        x = tf.keras.layers.Concatenate()([x, skip_b3.output])

        # upsample and concat with block 1
        skip_b1 = base.get_layer('block_1_expand_relu')
        n_filters = tf.keras.backend.int_shape(skip_b1.output)[-1]
        x = self._upconv(x, n_filters=n_filters, kernel_size=3, strides=2)
        x = tf.keras.layers.Concatenate()([x, skip_b1.output])

        if self._mode == 'binary':
            x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(x)
            out = tf.keras.activations.sigmoid(x)
        
        else:
            x = tf.keras.layers.Conv2DTranspose(filters=self._n_classes, kernel_size=3, strides=2, padding='same')(x)
            out = tf.keras.activations.softmax(x)
        
        return tf.keras.models.Model(inputs=base.input, outputs=out)
