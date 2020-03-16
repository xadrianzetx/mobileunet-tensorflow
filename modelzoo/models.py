import tensorflow as tf


class FastSCNN:

    def __init__(self, input_shape, **kwargs):
        self.input_shape = input_shape
        self.bin_sizes = kwargs.get('bin_sizes', (2, 4, 6, 8))
        self._mode = kwargs.get('mode', 'binary')
        self._n_classes = kwargs.get('n_classes', 1)

    def __call__(self, **kwargs):
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
        gfe = self._bottleneck_block(
            lds,
            n_filters=64,
            kernel_size=3,
            strides=2,
            expansion=6,
            n=3
        )

        gfe = self._bottleneck_block(
            gfe,
            n_filters=96,
            kernel_size=3,
            strides=2,
            expansion=6,
            n=3
        )

        gfe = self._bottleneck_block(
            gfe,
            n_filters=128,
            kernel_size=3,
            strides=1,
            expansion=6,
            n=3
        )

        # pyramid pooling
        ppl = self._pyramid_pooling_block(gfe)

        # feature fusion module w/ low resolution reshape
        ff_hires = self._conv_block(
            lds,
            n_filters=1,
            kernel_size=1,
            strides=1,
            relu=False
        )

        hires_shape = tf.keras.backend.int_shape(ff_hires)

        ff_lowres = tf.keras.layers.UpSampling2D((4, 4))(ppl)
        ff_lowres = self._conv_block_dc(ff_lowres, kernel_size=3, strides=1)

        ff_lowres = tf.keras.layers.Conv2D(
            128,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None
        )(ff_lowres)

        ff_lowres = tf.keras.layers.Lambda(
            lambda x: tf.image.resize(x, (hires_shape[1], hires_shape[2]))
        )(ff_lowres)

        ff = tf.keras.layers.add([ff_hires, ff_lowres])
        ff = tf.keras.layers.BatchNormalization()(ff)
        ff = tf.keras.activations.relu(ff)

        # classifier
        classifier = self._conv_block_sc(
            ff,
            n_filters=128,
            kernel_size=3,
            strides=1
        )

        classifier = self._conv_block_sc(
            classifier,
            n_filters=128,
            kernel_size=3,
            strides=1
        )

        classifier = self._conv_block(
            classifier,
            n_filters=self._n_classes,
            kernel_size=1,
            strides=1,
            relu=False
        )

        classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)

        if self._mode == 'binary':
            # pixel-wise binary classification
            outputs = tf.keras.activations.sigmoid(classifier)

        else:
            # pixel-wise multiclass classification
            outputs = tf.keras.activations.softmax(classifier)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name='fast_scnn')

    @staticmethod
    def _conv_block(inputs, n_filters, kernel_size, strides, relu=True):
        x = tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            strides,
            padding='same'
        )(inputs)

        x = tf.keras.layers.BatchNormalization()(x)

        if relu:
            # point-wise conv does not use non-linearity
            x = tf.keras.activations.relu(x)

        return x

    @staticmethod
    def _conv_block_sc(inputs, n_filters, kernel_size, strides):
        x = tf.keras.layers.SeparableConv2D(
            n_filters,
            kernel_size,
            strides,
            padding='same'
        )(inputs)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x

    @staticmethod
    def _conv_block_dc(inputs, kernel_size, strides):
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides,
            depth_multiplier=1,
            padding='same'
        )(inputs)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x

    def _bottleneck(self, inputs, n_filters, kernel_size, strides, expansion, skip=False):
        # channles expansion factor
        exp_channel = tf.keras.backend.int_shape(inputs)[-1] * expansion

        # bottleneck residual block transfers the input from c to c'
        # channels with expansion factor t
        x = self._conv_block(
            inputs,
            n_filters=exp_channel,
            kernel_size=1,
            strides=1
        )

        x = self._conv_block_dc(
            x,
            kernel_size=kernel_size,
            strides=strides
        )

        x = self._conv_block(
            x,
            n_filters=n_filters,
            kernel_size=1,
            strides=1,
            relu=False
        )

        if skip:
            tf.keras.layers.add([x, inputs])

        return x

    def _bottleneck_block(self, inputs, n_filters, kernel_size, strides, expansion, n):
        x = self._bottleneck(
            inputs,
            n_filters,
            kernel_size,
            strides=strides,
            expansion=expansion
        )

        for _ in range(n - 1):
            # add remaining n - 1 bottleneck layers with skip connections
            x = self._bottleneck(
                x,
                n_filters,
                kernel_size,
                strides=1,
                expansion=expansion,
                skip=True
            )

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
            ppl = tf.keras.layers.AveragePooling2D(
                pool_size=ppm_size,
                strides=ppm_size
            )(inputs)

            ppl = tf.keras.layers.Conv2D(
                128,
                kernel_size=3,
                strides=2,
                padding='same'
            )(ppl)

            ppl = tf.keras.layers.Lambda(
                lambda x: tf.image.resize(x, (w, h))
            )(ppl)

            concat_tensors.append(ppl)

        return tf.keras.layers.concatenate(concat_tensors)


class MobileUNet:

    def __init__(self, input_shape, **kwargs):
        self._input_shape = input_shape
        self._trainable = kwargs.get('train_encoder', True)
        self._n_classes = kwargs.get('n_classes', 1)
        self._decay = kwargs.get('weight_decay', True)
        self._mode = kwargs.get('mode', 'binary')

    def __call__(self, **kwargs):
        """
        Builds U-Net with pre trainded
        MobileNetV2 backbone

        Encoder is using ImageNet weights and can be
        frozen or trained with decoder

        :return:    tf.keras.models.Model
                    Ready to compile MobileUnet
        """
        depthwise_decoder = kwargs.get('depthwise_decoder', True)

        base = tf.keras.applications.MobileNetV2(
            input_shape=self._input_shape,
            include_top=False,
            weights='imagenet'
        )

        if not self._trainable:
            base.trainable = False

        base_out = base.get_layer('block_16_project')

        # encoder skip connections
        # block 13
        skip_b13 = base.get_layer('block_13_expand_relu')
        s13_filters = tf.keras.backend.int_shape(skip_b13.output)[-1]

        # block 6
        skip_b6 = base.get_layer('block_6_expand_relu')
        s6_filters = tf.keras.backend.int_shape(skip_b6.output)[-1]

        # block 3
        skip_b3 = base.get_layer('block_3_expand_relu')
        s3_filters = tf.keras.backend.int_shape(skip_b3.output)[-1]

        # block 1
        skip_b1 = base.get_layer('block_1_expand_relu')
        s1_filters = tf.keras.backend.int_shape(skip_b1.output)[-1]

        if depthwise_decoder:
            # bridge first
            x = self._residual_block(
                base_out.output,
                n_filters=s13_filters,
                kernel_size=3, strides=1
            )

            x = self._residual_block(
                x,
                n_filters=s13_filters,
                kernel_size=3,
                strides=1
            )

            # and start going up
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = tf.keras.layers.Concatenate()([x, skip_b13.output])

            x = self._residual_block(
                x,
                n_filters=s6_filters,
                kernel_size=3,
                strides=1
            )

        else:
            x = self._upconv(
                base_out.output,
                n_filters=s6_filters,
                kernel_size=3,
                strides=2
            )

            x = tf.keras.layers.Concatenate()([x, skip_b13.output])

        # upsample and concat with block 6
        if depthwise_decoder:
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = tf.keras.layers.Concatenate()([x, skip_b6.output])

            x = self._residual_block(
                x,
                n_filters=s3_filters,
                kernel_size=3,
                strides=1
            )

        else:
            x = self._upconv(
                x,
                n_filters=s3_filters,
                kernel_size=3,
                strides=2
            )

            x = tf.keras.layers.Concatenate()([x, skip_b6.output])

        # upsample and concat with block 3
        if depthwise_decoder:
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = tf.keras.layers.Concatenate()([x, skip_b3.output])

            x = self._residual_block(
                x,
                n_filters=s1_filters,
                kernel_size=3,
                strides=1
            )

        else:
            x = self._upconv(
                x,
                n_filters=s1_filters,
                kernel_size=3,
                strides=2
            )

            x = tf.keras.layers.Concatenate()([x, skip_b3.output])

        # upsample and concat with block 1
        if depthwise_decoder:
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = tf.keras.layers.Concatenate()([x, skip_b1.output])

            x = self._residual_block(
                x,
                n_filters=64,
                kernel_size=3,
                strides=1
            )

        else:
            x = self._upconv(
                x,
                n_filters=64,
                kernel_size=3,
                strides=2
            )

            x = tf.keras.layers.Concatenate()([x, skip_b1.output])

        if self._mode == 'binary':
            x = tf.keras.layers.UpSampling2D((2, 2))(x)

            x = tf.keras.layers.SeparableConv2D(
                1,
                kernel_size=3,
                strides=1,
                padding='same'
            )(x)

            out = tf.keras.activations.sigmoid(x)

        else:
            x = tf.keras.layers.UpSampling2D((2, 2))(x)

            x = tf.keras.layers.SeparableConv2D(
                self._n_classes,
                kernel_size=3,
                strides=1,
                padding='same'
            )(x)

            out = tf.keras.activations.softmax(x)

        return tf.keras.models.Model(inputs=base.input, outputs=out)

    def _upconv(self, inputs, n_filters, kernel_size, strides):
        if self._decay:
            # L2 regularization on weights
            x = tf.keras.layers.Conv2DTranspose(
                n_filters,
                kernel_size,
                strides,
                padding='same',
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2())(inputs)

        else:
            x = tf.keras.layers.Conv2DTranspose(
                n_filters,
                kernel_size,
                strides,
                padding='same',
                use_bias=False)(inputs)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x

    def _conv_ds(self, inputs, n_filters, kernel_size, strides):
        if self._decay:
            # L2 regularization on weights
            x = tf.keras.layers.SeparableConv2D(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2())(inputs)

        else:
            x = tf.keras.layers.SeparableConv2D(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same')(inputs)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        return x

    def _residual_block(self, inputs, n_filters, kernel_size, strides):
        x = self._conv_ds(
            inputs,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides
        )

        x = self._conv_ds(
            x,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides
        )

        # depthwise conv skip connection
        if self._decay:
            skip = tf.keras.layers.SeparableConv2D(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2())(x)

        else:
            skip = tf.keras.layers.SeparableConv2D(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same')(x)

        skip = tf.keras.layers.BatchNormalization()(skip)
        skip = tf.keras.activations.relu(skip)

        out = tf.keras.layers.Add()([skip, x])

        return out


class MobileFPNet:

    def __init__(self, input_shape, **kwargs):
        self._input_shape = input_shape
        self._backbone = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        self._backbone.trainable = kwargs.get('train_encoder', True)

    def __call__(self, **kwargs):
        """
        Creates instance of FPN Net with MobileNetV2 backbone.

        based on:
        https://arxiv.org/pdf/1612.03144.pdf
        https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/models/fpn.py

        Net has been modified to fully compile
        to Edge TPU with 192x192x3 input size.

        :return:    tf.keras.models.Model
                    FPN net instance
        """
        segname = 'block_{}_expand_relu'
        blocks = [13, 6, 3, 1]
        skips = [self._backbone.get_layer(segname.format(i)) for i in blocks]
        backbone_out = self._backbone.get_layer('block_16_project')

        p5 = self._fpn_block(backbone_out.output, skips[0].output)
        p4 = self._fpn_block(p5, skips[1].output)
        p3 = self._fpn_block(p4, skips[2].output)
        p2 = self._fpn_block(p3, skips[3].output)

        s5 = self._conv_block(p5, 128)
        s4 = self._conv_block(p4, 128)
        s3 = self._conv_block(p3, 128)
        s2 = self._conv_block(p2, 128)

        s5 = tf.keras.layers.UpSampling2D(
            size=(8, 8),
            interpolation='nearest'
        )(s5)

        s4 = tf.keras.layers.UpSampling2D(
            size=(4, 4),
            interpolation='nearest'
        )(s4)

        s3 = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation='nearest'
        )(s3)

        concat = [s5, s4, s3, s2]
        x = tf.keras.layers.Concatenate()(concat)
        x = tf.keras.layers.Conv2D(
            64,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(
            1,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
        )(x)

        out = tf.keras.layers.Activation('sigmoid')(x)
        model = tf.keras.models.Model(
            inputs=self._backbone.input,
            outputs=out
        )

        return model

    @staticmethod
    def _fpn_block(inputs, skip):
        inputs = tf.keras.layers.Conv2D(
            256,
            kernel_size=1,
            padding='same',
            kernel_initializer='he_uniform'
        )

        skip = tf.keras.layers.SeparableConv2D(
            256,
            kernel_size=1,
            kernel_initializer='he_uniform'
        )(skip)

        up = tf.keras.layers.UpSampling2D((2, 2))(inputs)
        out = tf.keras.layers.Add()([up, skip])

        return out

    @staticmethod
    def _conv_block(inputs, filters):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
        )(inputs)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        return x


class DeepLabV3Plus:

    def __init__(self, input_shape, **kwargs):
        self._input_shape = input_shape

    def __call__(self, **kwargs):
        pass
