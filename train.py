import json
import config
import argparse
import tensorflow as tf
from utils.dataset import CULaneImageGenerator
from modelzoo.models import MobileUNet
from modelzoo.losses import focal_tversky_loss
from modelzoo.metrics import dice_coefficient


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data-train', type=str, default=config.TRAIN_PATH)
    parser.add_argument('--data-valid', type=str, default=config.VALID_PATH)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--model-name', type=str, default='model')

    return parser.parse_args()


def train():
    args = arguments()
    
    # model instance
    model = MobileUNet(mode='binary', input_shape=config.IMG_SIZE, train_encoder=config.TRAIN_ENCODER).build()
    loss = focal_tversky_loss(alpha=config.LOSS_ALPHA, beta=config.LOSS_BETA, gamma=config.LOSS_GAMMA)
    optimizer = tf.keras.optimizers.Adam()

    metrics = [tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.Precision(), dice_coefficient()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.LOGDIR)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=config.SAVE_PATH)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.0001)

    if args.mode = 'debug':
        # sets both generators to output batch from debug list of images
        # rest of the config is the same as training mode
        train_g = CULaneImageGenerator(
            path=args.data-train,
            lookup_name=config.DEBUG_TRAIN_LOOKUP,
            batch_size=config.DEBUG_BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=True,
            augmentations=config.AUGMENTATIONS
        )

        valid_g = CULaneImageGenerator(
            path=args.data-train,
            lookup_name=config.DEBUG_TRAIN_LOOKUP,
            batch_size=config.DEBUG_BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=False
        )

        train_generator = tf.data.Dataset.from_generator(
            generator=train_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(config.GEN_IMG_OUT_SHAPE, config.GEN_MASK_OUT_SHAPE)
        )

        valid_generator = tf.data.Dataset.from_generator(
            generator=valid_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(config.GEN_IMG_OUT_SHAPE, config.GEN_MASK_OUT_SHAPE)
        )

    else:
        # generators set to training mode
        train_g = CULaneImageGenerator(
            path=args.data-train,
            lookup_name=config.TRAIN_LOOKUP,
            batch_size=config.BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=True,
            augmentations=config.AUGMENTATIONS
        )

        valid_g = CULaneImageGenerator(
            path=args.data-train,
            lookup_name=config.VALID_LOOKUP,
            batch_size=config.BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=False
        )

        train_generator = tf.data.Dataset.from_generator(
            generator=train_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(config.GEN_IMG_OUT_SHAPE, config.GEN_MASK_OUT_SHAPE)
        )

        valid_generator = tf.data.Dataset.from_generator(
            generator=valid_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(config.GEN_IMG_OUT_SHAPE, config.GEN_MASK_OUT_SHAPE)
        )
    
    # train model
    history = model.fit_generator(
            train_generator,
            epochs=args.epochs,
            callbacks=[tensorboard, model_checkpoint, reduce_lr],
            validation_data=valid_generator,
            shuffle=False
        )
    
    # save model and training log
    model.save('{}.h5'.format(args.model-name))

    with open('{}_logs.json'.format(args.model-name), 'w') as file:
        log = {args.model-name: history}
        json.dump(log, file, indent=4)


if __name__ == "__main__":
    train()
