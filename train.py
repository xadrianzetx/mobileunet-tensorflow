import os
import json
import config
import argparse
import tensorflow as tf
from utils.dataset import CULaneImageGenerator
from modelzoo.models import MobileUNet
from modelzoo.losses import focal_tversky_loss
from modelzoo.metrics import dice_coefficient


def jsonify_history(history):
    # converts history.history to serializable object
    for key in history.keys():
        history[key] = [str(x) for x in history[key]]
    
    return history


def train():
    args = {
        # passed by docker run -e
        'mode': os.environ['--mode'],
        'data-train': os.environ['--data-train'],
        'epochs': os.environ['--epochs'],
        'model-name': os.environ['--model-name']
    }
    
    # model instance
    model = MobileUNet(mode='binary', input_shape=config.IMG_SIZE, train_encoder=config.TRAIN_ENCODER).build()
    loss = focal_tversky_loss(alpha=config.LOSS_ALPHA, beta=config.LOSS_BETA, gamma=config.LOSS_GAMMA)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LR)

    metrics = [tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.Precision(), dice_coefficient()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.LOGDIR)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=config.SAVE_PATH)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.0001)

    if args['mode'] == 'debug':
        # sets both generators to output batch from debug list of images
        # rest of the config is the same as training mode
        print('#' * 20, args['mode'])

        train_g = CULaneImageGenerator(
            path=args['data-train'],
            lookup_name=config.DEBUG_TRAIN_LOOKUP,
            batch_size=config.DEBUG_BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=True,
            augmentations=config.AUGMENTATIONS
        )

        valid_g = CULaneImageGenerator(
            path=args['data-train'],
            lookup_name=config.DEBUG_TRAIN_LOOKUP,
            batch_size=config.DEBUG_BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=False
        )

        train_generator = tf.data.Dataset.from_generator(
            generator=train_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(config.DEBUG_GEN_IMG_OUT_SHAPE, config.DEBUG_GEN_MASK_OUT_SHAPE)
        )

        valid_generator = tf.data.Dataset.from_generator(
            generator=valid_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(config.DEBUG_GEN_IMG_OUT_SHAPE, config.DEBUG_GEN_MASK_OUT_SHAPE)
        )

    else:
        # generators set to training mode
        print('#' * 20, args['mode'])

        train_g = CULaneImageGenerator(
            path=args['data-train'],
            lookup_name=config.TRAIN_LOOKUP,
            batch_size=config.BATCH_SIZE,
            size=config.IMG_SIZE,
            augment=True,
            augmentations=config.AUGMENTATIONS
        )

        valid_g = CULaneImageGenerator(
            path=args['data-train'],
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
            epochs=int(args['epochs']),
            callbacks=[tensorboard, model_checkpoint, reduce_lr],
            validation_data=valid_generator,
            shuffle=False
        )
    
    # save model and training log
    model.save('{}.h5'.format(os.path.join(config.SAVE_PATH, args['model-name'])))
    logpath = os.path.join(config.LOGDIR, args['model-name'])

    with open('{}_logs.json'.format(logpath), 'w') as file:
        log = jsonify_history(history.history)
        json.dump(log, file, indent=4)


if __name__ == "__main__":
    train()
