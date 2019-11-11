import os
import json
import yaml
import tensorflow as tf
from utils.dataset import NightRideImageGenerator
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
        'model-name': os.environ['--model-name'],
        'checkpoint-name': os.environ['--checkpoint']
    }

    with open('config.yaml', 'r') as file:
        config = yaml.load(file)

    if args['mode'] == 'train' or args['mode'] == 'debug':
        # new model instance
        input_shape = tuple(config['shapes']['image'])
        train_enc = config['training']['train_encoder']

        unet = MobileUNet(
            mode='binary',
            input_shape=input_shape,
            train_encoder=train_enc
        )

        model = unet.build()

    else:
        # resume training from checkpoint
        path = os.path.join(config['paths']['checkpoint'], args['checkpoint-name'])

        custom_obj = {
            'focal_tversky': focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75),
            'dice': dice_coefficient()
        }

        # load from .h5 file
        model = tf.keras.models.load_model(
            path,
            custom_objects=custom_obj,
            compile=False
        )

    # load loss params from config
    loss_params = config['loss']

    loss = focal_tversky_loss(
        alpha=loss_params['alpha'],
        beta=loss_params['beta'],
        gamma=loss_params['gamma']
    )

    # init optimizer
    optimizer = tf.keras.optimizers.Adam(lr=config['training']['lr'])

    metrics = [
        tf.keras.metrics.MeanIoU(num_classes=2),
        tf.keras.metrics.Precision(),
        dice_coefficient()
    ]

    # compile new model or recompile from checkpoint
    # when anything has been changed
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config['paths']['logs'], update_freq='batch')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=config['paths']['checkpoint'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.0001)

    if args['mode'] == 'debug':
        # sets both generators to output batch from debug list of images
        # rest of the config is the same as training mode
        print('#' * 20, args['mode'])

        train_g = NightRideImageGenerator(
            path=args['data-train'],
            lookup_name=config['lookups']['debug'],
            batch_size=config['batch']['debug_size'],
            size=tuple(config['shapes']['image']),
            augment=True,
            augmentations=config['training']['augmentations'],
            augment_proba=config['training']['augment_proba']
        )

        valid_g = NightRideImageGenerator(
            path=args['data-train'],
            lookup_name=config['lookups']['debug'],
            batch_size=config['batch']['debug_size'],
            size=tuple(config['shapes']['image']),
            augment=False
        )

        # generator output shapes
        image_out = tuple(config['shapes']['debug_image_generator'])
        mask_out = tuple(config['shapes']['debug_mask_generator'])

        train_generator = tf.data.Dataset.from_generator(
            generator=train_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(image_out, mask_out)
        )

        valid_generator = tf.data.Dataset.from_generator(
            generator=valid_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(image_out, mask_out)
        )

    else:
        # generators set to training mode
        print('#' * 20, args['mode'])

        train_g = NightRideImageGenerator(
            path=args['data-train'],
            lookup_name=config['lookups']['train'],
            batch_size=config['batch']['size'],
            size=tuple(config['shapes']['image']),
            augment=True,
            augmentations=config['training']['augmentations'],
            augment_proba=config['training']['augment_proba']
        )

        valid_g = NightRideImageGenerator(
            path=args['data-train'],
            lookup_name=config['lookups']['valid'],
            batch_size=config['batch']['size'],
            size=tuple(config['shapes']['image']),
            augment=False
        )

        # generator output shapes
        image_out = tuple(config['shapes']['image_generator'])
        mask_out = tuple(config['shapes']['mask_generator'])

        train_generator = tf.data.Dataset.from_generator(
            generator=train_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(image_out, mask_out)
        )

        valid_generator = tf.data.Dataset.from_generator(
            generator=valid_g,
            output_types=(tf.float16, tf.float16),
            output_shapes=(image_out, mask_out)
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
    model.save('{}.h5'.format(os.path.join(config['paths']['checkpoint'], args['model-name'])))
    logpath = os.path.join(config['paths']['logs'], args['model-name'])

    with open('{}_logs.json'.format(logpath), 'w') as file:
        log = jsonify_history(history.history)
        json.dump(log, file, indent=4)


if __name__ == "__main__":
    train()
