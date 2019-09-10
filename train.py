import tensorflow as tf
from scnn.dataset import CULaneImageGenerator
from scnn.models import FastSCNN


# make config out of those
LIST_PATH = ''
TRAIN_LIST = ''
VALID_LIST = ''

BATCH_SIZE = 10
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 30
IMAGE_SIZE = (720, 1080, 3)
MASK_SIZE = (720, 1080)

train_g = CULaneImageGenerator(path=LIST_PATH, batch_size=BATCH_SIZE, lookup_name=TRAIN_LIST, augment=True)
valid_g = CULaneImageGenerator(path=LIST_PATH, batch_size=BATCH_SIZE, lookup_name=VALID_LIST, augment=False)

train_gen = tf.data.Dataset.from_generator(train_g, output_types=(tf.int32, tf.int32), output_shapes=(IMAGE_SIZE, MASK_SIZE))
valid_gen = tf.data.Dataset.from_generator(valid_g, output_types=(tf.int32, tf.int32), output_shapes=(IMAGE_SIZE, MASK_SIZE))

# optimizer same as in paper
optimizer = tf.keras.optimizers.SGD()
metrics = ['accuracy']

# model instance
m = FastSCNN(mode='binary', input_shape=IMAGE_SIZE)
model = m.compile(optimizer=optimizer, metrics=metrics)

# callbacks - TODO
callbacks = None

# oh and implement debug mode for Google AI platform

history = model.fit_generator(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=1,
    verbose=1,
    callbacks=None,
    validation_data=valid_gen,
    validation_steps=VALIDATION_STEPS,
    shuffle=Fale
)
