# paths
TRAIN_PATH = './data'
VALID_PATH = './data'
SAVE_PATH = 'data/models'
LOAD_PATH = 'data/models'
LOGDIR = 'data/logs'

# lookups
TRAIN_LOOKUP = 'train.txt'
VALID_LOOKUP = 'valid.txt'
DEBUG_TRAIN_LOOKUP = 'debug_list.txt'

# batch params
BATCH_SIZE = 5
DEBUG_BATCH_SIZE = 2
IMG_SIZE = (512, 512, 3)
MASK_SIZE = (512, 512, 1)
GEN_IMG_OUT_SHAPE = (BATCH_SIZE, 512, 512, 3)
GEN_MASK_OUT_SHAPE = (BATCH_SIZE, 512, 512, 1)
DEBUG_GEN_IMG_OUT_SHAPE = (DEBUG_BATCH_SIZE, 512, 512, 3)
DEBUG_GEN_MASK_OUT_SHAPE = (DEBUG_BATCH_SIZE, 512, 512, 1)

# loss params
LOSS_ALPHA = 0.7
LOSS_BETA = 0.3
LOSS_GAMMA = 0.75
LOSS_SMOOTH = 1e-6

# train params
LR = 0.001
EPOCHS = 50
DEBUG_EPOCHS = 2
TRAIN_ENCODER = True
AUGMENTATIONS = ('flip', 'rotate', 'brightness')
