import os
import logging

CLI_COMMAND_TRAIN = 'train'
CLI_COMMAND_EVALUATE = 'evaluate'

TOKEN_START = '<start>'
TOKEN_END = '<end>'

EPOCHS = 15
EMBEDDING_DIM = 256
LSTM_UNITS = 1024
BATCH_SIZE = 32

# Initialzed as the ArgumentParser parses argument
DATASET_SIZE = None  # For taking the complete dataset (= None)

# Logs each train step with great detail.
DEBUG_MODE = True

DIR_CACHE = os.path.join(os.path.abspath('.'), 'cache')
DIR_CHECKPOINTS = os.path.join(os.path.abspath('.'), 'checkpoints')
DIR_LOGS = os.path.join(os.path.abspath('.'), 'logs')

PATH_CACHE_DIR = os.path.join(DIR_CACHE,
                              f'cache_{DATASET_SIZE}_{BATCH_SIZE}')

# Initialzed as the ArgumentParser parses argument
PATH_LOG_FILE = None
PATH_CHECKPOINT_DIR = None
PATH_CHECKPOINT = None

# Initialzed as the ArgumentParser parses argument
# True => Use 1st columnt in dataset file as target and 2nd as input while training else vice versa.
REVERSE_DATA = None


def initialize_logger():
    # Create directory to save all logs data
    if not (os.path.exists(DIR_LOGS) and os.path.isdir(DIR_LOGS)):
        os.mkdir(DIR_LOGS)

    # Setup logging config
    logging.basicConfig(
        level=logging.DEBUG,
        filename=PATH_LOG_FILE,
        format='[%(asctime)s -- %(filename)s] %(levelname)s: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    LOGGER = logging.getLogger(__name__)

    # Create directory to save all checkpoints
    # if not (os.path.exists(DIR_CHECKPOINTS) and os.path.isdir(DIR_CHECKPOINTS)):
    #    os.mkdir(DIR_CHECKPOINTS)

    # Create directory to save all cache data
    if not (os.path.exists(DIR_CACHE) and os.path.isdir(DIR_CACHE)):
        os.mkdir(DIR_CACHE)
        LOGGER.info(f'Created directory => {DIR_CACHE}')

    # Create directory to save all caches
    if not (os.path.exists(PATH_CACHE_DIR) and os.path.isdir(PATH_CACHE_DIR)):
        os.mkdir(PATH_CACHE_DIR)
        LOGGER.info(f'Created directory => {PATH_CACHE_DIR}')
