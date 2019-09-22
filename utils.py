import unicodedata
import re
import io
import pickle
import logging
import os

import tensorflow as tf

K = tf.keras
KP = tf.keras.preprocessing


class Constants:
    TOKEN_START = '<start>'
    TOKEN_END = '<end>'

    EPOCHS = 15
    EMBEDDING_DIM = 256
    LSTM_UNITS = 1024
    BATCH_SIZE = 32
    DATASET_SIZE = 30000  # For taking the complete dataset (= None)

    # Logs each train step with great detail.
    DEBUG_MODE = True

    DIR_CACHE = os.path.join(os.path.abspath('.'), 'cache')
    DIR_CHECKPOINTS = os.path.join(os.path.abspath('.'), 'checkpoints')
    DIR_LOGS = os.path.join(os.path.abspath('.'), 'logs')

    PATH_CACHE_DIR = os.path.join(DIR_CACHE,
                                  f'cache_{DATASET_SIZE}_{BATCH_SIZE}')

    PATH_LOG_FILE = os.path.join(DIR_LOGS,
                                 f'app_debug_{DATASET_SIZE}_{BATCH_SIZE}.log')

    PATH_CHECKPOINT_DIR = os.path.join(DIR_CHECKPOINTS,
                                       f'ckpt_{DATASET_SIZE}_{BATCH_SIZE}')

    PATH_CHECKPOINT = os.path.join(PATH_CHECKPOINT_DIR, 'ckpt')

    def __init__(self):
        # Create directory to save all logs data
        if not (os.path.exists(self.DIR_LOGS) and os.path.isdir(self.DIR_LOGS)):
            os.mkdir(self.DIR_LOGS)

        # Setup logging config
        logging.basicConfig(
            level=logging.DEBUG,
            filename=self.PATH_LOG_FILE,
            format='[%(asctime)s -- %(threadName)s] %(levelname)s: %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )

        # Create directory to save all checkpoints
        if not (os.path.exists(self.DIR_CHECKPOINTS) and os.path.isdir(self.DIR_CHECKPOINTS)):
            os.mkdir(self.DIR_CHECKPOINTS)

        # Create directory to save all cache data
        if not (os.path.exists(self.DIR_CACHE) and os.path.isdir(self.DIR_CACHE)):
            os.mkdir(self.DIR_CACHE)
            logging.info(f'Created directory => {self.DIR_CACHE}')

        # Create directory to save all caches
        if not (os.path.exists(self.PATH_CACHE_DIR) and os.path.isdir(self.PATH_CACHE_DIR)):
            os.mkdir(self.PATH_CACHE_DIR)
            logging.info(f'Created directory => {self.PATH_CACHE_DIR}')


_ = Constants()


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = f'{Constants.TOKEN_START} {w} {Constants.TOKEN_END}'
    return w


def create_dataset(path, num_examples):
    """
    1. Remove the accents
    2. Clean the sentences
    3. Return word pairs in the format: [ENGLISH, SPANISH]
    """

    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [
        [preprocess_sentence(w) for w in l.split('\t')]
        for l in lines[:num_examples]
    ]

    return zip(*word_pairs)


def max_length(iterable):
    """
    Arguments:
        iterable {Iteratable} -- An object which can be iterated 
            using a python for loop

    Returns:
        [int] -- Max length of all objects length inside the iterable
    """
    return max(len(t) for t in iterable)


def tokenize(lang):
    """Creates a Tokenizer for the given list of sentences
    Tokenizes all the sentences in `lang`

    Arguments:
        lang {list} -- List of the strings each representing a sentence

    Returns:
        [list] -- List of sequences of each text string
        [KP.text.Tokenizer] -- Object of the tokenizer 
    """
    lang_tokenizer = KP.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    text_seq = lang_tokenizer.texts_to_sequences(lang)
    text_seq = KP.sequence.pad_sequences(text_seq, padding='post')

    return text_seq, lang_tokenizer


def _cache_data(func, path, *args, **kwargs):
    """Called by [load_cached_data] to cache the output of the provide function

    Arguments:
        func {function} -- Function used to calculate the ouput
        path {str} -- Path of the cache file

        args, kwargs -- `func` (function) parameters
    Raises:
        e: Exception raised either while executing `func` function or while serializing
            the data 
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(func(*args, **kwargs), f)
            logging.info(f'Succesfully cached output => {path}')
    except Exception as e:
        logging.error(str(e))
        raise e


def load_cached_data(func, cache_data=True, *args, **kwargs):
    """Loads the ouput from the cached file.
    If data is not already cached then calls the function `_cache_data`
    to cache it and returns the output

    Arguments:
        func {function} -- Function used to calculate the ouput
        cache_data {boolean} -- Where to recalculate the output and cache it again 

        args, kwargs -- `func` (function) parameters
    Returns:
        Ouput obtained after passing the function parameters
    """
    path = os.path.join(Constants.DIR_CACHE, f'cache_{func.__name__}.pkl')

    if not (os.path.exists(path) and os.path.isfile(path)) or cache_data:
        _cache_data(func, path, *args, **kwargs)

    with open(path, 'rb') as f:
        out = pickle.load(f)

    logging.info(f'Loaded cached data from => {path}')
    return out
