
import unicodedata
import re
import io
import pickle
import logging
import os


class Constants:
    TOKEN_START = '<start>'
    TOKEN_END = '<end>'

    CACHE_DIR = 'cache'

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

        # Create directory to save all cache data
        if not (os.path.exists(self.CACHE_DIR) and os.path.isdir(self.CACHE_DIR)):
            logging.debug(f'Created directory => {self.CACHE_DIR}')
            os.mkdir(Constants.CACHE_DIR)


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
    path = os.path.join(Constants.CACHE_DIR, f'cache_{func.__name__}.pkl')

    if not (os.path.exists(path) and os.path.isfile(path)) or cache_data:
        _cache_data(func, path, *args, **kwargs)

    with open(path, 'rb') as f:
        out = pickle.load(f)
    logging.info(f'Load cache data from => {path}')
    return out
