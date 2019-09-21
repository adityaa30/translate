
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
        self.logger = logging.Logger('Machine-Translation')


constants = Constants()
logger = constants.logger

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
    try:
        pickle.dump(func(*args, **kwargs), path)
        logger.info(f'Succesfully cached output to {path}')
    except Exception as e:
        logger.error(str(e))
        raise e


def load_cached_data(func, *args, **kwargs):
    path = os.path.join(Constants.CACHE_DIR, f'cache_{func.__name__}.pkl')

    if not (os.path.exists(path) and os.path.isfile(path)):
        _cache_data(func, path, *args, **kwargs)

    out = pickle.load(path)
    logger.info(f'Load cache data from {path}')
    return out
