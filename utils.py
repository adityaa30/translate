import unicodedata
import re
import io
import pickle
import os
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf

import constant

LOGGER = logging.getLogger(__name__)

K = tf.keras
KP = tf.keras.preprocessing


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
    w = f'{constant.TOKEN_START} {w} {constant.TOKEN_END}'
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
            LOGGER.info(f'Succesfully cached output => {path}')
    except Exception as e:
        LOGGER.error(str(e))
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
    path = os.path.join(constant.DIR_CACHE, f'cache_{func.__name__}.pkl')

    if not (os.path.exists(path) and os.path.isfile(path)) or cache_data:
        _cache_data(func, path, *args, **kwargs)

    with open(path, 'rb') as f:
        out = pickle.load(f)

    LOGGER.info(f'Loaded cached data from => {path}')
    return out


def plot_attention(attention, sentence, predicted_sentence):
    """Plots the attention weights

    Arguments:
        attention {list<ndarray>} -- Attention values for each time t unit
        sentence {list<str>} -- List of all words in the input sentence 
        predicted_sentence {list<str>} -- List of all words in the output sentence
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    # fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
