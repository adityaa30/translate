import os
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split

import constant
from utils import (create_dataset, tokenize, load_cached_data)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

K = tf.keras
KP = tf.keras.preprocessing
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:

    def __init__(self):
        # Download the dataset file
        self.path_to_zip = K.utils.get_file(
            'spa-eng.zip',
            origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
            extract=True
        )

        self.path_to_file = os.path.join(
            os.path.dirname(self.path_to_zip),
            'spa-eng',
            'spa.txt'
        )
        self.input_seq, \
            self.target_seq, \
            self.tokenizer_input, \
            self.tokenizer_target = self.load_dataset()

        # Creating training and validation sets using an 80-20 split
        self.train_input, self.val_input, self.train_target, self.val_target = train_test_split(
            self.input_seq, self.target_seq,
            test_size=0.2
        )

        # Log the dataset details
        LOGGER.debug(f'Input data (train, val) => '
                     f'({len(self.train_input)}, {len(self.val_input)})')

        LOGGER.debug(f'Target data (train, val) => '
                     f'({len(self.train_target)}, {len(self.val_target)})')

        self.buffer_size = len(self.train_input)
        self.steps_per_epoch = len(self.train_input) // constant.BATCH_SIZE
        self.vocab_size_input = len(self.tokenizer_input.word_index) + 1
        self.vocab_size_target = len(self.tokenizer_target.word_index) + 1
        self.max_length_target = self.max_length(self.target_seq)
        self.max_length_input = self.max_length(self.input_seq)

        # Create [tf.data.Dataset] pipeline considering bucketing of each batch
        self.train_dataset = tf.data.Dataset \
            .from_tensor_slices((self.train_input, self.train_target)) \
            .shuffle(self.buffer_size) \
            .batch(constant.BATCH_SIZE, drop_remainder=True)

    def load_dataset(self, num_examples=constant.DATASET_SIZE):
        """Loads the dataset
        Calls [create_dataset] to retrive the processed data

        Keyword Arguments:
            num_examples {int} -- Total dataset size 
                (default: {None} refers to complete dataset)

        Returns:
            [list] -- List of sequences of input & output dataset
            [KP.text.Tokenizer] -- Object of the tokenizer of input & output dataset
        """
        targ_lang, inp_lang = load_cached_data(
            create_dataset, False,
            self.path_to_file, num_examples
        )

        input_text_seq, inp_lang_tokenizer = tokenize(inp_lang)
        target_text_seq, targ_lang_tokenizer = tokenize(targ_lang)

        LOGGER.info('Loaded the dataset')
        return input_text_seq, target_text_seq, inp_lang_tokenizer, targ_lang_tokenizer

    @staticmethod
    def max_length(iterable):
        """
        Arguments:
            iterable {Iteratable} -- An object which can be iterated 
                using a python for loop

        Returns:
            [int] -- Max length of all objects length inside the iterable
        """
        return max(len(t) for t in iterable)
