import os
import logging
import numpy as np
import tensorflow as tf

import translate.constant as constant
from translate.utils import (create_dataset, tokenize, load_cached_data)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

K = tf.keras
KP = tf.keras.preprocessing
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:

    def __init__(self, data_path=None, data_size=None):
        if data_path is None:
            LOGGER.warning(f'No path specified for dataset. Downloading English-Spanish dataset from => '
                           f"{'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'}")
            # Download the dataset file
            path_to_zip = K.utils.get_file(
                'spa-eng.zip',
                origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
                extract=True
            )

            self.path_to_file = os.path.join(
                os.path.dirname(path_to_zip),
                'spa-eng',
                'spa.txt'
            )
        else:
            self.path_to_file = data_path

        self.input_seq, \
            self.target_seq, \
            self.tokenizer_input, \
            self.tokenizer_target = self.load_dataset(data_size)

        # Creating training and validation sets using an 80-20 split
        self.train_input, \
            self.val_input, \
            self.train_target, \
            self.val_target = self.split_data(test_size=0.2)

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

        LOGGER.info(f'Data vocab size (input, output) => '
                    f'({self.vocab_size_input}, {self.vocab_size_target})')
        LOGGER.info(f'Data max length (input, output) => '
                    f'({self.max_length_input}, {self.max_length_target})')

        # Create [tf.data.Dataset] pipeline considering bucketing of each batch
        self.train_dataset = tf.data.Dataset \
            .from_tensor_slices((self.train_input, self.train_target)) \
            .shuffle(self.buffer_size) \
            .batch(constant.BATCH_SIZE, drop_remainder=True)

    def load_dataset(self, num_examples=None):
        """Loads the dataset
        Calls [create_dataset] to retrive the processed data

        Keyword Arguments:
            num_examples {int} -- Total dataset size 
                (default: {None} refers to complete dataset)

        Returns:
            [list] -- List of sequences of input & output dataset
            [KP.text.Tokenizer] -- Object of the tokenizer of input & output dataset
        """
        LOGGER.info(f'Loading & processing dataset from => '
                    f'{self.path_to_file}')

        # NOTE: Temporarily disabled caching of the dataset
        inp_lang, targ_lang = load_cached_data(
            create_dataset, False,
            self.path_to_file, num_examples
        )
        # inp_lang, targ_lang = create_dataset(self.path_to_file, num_examples)

        if constant.REVERSE_DATA:
            # Swap the input and ouput dataset
            inp_lang, targ_lang = targ_lang, inp_lang

        input_text_seq, inp_lang_tokenizer = tokenize(inp_lang)
        target_text_seq, targ_lang_tokenizer = tokenize(targ_lang)

        input_text_seq = np.array(input_text_seq)
        target_text_seq = np.array(target_text_seq)

        LOGGER.info('Successfully loaded the dataset')
        LOGGER.info(f'Processed input data => {inp_lang[:5]}')
        LOGGER.info(f'Tokenized input data => {input_text_seq[:5]}')
        LOGGER.info(f'Processed target data => {targ_lang[:5]}')
        LOGGER.info(f'Tokenized target data => {target_text_seq[:5]}')

        return input_text_seq, target_text_seq, inp_lang_tokenizer, targ_lang_tokenizer

    def split_data(self, test_size=0.2):
        """Split arrays or matrices into random train and test subsets

        Keyword Arguments:
            test_size {float} -- should be between 0.0 and 1.0 and represent the
             proportion of the dataset to include in the test split (default: {0.2})

        Returns:
            Training and validation data
        """
        assert(self.input_seq.shape[0] == self.target_seq.shape[0])

        split_size = int(self.input_seq.shape[0] * (1 - test_size))
        LOGGER.info(f'Test size for dataset => {test_size} (= {split_size})')

        idx = np.random.permutation(self.input_seq.shape[0])
        training_idx, val_idx = idx[:split_size], idx[split_size:]
        return self.input_seq[training_idx, :], self.input_seq[val_idx, :], self.target_seq[training_idx, :], self.target_seq[val_idx, :]

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
