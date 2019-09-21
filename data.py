from utils import (
    Constants,
    create_dataset,
    tokenize,
    max_length,
    load_cached_data
)

import os
import logging

import tensorflow as tf
from sklearn.model_selection import train_test_split

K = tf.keras
KP = tf.keras.preprocessing
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:

    def __init__(self, batch_size=64):
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
        self.input_seq, self.target_seq, self.tokenizer_input, self.tokenizer_target = self.load_dataset()

        # Creating training and validation sets using an 80-20 split
        self.train_input, self.val_input, self.train_target, self.val_target = train_test_split(
            self.input_seq, self.target_seq,
            test_size=0.2
        )

        # Log the dataset details
        logging.debug(f'Input data (train, val) => '
                      '({len(self.train_input)}, {len(self.val_input)})')

        logging.debug(f'Target data (train, val) => '
                      '({len(self.train_target)}, {len(self.val_target)})')

        self.buffer_size = len(self.train_input)
        self.batch_size = batch_size
        self.steps_per_epoch = len(self.train_input) // self.batch_size
        self.vocab_size_input = len(self.tokenizer_input.word_index)
        self.vocab_size_target = len(self.tokenizer_target.word_index)

        # Create [tf.data.Dataset] pipeline considering bucketing of each batch
        self.train_dataset = tf.data.Dataset \
            .from_tensor_slices((self.train_input, self.train_target)) \
            .map(lambda x, y: self.remove_unecessary_zero(x, y), num_parallel_calls=AUTOTUNE) \
            .shuffle(self.buffer_size) \
            .batch(self.batch_size, drop_remainder=True)

    @staticmethod
    def remove_unecessary_zero(x, y):
        # TODO: Complete the function
        return x, y

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
        targ_lang, inp_lang = load_cached_data(
            create_dataset, False,
            self.path_to_file, num_examples
        )

        input_text_seq, inp_lang_tokenizer = tokenize(inp_lang)
        target_text_seq, targ_lang_tokenizer = tokenize(targ_lang)

        logging.info('Loaded the dataset')
        return input_text_seq, target_text_seq, inp_lang_tokenizer, targ_lang_tokenizer
