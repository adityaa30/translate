import os

import tensorflow as tf

from utils import (Constants,
                   load_cached_data,
                   unicode_to_ascii,
                   preprocess_sentence,
                   create_dataset)

K = tf.keras

# Download the file
path_to_zip = K.utils.get_file(
    'spa-eng.zip',
    origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True
)

path_to_file = os.path.join(os.path.dirname(path_to_zip), 'spa-eng', 'spa.txt')

en, sp = load_cached_data(create_dataset, path_to_file, None)
print(en[-1])
print(sp[-1])
