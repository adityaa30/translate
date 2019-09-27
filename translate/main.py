import os
import time
import logging

from argparse import ArgumentParser
import numpy as np
import tensorflow as tf

import translate.constant as constant
from translate.utils import (preprocess_sentence,
                             plot_attention,
                             readable_dir,
                             readable_file)
from translate.data import Dataset
from translate.models import (Encoder, BahdanauAttention, Decoder)

K = tf.keras
KP = tf.keras.preprocessing
KC = tf.keras.callbacks
KL = tf.keras.losses
KO = tf.keras.optimizers

# Create an ArgumentParser instance
parser = ArgumentParser()
sub_parser = parser.add_subparsers(dest='sub_command')
train_sub_parser = sub_parser.add_parser(name=constant.CLI_COMMAND_TRAIN)
train_sub_parser.add_argument(
    '-b', '--batch-size',
    action='store',
    type=int,
    required=False,
    default=32,
    help='Batch size to use while training'
)

train_sub_parser.add_argument(
    '-D', '--dataset-size',
    action='store',
    type=int,
    required=False,
    default=None,
    help='Dataset size to use while training'
)

evaluate_sub_parser = sub_parser.add_parser(name=constant.CLI_COMMAND_EVALUATE)
evaluate_sub_parser.add_argument(
    '-s', '--sentence',
    action='store',
    type=str,
    required=True,
    help='Sentence string to be translated'
)

parser.add_argument(
    '-d', '--dataset-path',
    action='store',
    type=readable_dir,
    required=False,
    help='Path of the directory containing the dataset.\n'
    'Downloads English-Spanish dataset by default.'
)

parser.add_argument(
    '-l', '--log-file',
    action='store',
    type=readable_file,
    required=False,
    help='Path to the .log file. By default logs will be added to `app.log`',
    # default='app.log' => Created automatically below while parsing params
)

parser.add_argument(
    '-c', '--checkpoint-dir',
    action='store',
    type=readable_dir,
    required=True,
    help='Loads the checkpoint from the given directory if present.\n'
    'Saves a checkpoint at the end of every 2 epochs.'
)

args = parser.parse_args()
print(args)

if args.sub_command == constant.CLI_COMMAND_TRAIN:
    constant.BATCH_SIZE = args.batch_size
    constant.DATASET_SIZE = args.dataset_size

constant.PATH_CHECKPOINT_DIR = os.path.abspath(args.checkpoint_dir)
constant.PATH_CHECKPOINT = os.path.join(constant.PATH_CHECKPOINT_DIR, 'ckpt')

log_file_name = None
if args.log_file:
    constant.PATH_LOG_FILE = os.path.join(constant.DIR_LOGS, args.log_file)


constant.initialize_logger()

# Get the logger instance with updated configuration
LOGGER = logging.getLogger(__name__)

data = Dataset(data_path=args.dataset_path,
               data_size=args.dataset_size)
               
optimizer = KO.Adam()
loss_object = KL.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)

encoder = Encoder(
    data.vocab_size_input,
    constant.EMBEDDING_DIM,
    constant.LSTM_UNITS,
    constant.BATCH_SIZE
)
attention_layer = BahdanauAttention(units=10)
decoder = Decoder(
    data.vocab_size_target,
    constant.EMBEDDING_DIM,
    constant.LSTM_UNITS,
    constant.BATCH_SIZE
)

LOGGER.info(f'Initialized Encoder, Attention Layer & Decoder')

# log the model details
example_input_batch, example_target_batch = next(iter(data.train_dataset))
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
attention_result, attention_weights = attention_layer(sample_hidden,
                                                      sample_output)
sample_decoder_output, sample_decoder_state, _ = decoder(tf.random.uniform((constant.BATCH_SIZE, 1)),
                                                         sample_hidden,
                                                         sample_output)

LOGGER.info(f'Encoder output shape => '
            f'(batch size, sequence length, units) => {sample_output.shape}')
LOGGER.info(f'Encoder Hidden state shape => '
            f'(batch size, units) => {sample_hidden.shape}')

LOGGER.info(f'Attention result shape => '
            f'(batch size, units) => {attention_result.shape}')
LOGGER.info(f'Attention weights shape => '
            f'(batch_size, sequence_length, 1) => {attention_weights.shape}')

LOGGER.info(f'Decoder output shape => '
            f'(batch_size, vocab size) => {sample_decoder_output.shape}')
LOGGER.info(f'Decoder Hidden state shape => '
            f'(batch_size, units) => {sample_decoder_state.shape}')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Create a checkpoint
checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    encoder=encoder,
    decoder=decoder
)


@tf.function
def train_step(inp, target, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([data.tokenizer_target.word_index[constant.TOKEN_START]]
                                   * constant.BATCH_SIZE, axis=1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input,
                                                 dec_hidden,
                                                 enc_output)

            loss += loss_function(target[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train():
    LOGGER.debug(f'Starting training')
    for epoch in range(constant.EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, target)) in enumerate(data.train_dataset.take(data.steps_per_epoch)):
            batch_loss = train_step(inp, target, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                LOGGER.debug(f'Epoch {epoch + 1} '
                             f'Batch {batch} '
                             f'Loss {batch_loss.numpy()}')

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=constant.PATH_CHECKPOINT)
            LOGGER.debug(f'Epoch {epoch + 1} Checkpoint saved')

        LOGGER.debug(f'Epoch {epoch + 1} '
                     f'Loss {total_loss / data.steps_per_epoch}')
        LOGGER.debug(f'Time taken for 1 epoch {time.time() - start} sec')


def evaluate(sentence):
    """Input to the decoder at each time step is its previous predictions along
    with the hidden state and the encoder output.

    Arguments:
        sentetnce {str} -- Sentence which will be converted into spanish
    """
    attention_plot = np.zeros((data.max_length_target, data.max_length_input))
    sentence = preprocess_sentence(sentence)
    print(sentence)

    inputs = [data.tokenizer_input.word_index[i]
              for i in sentence.strip().split(' ')]

    inputs = KP.sequence.pad_sequences([inputs],
                                       maxlen=data.max_length_input,
                                       padding='post')

    inputs = tf.convert_to_tensor(inputs)
    result = ''
    initial_hidden_state = [tf.zeros((1, constant.LSTM_UNITS))]
    enc_out, enc_hidden = encoder(inputs, initial_hidden_state)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([data.tokenizer_target.word_index[constant.TOKEN_START]],
                               axis=0)

    for t in range(data.max_length_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += data.tokenizer_target.index_word[predicted_id] + ' '

        if data.tokenizer_target.index_word[predicted_id] == constant.TOKEN_END:
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    LOGGER.debug(f'Input: {sentence}')
    LOGGER.debug(f'Predicted translation: {result}')

    result = result.strip().split(' ')
    sentence = sentence.strip().split(' ')

    attention_plot = attention_plot[:len(result), :len(sentence)]
    plot_attention(attention_plot, sentence, result)


def restore_checkpoints():
    try:
        checkpoint.restore(tf.train.latest_checkpoint(
            constant.PATH_CHECKPOINT_DIR))
        LOGGER.info(f'Loaded weights from => '
                    f'{constant.PATH_CHECKPOINT_DIR}')
    except Exception as e:
        LOGGER.error(f'Error while loading trained weights => {e}')
        LOGGER.warning(f'Cannot load weights from {constant.PATH_CHECKPOINT_DIR}, '
                       f'initialized model with no pre-trained weights.')


restore_checkpoints()

if args.sub_command == constant.CLI_COMMAND_EVALUATE:
    translate(args.sentence)
elif args.sub_command == constant.CLI_COMMAND_TRAIN:
    train()
else:
    LOGGER.warning(f'Incorrect sub-command used "{args.sub_command}"')
