import time
import numpy as np
import tensorflow as tf

import constant
from utils import (preprocess_sentence, plot_attention)
from data import Dataset
from models import (Encoder, BahdanauAttention, Decoder)

K = tf.keras
KP = tf.keras.preprocessing
KC = tf.keras.callbacks
KL = tf.keras.losses

data = Dataset()
optimizer = tf.keras.optimizers.Adam()
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
constant.LOGGER.info(f'Initialized Encoder, Attention Layer & Decoder')

# constant.LOGGER the model details
example_input_batch, example_target_batch = next(iter(data.train_dataset))
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
attention_result, attention_weights = attention_layer(sample_hidden,
                                                      sample_output)
sample_decoder_output, sample_decoder_state, _ = decoder(tf.random.uniform((constant.BATCH_SIZE, 1)),
                                                         sample_hidden,
                                                         sample_output)

constant.LOGGER.debug(f'Encoder output shape => '
              f'(batch size, sequence length, units) => {sample_output.shape}')
constant.LOGGER.debug(f'Encoder Hidden state shape => '
              f'(batch size, units) => {sample_hidden.shape}')

constant.LOGGER.debug(f'Attention result shape => '
              f'(batch size, units) => {attention_result.shape}')
constant.LOGGER.debug(f'Attention weights shape => '
              f'(batch_size, sequence_length, 1) => {attention_weights.shape}')

constant.LOGGER.debug(f'Decoder output shape => '
              f'(batch_size, vocab size) => {sample_decoder_output.shape}')
constant.LOGGER.debug(f'Decoder Hidden state shape => '
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
    constant.LOGGER.info(f'Starting training')
    for epoch in range(constant.EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, target)) in enumerate(data.train_dataset.take(data.steps_per_epoch)):
            batch_loss = train_step(inp, target, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                constant.LOGGER.debug(f'Epoch {epoch + 1} '
                              f'Batch {batch} '
                              f'Loss {batch_loss.numpy()}')

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=constant.PATH_CHECKPOINT)
            constant.LOGGER.debug(f'Epoch {epoch + 1} Checkpoint saved')

        constant.LOGGER.debug(f'Epoch {epoch + 1} '
                      f'Loss {total_loss / data.steps_per_epoch}')
        constant.LOGGER.debug(f'Time taken for 1 epoch {time.time() - start} sec')


def evaluate(sentence):
    """Input to the decoder at each time step is its previous predictions along 
    with the hidden state and the encoder output.

    Arguments:
        sentetnce {str} -- Sentence which will be converted into spanish
    """
    attention_plot = np.zeros((data.max_length_target, data.max_length_input))
    sentence = preprocess_sentence(sentence)

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

    constant.LOGGER.debug(f'Input: {sentence}')
    constant.LOGGER.debug(f'Predicted translation: {result}')

    result = result.strip().split(' ')
    sentence = sentence.strip().split(' ')

    attention_plot = attention_plot[:len(result), :len(sentence)]
    plot_attention(attention_plot, sentence, result)


# train()
try:
    checkpoint.restore(tf.train.latest_checkpoint(
        constant.PATH_CHECKPOINT_DIR))
    constant.LOGGER.info(f'Loaded weights from => '
                 f'{constant.PATH_CHECKPOINT_DIR}')
except Exception as e:
    constant.LOGGER.error(f'Error while loading trained weights => {e}')

translate(u'hace mucho frio aqui.')


print('Debug breakpoint')
