import time
import logging
import tensorflow as tf

from utils import Constants
from data import Dataset
from models import (Encoder, BahdanauAttention, Decoder)

K = tf.keras
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
    Constants.EMBEDDING_DIM,
    Constants.LSTM_UNITS,
    Constants.BATCH_SIZE
)
attention_layer = BahdanauAttention(units=10)
decoder = Decoder(
    data.vocab_size_target,
    Constants.EMBEDDING_DIM,
    Constants.LSTM_UNITS,
    Constants.BATCH_SIZE
)
logging.info(f'Initialized Encoder, Attention Layer & Decoder')

# Logging the model details
example_input_batch, example_target_batch = next(iter(data.train_dataset))
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
attention_result, attention_weights = attention_layer(sample_hidden,
                                                      sample_output)
sample_decoder_output, sample_decoder_state, _ = decoder(tf.random.uniform((Constants.BATCH_SIZE, 1)),
                                                         sample_hidden,
                                                         sample_output)

logging.debug(f'Encoder output shape => '
              f'(batch size, sequence length, units) => {sample_output.shape}')
logging.debug(f'Encoder Hidden state shape => '
              f'(batch size, units) => {sample_hidden.shape}')

logging.debug(f'Attention result shape => '
              f'(batch size, units) => {attention_result.shape}')
logging.debug(f'Attention weights shape => '
              f'(batch_size, sequence_length, 1) => {attention_weights.shape}')

logging.debug(f'Decoder output shape => '
              f'(batch_size, vocab size) => {sample_decoder_output.shape}')
logging.debug(f'Decoder Hidden state shape => '
              f'(batch_size, units) => {sample_decoder_state.shape}')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Callbacks
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
        dec_input = tf.expand_dims([data.tokenizer_target.word_index['<start>']]
                                   * Constants.BATCH_SIZE, 1)

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
    logging.info(f'Starting training')
    for epoch in range(Constants.EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, target)) in enumerate(data.train_dataset.take(data.steps_per_epoch)):
            batch_loss = train_step(inp, target, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                logging.debug(f'Epoch {epoch + 1} '
                              f'Batch {batch} '
                              f'Loss {batch_loss.numpy()}')

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=Constants.PATH_CHECKPOINT)
            logging.debug(f'Epoch {epoch + 1} Checkpoint saved')

        logging.debug(f'Epoch {epoch + 1} '
                      f'Loss {total_loss / data.steps_per_epoch}')
        logging.debug(f'Time taken for 1 epoch {time.time() - start} sec')


train()
print('Debug breakpoint')
