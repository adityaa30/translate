import logging
import tensorflow as tf

K = tf.keras
KL = tf.keras.layers

"""Model pseuda-code model summary

Notations:
1) FC = Fully connected (dense) layer
2) EO = Encoder output
3) H = hidden state
4) X = input to the decoder

pseduo-code:
1) score = FC(tanh(FC(EO) + FC(H)))
2) attention weights = softmax(score, axis = 1). Softmax by default is applied on the last axis but
    here we want to apply it on the 1st axis, since the shape of score is
    (batch_size, max_length, hidden_size). Max_length is the length of our input. 
    Since we are trying to assign a weight to each input, softmax should be applied on that axis.
3) context vector = sum(attention weights * EO, axis = 1). Same reason as above for choosing axis as 1.
4) embedding output = The input to the decoder X is passed through an embedding layer.
5) merged vector = concat(embedding output, context vector)
6) This merged vector is then given to the GRU

Shapes can be seen in app*.log file
"""


class Encoder(K.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = KL.Embedding(vocab_size, embedding_dim)
        self.gru = KL.GRU(self.enc_units,
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = KL.Dense(units)
        self.W2 = KL.Dense(units)
        self.V = KL.Dense(1)

    def call(self, query, values):  # (self, hidden, enc_output)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) +
                                  self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = KL.Embedding(vocab_size, embedding_dim)
        self.gru = KL.GRU(self.dec_units,
                          return_sequences=True,
                          return_state=True,
                          recurrent_initializer='glorot_uniform')
        self.fc = KL.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
