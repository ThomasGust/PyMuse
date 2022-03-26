import numpy as np
import tensorflow as tf


def get_vocab(joined_string):
    return sorted(set(joined_string))


def vectorize_vocab(vocab):
    return {u: i for i, u in enumerate(vocab)}, np.array(vocab)


def vectorize_string(char2idx, string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output


def get_lstm_layer(rnnu, return_sequences=True, recurrent_initializer='glorot_uniform', recurrent_activation='sigmoid',
                   stateful=True):
    lstm = tf.keras.layers.LSTM(
        rnnu,
        return_sequences=return_sequences,
        recurrent_initializer=recurrent_initializer,
        recurrent_activation=recurrent_activation,
        stateful=stateful
    )
    return lstm


class EuterpeModelRNN(tf.keras.Model):

    def __init__(self):
        super(EuterpeModelRNN, self).__init__()

    def call(self, x):
        return x


class EuterpeModelLSTM(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024, batch_size=32):
        super(EuterpeModelLSTM, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
        self.l1 = get_lstm_layer(rnnu=rnn_units)
        self.o = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embed(x)
        x = self.l1(x)
        return self.o(x)

    def get_loss(self, labels, logits):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss

    def get_batch(self, vectorized_songs, seq_length, batch_size):
        # the length of the vectorized songs string
        n = len(vectorized_songs) - 1
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(n - seq_length, batch_size)

        '''TODO: construct a list of input sequences for the training batch'''
        input_batch = [vectorized_songs[i: i + seq_length] for i in idx]
        # input_batch = # TODO
        '''TODO: construct a list of output sequences for the training batch'''
        output_batch = [vectorized_songs[i + 1: i + seq_length + 1] for i in idx]
        # output_batch = # TODO

        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            yh = self(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, yh, from_logits=True)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class EuterpeLSTM:

    def __init__(self):
        pass

    @staticmethod
    def train_euterpe_lstm_model(vectorized, vocab, output_path, num_training_iterations=2000, batch_size=4,
                                 seq_length=100, learning_rate=5e-3, embedding_dim=256, rnn_units=1024):
        assert isinstance(vectorized, np.ndarray)
        model = EuterpeModelLSTM(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units,
                                 batch_size=batch_size)

        model.optimizer = tf.keras.optimizers.Adam(learning_rate)

        for iter in range(num_training_iterations):
            xb, yb = model.get_batch(vectorized, seq_length, batch_size)
            model.train_step(xb, yb)
        model.save_weights(output_path)

class EuterpeModelAutoEncoder(tf.keras.Model):

  def __init__(self):
    super(EuterpeModelAutoEncoder, self).__init__()

  def call(self, x):
    return x
  
class EuterpeModelTransformer(tf.keras.Model):

  def __init__(self):
    super(EuterpeModelTransformer, self).__init__()

  def call(self, x):
    pass