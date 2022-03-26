import numpy as np
import tensorflow as tf
import os
import pickle as pkl


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

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(EuterpeModelRNN, self).__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
    
    def get_batch(self, vectorized, seq_length, batch_size):
        n = len(vectorized) - 1
        idx = np.random.choice(n - seq_length, batch_size)

        input_batch = [vectorized[i: i + seq_length] for i in idx]
        output_batch = [vectorized[i + 1: i + seq_length + 1] for i in idx]

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

    def get_batch(self, vectorized, seq_length, batch_size):
        n = len(vectorized) - 1
        idx = np.random.choice(n - seq_length, batch_size)

        input_batch = [vectorized[i: i + seq_length] for i in idx]
        output_batch = [vectorized[i + 1: i + seq_length + 1] for i in idx]

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


def train_euterpe_lstm_model(vectorized, vocab, output_path, output_name, num_training_iterations=3000, batch_size=32,
                             seq_length=100, learning_rate=5e-3, embedding_dim=256, rnn_units=1024):
    assert isinstance(vectorized, np.ndarray)
    model = EuterpeModelLSTM(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units,
                             batch_size=batch_size)

    model.optimizer = tf.keras.optimizers.Adam(learning_rate)

    for iter in range(num_training_iterations):
        xb, yb = model.get_batch(vectorized, seq_length, batch_size)
        model.train_step(xb, yb)
        print(f"Just finished train step: {iter}")

    out = os.path.join(output_path, output_name)
    os.makedirs(out)
    model.save_weights(out)

    with open(os.path.join(out, "loadparams.config"), "wb") as f:
        b = pkl.dumps([len(vocab), embedding_dim, rnn_units, batch_size])
        pkl.dump(b, f)

    with open(os.path.join(out, "in.config"), "wb") as f:
        b = pkl.dumps([vectorized, vocab, output_path, output_name, num_training_iterations, batch_size, seq_length,
                       learning_rate, embedding_dim, rnn_units])
        pkl.dump(b, f)


class EuterpeLSTM:

    def __init__(self):
        self.model = None

    def load_euterpe_lstm_model(self, path, config_path, vocab_size, embedding_dim, rnn_units, batch_size=1):
        with open(os.path.join(config_path, "loadparams.config"), "rb") as f:
            ps = pkl.load(f)
            ps = pkl.loads(ps)
            vocab_size = ps[0]
            embedding_dim = ps[1]
            rnn_units = ps[2]

        model = EuterpeModelLSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
                                 batch_size=batch_size)
        model.build(tf.TensorShape([1, None]))
        model.load_weights(path)
        self.model = model
        return model

    def predict_euterpe_lstm_model(self, model, start_string, generation_length, char2idx, idx2char):
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        model.reset_states()

        for i in range(generation_length):
            predictions = model(input_eval)

            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])

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