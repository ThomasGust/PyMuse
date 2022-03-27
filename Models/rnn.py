import tensorflow as tf
import numpy as np
import pickle as pkl
import os

class GRURNNModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        super(GRURNNModel, self).__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_batch_size=[batch_size, None])
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


class GRURNN:

    def __init__(self):
        self.model = None

    def train_euterpe_rnn(self, vectorized, vocab_size, embedding_dim, rnn_units, batch_size, learning_rate, epochs,
                          steps_per_epoch, sequence_length, output_path, output_name):
        self.model = GRURNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
                                     batch_size=batch_size)
        self.model.optimizer = tf.keras.optimizers.Adam(learning_rate)

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                xb, yb = self.model.get_batch(vectorized=vectorized, seq_length=sequence_length,
                                              batch_size=batch_size)
                self.model.train_step(xb, yb)
                print(f"Completed step {step + 1} of epoch {epoch + 1}")
            print()
            print(f"Completed epoch {epoch}")
        out = os.path.join(output_path, output_name)
        os.makedirs(out)
        self.model.save_weights(out)

        with open(os.path.join(out, "loadparams.config"), "wb") as f:
            b = pkl.dumps([vocab_size, embedding_dim, rnn_units, batch_size])
            pkl.dump(b, f)

        with open(os.path.join(out, "in.config"), "wb") as f:
            b = pkl.dumps([vectorized, output_path, output_name, steps_per_epoch, epochs, batch_size, sequence_length,
                           learning_rate, embedding_dim, rnn_units])
            pkl.dump(b, f)

    def load_euterpe_rnn_model(self, path, config_path, vocab_size, embedding_dim, rnn_units, batch_size=1):
        with open(os.path.join(config_path, "loadparams.config"), "rb") as f:
            ps = pkl.load(f)
            ps = pkl.loads(ps)
            vocab_size = ps[0]
            embedding_dim = ps[1]
            rnn_units = ps[2]

        self.model = GRURNN(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
                                     batch_size=batch_size)
        self.model.build(tf.TensorShape([1, None]))
        self.model.load_weights(path)
        return self.model