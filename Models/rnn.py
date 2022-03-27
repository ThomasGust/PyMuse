import tensorflow as tf
import numpy as np
import pickle as pkl
import os
from sound_utils import abc2midipy, midi2wav
from music21 import ABCHand
from text_utils import extract_song_snippet

class GRURNNModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        super(GRURNNModel, self).__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
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
        self.char2idx = None
        self.idx2char = None

    def train_rnn(self, vectorized, vocab_size, embedding_dim, char2idx, idx2char, rnn_units, batch_size,
                          learning_rate, epochs,
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
            print(f"Completed epoch {epoch + 1}")
        out = os.path.join(output_path, output_name)
        os.makedirs(out)
        self.model.save_weights(out)
        self.char2idx = char2idx
        self.idx2char = idx2char

        with open(os.path.join(out, "loadparams.config"), "wb") as f:
            b = pkl.dumps([vocab_size, embedding_dim, rnn_units, batch_size, char2idx, idx2char])
            pkl.dump(b, f)

        with open(os.path.join(out, "in.config"), "wb") as f:
            b = pkl.dumps([vectorized, output_path, output_name, steps_per_epoch, epochs, batch_size, sequence_length,
                           learning_rate, embedding_dim, rnn_units])
            pkl.dump(b, f)

    def load_rnn_model(self, path, config_path, vocab_size, embedding_dim, rnn_units, batch_size=1):
        with open(os.path.join(config_path, "loadparams.config"), "rb") as f:
            ps = pkl.load(f)
            ps = pkl.loads(ps)
            vocab_size = ps[0]
            embedding_dim = ps[1]
            rnn_units = ps[2]
            self.char2idx = ps[4]
            self.idx2char = ps[5]

        self.model = GRURNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
                                 batch_size=batch_size)
        self.model.build(tf.TensorShape([1, None]))
        self.model.load_weights(path)
        return self.model

    def predict_rnn_model(self, start_seed, generation_length, format, fp):
        input_eval = [self.char2idx[s] for s in start_seed]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        self.model.reset_states()

        for i in range(generation_length):
            predictions = self.model(input_eval)

            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])
        abc = start_seed + ''.join(text_generated)
        abc = extract_song_snippet(abc)[0]
        try:
            if format == "abc" and fp is None:
                return abc
            elif format == "midi" and fp is None:
                with open("tmp.abc", "w") as f:
                    f.flush()
                    f.write(abc)
                o = abc2midipy("tmp.abc")
                os.remove("tmp.abc")
                return o
            elif format == "abc" and fp is not None:
                with open(f"{fp}.abc", "w") as f:
                    f.flush()
                    f.write(abc)
                return abc
            elif format == "midi" and fp is not None:
                with open("tmp.abc", "w") as f:
                    f.flush()
                    f.write(abc)
                o = abc2midipy("tmp.abc", fp)
                os.remove("tmp.abc")
                return o
            elif format == "wav" and fp is None:
                raise TypeError("Value: fp cannot be none to save in .wav format")
            elif format == "wav" and fp is not None:
                with open("tmp.abc", "w") as f:
                    f.flush()
                    f.write(abc)
                abc2midipy("tmp.abc", "tmp")
                os.remove("tmp.abc")
                midi2wav("tmp.mid", f"{fp}.wav")
                os.remove("tmp.mid")
                return f"{fp}.wav"
        except Exception as e:
            print(e)
            self.predict_rnn_model(start_seed=start_seed, generation_length=generation_length, format=format, fp=fp)

    def generate_wav_batch(self, n, output_directory):
        if os.path.isdir(output_directory) is False:
            os.makedirs(output_directory)
        else:
            pass
        
        for i in range(n):
            self.predict_lstm_model(start_seed="X", generation_length=1000, format="wav", fp=os.path.join(output_directory, i+1))
        return output_directory