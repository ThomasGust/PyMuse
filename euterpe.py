import numpy as np
import tensorflow as tf

def get_vocab(joined_string):
  return sorted(set(joined_string))

def vectorize_vocab(vocab):
  return {u:i for i, u in enumerate(vocab)}, np.array(vocab)

def vectorize_string(char2idx, string):
  return np.array([char2idx[i]] for i in string)

def get_lstm_layer(rnnu, return_sequences=True, recurrent_initializer='glorot_uniform', recurrent_activation='sigmoid', stateful=True):
  lstm = tf.keras.layers.LSTM(
    rnnu,
    return_sequences = return_sequences,
    recurrent_initializer = recurrent_initializer,
    recurrent_activation = recurrent_activation,
    stateful = stateful
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
    self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size, None])
    self.l1 = get_lstm_layer(rnnu=rnn_units)
    self.o = tf.keras.layers.Dense(vocab_size)

  def call(self, x):
    x = self.embed(x)
    x = self.l1(x)
    return self.o(x)
  
  def get_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss

  def get_batch(vectorized, sequence_length, bs):
    n = vectorized.shape[0] - 1
    randindx = np.random.choice(n-sequence_length, bs)
    
    ibatch = [vectorized[i : i+sequence_length] for i in randindx]
    obatch = [vectorized[i+1 : i+sequence_length+1] for i in randindx]

    x = np.reshape(ibatch, [bs, sequence_length])
    y = np.reshape(obatch, [bs, sequence_length])
    return x, y
  
  @tf.function
  def train_step(self, x, y):

    with tf.GradientTape as tape:
      yh = self(x)
      loss = tf.keras.losses.sparse_categorical_crossentropy(y, yh, from_logits=True)
    
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradient(zip(gradients, self.trainable_variables))
    return loss
  
  def train(self, lr):
    self.optimizer = tf.keras.optimizers.Adam(lr)

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