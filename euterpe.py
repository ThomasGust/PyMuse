import tensorflow as tf

class EuterpeModelRNN(tf.keras.Model):

  def __init__(self):
    super(EuterpeModelRNN, self).__init__()

  def call(self, x):
    return x

class EuterpeModelLSTM(tf.keras.Model):
  
  def __init__(self):
    super(EuterpeModelLSTM, self).__init()

  def call(self, x):
    return x

class EuterpeModelAutoEncoder(tf.keras.Model):

  def __init__(self):
    super(EuterpeModelAutoEncoder, self).__init()

  def call(self, x):
    return x
  
class EuterpeModelTransformer(tf.keras.Model):

  def __init__(self):
    super(EuterpeModelTransformer, self).__init__()

  def call(self, x):
    pass