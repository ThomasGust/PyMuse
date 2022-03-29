# *PyMuse*
PyMuse is a python tool to create music and art with the help of artificial intelligence. (Readme and docs are in super alpha so don't use them rn)

# Usage
  ## Train LSTM
  ```python
  from Models.lstm import ABCLSTM
  from text_utils import get_vocab
  # Will fill out all of the docs and readme stuff later
  
  songs_joined = "\n\n".join(songs)
  vocab = get_vocab(songs_joined)
  char2idx = {u: i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)
  vectorized_songs = vectorize_string(char2idx, songs_joined)

  model = ABCLSTM()
  model.train_lstm(batch_size=32, char2idx=char2idx, idx2char=idx2char, learning_rate=1e-3, embedding_dim=256,
                   steps_per_epoch=5000, epochs=1, vocab_size=len(vocab), vectorized=vectorized_songs,
                   sequence_length=100, rnn_units=1024, output_path="Models\\train_lstm", output_name="name")
  ```

  ## Train RNN
  ```python
  from Models.rnn import ABCGRURNN
  from text_utils import get_vocab
  # Will fill out all of the docs and readme stuff later
  songs_joined = "\n\n".join(songs)
  vocab = get_vocab(songs_joined)
  char2idx = {u: i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)
  vectorized_songs = vectorize_string(char2idx, songs_joined)

  model = ABCGRURNN()
  model.train_lstm(batch_size=32, char2idx=char2idx, idx2char=idx2char, learning_rate=1e-3, embedding_dim=256,
                   steps_per_epoch=5000, epochs=1, vocab_size=len(vocab), vectorized=vectorized_songs,
                   sequence_length=100, rnn_units=1024, output_path="Models\\train_rnn", output_name="name")
  ```

  ## Train SEQ2SEQ
  ```python
  from Models.seq2seq import ABCSEQ2SEQ
  # Will fill out all of the docs and readme stuff later
  ```
  
  ## Infer with ABCLSTM
  ```python
  from models.lstm import ABCLSTM
  # Will fill out all of the docs and readme stuff later
  model = ABCLSTM()
  model.load_lstm_model(path="path\\to\\file", config_path="path\\to\\file", rnn_units=1024,
                      embedding_dim=256, vocab_size=len(vocab))
  preds = model.predict_lstm_model(start_seed="X", generation_length=1000, format="midi", fp="out1")
  ```
  
  ## Infer with ABCGRURNN
  ```python
  from models.rnn import ABCGRURNN
  # Will fill out all of the docs and readme stuff later
  ```
