"""Recurrent language models: a plain RNN and an LSTM.

Both models do the same job. Given a batch of token-id sequences, they output,
for EVERY position, a score for every word in the vocabulary: "how likely is
each word to come next?". Their forward pass takes

    input_ids: (batch, seq_len)  LongTensor of token ids

and returns

    logits:    (batch, seq_len, vocab_size)  raw (unnormalised) scores

which is exactly what evaluate.py and train.py expect from any model.

The layers, in order (identical in both classes except step 3):

1.  Embedding      token id -> a learnable vector of `embedding_dim` (100)
                   numbers. Starts random, is trained with everything else.
2.  Dropout        randomly zeroes a fraction of the numbers while TRAINING
                   (not when evaluating), which fights overfitting.
3.  Recurrent      reads the sequence left to right, one token at a time,
                   keeping a "memory" vector (the hidden state) of size
                   `hidden_size` that it updates after every token.
                   THIS is the only difference between the two classes.
4.  Dropout        same idea again, on the recurrent layers' output.
5.  Linear         hidden vector -> one score per vocabulary word.

No future leakage: the recurrent layers are one-directional (left to right),
so the output at position t depends only on input tokens 0..t.

Difference between the two classes
----------------------------------
RNNLanguageModel   Memory update:   h_t = tanh(W_x x_t + W_h h_{t-1} + b)
    The old memory is pushed through a matrix and a squashing function at
    EVERY step. To learn from something that happened 30 words ago, the
    learning signal (gradient) has to travel back through 30 such squashes,
    and each one tends to shrink it (the "vanishing gradient" problem; it
    can also blow up). So in practice the plain RNN mostly learns short-range
    patterns.

LSTMLanguageModel  Memory update:   c_t = f_t * c_{t-1} + i_t * g_t
                                    h_t = o_t * tanh(c_t)
    Besides h it keeps a separate "cell state" c, and three learned gates
    (each a number between 0 and 1 per memory slot): forget f (how much old
    memory to keep), input i (how much new information to write) and output
    o (how much of the memory to expose). The cell state is updated mostly by
    ADDITION, with f close to 1 meaning "keep this unchanged". The gradient
    can then flow back along c with little shrinking, so the signal from
    far-away words survives, and the model can LEARN to hold on to
    information (like the subject of a sentence or the film being discussed)
    for many steps. That is why the LSTM handles longer dependencies better.
"""

import torch.nn as nn


class RNNLanguageModel(nn.Module):
    """Embedding -> dropout -> tanh RNN -> dropout -> linear."""

    def __init__(self, vocab_size, pad_id, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        # padding_idx: the <pad> row stays all zeros and is never updated.
        # Every other row starts at random values and IS trained.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        # batch_first=True: tensors are (batch, seq_len, features).
        # `dropout=` here applies BETWEEN stacked layers; PyTorch ignores it
        # (with a warning) when there is only one layer, so we pass 0 then.
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_layers, nonlinearity="tanh",
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embedding_dim)
        x, _ = self.rnn(x)                           # (batch, seq_len, hidden_size); _ is the final state, unused
        return self.output(self.dropout(x))          # (batch, seq_len, vocab_size)


class LSTMLanguageModel(nn.Module):
    """Embedding -> dropout -> LSTM -> dropout -> linear."""

    def __init__(self, vocab_size, pad_id, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embedding_dim)
        x, _ = self.lstm(x)                          # _ is the final (hidden, cell) state, unused
        return self.output(self.dropout(x))          # (batch, seq_len, vocab_size)
