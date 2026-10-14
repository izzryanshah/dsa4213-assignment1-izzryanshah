"""Language models: a plain RNN, an LSTM and a causal Transformer.

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

import torch
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

    def hidden_states(self, input_ids):
        """The vector at every position just before the output layer: (batch, seq_len, hidden_size).
        Used by extract_features.py (Part III); forward() below is built on it."""
        x = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embedding_dim)
        x, _ = self.rnn(x)                           # (batch, seq_len, hidden_size); _ is the final state, unused
        return x

    def forward(self, input_ids):
        return self.output(self.dropout(self.hidden_states(input_ids)))  # (batch, seq_len, vocab_size)


class LSTMLanguageModel(nn.Module):
    """Embedding -> dropout -> LSTM -> dropout -> linear."""

    def __init__(self, vocab_size, pad_id, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, vocab_size)

    def hidden_states(self, input_ids):
        """The vector at every position just before the output layer: (batch, seq_len, hidden_size)."""
        x = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embedding_dim)
        x, _ = self.lstm(x)                          # _ is the final (hidden, cell) state, unused
        return x

    def forward(self, input_ids):
        return self.output(self.dropout(self.hidden_states(input_ids)))  # (batch, seq_len, vocab_size)


class TransformerLanguageModel(nn.Module):
    """Embedding + position -> dropout -> N decoder blocks -> LayerNorm -> linear.

    Unlike the RNN/LSTM, a Transformer does not read left to right. Self-attention
    lets every position look at every other position AT ONCE. So we must (1) tell
    it where each word sits (positional embedding) and (2) stop it from looking
    at later words (causal mask), otherwise it could just copy the answer.

    Each block (pre-norm style) does:
        x = x + SelfAttention(LayerNorm(x))   # each word gathers info from earlier words
        x = x + FeedForward(LayerNorm(x))     # each word is processed on its own
    The "x = x + ..." parts are residual connections: they let the signal and
    the gradient flow straight through, which makes deep stacks trainable.
    """

    def __init__(self, vocab_size, pad_id, embedding_dim, hidden_size, num_layers, dropout,
                 num_heads, max_positions):
        super().__init__()
        self.max_positions = max_positions
        # Word embedding: random, trainable, <pad> row fixed at zero (as in the RNN/LSTM).
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # Positional embedding: one learned 100-number vector per position 0..max_positions-1.
        # It is ADDED to the word vector so the model knows "this is word number 5".
        self.position_embedding = nn.Embedding(max_positions, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # One block. Despite PyTorch's name "EncoderLayer", with a causal mask it is
        # exactly a GPT-style decoder block (self-attention + feed-forward).
        #   nhead           number of attention heads (embedding_dim must divide by it)
        #   dim_feedforward width of the feed-forward layer (our config's hidden_size)
        #   norm_first=True LayerNorm BEFORE each sub-layer (more stable to train)
        block = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                           dim_feedforward=hidden_size, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        # Stack num_layers copies (each with its own weights). enable_nested_tensor=False
        # only silences a PyTorch warning about a speed trick we do not use.
        self.blocks = nn.TransformerEncoder(block, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(embedding_dim)  # pre-norm blocks need one last LayerNorm
        self.output = nn.Linear(embedding_dim, vocab_size)

    def hidden_states(self, input_ids):
        """The vector at every position just before the output layer (after the final LayerNorm):
        (batch, seq_len, embedding_dim)."""
        seq_len = input_ids.size(1)
        if seq_len > self.max_positions:
            raise ValueError(f"sequence length {seq_len} is longer than max_positions "
                             f"{self.max_positions}; raise options.max_positions in the config")
        positions = torch.arange(seq_len, device=input_ids.device)  # [0, 1, ..., seq_len-1]
        x = self.embedding(input_ids) + self.position_embedding(positions)  # (batch, seq_len, 100)
        x = self.dropout(x)

        # --- the causal mask, line by line ---------------------------------
        # 1. torch.ones(seq_len, seq_len): a square table of ones. Row i is "the word
        #    at position i that is asking"; column j is "the word at position j it may look at".
        # 2. dtype=torch.bool: make them True/False instead of 1.0.
        # 3. device=...: create it on the same device (CPU/GPU) as the data.
        # 4. torch.triu(..., diagonal=1): keep only the part strictly ABOVE the diagonal,
        #    set everything else to False. Above the diagonal means column j > row i,
        #    i.e. positions in the FUTURE of i.
        # 5. PyTorch reads True as "blocked": those scores are set to -infinity before
        #    the softmax, so the attention weight on a future word is exactly 0.
        #    Row 0 sees only word 0, row 1 sees words 0-1, ..., the last row sees all.
        #    (The diagonal itself stays False, so a word can look at itself.)
        future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
                            diagonal=1)
        x = self.blocks(x, mask=future)  # (batch, seq_len, 100)
        return self.final_norm(x)

    def forward(self, input_ids):
        return self.output(self.hidden_states(input_ids))  # (batch, seq_len, vocab_size)
