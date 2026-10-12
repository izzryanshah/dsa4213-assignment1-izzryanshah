"""Short tests for src/models.py (tiny made-up inputs, no dataset needed).

Run with:   python3 tests/test_models.py --config configs/test_models.json

Checks, for the RNN, the LSTM and the Transformer: output shape, no leakage from future
tokens, the <pad> embedding row stays zero, and the embedding is trainable.
Saves a pass/fail record to results/test_models.json.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import LSTMLanguageModel, RNNLanguageModel, TransformerLanguageModel  # noqa: E402
from runconfig import add_config_args, load_config, save_run  # noqa: E402

PAD_ID, NUM_SPECIAL = 0, 3


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    config, run = load_config(parser.parse_args())
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    vocab_total = config["vocab_size"] + NUM_SPECIAL
    batch, seq = config["batch_size"], config["options"]["seq_len"]
    passed = []
    for name, cls in [("rnn", RNNLanguageModel), ("lstm", LSTMLanguageModel),
                      ("transformer", TransformerLanguageModel)]:
        extra = (config["options"]["num_heads"], config["options"]["max_positions"]) if name == "transformer" else ()
        model = cls(vocab_total, PAD_ID, config["embedding_dim"], config["hidden_size"],
                    config["layers"], config["dropout"], *extra)
        model.eval()  # dropout off, so outputs are deterministic
        x = torch.randint(NUM_SPECIAL, vocab_total, (batch, seq))

        logits = model(x)
        assert logits.shape == (batch, seq, vocab_total), (name, logits.shape)

        # No leakage: changing the LAST token must not change any earlier output.
        x2 = x.clone()
        x2[:, -1] = NUM_SPECIAL + (x2[:, -1] - NUM_SPECIAL + 1) % config["vocab_size"]
        assert torch.allclose(logits[:, :-1], model(x2)[:, :-1]), f"{name}: future token changed earlier output"
        assert not torch.allclose(logits[:, -1], model(x2)[:, -1]), f"{name}: last output ignores its own input"

        assert model.embedding.weight.requires_grad, f"{name}: embedding not trainable"
        assert model.embedding.weight[PAD_ID].abs().sum().item() == 0, f"{name}: <pad> row not zero"
        assert model.embedding.weight[NUM_SPECIAL:].abs().sum().item() > 0, f"{name}: embedding not initialised"

        # Gradients reach the embedding, and the <pad> row stays zero.
        model.train()
        model(x).sum().backward()
        assert model.embedding.weight.grad[NUM_SPECIAL:].abs().sum().item() > 0, f"{name}: no gradient to embedding"
        assert model.embedding.weight.grad[PAD_ID].abs().sum().item() == 0, f"{name}: <pad> row got a gradient"
        passed.append(name)

    print("All model tests passed:", ", ".join(passed))
    save_run("results", run, "tests/test_models.py", config, metrics={"passed": True, "models": passed})


if __name__ == "__main__":
    main()
