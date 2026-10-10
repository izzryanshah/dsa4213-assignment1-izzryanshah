"""Sanity-check the data pipeline with a trivial, UNTRAINED language model.

This script does not train anything. It builds the simplest possible
word-level language model (an embedding layer feeding straight into a
linear layer, with random/untrained weights) and measures its perplexity on
the validation set.

Why this is useful even though the model is untrained: with random weights,
the model has learned nothing about which words actually follow which, so
its predicted probability for the next word is close to *uniform* over the
whole vocabulary (every word looks equally likely). Perplexity is literally
"the effective number of equally-likely choices the model is confused
between" — for a perfectly uniform guess over V words, perplexity is
exactly V (see the long comment near the bottom for the maths). So:

    untrained perplexity ≈ vocabulary size

is the expected result. If it came out far from the vocabulary size, that
would point to a bug — e.g. padding leaking into the loss, token ids not
matching the vocabulary, or the model being wired up wrong — rather than
anything about how "good" the model is (it isn't trained, so "good" doesn't
apply yet). Once a real model is trained later, a MUCH lower perplexity is
what shows training actually worked.

Usage:
    python src/sanity_check.py --config configs/sanity_check.json
    python src/sanity_check.py --config configs/sanity_check.json --batch-size 64

Settings (seed, embedding size, batch size, max review length, ...) come from
the config file; see runconfig.py for how overrides work.
"""

import argparse
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from data import check_config_matches_data, load_processed_data
from runconfig import add_config_args, load_config, save_run

class BigramLM(nn.Module):
    """The simplest possible language model: predict the next token using
    only the current token's embedding (no memory of anything earlier).

    This still obeys the project rule that a language model may only use
    PRECEDING tokens to predict each target — it just uses the smallest
    possible amount of preceding context (one token), which is exactly what
    makes it a good sanity check rather than a real model to train.
    """

    def __init__(self, vocab_size, pad_id, embedding_dim):
        super().__init__()
        # padding_idx tells nn.Embedding to always keep the <pad> row at all
        # zeros and never update it — harmless here since nothing is
        # trained, but it is the correct habit for every later script.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len) token ids.
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        logits = self.linear(embedded)        # (batch, seq_len, vocab_size)
        return logits


def batched_reviews(reviews, batch_size):
    """Yield successive chunks of `batch_size` reviews at a time."""
    for start in range(0, len(reviews), batch_size):
        yield reviews[start : start + batch_size]


def compute_perplexity(model, reviews, pad_id, vocab_size, batch_size, device, max_length):
    """Average next-token cross-entropy (in nats), converted to perplexity,
    over every non-padding position in `reviews`.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_targets = 0

    # A few reviews run past 1000 tokens. With a 20,000-word vocabulary, the
    # (batch, seq_len, vocab_size) logits tensor for a batch padded up to
    # one of those outliers can reach several gigabytes, which is slow and
    # memory-hungry for what is meant to be a quick sanity check. Truncating
    # every review to `max_length` tokens bounds that worst case, and barely
    # changes the *average* loss since the vast majority of reviews are
    # already shorter than that.
    reviews = [r[:max_length] for r in reviews]

    # Review lengths are still uneven below that cap. pad_sequence pads
    # every review in a batch up to the LONGEST one in that batch, so a
    # longer review dragged into a batch of otherwise-short reviews wastes
    # padding. Sorting by length first means each batch only groups
    # similarly-long reviews together, keeping padding (and memory) small.
    # This changes nothing about the final average, since every review is
    # still visited exactly once.
    reviews = sorted(reviews, key=len)

    model.eval()
    with torch.no_grad():  # untrained and not training here, so no need for gradients
        for batch in batched_reviews(reviews, batch_size):
            # Each review is [t_0, t_1, ..., t_{L-1}] (ending in <eos>).
            # Predicting "the next token from the preceding one" means the
            # model sees t_0..t_{L-2} as input and must predict t_1..t_{L-1}.
            inputs = [torch.as_tensor(r[:-1], dtype=torch.long) for r in batch if len(r) > 1]
            targets = [torch.as_tensor(r[1:], dtype=torch.long) for r in batch if len(r) > 1]
            if not inputs:
                continue

            # Reviews in a batch have different lengths; pad_sequence stacks
            # them into one (batch, max_len) tensor, padding the short ones
            # with <pad> so they all line up.
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_id).to(device)

            logits = model(padded_inputs)  # (batch, max_len, vocab_size)
            # CrossEntropyLoss expects (N, C) vs (N,), so flatten the batch
            # and time dimensions together. ignore_index=pad_id above makes
            # it skip every padded position, as the project rules require.
            loss_sum = criterion(logits.view(-1, vocab_size), padded_targets.view(-1))

            num_real_targets = (padded_targets != pad_id).sum().item()
            total_loss += loss_sum.item()
            total_targets += num_real_targets

    avg_nll = total_loss / total_targets
    perplexity = math.exp(avg_nll)
    return perplexity, avg_nll, total_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check the data pipeline with an untrained LM.")
    add_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    opts = config["options"]  # processed_dir, max_length

    # Record the seed: it controls the model's random initial weights, so
    # the exact perplexity number is reproducible.
    print(f"Random seed: {config['seed']}")
    torch.manual_seed(config["seed"])

    # This untrained, 2-million-parameter model on 5,000 validation reviews
    # is tiny by deep-learning standards, so plain CPU is fast enough here
    # and keeps this sanity check simple and dependency-free.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_processed_data(opts["processed_dir"])
    check_config_matches_data(config, data)
    vocab_size = len(data.vocab)

    model = BigramLM(vocab_size, pad_id=data.vocab.pad_id, embedding_dim=config["embedding_dim"]).to(device)

    val_reviews = data.val.reviews()
    perplexity, avg_nll, num_targets = compute_perplexity(
        model, val_reviews, data.vocab.pad_id, vocab_size, config["batch_size"], device, opts["max_length"]
    )

    print(f"Vocabulary size: {vocab_size}")
    print(f"Untrained validation perplexity: {perplexity:.1f}")
    print(f"(average next-token loss: {avg_nll:.4f} nats, over {num_targets} non-padding tokens)")

    results_path = save_run(
        args.results_dir, run, "src/sanity_check.py", config,
        metrics={
            "val_perplexity_untrained": perplexity,
            "val_avg_nll_untrained": avg_nll,
            "val_num_target_tokens": num_targets,
        },
        details={"vocab_size_with_special_tokens": vocab_size},
    )
    print(f"Saved run record to {results_path}")


if __name__ == "__main__":
    main()
