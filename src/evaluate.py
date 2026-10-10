"""One shared evaluation function for EVERY language model in this project.

The assignment compares neural language models against an n-gram model, and
that comparison is only fair if they are all scored on exactly the same
thing: the same reviews, cut the same way, with the same list of "target"
tokens being predicted. If each model had its own evaluation loop, it would
be easy for them to drift apart (one truncates reviews, another counts the
<eos> token, ...) and the perplexities would no longer be comparable.

So all of that logic lives here, ONCE, in `evaluate()`. The only thing that
differs between models is a tiny step: "given these input tokens and these
target tokens, what is the log-probability the model gave each target?"
That step is called a *scorer* and has this one-method interface:

    scorer.target_log_probs(inputs, targets) -> tensor of shape (batch, seq_len)

  * `inputs`  : (batch, seq_len) LongTensor of token ids the model may read.
  * `targets` : (batch, seq_len) LongTensor; targets[b, t] is the token that
                follows inputs[b, t] in the review (i.e. inputs shifted by 1).
  * returns   : out[b, t] = log P(targets[b, t] | inputs[b, 0..t]).
                Natural log. Values at <pad> positions are ignored.

The "no future leakage" rule is part of this contract: out[b, t] may depend
ONLY on inputs[b, 0..t], never on inputs[b, t+1:] or on the targets.

  * Neural models: wrap any `nn.Module` whose forward(input_ids) returns
    logits of shape (batch, seq_len, vocab_size). `evaluate()` does the
    wrapping for you (see `NeuralScorer`).
  * N-gram model: write a small class with a `target_log_probs` method that,
    for each position t, looks at the last (n-1) tokens of inputs[b, 0..t]
    and returns the log-probability it assigns to targets[b, t].

Because both kinds go through `evaluate()`, they are guaranteed to be scored
on the identical sequence of target tokens. To let you PROVE that, the result
includes `num_targets` and `target_checksum` (a fingerprint of the exact
target ids, in order): if two models' results show the same two values, they
were scored on exactly the same tokens.
"""

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class NeuralScorer:
    """Adapts a neural language model to the scorer interface above."""

    def __init__(self, model):
        self.model = model
        # Wherever the model's weights live (CPU or GPU), the input tensors
        # need to be moved there too.
        self.device = next(model.parameters()).device

    def target_log_probs(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(inputs)  # (batch, seq_len, vocab_size)
        # log_softmax turns raw scores into log-probabilities that sum (after
        # exp) to 1 over the vocabulary. We then pick out, at every position,
        # only the log-probability of the token that actually came next.
        # `.float()` keeps this accurate even if the model ran in half precision.
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # gather needs an index tensor with a trailing dimension, hence the
        # unsqueeze(-1) / squeeze(-1) pair.
        picked = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return picked.cpu()


def evaluate(model, split, pad_id, batch_size, max_length):
    """Score `model` on `split`; return average next-token cross-entropy and perplexity.

    Args:
        model: either an `nn.Module` (forward(input_ids) -> logits) or any
            object with a `target_log_probs(inputs, targets)` method (see the
            module docstring), e.g. the n-gram model.
        split: a `SplitData` from data.py (anything with a `.reviews()` method),
            or directly a list of reviews, each a list/array of token ids
            ending in <eos>.
        pad_id: the id of the <pad> token. Padded positions are excluded.
        batch_size: reviews per batch (from the run's config). Only affects
            speed/memory, never the result.
        max_length: if given, every review is cut to its first `max_length`
            tokens BEFORE scoring. Pass the same value for every model you
            want to compare. None means no cutting. Both
            batch_size and max_length are required (no hidden defaults): the
            calling script must pass them from its config.

    Returns a dict with:
        avg_cross_entropy: mean of -log P(target) over all real targets, in nats.
        perplexity:        exp(avg_cross_entropy).
        num_targets:       how many target tokens were scored.
        target_checksum:   fingerprint of the exact target ids scored, in order.
    """
    scorer = NeuralScorer(model) if isinstance(model, nn.Module) else model
    reviews = split.reviews() if hasattr(split, "reviews") else split

    if max_length is not None:
        reviews = [r[:max_length] for r in reviews]
    # A review needs at least 2 tokens to have one (input, target) pair.
    reviews = [r for r in reviews if len(r) > 1]
    # Sorting by length keeps each batch's padding small. It is a stable sort
    # of a list that depends only on the data, so every model sees reviews in
    # the same order. The average is unaffected by order anyway.
    reviews = sorted(reviews, key=len)

    was_training = isinstance(model, nn.Module) and model.training
    if isinstance(model, nn.Module):
        model.eval()  # switch off dropout etc. so scoring is deterministic

    total_nll = 0.0  # running sum of -log P(target), in float64 for accuracy
    num_targets = 0
    fingerprint = hashlib.sha256()

    with torch.no_grad():  # evaluation only: no gradients needed
        for start in range(0, len(reviews), batch_size):
            batch = reviews[start : start + batch_size]
            # Each review is [t_0, ..., t_{L-1}]. The model reads t_0..t_{L-2}
            # and must predict t_1..t_{L-1}: input at position i, target is
            # the token right after it. Each target thus only ever has
            # EARLIER tokens available as input.
            inputs = pad_sequence(
                [torch.as_tensor(r[:-1], dtype=torch.long) for r in batch],
                batch_first=True, padding_value=pad_id,
            )
            targets = pad_sequence(
                [torch.as_tensor(r[1:], dtype=torch.long) for r in batch],
                batch_first=True, padding_value=pad_id,
            )

            log_probs = scorer.target_log_probs(inputs, targets)  # (batch, seq_len)

            # The mask is True exactly where a real (non-padding) target is.
            # Everything below uses it, so padding can never affect the result.
            mask = targets != pad_id
            total_nll -= log_probs[mask].double().sum().item()
            num_targets += int(mask.sum().item())
            fingerprint.update(targets[mask].numpy().astype("int64").tobytes())

    if was_training:
        model.train()

    avg_cross_entropy = total_nll / num_targets
    return {
        "avg_cross_entropy": avg_cross_entropy,
        "perplexity": math.exp(avg_cross_entropy),
        "num_targets": num_targets,
        "target_checksum": fingerprint.hexdigest(),
    }
