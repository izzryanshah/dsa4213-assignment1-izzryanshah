"""Generate text from a language model: continue each prompt, one word at a time.

How generation works
--------------------
A language model, given the words so far, gives a probability for EVERY word
in the vocabulary being the next one. To write text we repeat:

    1. ask the model for those next-word probabilities,
    2. pick ONE word at random, weighted by those probabilities ("sampling"),
    3. add that word to the text and go back to step 1,

until we have produced the maximum number of new words or the model produces
the end-of-review token <eos>.

Temperature and top-k: the two dials that reshape step 2
--------------------------------------------------------
Say the model thinks the next word is one of four, with probabilities
[0.50, 0.30, 0.15, 0.05].

TEMPERATURE re-weights the probabilities before picking (each probability is
raised to the power 1/temperature, then everything is rescaled to add up to 1):
  * temperature 1.0 leaves them unchanged: [0.50, 0.30, 0.15, 0.05].
    You sample from exactly what the model believes.
  * LOWER than 1 (e.g. 0.5) sharpens: [0.685, 0.247, 0.062, 0.007]. The likely
    words become even more likely, so text is safer and more repetitive.
    As temperature -> 0 this turns into "always pick the single likeliest word".
  * HIGHER than 1 (e.g. 2.0) flattens: [0.379, 0.294, 0.207, 0.120]. Unlikely
    words get a bigger chance, so text is more varied and more often nonsense.
    As temperature -> infinity every word becomes equally likely.

TOP-K throws away everything except the k likeliest words, then rescales what
is left to add up to 1. With k = 2 the example above becomes [0.625, 0.375,
0, 0]. This cuts off the "long tail": with a 20,000-word vocabulary, thousands
of very unlikely words together hold a few percent of the probability, so
without top-k they get picked now and then and produce odd words. A SMALL k
gives safer, less varied text; a LARGE k (or k = 0, which means "off") allows
more variety and more mistakes.

The two combine: temperature reshapes the probabilities, top-k then removes
the tail. Order used here: banned words are removed first, then temperature,
then top-k (so top-k means "the k likeliest ALLOWED words").

Reproducibility: every (prompt, sample number) gets its own random generator
seeded from the fixed seed, the sample number and the prompt text. The same
settings give identical text, and adding or reordering prompts does not change
the text of the others.

One function for every model
----------------------------
`generate()` only needs an object with one method:

    model.next_token_log_probs(context_ids) -> array of length vocab_size

where `context_ids` is the list of ALL token ids so far (prompt + generated)
and the result holds the natural-log probability of each possible next word,
using only those preceding tokens. The n-gram model provides this method
itself (it looks at the last two ids). Any neural `nn.Module` whose
forward(input_ids) returns logits of shape (batch, seq_len, vocab_size) is
wrapped automatically by `NeuralNextToken`, which reads the logits at the LAST
position. So future LSTM/Transformer models need no changes here beyond adding
a loader for their saved weights in `build_model`.

Usage:
    python src/generate.py --config configs/generations_ngram.json
    python src/generate.py --config configs/generations_ngram.json \\
        --set options.temperature=0.7 --set options.top_k=10 --run-name generations_ngram_t07
Prompts are read from the file named by options.prompts_path (see read_prompts).
The results go to results/<run>.json; the config is named generations_<model>.json
so the output is results/generations_<model>.json.
"""

import argparse
import json
import random
import zlib

import numpy as np
import torch
import torch.nn as nn

from data import check_config_matches_data, clean_and_tokenize, load_processed_data
from ngram import KneserNeyTrigram, build_events
from runconfig import add_config_args, load_config, save_run


# ------------------------------------------------------------------ model wrapper
class NeuralNextToken:
    """Adapts a neural language model to the `next_token_log_probs` interface."""

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()  # switch off dropout so generation is not randomly noisy

    def next_token_log_probs(self, context_ids):
        ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)  # (1, seq_len)
        with torch.no_grad():  # generating, not training: no gradients needed
            logits = self.model(ids)[0, -1]  # scores for the NEXT word, after the last token
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------ sampling
def next_token_probs(log_probs, temperature, top_k, banned_ids):
    """The probabilities words are actually sampled from, after the three dials.

    Order: ban words -> temperature -> top-k -> rescale to sum to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    # log(p) / T, then softmax, is the same as p**(1/T) rescaled -- see the docstring.
    scores = np.asarray(log_probs, dtype=np.float64) / temperature
    scores[list(banned_ids)] = -np.inf  # a banned word can never be picked
    if top_k > 0:
        allowed = np.isfinite(scores).sum()
        if top_k < allowed:
            # indices of everything EXCEPT the k highest scores
            drop = np.argpartition(scores, len(scores) - top_k)[: len(scores) - top_k]
            scores[drop] = -np.inf
    scores -= scores.max()  # subtract the max before exp() so it cannot overflow
    probs = np.exp(scores)
    return probs / probs.sum()


def generate(model, prompt_ids, max_new_tokens, temperature, top_k, banned_ids, eos_id, stop_at_eos, rng):
    """Continue `prompt_ids` by up to `max_new_tokens` tokens.

    Returns (new_ids, ended_with_eos). When the model produces <eos> and
    `stop_at_eos` is true, generation stops and the <eos> is NOT included in
    `new_ids`.
    """
    context = list(prompt_ids)
    new_ids = []
    for _ in range(max_new_tokens):
        log_probs = model.next_token_log_probs(context)
        probs = next_token_probs(log_probs, temperature, top_k, banned_ids)
        next_id = int(rng.choice(len(probs), p=probs))
        if stop_at_eos and next_id == eos_id:
            return new_ids, True
        new_ids.append(next_id)
        context.append(next_id)
    return new_ids, False


def make_rng(seed, sample_index, prompt_text):
    """A random generator unique to (seed, sample number, prompt) but reproducible."""
    return np.random.default_rng([seed, sample_index, zlib.crc32(prompt_text.encode("utf-8"))])


# ------------------------------------------------------------------ prompts
def read_prompts(path):
    """One prompt per line. Blank lines and lines starting with '#' are skipped."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        raise SystemExit(f"Prompts file '{path}' not found. Create it with one prompt per line "
                         "(or point options.prompts_path at another file).")
    prompts = [line for line in lines if line and not line.startswith("#")]
    if not prompts:
        raise SystemExit(f"No prompts found in '{path}'.")
    return prompts


def encode_prompt(prompt, vocab):
    """Prompt text -> token ids, with the same cleaning/tokenizing as the training data."""
    tokens = clean_and_tokenize(prompt)
    if not tokens:
        raise SystemExit(f"Prompt '{prompt}' has no tokens after cleaning.")
    ids = vocab.encode(tokens)
    return tokens, ids


# ------------------------------------------------------------------ building models
def build_model(config, data):
    """Return an object with next_token_log_probs for the model named in the config."""
    opts = config["options"]
    if opts["model"] == "ngram":
        # The discounts were tuned on the validation split by src/ngram.py and saved
        # in its run record; fitting the counts again takes about a second.
        with open(opts["ngram_results_path"]) as f:
            ngram_run = json.load(f)
        if ngram_run["config"]["vocab_size"] != config["vocab_size"]:
            raise SystemExit("The n-gram run record was made with a different vocab_size than this config.")
        d = ngram_run["details"]["best_discounts"]
        model = KneserNeyTrigram(len(data.vocab), discounts=(d["d3"], d["d2"], d["d1"]))
        model.fit(*build_events(data.train, model.bos_id))
        source = {"ngram_results_path": opts["ngram_results_path"], "ngram_run_date": ngram_run["date"],
                  "ngram_run_git_commit": ngram_run["git_commit"], "discounts": d}
        return model, source
    raise SystemExit(f"Model '{opts['model']}' is not available yet. Neural models need a trained, saved "
                     "model first; add a loader for it here (wrap it with NeuralNextToken).")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text continuations of prompts.")
    add_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    opts = config["options"]
    seed = config["seed"]
    print(f"Random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_processed_data(opts["processed_dir"])
    check_config_matches_data(config, data)
    vocab = data.vocab

    model, model_source = build_model(config, data)
    if isinstance(model, nn.Module):
        model = NeuralNextToken(model)

    banned_ids = [vocab.token_to_id[t] for t in opts["banned_tokens"]]
    prompts = read_prompts(opts["prompts_path"])

    generations = []
    for prompt in prompts:
        prompt_tokens, prompt_ids = encode_prompt(prompt, vocab)
        for sample_index in range(opts["num_samples"]):
            rng = make_rng(seed, sample_index, prompt)
            new_ids, ended = generate(model, prompt_ids, opts["max_new_tokens"], opts["temperature"],
                                      opts["top_k"], banned_ids, vocab.eos_id, opts["stop_at_eos"], rng)
            new_tokens = [vocab.id_to_token[i] for i in new_ids]
            generations.append({
                "prompt": prompt,
                "sample": sample_index,
                "prompt_tokens": prompt_tokens,
                "prompt_unk_tokens": sum(1 for i in prompt_ids if i == vocab.unk_id),
                "new_tokens": new_tokens,
                "num_new_tokens": len(new_tokens),
                "ended_with_eos": ended,
                "continuation": " ".join(new_tokens),
                "text": " ".join(prompt_tokens + new_tokens),
            })
            print(f"\n[{prompt}] (sample {sample_index})\n  {generations[-1]['continuation']}"
                  f"{'  <eos>' if ended else ''}")

    decoding = {
        "temperature": opts["temperature"],
        "top_k": opts["top_k"],
        "max_tokens": opts["max_new_tokens"],
        "seed": seed,
        "num_samples": opts["num_samples"],
        "banned_tokens": opts["banned_tokens"],
        "stop_at_eos": opts["stop_at_eos"],
    }
    results_path = save_run(
        args.results_dir, run, "src/generate.py", config,
        metrics={
            "num_prompts": len(prompts),
            "num_generations": len(generations),
            "num_ended_with_eos": sum(g["ended_with_eos"] for g in generations),
            "mean_new_tokens": float(np.mean([g["num_new_tokens"] for g in generations])),
        },
        details={"model": opts["model"], "prompts_path": opts["prompts_path"], **model_source},
        extra={"decoding": decoding, "generations": generations},
    )
    print(f"\nSaved generations to {results_path}")


if __name__ == "__main__":
    main()
