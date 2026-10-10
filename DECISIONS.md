# Design decisions

Audit written 2026-10-10 from the git history, every script, `CLAUDE.md`, `TOOL_USE.md` and `results/`.
From now on, every new design choice is added here **after** the user picks from 2-3 offered options (see `CLAUDE.md`).

**How to read an entry**
- **Value / Where**: the value, and where it is set. `configs/x.json:N` is the config line; `src/x.py:N` is the code line (line numbers as of 2026-10-10).
- **Chosen by**: who made the choice.
  - *User, CLAUDE.md* = one of the standing rules the user wrote in `CLAUDE.md`.
  - *User, prompt (date)* = the user's instruction in a prompt I can see.
  - *Assistant default* = chosen by Claude without the user being offered options.
  - *Unknown (earlier session)* = made in a session whose prompts I cannot see; the user may have specified it. Treated as NEEDS REVIEW.
- **Status**: `CONFIRMED` (user explicitly instructed it) or `NEEDS REVIEW` (made without an explicit user instruction, or cannot be verified).
- "Reason" for assistant defaults is the reason the assistant gave in code comments or `TOOL_USE.md` at the time, not a reason from the user.

---

## A. Data and text processing

### A1. Dataset file
- **Value:** IMDB Dataset of 50K Movie Reviews, `data/IMDB Dataset.csv` (Kaggle, 50,000 reviews, 25,000 positive / 25,000 negative).
- **Where:** `configs/data.json:16`, `configs/inspect_data.json:16`; documented in `README.md:5-19`.
- **Chosen by:** User (downloaded it manually; `README.md` Kaggle link). The assignment's required dataset is not recorded in the repo.
- **Alternatives:** other review datasets (not recorded as considered).
- **Reason:** not recorded. `README.md:11` still contains an unfilled placeholder `<PASTE THE EXACT KAGGLE DATASET URL YOU USED HERE>` next to the real link on line 19.
- **Status:** CONFIRMED (dataset), README placeholder still to fix.

### A2. Subsample size: none for counting-based work; decided per-script for future neural training
- **Value:** `null` (no subsampling) for `data.py`/vocabulary/n-gram, which all use the full 50,000 reviews. Whether to subsample is deferred to each future neural-LM training script individually, not fixed globally.
- **Where:** `configs/data.json:13`; handled at `src/data.py:280-287`.
- **Chosen by:** User, prompt (2026-10-10, reviewing NEEDS REVIEW): picked "keep full data for counting models, subsample only for neural training later" over keeping full data for everything, or subsampling to ~12,500 globally now.
- **Alternatives considered:** (1) keep full data for everything, including later neural training; (2) subsample to ~12,500 reviews now, globally, matching `inspect_data.py`'s suggestion.
- **Reason:** the n-gram and vocabulary benefit from more data and already ran fine on the full set; Colab-GPU trainability is a concern specific to the (not yet written) neural LM scripts, so that call is better made per-script when their actual memory/time cost is known.
- **Status:** CONFIRMED

### A3. Word budget for the subsample suggestion
- **Value:** 3,000,000 words; suggestion rounded down to a multiple of 500; shows 3 example reviews, 500 characters each.
- **Where:** `configs/inspect_data.json:17-20`; used at `src/inspect_data.py:97-103`.
- **Chosen by:** Unknown (earlier session) for the 3M figure; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep as-is" over changing the number.
- **Alternatives considered:** a different word-budget number.
- **Reason:** per A2, this script's suggestion is now effectively superseded (no global subsampling is being used) — harmless to leave, not worth further tuning.
- **Status:** CONFIRMED

### A4. Word counting in the inspect script
- **Value:** rough whitespace-split word count (not the project tokenizer).
- **Where:** `src/inspect_data.py:68`.
- **Chosen by:** Assistant default, originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep as-is" over switching to the real tokenizer.
- **Alternatives considered:** use the real tokenizer for an exact match with `data.py`'s count.
- **Reason:** low impact (only affects a now-superseded printed suggestion, per A3); not worth the slower real-tokenizer pass for this diagnostic script.
- **Status:** CONFIRMED

### A5. Text cleaning: strip HTML tags, lowercase
- **Value:** every `<...>` tag (e.g. `<br />`) is replaced by a space; text is lowercased; nothing else is cleaned (no number, URL, HTML-entity, or accent handling).
- **Where:** `src/data.py:58` (tag regex), `src/data.py:71` (`.lower()`).
- **Chosen by:** User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over adding number normalization or broader cleaning (HTML entities, etc.).
- **Alternatives considered:** keep `<br />` as a token; add a `<num>` token for digit runs; decode HTML entities and other broader normalization.
- **Reason:** simplicity; matches the literal instruction; broader cleaning would add more assistant-chosen judgment calls for little expected benefit on this task.
- **Status:** CONFIRMED

### A6. Tokenisation rule
- **Value:** word-level, using the regex `\w+|[^\s\w]` — words (Unicode-aware) and each punctuation character as its own token. (So "don't" becomes `don`, `'`, `t`.)
- **Where:** `src/data.py:65`.
- **Chosen by:** *Word-level* = User, CLAUDE.md:6. *The specific regex* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over keeping numbers as single tokens or switching to a library tokenizer (e.g. NLTK).
- **Alternatives considered:** whitespace split; NLTK `word_tokenize`; dropping punctuation; keeping numbers as single tokens; keeping contractions whole.
- **Reason:** simplest, fully transparent for a from-scratch PyTorch learner (CLAUDE.md: prefer simple code); matches the literal instruction to keep punctuation as separate tokens.
- **Status:** CONFIRMED

### A7. Special tokens and their ids
- **Value:** `<pad>`=0, `<unk>`=1, `<eos>`=2.
- **Where:** `src/data.py:50-51`.
- **Chosen by:** `<pad>` and `<unk>` existing = User, CLAUDE.md:6. Their ids and the extra `<eos>` token = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over adding a `<bos>` token too.
- **Alternatives considered:** also adding a `<bos>` start token (would unify with the n-gram's separate invisible BOS symbol, D5, but requires touching `data.py`, `sanity_check.py`, and `ngram.py`).
- **Status:** CONFIRMED

### A8. End-of-document marker
- **Value:** one `<eos>` token appended to the end of every review; no start-of-review token in the data. `<eos>` is a normal predicted token for language models.
- **Where:** `src/data.py:177`.
- **Chosen by:** User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep current convention" over adding a symmetric `<bos>`.
- **Alternatives considered:** no end marker; adding `<bos>` as well (A7).
- **Reason:** simple; already used consistently by `sanity_check.py` and `ngram.py` (both score `<eos>` as an ordinary target); lets a language model learn where reviews end.
- **Status:** CONFIRMED

### A9. Vocabulary: size, source, and rule
- **Value:** the 20,000 most frequent words of the TRAINING split, plus the 3 special tokens (20,003 total); no minimum-frequency cut; everything else maps to `<unk>`. Allowed range enforced: 10,000-20,000.
- **Where:** `configs/data.json:12` (size); `src/data.py:149-165` (built from training reviews only); `src/data.py:260` (range check).
- **Chosen by:** *One shared vocabulary with `<unk>`/`<pad>`* = User, CLAUDE.md:6. *Size 20,000* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 20,000" over reducing to 10,000 or another value. *Training-only source* = treated as settled (standard practice to avoid leaking val/test information into the vocabulary; not re-litigated as a genuine alternative).
- **Alternatives considered:** 10,000 (smaller, more `<unk>`, faster/smaller model); a specific value in between.
- **Reason:** lower `<unk>` rate (train 2.07%, val/test ~2.2%) outweighed the smaller/faster model from a reduced vocabulary, for this dataset size.
- **Status:** CONFIRMED

### A10. Split ratios and stratification
- **Value:** 80% train / 10% validation / 10% test, stratified by sentiment label, done before the vocabulary is built.
- **Where:** `configs/data.json:18-19`; `src/data.py:119-134`.
- **Chosen by:** *Fixed splits, created once, saved and reused* = User, CLAUDE.md:7. *The 80/10/10 ratio and stratification* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 80/10/10, stratified" over 90/5/5 or a custom ratio.
- **Alternatives considered:** 90/5/5 (more train data, noisier val/test); a custom ratio.
- **Reason:** standard default; stratification keeps the 50/50 label balance in every split; 5,000-review val/test sets are large enough for stable tuning/evaluation at this dataset size.
- **Status:** CONFIRMED

### A11. Random seed
- **Value:** 0, in every config (data split, example picking, model initialisation, n-gram).
- **Where:** `configs/data.json:14`, `configs/inspect_data.json:14`, `configs/sanity_check.json:14`, `configs/ngram.json:14`, `configs/test_evaluate.json`.
- **Chosen by:** *Setting and recording a seed* = User, CLAUDE.md:8. *The value 0* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep seed 0" over picking a different value; open to multi-seed averaging later for final neural models without redoing these results.
- **Alternatives considered:** a different single seed; multi-seed averaging (deferred, not rejected).
- **Status:** CONFIRMED

---

## B. Sanity-check model

### B1. Untrained bigram sanity model
- **Value:** bigram language model (embedding -> linear layer over the vocabulary), randomly initialised with PyTorch's default init, not trained.
- **Where:** `src/sanity_check.py:43-67`; `configs/sanity_check.json:2,6,7`.
- **Chosen by:** Unknown (earlier session; the user asked for a sanity check of the data pipeline) for the original choice; User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the bigram sanity check" over also adding a uniform-distribution baseline.
- **Alternatives considered:** an RNN; a uniform-distribution baseline (judged to add little, since the bigram already lands close to that number).
- **Status:** CONFIRMED

### B2. Embedding dimension 100
- **Value:** 100.
- **Where:** `configs/sanity_check.json:5`.
- **Chosen by:** User, CLAUDE.md:9 ("All neural models use embedding dimension 100, to match GloVe 100d").
- **Status:** CONFIRMED

### B3. Sanity-check truncation and batch size
- **Value:** reviews truncated to 256 tokens; batch size 32; reviews sorted by length before batching.
- **Where:** `configs/sanity_check.json:9,17`; `src/sanity_check.py:89,98`.
- **Chosen by:** Assistant default, introduced as a performance fix (a padded batch with a very long review made the logits tensor several GB). User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 256/32" after seeing the verified <0.2% effect on the result.
- **Alternatives considered:** no truncation with smaller batches; a different truncation length or batch size.
- **Status:** CONFIRMED

### B4. Loss ignores padding, uses only preceding tokens
- **Value:** `CrossEntropyLoss(ignore_index=pad_id)` in the sanity check; masking of padding in `evaluate()`; each target predicted from earlier tokens only.
- **Where:** `src/sanity_check.py:78`; `src/evaluate.py:138`.
- **Chosen by:** User, CLAUDE.md:10-11.
- **Status:** CONFIRMED

---

## C. Evaluation function (`src/evaluate.py`)

### C1. What counts as a target token
- **Value:** for a review `[t0 ... tL-1]` the model reads `t0..tL-2` and is scored on `t1..tL-1`. So the first token of each review is never scored, and `<eos>` and `<unk>` ARE scored as ordinary targets. Reviews with fewer than 2 tokens are skipped.
- **Where:** `src/evaluate.py:104` (skip), `:126`, `:130` (shift), `:138` (padding mask).
- **Chosen by:** Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over excluding `<eos>`/`<unk>` from scoring.
- **Alternatives considered:** score `t0` too (needs a start symbol); exclude `<eos>` or `<unk>` targets; per-word rather than per-token normalisation.
- **Reason:** simplest definition consistent with "previous tokens only" and with no start token in the data; already used consistently by both models.
- **Status:** CONFIRMED

### C2. Perplexity definition and scoring truncation
- **Value:** perplexity = exp(mean negative log-likelihood per scored token, natural log). No truncation by default (`max_length` is passed explicitly by each caller); currently sanity_check.py uses 256, ngram.py uses null (full reviews) — their numbers are not directly comparable to each other as a result.
- **Where:** `src/evaluate.py:149`; `configs/ngram.json:17` (`max_length`: null), `configs/sanity_check.json:17` (256).
- **Chosen by:** Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep per-script truncation as-is for now" — each script's number stands alone; when comparing two real models later, the same `max_length` must be passed to both (already supported via config).
- **Alternatives considered:** standardizing one `max_length` across every future model-comparison run.
- **Status:** CONFIRMED (policy for future comparisons: pass matching `max_length` explicitly, rather than a fixed global standard)

### C3. Scorer interface
- **Value:** all models are scored by one `evaluate()`; neural models via `NeuralScorer` (log-softmax + gather), the n-gram model via its own `target_log_probs`. A SHA-256 fingerprint of the scored targets (`target_checksum`) is saved so two models' results can be shown to use identical targets.
- **Where:** `src/evaluate.py:49-70` (NeuralScorer), `:116` (fingerprint).
- **Chosen by:** Assistant default, in response to the user's prompt (2026-10-10) that both models be scored on exactly the same targets. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the shared evaluate() + checksum design" over dropping the checksum.
- **Alternatives considered:** dropping the checksum (less code, but loses the automatic proof of identical targets).
- **Status:** CONFIRMED

---

## D. N-gram model (`src/ngram.py`)

### D1. N-gram order
- **Value:** trigram (2 tokens of context).
- **Where:** `configs/ngram.json:7`; enforced at `src/ngram.py:277-278`.
- **Chosen by:** User, prompt (2026-10-10: "a trigram language model").
- **Status:** CONFIRMED

### D2. Smoothing method
- **Value:** interpolated Kneser-Ney (absolute discounting; lower orders use continuation counts).
- **Where:** `src/ngram.py:219-241`.
- **Chosen by:** The user allowed "interpolated Kneser-Ney smoothing (or linear interpolation of unigram, bigram and trigram)". The choice between the two was Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep Kneser-Ney" over switching to or also computing linear interpolation.
- **Alternatives considered:** linear interpolation of unigram/bigram/trigram estimates with weights tuned on validation (would let the two smoothing methods be compared, but needs new code and a rerun); modified Kneser-Ney (three discounts per order); Witten-Bell; add-k.
- **Reason:** Kneser-Ney is the standard strong smoothing method for n-gram language models.
- **Status:** CONFIRMED

### D3. One discount per order (D3, D2, D1)
- **Value:** three separate absolute discounts (trigram, bigram, unigram level), each tuned. Chosen values: D3=0.9, D2=0.8, D1=0.1.
- **Where:** `src/ngram.py:219-241` (formulas), `:255-262` (tuning); chosen values are saved in `results/ngram.json` under `details.best_discounts`.
- **Chosen by:** Per-order structure = Assistant default, originally. The values = result of the validation grid search the user asked for ("Tune any smoothing or interpolation weights on the VALIDATION set only"). User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 3 discounts as-is, note D1 is uninformative" over a single shared discount or the closed-form n1/(n1+2n2) estimator.
- **Alternatives considered:** one shared discount for all three orders; the closed-form estimate D = n1/(n1 + 2·n2); modified KN with count-dependent discounts.
- **Note on D1:** validation cross-entropy is identical to 6 decimals for every D1 in the grid. Mechanism: 20,002 of the 20,003 vocabulary entries appear in the continuation-count table, so the unigram-level discount and the uniform-floor term almost exactly cancel. The reported D1=0.1 is therefore just the first value in the grid (ties go to the first minimum), not a meaningful optimum — kept and stated as-is (a legitimate finding) rather than disguised.
- **Status:** CONFIRMED

### D4. Discount grid and how it was tuned
- **Value:** exhaustive grid over D3, D2, D1 ∈ {0.1, 0.2, ..., 0.9, 0.95} (1,000 combinations), minimising validation cross-entropy. The test split is evaluated once, after the discounts are fixed.
- **Where:** `configs/ngram.json:18`; `src/ngram.py:255-262` (grid search), `:304-306` (tuning on validation), `:316-318` (test evaluated after).
- **Chosen by:** Validation-only tuning = User, prompt (2026-10-10) and CLAUDE.md:7. The grid values = Assistant default.
- **History:** the first run used {0.5, ..., 0.95}; best D1 sat on the lower edge, so the assistant widened the grid to 0.1-0.95 and re-ran. The assistant had already seen the test number from the first run; the widening was prompted by the validation edge (the test number was identical in both runs). Logged in `TOOL_USE.md`.
- **Alternatives considered:** coarser/finer grid; coordinate search; continuous optimiser; redoing the tuning cleanly (validation-only, grid fixed in advance) before ever looking at test again.
- **User's decision (2026-10-10, reviewing NEEDS REVIEW):** "Accept it — test number was unaffected, keep current results." Explicitly signed off as a flagged incident: the test perplexity (116.5) was identical across both grid runs, so no actual leakage occurred in this instance, but the sequence (seeing test, then revising a validation-tuning choice) is acknowledged as a methodological risk that happened and is on the record rather than quietly accepted.
- **Status:** CONFIRMED (accepted with the risk explicitly logged, not silently resolved)

### D5. Start-of-review context symbol
- **Value:** one invisible start symbol (id = vocabulary size) precedes each review as context for the first scored token; never predicted. In training, only predictions of tokens 1..L-1 are counted.
- **Where:** `src/ngram.py:156` (symbol), `:99-118` (training events), `:117` (context at review start).
- **Chosen by:** Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the invisible BOS symbol" over backing off to the bigram for the first token.
- **Alternatives considered:** no start symbol (back off to the bigram for the first target); two start symbols; counting `t0` as a predicted token.
- **Status:** CONFIRMED

### D6. Uniform floor over the vocabulary
- **Value:** the unigram level is interpolated with a uniform distribution over all `vocab_size` = 20,003 entries (including `<pad>`, which is never a target).
- **Where:** `src/ngram.py:222-224`.
- **Chosen by:** Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the uniform floor over all 20,003 entries" over excluding `<pad>`.
- **Alternatives considered:** exclude `<pad>` from the floor (judged negligible: 1 part in 20,003); a separate `<unk>` rule; no floor (risking zero probabilities).
- **Reason:** guarantees every token gets nonzero probability in every context, so perplexity is finite; excluding `<pad>` would have negligible effect.
- **Status:** CONFIRMED

### D7. What "model size" and "training time" mean
- **Value:** size = number of distinct stored trigram + bigram + unigram entries (6,445,342), with the context-summary rows reported separately; lower orders store continuation counts, not raw counts. Training time = time to count n-grams from the training split, excluding data loading and discount tuning (tuning time is reported separately).
- **Where:** `src/ngram.py:190-201` (size), `:295-298` (timing).
- **Chosen by:** Assistant default (interpretation of the user's "training time and model size (number of stored n-gram counts)"), originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep these definitions" over folding tuning time into training time.
- **Alternatives considered:** count raw counts at every order; include context tables; include tuning in training time (would conflate counting and hyperparameter search).
- **Status:** CONFIRMED

### D8. Evaluation batch size
- **Value:** 32 (speed only; verified not to change the result).
- **Where:** `configs/ngram.json:9`. **Chosen by:** Assistant default, originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep 32" (trivial, no effect on results). **Status:** CONFIRMED

---

## E. Record-keeping and tooling

### E1. Config files: JSON, one shared schema
- **Value:** JSON files in `configs/`, all containing the same 13 keys (null where not applicable) plus an `options` block.
- **Where:** `configs/*.json`; `src/runconfig.py:53-68`.
- **Chosen by:** JSON and shared schema = User, prompt (2026-10-10, selected from options offered). The `options` block for script-specific settings = Assistant default, originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep the options block" over formalizing per-script schemas.
- **Alternatives considered:** YAML, Python dict module; per-script keys, shared core + script section; formalized (non-free-form) per-script option schemas.
- **Status:** CONFIRMED (format, schema, and `options` block)

### E2. Run records
- **Value:** every run writes `results/<run>.json` with: config used, seed, git commit (+ a flag for uncommitted code), hardware (CPU or GPU name), date, per-epoch losses, metrics, details.
- **Where:** `src/runconfig.py:144-224`. **Chosen by:** User, prompts (2026-10-10).
- **Note:** `git_dirty` is `true` in all current records because the code is not committed; the stored hash identifies the last commit (8764c2a), not the code that actually ran.
- **Status:** CONFIRMED

### E3. Python used to run scripts
- **Value:** `/usr/local/bin/python3` (the default `python` on this machine, from Anaconda, has no PyTorch).
- **Chosen by:** Assistant default (environment workaround), originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep using /usr/local/bin/python3 as-is" over setting up a dedicated project virtual environment.
- **Alternatives considered:** a dedicated venv with `requirements.txt` pinned (cleaner isolation, one-time setup cost, no effect on existing results).
- **Status:** CONFIRMED

---

## F. Generation

### F1. Generation prompts
- **Value:** three prompts: `this movie was one of the best` / `i was really disappointed by the acting and` / `the story follows a young man who`.
- **Where:** `prompts.txt:5`, `:8`, `:11` (reasons are in the `#` comment line under each prompt).
- **Chosen by:** User, prompt (2026-10-10). The prompts were suggested in a chat response that the user pasted and adopted; no prompt-reading or generation script exists yet.
- **Alternatives:** not recorded.
- **Reason (as given with the prompts):** all three are lowercase and use common words, so they match the preprocessing and should not hit `<unk>` (checked 2026-10-10: every token is in the 20,003-entry vocabulary). Each tests something different: (1) a positive opening — does the continuation stay positive or drift into a contradiction; (2) a negative opening — the same test in the other direction, and ending on "and" forces the model to continue a clause mid-thought; (3) a neutral plot opening — reviews often summarise plots, so this tests whether a model can keep a short story coherent (does "who" lead to something sensible, does the subject stay consistent).
- **Status:** CONFIRMED

---

### G1. Default decoding settings (2026-10-11)
- **Value:** temperature 1.0; top-k 40; maximum new tokens 50; 1 sample per prompt; never sample `<pad>` or `<unk>`; stop when the model produces `<eos>`; seed 0 (the project seed, see A11).
- **Where:** `configs/generations_ngram.json:20-25`; used in `src/generate.py:103-142`. The same values are meant to be used for every model so outputs are comparable.
- **Chosen by:** User, prompt (2026-10-11), each picked from options offered.
- **Alternatives offered:** temperature 1.0 / 0.8 / 0.5; top-k 40 / 10 / off; length 50 tokens x 1 sample / 50 x 3 / 100 x 1; special tokens: ban `<pad>`+`<unk>` and stop at `<eos>` / ban `<pad>` only and stop at `<eos>` / ban all three and never stop early.
- **Reason:** user selected the options (1.0 keeps the model's own distribution, so models are compared as they are; top-k 40 removes the rare nonsense words; ban `<pad>`/`<unk>` keeps text readable; stopping at `<eos>` ends like a real review). No further reason was given by the user beyond the selection.
- **Status:** CONFIRMED

### G2. Prompts file and its format
- **Value:** prompts are read from `prompts.txt` (a file the user is writing); one prompt per line; blank lines and lines starting with `#` are skipped. The prompts are cleaned and tokenized with the same function as the training data; words not in the vocabulary become `<unk>`.
- **Where:** `configs/generations_ngram.json:19`; `src/generate.py:150-161` (`read_prompts`), `:164-170` (`encode_prompt`).
- **Chosen by:** file name `prompts.txt` = User, prompt (2026-10-11). The line format and `#` comment rule = Assistant default (chosen so the file can carry `#` reason lines under each prompt; see F1).
- **Alternatives:** JSON list of prompts; one prompt per paragraph.
- **Status:** file name CONFIRMED; format NEEDS REVIEW. `prompts.txt` (the F1 prompts, renamed by the user from `prompt.txt`) exists and was used for `results/generations_ngram.json`.

### G3. Random seed per generation
- **Value:** each (prompt, sample number) gets its own random generator seeded from `[seed, sample number, CRC32 of the prompt text]`, so output depends only on those three things.
- **Where:** `src/generate.py:144-146`.
- **Chosen by:** Assistant default. **Alternatives:** one generator seeded once at the start (simpler, but adding or reordering prompts would change every other prompt's text).
- **Status:** NEEDS REVIEW

### G4. Order of the sampling dials, and their edge cases
- **Value:** banned words are removed first, then temperature is applied, then top-k (so top-k keeps the k likeliest *allowed* words), then the probabilities are rescaled. Temperature must be greater than 0 (0 would mean "always the likeliest word" and is not supported); top-k 0 means off.
- **Where:** `src/generate.py:103-121`.
- **Chosen by:** Assistant default. **Alternatives:** top-k before banning (could leave fewer than k usable words); support temperature 0 as greedy decoding.
- **Status:** NEEDS REVIEW

### G5. How generated text is stored
- **Value:** the stopping `<eos>` is not included in the stored tokens (a flag `ended_with_eos` records it); text is the prompt tokens plus new tokens joined with single spaces, so punctuation is separate (e.g. `can ' t`); no detokenization.
- **Where:** `src/generate.py:137-141` (eos), `:237-238` (text).
- **Chosen by:** Assistant default. **Alternatives:** keep `<eos>` in the output; detokenize punctuation back onto words.
- **Status:** NEEDS REVIEW

### G6. How the n-gram model is prepared for generation
- **Value:** the counts are re-fitted from the training split (about 1 second); the discounts are NOT re-tuned but read from the saved n-gram run (`results/ngram.json`, `details.best_discounts`), whose date and commit are stored in the generation record.
- **Where:** `src/generate.py:174-191`; `configs/generations_ngram.json:18`.
- **Chosen by:** Assistant default. **Alternatives:** save and load a fitted model file; re-tune on every generation run.
- **Status:** NEEDS REVIEW

### G7. How neural models plug in
- **Value:** any `nn.Module` with `forward(ids) -> logits` is wrapped by `NeuralNextToken`, which feeds the whole context (prompt + generated so far) each step and reads the last position's logits. No neural model is loadable yet (`build_model` stops with a message).
- **Where:** `src/generate.py:87-97`, `:174-190`.
- **Chosen by:** Assistant default. **Alternatives:** require each model to define its own generation step (e.g. caching hidden states for speed).
- **Status:** NEEDS REVIEW

### G8. Generation files are full run records
- **Value:** `results/generations_<model>.json` is a normal run record (config, seed, commit, hardware, date, metrics) with the decoding settings (temperature, top_k, max_tokens, seed, plus the other settings) and the generated texts as extra top-level fields. The file name comes from the config file name (`configs/generations_ngram.json` -> `generations_ngram`).
- **Where:** `src/generate.py:252-263`; `src/runconfig.py:189-225` (`extra`).
- **Chosen by:** Assistant default (to satisfy both "every run saves its record" and "generation files store their decoding settings"). **Alternatives:** a separate, lighter file format for generations.
- **Status:** NEEDS REVIEW

---

## NEEDS REVIEW summary

**All 24 items resolved 2026-10-10.** The user reviewed every one (2-3 options with trade-offs each, per CLAUDE.md's record-keeping rules) and confirmed every current value as-is: A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, B1, B3, C1, C2, C3, D2, D3, D4, D5, D6, D7, D8, E1, E3. See each entry above for the specific options offered and the reasoning. No reruns were needed since nothing changed.

One item (D4) was accepted rather than fixed: the n-gram discount grid was widened and re-run after the test-set perplexity from the first run had already been seen. The test number was identical in both runs, so no actual leakage occurred this time, but the user explicitly signed off on this as a logged methodological risk rather than it being silently resolved.

**New since that review (2026-10-11), pending the user:** G2 (prompt file format), G3, G4, G5, G6, G7, G8.
