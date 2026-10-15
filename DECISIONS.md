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
- **Status:** CONFIRMED (file name 2026-10-11; format reviewed by the user on 2026-10-14 and kept as is). `prompts.txt` (the F1 prompts, renamed by the user from `prompt.txt`) exists and was used for `results/generations_ngram.json`.

### G3. Random seed per generation
- **Value:** each (prompt, sample number) gets its own random generator seeded from `[seed, sample number, CRC32 of the prompt text]`, so output depends only on those three things.
- **Where:** `src/generate.py:144-146`.
- **Chosen by:** Assistant default. **Alternatives:** one generator seeded once at the start (simpler, but adding or reordering prompts would change every other prompt's text).
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### G4. Order of the sampling dials, and their edge cases
- **Value:** banned words are removed first, then temperature is applied, then top-k (so top-k keeps the k likeliest *allowed* words), then the probabilities are rescaled. Temperature must be greater than 0 (0 would mean "always the likeliest word" and is not supported); top-k 0 means off.
- **Where:** `src/generate.py:103-121`.
- **Chosen by:** Assistant default. **Alternatives:** top-k before banning (could leave fewer than k usable words); support temperature 0 as greedy decoding.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### G5. How generated text is stored
- **Value:** the stopping `<eos>` is not included in the stored tokens (a flag `ended_with_eos` records it); text is the prompt tokens plus new tokens joined with single spaces, so punctuation is separate (e.g. `can ' t`); no detokenization.
- **Where:** `src/generate.py:137-141` (eos), `:237-238` (text).
- **Chosen by:** Assistant default. **Alternatives:** keep `<eos>` in the output; detokenize punctuation back onto words.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### G6. How the n-gram model is prepared for generation
- **Value:** the counts are re-fitted from the training split (about 1 second); the discounts are NOT re-tuned but read from the saved n-gram run (`results/ngram.json`, `details.best_discounts`), whose date and commit are stored in the generation record.
- **Where:** `src/generate.py:174-191`; `configs/generations_ngram.json:18`.
- **Chosen by:** Assistant default. **Alternatives:** save and load a fitted model file; re-tune on every generation run.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### G7. How neural models plug in
- **Value:** any `nn.Module` with `forward(ids) -> logits` is wrapped by `NeuralNextToken`, which feeds the whole context (prompt + generated so far) each step and reads the last position's logits. No neural model is loadable yet (`build_model` stops with a message).
- **Where:** `src/generate.py:87-97`, `:174-190`.
- **Chosen by:** Assistant default. **Alternatives:** require each model to define its own generation step (e.g. caching hidden states for speed).
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### G8. Generation files are full run records
- **Value:** `results/generations_<model>.json` is a normal run record (config, seed, commit, hardware, date, metrics) with the decoding settings (temperature, top_k, max_tokens, seed, plus the other settings) and the generated texts as extra top-level fields. The file name comes from the config file name (`configs/generations_ngram.json` -> `generations_ngram`).
- **Where:** `src/generate.py:252-263`; `src/runconfig.py:189-225` (`extra`).
- **Chosen by:** Assistant default (to satisfy both "every run saves its record" and "generation files store their decoding settings"). **Alternatives:** a separate, lighter file format for generations.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## H. Training windows (`src/data.py`)

### H1. Context length
- **Value:** 64 tokens per window.
- **Where:** `configs/show_batch.json:7` (`context_length`); future training configs must set their own.
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** 32 (faster, less history); 128 (more context, about 2x compute per window, fewer windows).
- **Reason given:** accepted the recommended value (the user's own example).
- **Status:** CONFIRMED

### H2. Window stride
- **Value:** non-overlapping windows (stride = context length): window i starts at token i*L; each token is a target exactly once.
- **Where:** `src/data.py:255-296` (`WindowDataset`, `start = i * self.context_length`).
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** half overlap (2x windows, tokens counted twice); stride 1 (about 64x windows, slow and redundant).
- **Reason given:** accepted the recommended option.
- **Status:** CONFIRMED

### H3. Leftover tail of a split's token stream
- **Value:** the tail shorter than L+1 tokens is dropped (no padding); windows run across review boundaries (reviews are separated by `<eos>`).
- **Where:** `src/data.py:276-278` (`num_windows = (N - 1) // L`).
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** pad the tail with `<pad>` (keeps every token, needs loss masking).
- **Reason given:** accepted the recommended option. Windows crossing review boundaries followed from the user's wording ("a split's token stream").
- **Status:** CONFIRMED

### H4. Batch size
- **Value:** 64.
- **Where:** `configs/show_batch.json:9` (`batch_size`).
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** 32 (noisier, less memory); 128 (faster epochs, more memory, fewer updates).
- **Status:** CONFIRMED

### H5. Shuffling and last batch
- **Value:** train loader shuffles each epoch using a `torch.Generator` seeded from the config seed; val/test loaders do not shuffle; the final batch of an epoch may be smaller than `batch_size` (not dropped).
- **Where:** `src/data.py:299-315` (`make_dataloader`).
- **Chosen by:** Assistant default (shown in the design the user approved, but not offered as options); User, prompt (2026-10-11, reviewing NEEDS REVIEW) confirmed "keep those". **Alternatives considered:** `drop_last=True` for equal-size batches.
- **Status:** CONFIRMED

---

## I. Training script (`src/train.py`)

### I1. Early stopping and epoch cap
- **Value:** stop after 3 epochs without validation improvement; at most 20 epochs.
- **Where:** `configs/train_bigram.json:12` (`epochs`), `:20` (`options.patience`); used in `src/train.py` (`fit`).
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** N=2 / 10 epochs (fast, may stop on a noisy epoch); N=5 / 30 epochs (patient, slower).
- **Status:** CONFIRMED

### I2. Learning rate and gradient clipping
- **Value:** Adam, learning rate 1e-3; gradients clipped to total norm 1.0.
- **Where:** `configs/train_bigram.json:10` (`learning_rate`), `:21` (`options.grad_clip`); `src/train.py` (`fit`, `train_one_epoch`).
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** lr 3e-4 / clip 1.0 (steadier, slower); lr 1e-3 / clip 5.0 (looser).
- **Status:** CONFIRMED

### I3. Models behind `--model`
- **Value:** only `bigram` (the existing `BigramLM`) for now; others are added with one line in `MODELS`.
- **Where:** `src/train.py` (`MODELS`).
- **Chosen by:** User, prompt (2026-10-11), from options offered. **Alternatives considered:** bigram + LSTM; bigram + LSTM + Transformer (each needs its own design choices first).
- **Status:** CONFIRMED

### I4. Validation/test review length
- **Value:** reviews cut to their first 256 tokens before scoring with `evaluate()`; eval batch size 64.
- **Where:** `configs/train_bigram.json:22-23` (`options.eval_max_length` = user's choice; `options.eval_batch_size` = assistant default, affects speed only).
- **Chosen by:** `eval_max_length`: User, prompt (2026-10-11), from options offered. **Alternatives considered:** 64 (matches training windows); None (full reviews, slow).
- **Confirmed:** User, prompt (2026-10-11, reviewing NEEDS REVIEW) kept the eval batch size 64.
- **Status:** CONFIRMED

### I5. Training-script defaults
- **Value:** (a) the reported training loss is the token-weighted running average over the epoch's batches (computed while the weights are changing, dropout on), so it is not directly comparable to validation loss; (b) a checkpoint is saved only when validation loss is strictly lower than the best so far; (c) the best weights go to `checkpoints/<run>.pt` (git-ignored); (d) device is CUDA if available, else CPU; (e) training time is wall-clock seconds around the epoch loop (including validation, excluding data loading and the final test).
- **Where:** `src/train.py` (`train_one_epoch`, `fit`, `main`); `configs/train_bigram.json:19` (`options.checkpoint_dir`).
- **Chosen by:** Assistant default (shown in the design the user approved, not offered as options); User, prompt (2026-10-11, reviewing NEEDS REVIEW) confirmed "keep those". **Alternatives considered:** re-score the training set in eval mode after each epoch; `<=` instead of `<`; Apple MPS device.
- **Status:** CONFIRMED

---

## J. Results document (`RESULTS.md`, `src/update_results.py`, 2026-10-12)

The user's prompt of 2026-10-12 specified the document's sections, contents, the Checks list, the "My notes" headings, the Colab change and the CLAUDE.md rule. The items below are the values and mechanisms the prompt did not specify. The assistant chose them without offering options first (the prompt was a full specification), so all are `NEEDS REVIEW`.

### J1. Which runs belong to Data / Part I / Part II / Part III (revised 2026-10-12 after reading the assignment brief)
- **Value:** the brief defines the parts: Part I = n-gram, RNN, LSTM, Transformer comparison; Part II = three embedding settings on one architecture; Part III = downstream classifier on representations. Rules in order: `data`, `inspect_data`, `show_batch` -> Data; any run name containing `p2_` -> Part II; containing `p3_` -> Part III; `sanity_check`, `ngram`, `train_*`, `test_*`, `generations_*` -> Part I. **Naming convention for later runs: Part II runs start with `p2_`, Part III runs with `p3_`.** The bigram run (`train_bigram`) is not required by the brief; it is listed in Part I as a baseline. Each of Parts I and II shows its own perplexity/cost table and side-by-side generated text; Part III shows its run records.
- **Where:** `configs/update_results.json`, `options.part_rules`, `options.part_comparison_extra_runs` (runs from another part to include in a part's comparison, e.g. the reused Part I model in Part II).
- **Chosen by:** Parts: the brief (`Assignment1_2026.pdf`). Run-name convention and placing `train_bigram` in Part I: Assistant default.
- **Alternatives considered:** a `part` field inside each config; a separate Part for baselines.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is) (naming convention and bigram placement only)

### J2. "Close to the vocabulary size" threshold
- **Value:** untrained validation perplexity divided by vocabulary size must be between 0.5 and 2.0 to PASS.
- **Where:** `configs/update_results.json`, `options.untrained_ppl_to_vocab_ratio_range`.
- **Chosen by:** Assistant default.
- **Alternatives considered:** a tighter band (0.8 to 1.25); report the ratio with no pass/fail line.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### J3. How the exact command of each run is known
- **Value:** `save_run` now stores a `command` field (interpreter + arguments) in every new `results/<run>.json`. For older records, the command is copied from `results/RESULTS_LOG.md`; if not found there it is shown as "not recorded".
- **Where:** `src/runconfig.py` (`save_run`); lookup in `src/update_results.py` (`find_command`).
- **Chosen by:** Assistant default.
- **Alternatives considered:** re-run old runs to get commands stored (slow, changes their dates); write commands by hand into a file.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### J4. How the test-set and padding checks are evidenced
- **Value:** static scan of the source with Python `ast`: in `src/ngram.py` and `src/train.py`, `data.test` must be used exactly once, on a line after the `tune_discounts(...)` / `fit(...)` call, and no other `src/` file may use it. Padding: every `CrossEntropyLoss` in `src/train.py` and `src/sanity_check.py` must pass `ignore_index`, and the recorded test `test_padding_is_ignored` must be `passed`. Frozen embeddings: PASS needs a run with a pretrained `embedding_init` whose `details.frozen_embedding_max_abs_change` equals 0; `src/train.py` does not write that key yet, so a future Part II change must add it. Causality: any recorded test with "causal" in its name.
- **Where:** `configs/update_results.json` (`test_set_markers`, `padding_loss_files`, `padding_test_*`, `causality_test_keyword`, `frozen_embedding_detail_key`, `not_frozen_embedding_inits`); logic in `src/update_results.py` (`check_*`).
- **Chosen by:** Assistant default.
- **Alternatives considered:** run-time instrumentation of the data loader; only trusting the unit tests.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### J5. How "My notes" are protected
- **Value:** each section ends with `### My notes` and a pair of HTML-comment marker lines; text between the markers is copied back unchanged on every rebuild. If the markers are missing from the old file, the script stops (override with `--force`).
- **Where:** `src/update_results.py` (`extract_notes`, `notes_block`).
- **Chosen by:** Assistant default.
- **Alternatives considered:** keep notes in a separate file linked from RESULTS.md.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### J6. Figures, and `update_results.py` not saving a run record
- **Value:** matplotlib PNGs in `results/figures/`: tokens per split, n-gram discount tuning (D3 axis, D2/D1 at their chosen values), train/validation loss per trained run, and validation loss of all trained runs when there are two or more. Colours are the first slots of the dataviz skill's validated categorical palette. Runs named `test_*` are not plotted. `update_results.py` writes `RESULTS.md` and the figures but no `results/update_results.json`, because it is not an experiment run.
- **Where:** `configs/update_results.json` (`figure_colours`, `test_run_prefix`); `src/update_results.py` (`fig_*`).
- **Chosen by:** Assistant default.
- **Alternatives considered:** an interactive HTML report; saving a run record for each rebuild.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## K. Task list (`MY_TASKS.md`, `src/my_tasks.py`, 2026-10-12)

The user's prompt specified the six sections, checkboxes, planned days, overdue-first order, "never tick for me" and the `NEW FOR YOU:` rule. The assignment brief and the user's schedule are not in the repo, so the items below are assistant defaults, all `NEEDS REVIEW`.

### K1. Report and submission requirement list (revised 2026-10-12 from the brief)
- **Value:** 23 requirements taken from `Assignment1_2026.pdf` (Part I, II, III bullets, the report contents list, 6-page limit, citations, README, no credentials, submission, late policy), each tagged with the brief's points where known. Automatic evidence only for: the four Part I model families being present, the frozen-embedding check, README keywords, repo state. Everything else shows "not done" until the user ticks it. The split of each part's bullets into separate items is the assistant's.
- **Where:** `configs/update_results.json`, `options.report_requirements`.
- **Chosen by:** Content: the brief. Granularity: Assistant default.
- **Alternatives considered:** one item per part.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### K2. Schedule and planned days
- **Value:** `options.schedule` is empty, so every item shows "Planned: not set" and nothing can be overdue. Deadline is Fri 16 Oct 2026, 23:59 Singapore time (from the user's prompt).
- **Where:** `configs/update_results.json`, `options.schedule`, `options.deadline`, `options.timezone`.
- **Chosen by:** Deadline: User, prompt (2026-10-12). Empty schedule: Assistant default (no schedule was available to read).
- **Alternatives considered:** spread items evenly over the days left (would invent a schedule).
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### K3. What counts as "waiting on me now", and how interpretation items are counted
- **Value:** Waiting = empty schedule, unresolved NEEDS REVIEW decisions, configs with no results yet, sections with results but empty *My notes*, failing checks, uncommitted or unpushed git changes. "Results to interpret" has one item per section (Data, Part I, Part II, Part III), because *My notes* exist per section, not per run. "Runs to do" lists configs in `configs/` with no `results/<run>.json` plus `options.extra_items` (steps from the brief that no script can detect, such as scripts still to be written); time is taken from earlier runs of the same script.
- **Where:** `src/my_tasks.py` (`build_items`); `configs/update_results.json` (`tasks_ignore_configs`, `run_script_rules`, `extra_items`).
- **Chosen by:** Assistant default.
- **Alternatives considered:** one interpretation item per run; treat every NEEDS REVIEW item as blocking.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## L. Recurrent models (`src/models.py`, 2026-10-12)

### L1. Default size of the RNN and LSTM
- **Value:** hidden size 256, 2 recurrent layers, dropout 0.3, embedding dimension 100 (randomly initialised, trainable).
- **Where:** `configs/train_rnn.json` and `configs/train_lstm.json`, lines 3-4 (`layers`, `hidden_size`) and 11 (`dropout`); overridable with `--hidden-size`, `--layers`, `--dropout`.
- **Chosen by:** User, prompt (2026-10-12), from options offered. **Alternatives considered:** hidden 128 / 512; layers 1 / 3; dropout 0.1 / 0.5.
- **Status:** CONFIRMED

### L2. RNN nonlinearity
- **Value:** tanh.
- **Where:** `src/models.py` (`RNNLanguageModel`, `nonlinearity="tanh"`).
- **Chosen by:** User, prompt (2026-10-12), from options offered. **Alternatives considered:** relu.
- **Status:** CONFIRMED

### L3. Dropout placement and layout
- **Value:** dropout after the embedding, between stacked recurrent layers (PyTorch's `dropout=`, set to 0 when there is 1 layer) and before the output layer; unidirectional recurrent layers; no weight tying; the final hidden state is not carried between windows.
- **Where:** `src/models.py` (`forward` of both classes).
- **Chosen by:** Assistant default (shown in the design the user approved, not offered as options). **Alternatives:** no dropout on the embedding; tie the output layer to the embedding (needs hidden size = 100); carry the hidden state across consecutive windows.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### L4. Smoke-test data
- **Value:** the first 60 train / 40 val / 40 test reviews of the saved splits, copied to a scratch folder (not `data/processed`), 1 epoch, output to a scratch results folder.
- **Where:** scratchpad only (not a repository file).
- **Chosen by:** Assistant default for how to make the "small subset" (the user asked for a small subset). **Alternatives:** re-run `data.py` with `subsample_size` (would overwrite the saved splits).
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## M. Transformer language model (`src/models.py`, 2026-10-12)

### M1. Size: blocks, heads, feed-forward width
- **Value:** 2 decoder blocks, 4 attention heads (25 dims each), feed-forward width 256, dropout 0.3, embedding dimension 100 (random, trainable).
- **Where:** `configs/train_transformer.json` lines 3-4 (`layers`, `hidden_size`), 11 (`dropout`) and `options.num_heads`.
- **Chosen by:** User, prompt (2026-10-12), from options offered ("proceed with your recommended set"). **Alternatives considered:** blocks 1 / 4; heads 2 / 5; feed-forward 400 / 512.
- **Status:** CONFIRMED

### M2. Positional embedding table size
- **Value:** `max_positions` 256 (learned positional embeddings), while training windows stay at `context_length` 64. Positions 64-255 are never updated in training (they stay at their random start values), but evaluation (`eval_max_length` 256) feeds sequences up to 256.
- **Where:** `configs/train_transformer.json`, `options.max_positions`.
- **Chosen by:** User, prompt (2026-10-12), from options offered (256 / 64 with sliding window / train at 256). **Alternatives considered:** 64 with a sliding window in evaluate/generate; context_length 256.
- **Status:** CONFIRMED

### M3. Block style and other defaults
- **Value:** pre-norm blocks (LayerNorm before attention and feed-forward) plus one final LayerNorm; GELU activation; dropout after the embeddings and inside each block; positional embedding starts at PyTorch's default random values; no weight tying; causal mask built with `torch.triu` and passed to `nn.TransformerEncoder`.
- **Where:** `src/models.py`, class `TransformerLanguageModel`.
- **Chosen by:** Assistant default (shown in the design the user approved, not offered as options). **Alternatives:** post-norm blocks; ReLU; tied input/output embeddings.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### M4. Training settings copied from the LSTM
- **Value:** learning rate 0.001, batch size 64, epochs 20, context length 64, patience 3, grad clip 1.0, vocabulary 20000, seed 0.
- **Where:** `configs/train_transformer.json`.
- **Chosen by:** Assistant default (copied from `configs/train_lstm.json` so the comparison is matched; not offered as options). **Alternatives:** a smaller learning rate or warmup, which Transformers often prefer.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## N. Causality test (`tests/test_causality.py`, 2026-10-13)

### N1. Test settings
- **Value:** random sequences of length 12, batch 4, tolerance 1e-6 on the largest change in any probability; every position t from 0 to 11 is changed in turn; models are in eval mode (dropout off); every class in `models.py` is tested, found automatically. The bigram (in `sanity_check.py`) is not covered.
- **Where:** `configs/test_causality.json` (`batch_size`, `options.seq_len`, `options.tolerance`).
- **Chosen by:** Assistant default (the user specified what to check, not these values; not offered as options). **Alternatives:** longer sequences (up to 256); tolerance 1e-8 or exact equality; also testing in train mode with a fixed dropout seed.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## O. Benchmark (`src/benchmark.py`, 2026-10-13)

### O1. Workload, warm-up and devices
- **Value:** batch size 64 and context length 256 (the same as the evaluation settings, so the speed is for the exact scoring that gave the perplexities); 2 full warm-up passes; 3 timed repeats (mean and standard deviation); every neural model timed on the GPU and on the CPU, the n-gram on the CPU only (NumPy).
- **Where:** `configs/benchmark.json` (`batch_size`, `context_length`, `options.warmup_passes`, `options.repeats`, `options.devices`).
- **Chosen by:** User, prompt (2026-10-13): 3 repeats; the rest from options offered ("proceed with your recommended set"). **Alternatives considered:** 64/64 (training context) or batch 1; 1 warm-up pass or a few batches; GPU-only neural timing without CPU columns.
- **Status:** CONFIRMED

### O2. Other benchmark details
- **Value:** the timed region is the whole `evaluate()` call (sorting, padding and scoring included); the n-gram is refit from the training split with its saved discounts, untimed; the n-gram "parameters" are its stored counts, with a column saying so; n-gram training and tuning seconds are separate columns; a run stops if a model's perplexity differs from its saved test perplexity by more than 0.1% (neural models only; the n-gram was scored without the 256 cut) or if the scored targets differ between models.
- **Where:** `src/benchmark.py`; `configs/benchmark.json` (`options.perplexity_rel_tolerance`).
- **Chosen by:** Assistant default (shown in the design the user approved, not offered as options). **Alternatives:** time only the model forward pass on pre-built batches; use parameter counts only for the neural models.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## P. Matched generation for Part I (`src/generate.py`, `src/combine_generations.py`, 2026-10-13)

### P1. Shared decoding settings for all four models
- **Value:** temperature 1.0; top-k 40; 50 new tokens; seed 0; 1 sample per prompt; `<pad>` and `<unk>` never sampled; stop at `<eos>`; prompts from `prompts.txt`. Identical in `configs/generations_{ngram,rnn,lstm,transformer}.json`, and `combine_generations.py` stops if they differ.
- **Where:** `configs/generations_*.json`, `options.temperature`, `options.top_k`, `options.max_new_tokens`, `options.num_samples`, `seed`.
- **Chosen by:** User, prompt (2026-10-13), from options offered ("proceed with your recommended set"; the temperature, top-k and length also match the earlier choice in G3). **Alternatives considered:** temperature 0.8 / 0.5; top-k 10 / off; 100 or 30 tokens; 3 samples per prompt; another seed.
- **Status:** CONFIRMED

### P2. Loading neural models and the combined file
- **Value:** the neural model's size settings come from its training record (`results/train_<name>.json`), not from the generation config; the checkpoint's parameter count must match that record; generation runs on the GPU when available; the combined file `results/part1_generations.md` lists the shared settings, each source file's commit, hardware and date, and one section per prompt with the models in the order n-gram, RNN, LSTM, Transformer; it adds no commentary.
- **Where:** `src/generate.py` (`build_neural`), `src/combine_generations.py`, `configs/combine_generations.json`.
- **Chosen by:** Assistant default (shown in the design the user approved, not offered as options). **Alternatives:** model sizes typed into each generation config; one file per prompt; a table instead of block quotes.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## Q. Own word2vec embeddings (`src/train_word2vec.py`, 2026-10-13)

### Q1. Algorithm
- **Value:** Skip-gram (`sg=1`).
- **Where:** `configs/word2vec_cost.json`, `options.sg`.
- **Chosen by:** User, 2026-10-13, from Skip-gram / CBOW / config option. **Alternatives considered:** CBOW (faster, weaker on rare words).
- **Status:** CONFIRMED

### Q2. gensim settings and training text (confirmed by the user 2026-10-13; the repeat request re-affirmed Skip-gram as Q1)
- **Value:** window 5; min_count 1; epochs 10; negative 5; sample 1e-3; workers 1 (for reproducibility with a seed); vector size 100; seed 0. One "sentence" per training review (including its `<eos>`; `<unk>` is a normal token). `<pad>` row is all zeros.
- **Where:** `configs/word2vec_cost.json` (`epochs`, `embedding_dim`, `seed`, `options.*`); text built in `src/train_word2vec.py` (`reviews_as_word_lists`).
- **Chosen by:** User, 2026-10-13 (second request), option A of three offered: A window 5 / min_count 1 / epochs 10 (chosen); B window 5 / min_count 20 / epochs 10 (coverage 87% of types, 99.6% of tokens); C window 10 / min_count 1 / epochs 20 (about 4x the cost). **Other alternatives:** several workers (faster, not reproducible); `<eos>` dropped from the text.
- **Status:** CONFIRMED

---

## R. Part II setup (2026-10-13)

### R1. Architecture for Part II
- **Value:** LSTM (2 layers, hidden 256, context 64, dropout 0.3, lr 0.001, batch 64, up to 20 epochs, seed 0, as in `configs/train_lstm.json`). The random-embedding setting reuses the Part I run `train_lstm` (test perplexity 80.45, Colab T4); the two frozen-embedding runs must use exactly these settings except the embedding.
- **Where:** `configs/train_lstm.json` (to be copied to `configs/p2_*.json`).
- **Chosen by:** User, 2026-10-13. **Alternatives considered:** RNN; Transformer (not discussed in detail).
- **Status:** CONFIRMED

### R2. Initialising vocabulary words that have no vector (own word2vec and public GloVe)
- **Value:** rows without a vector are drawn from a normal distribution, mean 0, standard deviation = that of all real vectors in the same table, seeded with the run's seed. The `<pad>` row is always zeros. The table is then kept fixed. The same function is used for both tables.
- **Where:** `src/embedding_utils.py` (`fill_missing_rows`); called from `src/train_word2vec.py`. (The GloVe script, not yet written, must call it too.)
- **Chosen by:** User, 2026-10-13, from three options. **Alternatives considered:** mean of the known vectors (all missing words identical); all zeros.
- **Status:** CONFIRMED
- **Note:** with the chosen word2vec settings no row needed filling (coverage 100%), so this rule has been tested only on a small synthetic example for word2vec; it will matter for GloVe.

### S1. Which public GloVe file
- **Value:** `glove.6B.100d.txt` (Wikipedia + Gigaword, 6B tokens, 400k lower-cased words, 100-d), matched to our tokens by exact string. Missing words (including `<unk>`, `<eos>`) follow R2 (seeded small random normal, `<pad>` zeros); coverage is reported with `<unk>`/`<eos>` counted as missing, and also without them.
- **Where:** `configs/glove_coverage.json` (`options.glove_path`, `options.output_path`, `options.top_missing` = 20); `src/load_glove.py`.
- **Chosen by:** Assistant default (the request said "GloVe 100d" and R2 fixed the missing-word rule). **Alternatives:** `glove.twitter.27B.100d` (cased-agnostic tweets, different domain, 1.2M words); `glove.840B.300d` (wrong dimension). 6B is the one that matches our lower-cased text and 100-d.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

## T. Frozen-embedding runs (`src/train.py --embedding`, 2026-10-14)

### T1. How the option works and what stays identical
- **Value:** `--embedding random|w2v|glove` overrides the config's `embedding_init`. For w2v/glove the saved table is copied into the model's embedding layer and the WHOLE table (all 20,003 rows, including `<pad>`, `<unk>`, `<eos>` and words filled by R2) gets `requires_grad=False`, so Adam never changes it. The positional/other parts of the model are unaffected (the LSTM has no other embedding). Model construction order and seed are unchanged, so the LSTM and output layers start from the same random values as in the Part I run. The run configs `configs/p2_lstm_w2v.json` and `configs/p2_lstm_glove.json` differ from `configs/train_lstm.json` only in `embedding_init` and the added `options.embedding_paths`. Optimiser: Adam on the trainable parameters only (for random runs that is every parameter, as before).
- **Where:** `src/train.py` (`EMBEDDING_CHOICES`, `freeze_embedding_if_pretrained`, `load_embedding_matrix`, `check_embedding_unchanged`, `main`); `configs/p2_lstm_*.json`.
- **Chosen by:** User asked for the option, the freezing and the check; the config layout and run names (`p2_lstm_w2v`, `p2_lstm_glove`) are assistant defaults. **Alternatives:** one shared config with `--embedding` only; a hard-coded path table in `train.py` (rejected: rule 1).
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is) (config layout, run names)

### T2. The frozen-embedding check
- **Value:** a copy of the table is taken right after loading; it must be exactly equal (`torch.equal`, no tolerance) after the last epoch and again after the best checkpoint is reloaded, and the table must still be non-trainable; otherwise the script raises an error and does NOT save results. The record stores `frozen_embedding_max_abs_change` (0.0), the file path and its SHA-256, `total_parameters`, and `embedding_table_trainable`. `num_parameters` stays the trainable count (so a frozen run reports 6,033,699, not 8,033,999; the total is in `details.total_parameters`).
- **Where:** `src/train.py`; tested in `tests/test_frozen_embedding.py`.
- **Chosen by:** User (asked for a check that fails loudly); exact-equality and the reported parameter counts are assistant defaults. **Alternatives:** tolerance of 1e-8; report total instead of trainable parameters as `num_parameters`.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is) (parameter-count convention)

## U. Part II audit, generation and RESULTS.md section (2026-10-14)

### U1. Part II generation set-up
- **Value:** the two frozen-embedding LSTMs are generated with `configs/generations_p2_lstm_w2v.json` and `configs/generations_p2_lstm_glove.json`, which equal `configs/generations_lstm.json` except `embedding_init` and the `train_result` / `checkpoint` they load (temperature 1.0, top-k 40, 50 new tokens, seed 0, 1 sample, same `prompts.txt`). The random-embedding column reuses the Part I file `results/generations_lstm.json` (no new generation). `configs/combine_generations_p2.json` merges the three into `results/part2_generations.md` and refuses to merge if decoding settings or prompts differ.
- **Where:** `configs/generations_p2_lstm_*.json`, `configs/combine_generations_p2.json`; `src/combine_generations.py` (optional `options.title` and `options.labels`).
- **Chosen by:** User asked for "same prompts.txt and decoding settings as Part I"; reuse of the Part I LSTM generation and the file names are assistant defaults. **Alternatives:** regenerate the Part I LSTM text inside Part II (same output, since seed and settings are fixed; it would only add a duplicate run).
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

### U2. What the Part II section of RESULTS.md contains
- **Value:** the Part II runs and their loss figures; a comparison table and generated-text table that include the reused Part I runs `train_lstm` and `generations_lstm`; the embedding-table records `word2vec_cost` and `glove_coverage` (now assigned to Part II); and a "Part II set-up facts" block (settings differences, provenance, frozen check, embedding tables with coverage and cost, per-epoch loss facts), all read from the run records.
- **Where:** `configs/update_results.json` (`part_comparison_extra_runs`, `part_rules`, `part2_baseline_run`, `part2_own_embedding_run`, `part2_public_embedding_run`); `src/update_results.py` (`render_part2_facts`).
- **Chosen by:** Assistant default (the user asked for "what exists so far, facts only"). **Alternatives:** a separate audit file; no per-epoch table.
- **Status:** CONFIRMED (user reviewed each item one by one on 2026-10-14 and kept it as is)

---

## V. Audit follow-up (2026-10-14)

### V1. Reruns after the Part II audit
- **Value:** no reruns. A (clean-commit rerun of both Part II runs), B (rerun of the Part I LSTM on current code), C (all three at 40 epochs) and D (per-batch loss logging) were offered with Colab time estimates and consequences; the user chose none. Consequences accepted: the Part II records show `git_dirty: true` with the cause unknown; Part I LSTM ran on an older commit (equivalence rests on the code diff); all three runs stopped at the 20-epoch cap while still improving, so convergence is compared at a fixed 20-epoch budget; stability evidence is per-epoch only.
- **Where:** nothing changed; `MY_TASKS.md` has no rerun items.
- **Chosen by:** User, 2026-10-14. **Alternatives considered:** A, B, C, D (about 1.9 h, 64 min, 6 h, 3 h on a T4).
- **Status:** CONFIRMED

---

## W. Part III classification data (2026-10-14)

### W1. Maximum review length and truncation side
- **Value:** `max_length` 256, keep the FIRST 256 tokens ("head" truncation), pad on the right with `<pad>`. 37.0% of train, 37.0% of val and 38.2% of test reviews are truncated.
- **Where:** `configs/p3_prepare_classification.json` (`options.max_length`, `options.truncate`); used in `src/prepare_classification.py`.
- **Chosen by:** User, prompt (2026-10-14), from options offered. **Alternatives considered:** 128 (cheaper, about 85% truncated); 512 (fewer truncated, but beyond the Transformer's 256 positions); last-N tokens instead of first-N.
- **Reason (user):** option 1 as offered (matches the Transformer's 256 positions and the existing `eval_max_length`).
- **Status:** CONFIRMED

### W2. Details not covered by the user's choice
- **Value:** (a) the `<eos>` marker data.py appended to each review is dropped (it separates documents for the language models; it is not part of a review, and truncated reviews would lose it anyway); (b) `<unk>` rate is reported over real tokens only (no `<pad>`, no `<eos>`), both for the full review and for the kept part; (c) the saved files also store `lengths` (real tokens per row) so later code can ignore padding.
- **Where:** `src/prepare_classification.py` (`prepare_split`).
- **Chosen by:** Assistant default. **Alternatives considered:** keep `<eos>` when the review fits (then short and long reviews end differently).
- **Status:** NEEDS REVIEW

---

## X. Feature extraction (2026-10-14)

### X1. Which vector is pooled, and how padding is excluded
- **Value:** the pooled vector is the one fed to the model's output layer: output of the last recurrent layer (RNN/LSTM, size 256) or of the final LayerNorm (Transformer, size 100). `last` = hidden state at position `length - 1`; `mean` = sum over real positions (0/1 mask) divided by `length`. Padding is on the right and all models are causal, so real positions never depend on padding (checked by `tests/test_extract_features.py`). Models are put in eval mode (dropout off), all parameters set to `requires_grad False`, and the run stops if any weight changed. Only `hidden_states()` was added to `src/models.py`; `forward` computes the same numbers as before.
- **Where:** `src/extract_features.py` (`pool`, `extract_split`); `src/models.py` (`hidden_states` in each class).
- **Chosen by:** Assistant default (pooling methods themselves: user's request). **Alternatives considered:** pooling an earlier layer; pooling before the Transformer's final LayerNorm; concatenating last and mean.
- **Status:** NEEDS REVIEW

---

### X2. Part III comparison and pooling
- **Value:** compare the Part I LSTM and Part I Transformer (both random trainable embeddings, as trained), same pooling for both: `mean`. Runs: `p3_feat_lstm_mean`, `p3_feat_transformer_mean`; batch size 64 (speed only). Feature sizes: LSTM 256, Transformer 100.
- **Where:** `configs/p3_feat_lstm_mean.json`, `configs/p3_feat_transformer_mean.json` (`options.pooling`, `batch_size`); Colab section 12.
- **Chosen by:** comparison: User, prompt (2026-10-14), option 1 of 3. Pooling `mean`: Assistant default (it was named as the usual default in the options; the user did not specify). **Alternatives considered:** last vs mean on one model; full 2x2 grid; adding the Part II frozen-embedding LSTMs.
- **Status:** NEEDS REVIEW (pooling choice)

---

## Y. Part III classifier (2026-10-14)

### Y1. Classifier procedure
- **Value:** scikit-learn `LogisticRegression` (L2, lbfgs, `max_iter` 1000, `random_state` = seed 0), fitted on the train split for each C in [0.001, 0.01, 0.1, 1, 10, 100]; C chosen by best VALIDATION macro-F1 (ties go to the smaller C); the model trained at that C (no refit on train+val) predicts the test split once. Features are standardised with the training mean and standard deviation (the same for both feature sets). Identical procedure and seed for every feature set. All misclassified test reviews are saved with their raw text.
- **Where:** `configs/p3_classify.json` (`options.c_grid`, `selection_metric`, `standardise`, `max_iter`); `src/classify.py` (`select_c`).
- **Chosen by:** logistic regression, validation-only selection, single test evaluation, outputs: User, prompt (2026-10-14). The C grid, macro-F1 as the selection metric, standardisation, `max_iter`, no refit: Assistant defaults. **Alternatives considered:** a finer or wider C grid; selecting on validation accuracy (equal here: balanced classes); no standardisation; refitting on train+validation at the chosen C.
- **Status:** NEEDS REVIEW (the assistant defaults)

---

### Y2. Error-review material (counted properties only)
- **Value:** 15 random errors per variant (seed 0) with counted properties: tokens kept, truncated or not, `<unk>` count and share, negation-word count. Negation words: not, no, never, nothing, nobody, none, neither, nor, hardly, cannot, plus any n't contraction. The same properties are averaged over all errors and over all correct reviews. The patterns themselves are not written up: your `CLAUDE.md` says interpretation is the user's job.
- **Where:** `configs/p3_error_sample.json` (`options.n_sample`, `options.negation_words`); `src/error_sample.py`.
- **Chosen by:** Assistant default (the property list came from the user's prompt: negation, length, `<unk>`, sarcasm). **Alternatives considered:** a larger sample; a longer negation list; a sample balanced by true label.
- **Status:** NEEDS REVIEW

---

## Z. Report (2026-10-15)

### Z1. Comparison question
- **Value:** "Does a language model that predicts the next word better (lower test perplexity) also give better features for classifying review sentiment?" The report is organised around it (Part I perplexity ranking vs Part III accuracy ranking, with Part II as the embedding ablation).
- **Where:** `report/REPORT_DRAFT.md` (Section 1); no config or script depends on it.
- **Chosen by:** User, 2026-10-15 (answered the assistant's request for a question). **Alternatives considered:** the other two examples in the brief's "Experimental Focus": how the model families differ on this corpus; whether fixed embeddings help.
- **Status:** CONFIRMED

### Z2. Assistant-drafted report text (exception to the CLAUDE.md rule)
- **Value:** `report/REPORT_DRAFT.md` and `report/VERIFY.md` were drafted by the assistant, as were `results/part3_error_notes.md` (2026-10-14). The *My notes* blocks in `RESULTS.md` are untouched. Every number in the draft is tagged with its results file; analytical sentences are tagged `[OBS]`, `[HYP]` or `[CHECK]`. The user must verify and rewrite them in their own words (the brief grades the user's understanding and requires tool assistance to be cited).
- **Where:** `report/`, `results/part3_error_notes.md`; logged in `TOOL_USE.md`.
- **Chosen by:** User, prompt (2026-10-14 and 2026-10-15), an explicit override of "never write report text or interpret results".
- **Status:** CONFIRMED

---

## AA. Part III extension (2026-10-15)

### AA1. Extra representations and the five-way comparison
- **Value:** mean-pooled features, same settings as before (first 256 tokens, frozen model, batch size 64), for the Part I RNN (`train_rnn`) and the Part II LSTMs with frozen own word2vec (`p2_lstm_w2v`) and frozen GloVe (`p2_lstm_glove`) embeddings; `classify.py` then runs on all five feature sets with the unchanged procedure and seed 0 (`configs/p3_classify_five.json`, separate output files, so `part3.csv` is untouched). Spearman rho (scipy) between LM test perplexity and test accuracy, and between perplexity and test macro-F1, over the five rows; the sign is reported as computed (lower perplexity is better, higher accuracy is better, so agreement shows as a negative rho). The LM perplexity is the stored 256-token test value of each language model's training record.
- **Where:** `configs/p3_feat_rnn_mean.json`, `p3_feat_lstm_w2v_mean.json`, `p3_feat_lstm_glove_mean.json`, `p3_classify_five.json`, `p3_rank_table.json`; `src/rank_table.py`; Colab section 13.
- **Chosen by:** User, prompt (2026-10-15) (which models, same settings, the table and Spearman). Spearman via `scipy.stats.spearmanr` with its two-sided p-value and the table layout: Assistant default. **Alternatives considered:** Kendall tau; flipping the sign so that agreement is positive.
- **Status:** NEEDS REVIEW (Spearman details only)

### AA2. Window rescoring definition
- **Value:** the Part I test targets (each test review cut to its first 256 tokens; inputs = tokens 0..254, targets = tokens 1..255) are scored by `evaluate.evaluate()` unchanged, with a scorer that cuts each review's input into consecutive 64-token pieces (0-63, 64-127, ...) and runs the model on each piece separately (fresh recurrent state; Transformer positions 0-63 only). Each target is predicted from the tokens before it inside its own window. `num_targets` and `target_checksum` must equal the one-piece run (the script stops otherwise). It also recomputes the one-piece 256 value as a check of the script against the stored Part I value. Models: RNN, LSTM, Transformer (Part I checkpoints).
- **Where:** `src/rescore_windows.py` (`WindowedScorer`); `configs/p3_rescore_windows.json` (`options.window` 64, `options.max_length` 256, `batch_size` 64).
- **Chosen by:** User, prompt (2026-10-15): non-overlapping 64-token windows over the same first-256-token targets. The 64-token window starts at position 0 of the review (no tokens are shared between windows), and the first target of each later window sees only one input token of context: Assistant reading of "non-overlapping windows". **Alternatives considered:** giving each later window the previous window's last token as extra context; carrying the recurrent state across windows (that is not a window rescoring).
- **Status:** NEEDS REVIEW

### AA3. PCA procedure
- **Value:** `sklearn.decomposition.PCA(n_components=100, random_state=0)` fitted on the TRAIN split's raw LSTM features only; validation and test projected with the train-fitted PCA; then the unchanged classifier procedure (features standardised with the training mean and standard deviation of the 100 PCA outputs, C chosen on validation macro-F1, test scored once). Standardising after PCA gives every component equal variance (whitening).
- **Where:** `configs/p3_classify_pca.json` (`options.pca_components`); `src/classify.py`.
- **Chosen by:** User, prompt (2026-10-15): PCA to 100 dimensions fitted on train only, same procedure. Standardising after PCA, and PCA on the raw (not standardised) features: Assistant default. **Alternatives considered:** PCA on standardised features; no standardisation after PCA.
- **Status:** NEEDS REVIEW

### AA4. McNemar variant
- **Value:** counts from the saved error files (a review absent from a classifier's error file was classified correctly); statistic = (|n10 - n01| - 1)^2 / (n10 + n01) (continuity-corrected chi-square, 1 degree of freedom) with its p-value; the exact two-sided binomial p-value is also reported.
- **Where:** `src/mcnemar.py`; `configs/p3_mcnemar.json` (A = LSTM, B = Transformer).
- **Chosen by:** User (McNemar on the LSTM vs Transformer test predictions). Continuity correction and the extra exact p-value: Assistant default. **Alternatives considered:** the uncorrected statistic.
- **Status:** NEEDS REVIEW

---

## AB. Review text in git (2026-10-15)

### AB1. Error files with review text are not committed
- **Value:** `results/part3_errors_*.csv`, `results/part3_five_errors_*.csv` and `results/part3_error_sample_*.csv` (they hold verbatim IMDB review text) are git-ignored and untracked; the local copies stay and are rebuilt by `classify.py` and `error_sample.py`.
- **Where:** `.gitignore`; `git rm --cached` on the five tracked files. Earlier commits still contain them (history not rewritten).
- **Chosen by:** User (option 1 of 3, 2026-10-15). **Alternatives considered:** strip the `text` column; keep the files tracked.
- **Reason:** the dataset's redistribution terms are unclear and the brief says not to submit data you may not share.
- **Status:** confirmed

---

## NEEDS REVIEW summary

**All 24 items resolved 2026-10-10.** The user reviewed every one (2-3 options with trade-offs each, per CLAUDE.md's record-keeping rules) and confirmed every current value as-is: A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, B1, B3, C1, C2, C3, D2, D3, D4, D5, D6, D7, D8, E1, E3. See each entry above for the specific options offered and the reasoning. No reruns were needed since nothing changed.

One item (D4) was accepted rather than fixed: the n-gram discount grid was widened and re-run after the test-set perplexity from the first run had already been seen. The test number was identical in both runs, so no actual leakage occurred this time, but the user explicitly signed off on this as a logged methodological risk rather than it being silently resolved.

**New since that review (2026-10-11), pending the user:** G2 (prompt file format), G3, G4, G5, G6, G7, G8.

**H5 (shuffling and last batch) was resolved 2026-10-11:** the user kept the current values.

**I4 (eval batch size) and I5 (training-script defaults) were resolved 2026-10-11:** the user kept the current values.

**New since then (2026-10-12), pending the user:** J1-J6 (results document and `update_results.py`).

**New since then (2026-10-12), pending the user:** K1-K3 (task list).

**2026-10-12, after reading the brief:** J1 and K1 were revised (parts now follow the brief).

**New since that review (2026-10-12), pending the user:** L3 (dropout placement and layout), L4 (smoke-test data).

**New since then (2026-10-12), pending the user:** M3, M4 (Transformer defaults).

**New since then (2026-10-13), pending the user:** N1 (causality test settings).

**New since then (2026-10-13), pending the user:** O2 (benchmark details).

**New since then (2026-10-13), pending the user:** P2 (neural generation loading and combined file).

**New since then (2026-10-13), pending the user:** Q2 (word2vec settings). Q1 (Skip-gram) confirmed.

**2026-10-13:** R1 (Part II uses the LSTM) confirmed by the user.

**2026-10-13:** Q2 (word2vec settings) confirmed; R2 (missing-word fill) confirmed. Nothing new pending.

**New since then (2026-10-13), pending the user:** S1 (which GloVe file).

**New since then (2026-10-14), pending the user:** T1 (frozen-embedding option and configs), T2 (check and parameter-count convention). S1 still pending.

**New since then (2026-10-14), pending the user:** U1 (Part II generation set-up), U2 (Part II section of RESULTS.md).

**2026-10-14:** every remaining NEEDS REVIEW item (G2-G8, J1-J6, K1-K3, L3, L4, M3, M4, N1, O2, P2, S1, T1, T2, U1, U2: 28 in total) was reviewed one by one and kept as is by the user. V1 (no reruns) added and confirmed. Nothing is pending. Open fact, not a decision: the "Test set used only for final evaluation" check still shows FAIL because `src/benchmark.py` reads `data.test` (J4 kept the static scan).

**New since then (2026-10-14), pending the user:** W2 (classification data details). W1 confirmed.

**New since then (2026-10-14), pending the user:** W2 (above), X1 (feature extraction details), X2 (pooling `mean` for the LSTM vs Transformer comparison), Y1 (classifier grid and standardisation), Y2 (error-review properties).

**Current state (2026-10-15).** Pending the user: W2, X1, X2, Y1, Y2 (five items, all Part III). Everything earlier in this section was reviewed and kept as is by the user; the dated lines above are the history, not a to-do list. Z1 and Z2 are confirmed. Open fact, not a decision: the "Test set used only for final evaluation" check still shows FAIL because `src/benchmark.py` reads `data.test` (the statically scanned rule from J4 was kept), and "All models scored on identical test targets" still shows FAIL because the n-gram model scores 1,390,673 targets (no truncation) while the neural models score 981,353 (reviews cut to 256 tokens; decision C2). `results/part1_costs.csv` and the Part I table in `results/tables/` score all four models on the same 981,353 targets.

**New since then (2026-10-15), pending the user:** AA1 (Spearman details), AA2 (window rescoring definition), AA3 (PCA procedure), AA4 (McNemar variant).
