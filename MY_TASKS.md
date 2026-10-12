# My tasks

Rebuilt by `src/update_results.py` on Mon 12 Oct 2026, 21:58 Singapore time. **Deadline: Fri 16 Oct 2026, 11:59 PM SGT** (4 day(s) left; today is Mon 12 Oct).

Tick a box yourself with `[x]`; this file never ticks anything for you, and keeps your ticks and anything you write between the marker lines. Do not delete the `<!--id:...-->` tags at the end of each line.

Contents: [Waiting on me now](#waiting-on-me-now) · [Decisions to review](#decisions-to-review) · [Runs to do](#runs-to-do) · [Results to interpret](#results-to-interpret) · [Report and submission](#report-and-submission) · [Failed checks](#failed-checks)

## Waiting on me now

- [ ] Put your planned day for each task into `schedule` in `configs/update_results.json` (item id -> `YYYY-MM-DD`, or `section:<key>` -> date). Until then every item shows "Planned: not set" and nothing can be overdue *(Planned: not set)* <!--id:wait-schedule-->
- [ ] Review 20 NEEDS REVIEW decision(s): see [Decisions to review](#decisions-to-review) *(Planned: not set)* <!--id:wait-decisions-->
- [ ] Re-run `train_bigram` in Colab (it ran locally): see [Runs to do](#runs-to-do) *(Planned: not set)* <!--id:wait-rerun-train_bigram-->
- [ ] Start run `train_transformer`: see [Runs to do](#runs-to-do) *(Planned: not set)* <!--id:wait-run-train_transformer-->
- [ ] Write your notes for Data, Part I: see [Results to interpret](#results-to-interpret) *(Planned: not set)* <!--id:wait-notes-->
- [ ] Fix failing check: All models scored on identical test targets: see [Failed checks](#failed-checks) *(Planned: not set)* <!--id:wait-fail-all-models-scored-on-identical-test-targets-->
- [ ] Commit and push: 16 uncommitted file(s), 0 unpushed commit(s). The instructor can only see what is pushed *(Planned: not set)* <!--id:wait-commit-->
- [ ] Choose your comparison question (brief, "Experimental Focus": e.g. how model families differ on your corpus, whether fixed embeddings help, or whether better next-token prediction leads to better classification). The report is organised around it *(Planned: not set)* <!--id:w-question-->
  - Tell Claude your question; it is then logged in `DECISIONS.md`

### Added by me

<!-- my-tasks:waiting:start -->

<!-- my-tasks:waiting:end -->

## Decisions to review

- [ ] **G2** Prompts file and its format *(Planned: not set)* <!--id:dec-G2-->
  - Options: JSON list of prompts; one prompt per paragraph.
  - Current value: prompts are read from `prompts.txt` (a file the user is writing); one prompt per line; blank lines and lines starting with `#` are skipped. The prompts are clea...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G2 (line 268), which sets `Chosen by` and `Status`
- [ ] **G3** Random seed per generation *(Planned: not set)* <!--id:dec-G3-->
  - Options: one generator seeded once at the start (simpler, but adding or reordering prompts would change every other prompt's text).
  - Current value: each (prompt, sample number) gets its own random generator seeded from `[seed, sample number, CRC32 of the prompt text]`, so output depends only on those three ...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G3 (line 275), which sets `Chosen by` and `Status`
- [ ] **G4** Order of the sampling dials, and their edge cases *(Planned: not set)* <!--id:dec-G4-->
  - Options: top-k before banning (could leave fewer than k usable words); support temperature 0 as greedy decoding.
  - Current value: banned words are removed first, then temperature is applied, then top-k (so top-k keeps the k likeliest *allowed* words), then the probabilities are rescaled. T...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G4 (line 281), which sets `Chosen by` and `Status`
- [ ] **G5** How generated text is stored *(Planned: not set)* <!--id:dec-G5-->
  - Options: keep `<eos>` in the output; detokenize punctuation back onto words.
  - Current value: the stopping `<eos>` is not included in the stored tokens (a flag `ended_with_eos` records it); text is the prompt tokens plus new tokens joined with single spa...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G5 (line 287), which sets `Chosen by` and `Status`
- [ ] **G6** How the n-gram model is prepared for generation *(Planned: not set)* <!--id:dec-G6-->
  - Options: save and load a fitted model file; re-tune on every generation run.
  - Current value: the counts are re-fitted from the training split (about 1 second); the discounts are NOT re-tuned but read from the saved n-gram run (`results/ngram.json`, `det...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G6 (line 293), which sets `Chosen by` and `Status`
- [ ] **G7** How neural models plug in *(Planned: not set)* <!--id:dec-G7-->
  - Options: require each model to define its own generation step (e.g. caching hidden states for speed).
  - Current value: any `nn.Module` with `forward(ids) -> logits` is wrapped by `NeuralNextToken`, which feeds the whole context (prompt + generated so far) each step and reads the...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G7 (line 299), which sets `Chosen by` and `Status`
- [ ] **G8** Generation files are full run records *(Planned: not set)* <!--id:dec-G8-->
  - Options: a separate, lighter file format for generations.
  - Current value: `results/generations_<model>.json` is a normal run record (config, seed, commit, hardware, date, metrics) with the decoding settings (temperature, top_k, max_to...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry G8 (line 305), which sets `Chosen by` and `Status`
- [ ] **J1** Which runs belong to Data / Part I / Part II / Part III (revised 2026-10-12 after reading the assignment brief) *(Planned: not set)* <!--id:dec-J1-->
  - Options: a `part` field inside each config; a separate Part for baselines.
  - Current value: the brief defines the parts: Part I = n-gram, RNN, LSTM, Transformer comparison; Part II = three embedding settings on one architecture; Part III = downstream c...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry J1 (line 389), which sets `Chosen by` and `Status`
- [ ] **J2** "Close to the vocabulary size" threshold *(Planned: not set)* <!--id:dec-J2-->
  - Options: a tighter band (0.8 to 1.25); report the ratio with no pass/fail line.
  - Current value: untrained validation perplexity divided by vocabulary size must be between 0.5 and 2.0 to PASS.
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry J2 (line 396), which sets `Chosen by` and `Status`
- [ ] **J3** How the exact command of each run is known *(Planned: not set)* <!--id:dec-J3-->
  - Options: re-run old runs to get commands stored (slow, changes their dates); write commands by hand into a file.
  - Current value: `save_run` now stores a `command` field (interpreter + arguments) in every new `results/<run>.json`. For older records, the command is copied from `results/RESU...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry J3 (line 403), which sets `Chosen by` and `Status`
- [ ] **J4** How the test-set and padding checks are evidenced *(Planned: not set)* <!--id:dec-J4-->
  - Options: run-time instrumentation of the data loader; only trusting the unit tests.
  - Current value: static scan of the source with Python `ast`: in `src/ngram.py` and `src/train.py`, `data.test` must be used exactly once, on a line after the `tune_discounts(.....
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry J4 (line 410), which sets `Chosen by` and `Status`
- [ ] **J5** How "My notes" are protected *(Planned: not set)* <!--id:dec-J5-->
  - Options: keep notes in a separate file linked from RESULTS.md.
  - Current value: each section ends with `### My notes` and a pair of HTML-comment marker lines; text between the markers is copied back unchanged on every rebuild. If the marker...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry J5 (line 417), which sets `Chosen by` and `Status`
- [ ] **J6** Figures, and `update_results.py` not saving a run record *(Planned: not set)* <!--id:dec-J6-->
  - Options: an interactive HTML report; saving a run record for each rebuild.
  - Current value: matplotlib PNGs in `results/figures/`: tokens per split, n-gram discount tuning (D3 axis, D2/D1 at their chosen values), train/validation loss per trained run, ...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry J6 (line 424), which sets `Chosen by` and `Status`
- [ ] **K1** Report and submission requirement list (revised 2026-10-12 from the brief) *(Planned: not set)* <!--id:dec-K1-->
  - Options: one item per part.
  - Current value: 23 requirements taken from `Assignment1_2026.pdf` (Part I, II, III bullets, the report contents list, 6-page limit, citations, README, no credentials, submissio...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry K1 (line 437), which sets `Chosen by` and `Status`
- [ ] **K2** Schedule and planned days *(Planned: not set)* <!--id:dec-K2-->
  - Options: spread items evenly over the days left (would invent a schedule).
  - Current value: `options.schedule` is empty, so every item shows "Planned: not set" and nothing can be overdue. Deadline is Fri 16 Oct 2026, 23:59 Singapore time (from the user...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry K2 (line 444), which sets `Chosen by` and `Status`
- [ ] **K3** What counts as "waiting on me now", and how interpretation items are counted *(Planned: not set)* <!--id:dec-K3-->
  - Options: one interpretation item per run; treat every NEEDS REVIEW item as blocking.
  - Current value: Waiting = empty schedule, unresolved NEEDS REVIEW decisions, configs with no results yet, sections with results but empty *My notes*, failing checks, uncommitte...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry K3 (line 451), which sets `Chosen by` and `Status`
- [ ] **L3** Dropout placement and layout *(Planned: not set)* <!--id:dec-L3-->
  - Options: no dropout on the embedding; tie the output layer to the embedding (needs hidden size = 100); carry the hidden state across consecutive windows.
  - Current value: dropout after the embedding, between stacked recurrent layers (PyTorch's `dropout=`, set to 0 when there is 1 layer) and before the output layer; unidirectional...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry L3 (line 474), which sets `Chosen by` and `Status`
- [ ] **L4** Smoke-test data *(Planned: not set)* <!--id:dec-L4-->
  - Options: re-run `data.py` with `subsample_size` (would overwrite the saved splits).
  - Current value: the first 60 train / 40 val / 40 test reviews of the saved splits, copied to a scratch folder (not `data/processed`), 1 epoch, output to a scratch results folde...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry L4 (line 480), which sets `Chosen by` and `Status`
- [ ] **M3** Block style and other defaults *(Planned: not set)* <!--id:dec-M3-->
  - Options: post-norm blocks; ReLU; tied input/output embeddings.
  - Current value: pre-norm blocks (LayerNorm before attention and feed-forward) plus one final LayerNorm; GELU activation; dropout after the embeddings and inside each block; pos...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry M3 (line 502), which sets `Chosen by` and `Status`
- [ ] **M4** Training settings copied from the LSTM *(Planned: not set)* <!--id:dec-M4-->
  - Options: a smaller learning rate or warmup, which Transformers often prefer.
  - Current value: learning rate 0.001, batch size 64, epochs 20, context length 64, patience 3, grad clip 1.0, vocabulary 20000, seed 0.
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry M4 (line 508), which sets `Chosen by` and `Status`
- [ ] Part II: choose which one Part I architecture (RNN, LSTM or Transformer) gets the three embedding settings *(Planned: not set)* <!--id:dec-p2-arch-->
  - Decide after Part I results exist
- [ ] Part III: choose the labeled task and dataset, and the two or more representation choices to compare (brief examples: LSTM vs Transformer, two embedding variants, two pooling methods) *(Planned: not set)* <!--id:dec-p3-task-->
  - IMDB sentiment labels are already in `data/IMDB Dataset.csv`; the brief also allows a different dataset

### Added by me

<!-- my-tasks:decisions:start -->

<!-- my-tasks:decisions:end -->

## Runs to do

- [ ] Run `train_transformer` *(Planned: not set)* <!--id:run-train_transformer-->
  - Command: `python3 src/train.py --config configs/train_transformer.json` (in Colab: the same command with a leading `!`, from inside the repo folder)
  - Time: earlier run `train_bigram` of this script took 16,611 s (about 4.6 h) on CPU: Apple M2 (Darwin arm64)
  - Produces: `results/train_transformer.json`, then a new section in `RESULTS.md` after `update_results.py`
- [ ] Train the Transformer language model (Part I) *(Planned: not set)* <!--id:run-p1-transformer-->
  - Command (in Colab): `!python -u src/train.py --config configs/train_transformer.json --model transformer` (cell 4 in `notebooks/run_colab.ipynb`)
  - Time: not known yet; the RNN took 3,388 s and the LSTM 3,850 s on a Tesla T4
  - Produces: `results/train_transformer.json`, `checkpoints/train_transformer.pt`, loss-curve figure and a new Part I section in `RESULTS.md`
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Time inference for all Part I models on one common workload, in `benchmark.py` on the same Colab GPU for every model (brief: state hardware, budget and timing procedure) *(Planned: not set)* <!--id:run-p1-inference-->
  - Command: not available yet: no inference-timing script exists yet
  - Time: not known yet
  - Produces: `inference_tokens_per_second` in each model's record, shown in the Part I table
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Generate continuations from the same prompts with matched decoding for each neural model (in Colab) *(Planned: not set)* <!--id:run-p1-generations-->
  - Command: `python3 src/generate.py --config configs/generations_<model>.json` (the n-gram one is `configs/generations_ngram.json`; neural configs are written once the models exist)
  - Time: not known yet
  - Produces: `results/generations_<model>.json`; side-by-side text in `RESULTS.md` Part I
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Download public GloVe 100d vectors (GloVe 6B, Stanford NLP: nlp.stanford.edu/projects/glove) in Colab, never on this machine *(Planned: not set)* <!--id:run-p2-glove-->
  - Command: not available yet: the Colab cell is added together with the Part II code that loads the file; the vectors file must not be committed
  - Produces: the vectors file on Colab/Drive; cite Pennington et al. (2014) in the report
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Train your own word embeddings on the training split (Skip-gram, CBOW or GloVe), then freeze them *(Planned: not set)* <!--id:run-p2-own-->
  - Command: not available yet: no embedding-training script exists yet
  - Time: not known yet; the brief wants this cost reported separately from LM training
  - Produces: the vectors file and a `p2_` run record
  - May run locally (word2vec/embedding training is not Colab-only)
- [ ] Train the two new Part II language models (own frozen embeddings; public frozen embeddings), same architecture and procedure as the reused Part I model *(Planned: not set)* <!--id:run-p2-lms-->
  - Command: not available yet: `src/train.py` cannot load or freeze pretrained embeddings yet; name the runs `p2_*` so they land in Part II
  - Time: not known yet
  - Produces: `results/p2_*.json`, loss curves, the frozen-embeddings check in `RESULTS.md`
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Extract features from the frozen language models (`extract_features.py`, Colab only), then train the classifier for each representation choice (`classify.py`, may run locally; same classifier family and evaluation procedure) *(Planned: not set)* <!--id:run-p3-clf-->
  - Command: not available yet: no classifier script exists yet; name the runs `p3_*` so they land in Part III
  - Time: not known yet; needs the saved checkpoints from Parts I-II (save `checkpoints/` to Drive on Colab)
  - Produces: `results/p3_*.json` with accuracy or macro-F1
  - Feature extraction: Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Re-run `train_bigram` in Colab: it was run locally on CPU: Apple M2 (Darwin arm64), and Colab-only runs must come from the Colab GPU *(Planned: not set)* <!--id:rerun-train_bigram-->
  - Command (Colab cell): `!python -u src/train.py --config configs/train_bigram.json --model bigram --run-name train_bigram_colab` (a new run name, so the local result is kept)
  - Time: the local run took 16,611 s (about 4.6 h) on CPU; GPU time not known yet
  - Produces: `results/train_bigram_colab.json`, `checkpoints/train_bigram_colab.pt`; the local `results/train_bigram.json` stays as it is

### Added by me

<!-- my-tasks:runs:start -->

<!-- my-tasks:runs:end -->

## Results to interpret

- [ ] Write your notes for [Data](RESULTS.md#data) *(Planned: not set)* <!--id:interp-data-->
  - Runs in this section: `data`, `inspect_data`, `show_batch`
  - Where: the empty *My notes* block at the end of the Data section of `RESULTS.md`
- [ ] Write your notes for [Part I](RESULTS.md#part-i) *(Planned: not set)* <!--id:interp-part1-->
  - Runs in this section: `generations_ngram`, `ngram`, `sanity_check`, `test_evaluate`, `test_generate`, `test_models`, `test_train`, `train_bigram`, `train_lstm`, `train_rnn`
  - Where: the empty *My notes* block at the end of the Part I section of `RESULTS.md`

### Added by me

<!-- my-tasks:interpret:start -->

<!-- my-tasks:interpret:end -->

## Report and submission

- [ ] Part I (40 pts): n-gram, RNN, LSTM and Transformer trained on the same corpus; neural models use random trainable embeddings *(Planned: not set)* <!--id:req-p1-models-->
  - Status: not done (no run yet for: Transformer (architectures with results: bigram, bigram_lm, lstm, rnn, trigram_kneser_ney))
  - Where to look: `RESULTS.md` Part I
- [ ] Part I: test perplexity or cross-entropy on the common setup, padding excluded *(Planned: not set)* <!--id:req-p1-quality-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` Part I table and Checks
- [ ] Part I: training time, model size, inference time or throughput; hardware, training budget and timing procedure stated *(Planned: not set)* <!--id:req-p1-cost-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` Part I table
- [ ] Part I: training and validation loss curves, with convergence or overfitting discussed *(Planned: not set)* <!--id:req-p1-curves-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` figures; your *My notes*
- [ ] Part I: generated continuations compared (same prompts, matched decoding and lengths); repetition, local fluency, consistency discussed *(Planned: not set)* <!--id:req-p1-text-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` Part I, Generated text by prompt
- [ ] Part I: tradeoffs explained; observed differences kept separate from possible explanations (architecture, parameter count, context length, training budget) *(Planned: not set)* <!--id:req-p1-tradeoffs-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Part II (30 pts): one architecture, three embedding settings (random learned; own, frozen; public, frozen) with dimension, architecture, data and training procedure identical *(Planned: not set)* <!--id:req-p2-design-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` Part II
- [ ] Part II: vocabulary coverage of the public embeddings documented, and how missing words are initialised *(Planned: not set)* <!--id:req-p2-coverage-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Part II: the entire input-embedding table is fixed in both fixed settings (verified) *(Planned: not set)* <!--id:req-p2-frozen-->
  - Status: not done (Checks section: "Frozen embeddings unchanged" is NOT YET AVAILABLE)
  - Where to look: `RESULTS.md` Checks
- [ ] Part II: cost of training your own embeddings reported separately from LM training *(Planned: not set)* <!--id:req-p2-cost-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Part II: test quality, convergence, training stability and generated text compared; interpreted with embedding trainability and the extra data behind public embeddings in mind *(Planned: not set)* <!--id:req-p2-analysis-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Part III (20 pts): at least two representation choices compared on the same labeled data, with the same simple classifier family and evaluation procedure *(Planned: not set)* <!--id:req-p3-reps-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` Part III
- [ ] Part III: how each input becomes a fixed-length representation described (e.g. final hidden state, or mean of token states excluding padding) *(Planned: not set)* <!--id:req-p3-describe-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Part III: held-out metric (accuracy or macro-F1) reported and representative errors examined *(Planned: not set)* <!--id:req-p3-metric-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Part III: which representation works better explained; if different language models, whether the LM ranking agrees with the downstream ranking *(Planned: not set)* <!--id:req-p3-ranking-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Report is organised around one clear comparison question *(Planned: not set)* <!--id:req-question-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Report states: dataset and preprocessing; models and variants; evaluation criteria; main quantitative and qualitative observations; most important limitations, failure cases and lessons learned *(Planned: not set)* <!--id:req-contents-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `RESULTS.md` Data and Settings; your *My notes*
- [ ] Report PDF is at most 6 pages, excluding appendices and references *(Planned: not set)* <!--id:req-pages-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Citations for external code, datasets, pretrained models and substantial tool assistance (e.g. Claude Code, course notebooks, GloVe, the IMDB data) *(Planned: not set)* <!--id:req-citations-->
  - Status: not done (cannot be checked automatically; tick when finished)
  - Where to look: `README.md` Citation section; `TOOL_USE.md`
- [ ] README.md with data access, preprocessing instructions, dependencies, commands, configurations and random seeds for the main results *(Planned: not set)* <!--id:req-readme-->
  - Status: done (`README.md` matches every pattern ['(?i)data access', '(?i)preprocess', '(?i)requirements', '(?i)--config', '(?i)seed'] (keywords only; completeness is not checked))
  - Where to look: `README.md`
- [ ] Nothing submitted contains API keys, credentials or data you may not share (`data/` is git-ignored) *(Planned: not set)* <!--id:req-secrets-->
  - Status: not done (cannot be checked automatically; tick when finished)
- [ ] Submit the PDF report and the code (a GitHub link the instructor can open, or a zip) *(Planned: not set)* <!--id:req-submit-->
  - Status: not done (remote `origin` = git@github.com:izzryanshah/dsa4213-assignment1-izzryanshah.git; 16 uncommitted file(s); 0 unpushed commit(s). Whether the instructor has access cannot be checked from here)
- [ ] Find out the late-submission policy and the submission platform (the brief says both are announced separately) *(Planned: not set)* <!--id:req-late-->
  - Status: not done (cannot be checked automatically; tick when finished)

### Added by me

<!-- my-tasks:report:start -->

<!-- my-tasks:report:end -->

## Failed checks

- [ ] All models scored on identical test targets: FAIL *(Planned: not set)* <!--id:fail-all-models-scored-on-identical-test-targets-->
  - Evidence (Checks section of `RESULTS.md`): `ngram`: 1,390,673 targets, checksum `fd778f4c313d...`
  - Evidence (Checks section of `RESULTS.md`): `train_bigram`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_lstm`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_rnn`: 981,353 targets, checksum `4f77927a1cd9...`
  - Needs fixing: the evidence above shows what does not hold; after fixing, re-run `update_results.py`

### Added by me

<!-- my-tasks:failed:start -->

<!-- my-tasks:failed:end -->
