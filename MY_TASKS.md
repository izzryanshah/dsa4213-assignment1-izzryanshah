# My tasks

Rebuilt by `src/update_results.py` on Wed 14 Oct 2026, 17:59 Singapore time. **Deadline: Fri 16 Oct 2026, 11:59 PM SGT** (2 day(s) left; today is Wed 14 Oct).

Tick a box yourself with `[x]`; this file never ticks anything for you, and keeps your ticks and anything you write between the marker lines. Do not delete the `<!--id:...-->` tags at the end of each line.

Contents: [Waiting on me now](#waiting-on-me-now) · [Decisions to review](#decisions-to-review) · [Runs to do](#runs-to-do) · [Results to interpret](#results-to-interpret) · [Report and submission](#report-and-submission) · [Failed checks](#failed-checks)

## Waiting on me now

- [ ] Put your planned day for each task into `schedule` in `configs/update_results.json` (item id -> `YYYY-MM-DD`, or `section:<key>` -> date). Until then every item shows "Planned: not set" and nothing can be overdue *(Planned: not set)* <!--id:wait-schedule-->
- [ ] Review 3 NEEDS REVIEW decision(s): see [Decisions to review](#decisions-to-review) *(Planned: not set)* <!--id:wait-decisions-->
- [ ] Write your notes for Data, Part I, Part II, Part III: see [Results to interpret](#results-to-interpret) *(Planned: not set)* <!--id:wait-notes-->
- [ ] Fix failing check: Test set used only for final evaluation: see [Failed checks](#failed-checks) *(Planned: not set)* <!--id:wait-fail-test-set-used-only-for-final-evaluation-->
- [ ] Fix failing check: All models scored on identical test targets: see [Failed checks](#failed-checks) *(Planned: not set)* <!--id:wait-fail-all-models-scored-on-identical-test-targets-->
- [ ] Commit and push: 11 uncommitted file(s), 0 unpushed commit(s). The instructor can only see what is pushed *(Planned: not set)* <!--id:wait-commit-->
- [ ] Choose your comparison question (brief, "Experimental Focus": e.g. how model families differ on your corpus, whether fixed embeddings help, or whether better next-token prediction leads to better classification). The report is organised around it *(Planned: not set)* <!--id:w-question-->
  - Tell Claude your question; it is then logged in `DECISIONS.md`

### Added by me

<!-- my-tasks:waiting:start -->

<!-- my-tasks:waiting:end -->

## Decisions to review

- [ ] **W2** Details not covered by the user's choice *(Planned: not set)* <!--id:dec-W2-->
  - Options: keep `<eos>` when the review fits (then short and long reviews end differently).
  - Current value: (a) the `<eos>` marker data.py appended to each review is dropped (it separates documents for the language models; it is not part of a review, and truncated rev...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry W2 (line 644), which sets `Chosen by` and `Status`
- [ ] **X1** Which vector is pooled, and how padding is excluded *(Planned: not set)* <!--id:dec-X1-->
  - Options: pooling an earlier layer; pooling before the Transformer's final LayerNorm; concatenating last and mean.
  - Current value: the pooled vector is the one fed to the model's output layer: output of the last recurrent layer (RNN/LSTM, size 256) or of the final LayerNorm (Transformer, si...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry X1 (line 654), which sets `Chosen by` and `Status`
- [ ] **X2** Part III comparison and pooling *(Planned: not set)* <!--id:dec-X2-->
  - Options: last vs mean on one model; full 2x2 grid; adding the Part II frozen-embedding LSTMs.
  - Current value: compare the Part I LSTM and Part I Transformer (both random trainable embeddings, as trained), same pooling for both: `mean`. Runs: `p3_feat_lstm_mean`, `p3_fea...
  - Record your choice: tell Claude in chat; it is written to `DECISIONS.md` entry X2 (line 662), which sets `Chosen by` and `Status`
- [ ] Part II: choose which one Part I architecture (RNN, LSTM or Transformer) gets the three embedding settings *(Planned: not set)* <!--id:dec-p2-arch-->
  - Decide after Part I results exist
- [ ] Part III: choose the labeled task and dataset, and the two or more representation choices to compare (brief examples: LSTM vs Transformer, two embedding variants, two pooling methods) *(Planned: not set)* <!--id:dec-p3-task-->
  - IMDB sentiment labels are already in `data/IMDB Dataset.csv`; the brief also allows a different dataset

### Added by me

<!-- my-tasks:decisions:start -->

<!-- my-tasks:decisions:end -->

## Runs to do

- [ ] Download public GloVe 100d vectors (GloVe 6B, Stanford NLP: nlp.stanford.edu/projects/glove) in Colab, never on this machine *(Planned: not set)* <!--id:run-p2-glove-->
  - Command: `notebooks/run_colab.ipynb`, section 8 (downloads glove.6B.zip, then `python -u src/load_glove.py --config configs/glove_coverage.json`)
  - Time: a few minutes (about 822 MB download)
  - Produces: `data/embeddings/glove_100d.npy` (copied to Drive) and `results/glove_coverage.json` (coverage and 20 most frequent missing words); the GloVe file itself must not be committed; cite Pennington et al. (2014) in the report
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Train your own word embeddings on the training split (Skip-gram, CBOW or GloVe), then freeze them *(Planned: not set)* <!--id:run-p2-own-->
  - Command: `python3 src/train_word2vec.py --config configs/word2vec_cost.json` (already run once locally: `results/word2vec_cost.json`, `data/embeddings/w2v_100d.npy`)
  - Time: about 5 minutes on the laptop CPU (cost recorded separately from LM training)
  - Produces: the vectors file and `results/word2vec_cost.json`; still to do: copy `w2v_100d.npy` to Drive for the Colab LM runs
  - May run locally (word2vec/embedding training is not Colab-only)
- [ ] Train the two new Part II language models (own frozen embeddings; public frozen embeddings), same architecture and procedure as the reused Part I model *(Planned: not set)* <!--id:run-p2-lms-->
  - Command: `notebooks/run_colab.ipynb`, sections 9 and 10: `python -u src/train.py --config configs/p2_lstm_w2v.json --model lstm --embedding w2v` and `python -u src/train.py --config configs/p2_lstm_glove.json --model lstm --embedding glove` (needs `data/embeddings/w2v_100d.npy` and `glove_100d.npy` on Colab)
  - Time: about 1 hour each on a T4 (the Part I LSTM took 3,850 s)
  - Produces: `results/p2_*.json`, loss curves, the frozen-embeddings check in `RESULTS.md`
  - Colab only (CLAUDE.md rule 9): never run on this machine
- [ ] Extract features from the frozen language models (`extract_features.py`, Colab only), then train the classifier for each representation choice (`classify.py`, may run locally; same classifier family and evaluation procedure) *(Planned: not set)* <!--id:run-p3-clf-->
  - Command: not available yet: no classifier script exists yet; name the runs `p3_*` so they land in Part III
  - Time: not known yet; needs the saved checkpoints from Parts I-II (save `checkpoints/` to Drive on Colab)
  - Produces: `results/p3_*.json` with accuracy or macro-F1
  - Feature extraction: Colab only (CLAUDE.md rule 9): never run on this machine

### Added by me

<!-- my-tasks:runs:start -->

<!-- my-tasks:runs:end -->

## Results to interpret

- [ ] Write your notes for [Data](RESULTS.md#data) *(Planned: not set)* <!--id:interp-data-->
  - Runs in this section: `data`, `inspect_data`, `show_batch`
  - Where: the empty *My notes* block at the end of the Data section of `RESULTS.md`
- [ ] Write your notes for [Part I](RESULTS.md#part-i) *(Planned: not set)* <!--id:interp-part1-->
  - Runs in this section: `generations_lstm`, `generations_ngram`, `generations_rnn`, `generations_transformer`, `ngram`, `sanity_check`, `test_causality`, `test_evaluate`, `test_frozen_embedding`, `test_generate`, `test_models`, `test_train`, `train_bigram`, `train_bigram_colab`, `train_lstm`, `train_rnn`, `train_transformer`
  - Where: the empty *My notes* block at the end of the Part I section of `RESULTS.md`
- [ ] Write your notes for [Part II](RESULTS.md#part-ii) *(Planned: not set)* <!--id:interp-part2-->
  - Runs in this section: `generations_p2_lstm_glove`, `generations_p2_lstm_w2v`, `glove_coverage`, `p2_lstm_glove`, `p2_lstm_w2v`, `word2vec_cost`
  - Where: the empty *My notes* block at the end of the Part II section of `RESULTS.md`
- [ ] Write your notes for [Part III](RESULTS.md#part-iii) *(Planned: not set)* <!--id:interp-part3-->
  - Runs in this section: `p3_feat_lstm_mean`, `p3_feat_transformer_mean`, `p3_prepare_classification`
  - Where: the empty *My notes* block at the end of the Part III section of `RESULTS.md`

### Added by me

<!-- my-tasks:interpret:start -->

<!-- my-tasks:interpret:end -->

## Report and submission

- [ ] Part I (40 pts): n-gram, RNN, LSTM and Transformer trained on the same corpus; neural models use random trainable embeddings *(Planned: not set)* <!--id:req-p1-models-->
  - Status: done (all present: n-gram, RNN, LSTM, Transformer)
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
  - Status: done (Checks section: "Frozen embeddings unchanged" is PASS)
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
  - Status: not done (remote `origin` = git@github.com:izzryanshah/dsa4213-assignment1-izzryanshah.git; 11 uncommitted file(s); 0 unpushed commit(s). Whether the instructor has access cannot be checked from here)
- [ ] Find out the late-submission policy and the submission platform (the brief says both are announced separately) *(Planned: not set)* <!--id:req-late-->
  - Status: not done (cannot be checked automatically; tick when finished)

### Added by me

<!-- my-tasks:report:start -->

<!-- my-tasks:report:end -->

## Failed checks

- [ ] Test set used only for final evaluation: FAIL *(Planned: not set)* <!--id:fail-test-set-used-only-for-final-evaluation-->
  - Evidence (Checks section of `RESULTS.md`): `src/ngram.py`: `data.test` used at line(s) [333]; `tune_discounts(...)` called at line(s) [321]; single use after the call
  - Evidence (Checks section of `RESULTS.md`): `src/train.py`: `data.test` used at line(s) [332]; `fit(...)` called at line(s) [320]; single use after the call
  - Evidence (Checks section of `RESULTS.md`): other files in `src/` that use `data.test`: `src/benchmark.py` lines [152]
  - Evidence (Checks section of `RESULTS.md`): method: static scan of the source (Python `ast`), not a run-time trace
  - Needs fixing: the evidence above shows what does not hold; after fixing, re-run `update_results.py`
- [ ] All models scored on identical test targets: FAIL *(Planned: not set)* <!--id:fail-all-models-scored-on-identical-test-targets-->
  - Evidence (Checks section of `RESULTS.md`): `ngram`: 1,390,673 targets, checksum `fd778f4c313d...`
  - Evidence (Checks section of `RESULTS.md`): `p2_lstm_glove`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `p2_lstm_w2v`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_bigram`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_bigram_colab`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_lstm`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_rnn`: 981,353 targets, checksum `4f77927a1cd9...`
  - Evidence (Checks section of `RESULTS.md`): `train_transformer`: 981,353 targets, checksum `4f77927a1cd9...`
  - Needs fixing: the evidence above shows what does not hold; after fixing, re-run `update_results.py`

### Added by me

<!-- my-tasks:failed:start -->

<!-- my-tasks:failed:end -->
