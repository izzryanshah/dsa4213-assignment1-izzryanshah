# VERIFY: what to check in `report/REPORT_DRAFT.md` before you rewrite it

Written 2026-10-15 by the assistant. Section numbers refer to `REPORT_DRAFT.md`.

## 0. Problems with the inputs (read first)

1. **`report/template.docx` is a 0-byte file**, so it could not be read. The draft follows `~/Downloads/DSA4213_Assignment1_Report_Template.docx` (section list and page budgets). Check that this is the template you meant, and replace the empty file.
2. **Your message had two unfilled placeholders.** The brief path `[path to Assignment1_2026.pdf]` was not found on disk, so I used `ASSIGNMENT.md` (its restatement; the PDF is the authoritative text). `[YOUR COMPARISON QUESTION]` was set by your answer: *does better next-token prediction mean better classification?*
3. **All six *My notes* blocks in `RESULTS.md` are empty** (data, part1, part2, part3, settings, checks). There was nothing of yours to build on, so every analytical point in the draft is the assistant's. The assistant-drafted `results/part3_error_notes.md` (written earlier at your request) was used for the Part III error patterns; it is not your text either.
4. **`README.md` was out of date when this draft was written; it was rewritten on 2026-10-15.** It now lists the commands, configs and seeds for Parts I to III and the Colab steps. Check that its commands match what you actually ran, and fill in the course-notebook line in its last section.
5. **Figure 2 differs from the template.** The template asks for validation curves only; `results/figures/part2_loss_curves.pdf` shows training and validation loss, one panel per setting. Same for Figure 1 (the template says "training and validation", which matches).

## 1. Page-count estimate (main body, excluding references, tool-use statement and appendices)

Rough model: 600 prose words per page, 0.035 page per table row, 0.3 page per figure. Tags and source brackets are not counted. This is an estimate; check it in your actual template.

| Section | Budget (pages) | Estimate | Status |
|---|---|---|---|
| 1 Introduction | 0.3 | 0.19 | within |
| 2 Data | 0.5 | 0.46 | within |
| 3 Part I | 2.0 | **3.63** (1,618 prose words, 18 table rows, 1 figure) | **over by about 1.6** |
| 4 Part II | 1.5 | **1.74** (760 prose words) | **over by about 0.25** |
| 5 Part III | 1.0 | **1.86** (908 prose words, 10 table rows) | **over by about 0.85** |
| 6 Discussion | 0.4 | 0.33 | within |
| 7 Limitations | 0.3 | **0.35** | slightly over |
| **Total** | 6.0 | **about 8.6** | **over by about 2.6 pages** |

Where to cut: Section 3.3 observations and 3.5 interpretation (several bullets repeat each other); Section 3.4 (move more text to Appendix B); Section 5.3 and 5.4 (shorten the property list and merge bullets); the Table 2 and Table 5 notes. You can also shrink figures or tables.

## 2. Every [HYP] and [CHECK] item

**[HYP] items (each names an alternative and says whether the experiments can separate them):**

| # | Location | Claim | Alternative(s) named | Separable by our runs? |
|---|---|---|---|---|
| H1 | 3.3, timing caveat | run-to-run training-time spread comes from Colab session | code version (Part I LSTM on older commit) | no |
| H2 | 3.4 | LSTM's lower perplexity not visible in 3 samples because too few samples | perplexity differences of this size don't change fluency | no (needs larger rating) |
| H3 | 3.5, bullet 1 | LSTM beats RNN because of gating | parameter count; RNN limited by shared lr/clipping | no |
| H4 | 3.5, bullet 2 | Transformer worst because positions 64 to 255 untrained | fewest params; untuned LSTM settings; overfitting; too few epochs | no |
| H5 | 3.5, bullet 3 | trigram competitive with RNN because short context carries most information | RNN undertrained; evaluation-setup differences | no (no context-length experiment) |
| H6 | 4.3, bullet 3 | trainability costs little because Skip-gram vectors are good features | seed variation; older commit for random run; budget 20 epochs | no (single seed) |
| H7 | 4.3, bullet 4 | GloVe worst due to domain/tokenisation mismatch | vector scale; 88 filled rows; extra data itself | no (data, domain, scale change together) |
| H8 | 5.3, last bullet | sarcasm harder for Transformer features | sampling noise; 100-d vs 256-d; different C | no |
| H9 | 5.4 | perplexity predicts downstream quality | feature size, layer, untrained positions, C | no |
| H10 | 6 | cautious answer and the missing third data point | n/a | n/a |

**[CHECK] items (read the examples yourself):**

| # | Location | What to verify |
|---|---|---|
| C1 | 3.4 | fluency judgement ("all four drift off topic within about 20 tokens", no clear ranking): read `results/part1_generations.md` |
| C2 | 3.4 | sentiment-consistency judgement on the negative prompt (trigram "just plain sucks", Transformer "the dialogue is awful", RNN/LSTM mixed) |
| C3 | 4.3, last bullet | "no clear quality difference among the three LSTM settings": read `results/part2_generations.md` |
| C4 | 5.3, Table 7 pattern column and "Missed by both?" | four patterns and the review excerpts; the "missed by both" column was computed from the errors CSVs (3 of 4 reviews are missed by both classifiers) |
| C5 | 5.3, "Patterns, from 15 random errors" bullet | all counts ("about 8 of 15", etc.) come from the assistant's manual reading in `results/part3_error_notes.md`; re-read the 30 sampled reviews |
| C6 | 7 | wording of the "test set used only for final evaluation" FAIL |

The repetition count in 3.4 ("didn ' t" four times in the LSTM continuation) is an [OBS] I counted by eye in `results/part1_generations.md`; recount it.

## 3. Every number and its source

All results files are under `results/`. "Derived" means computed from results-file numbers, not stored in one.

| Number(s) in the draft | Source |
|---|---|
| 50,000 reviews, 25,000 / 25,000 labels | `inspect_data.json` |
| Table 1 reviews, tokens, `<unk>` rates; vocabulary 20,003 | `data.json` |
| Table 1 label counts per split; 37.0% / 38.2% truncated; 256-token head truncation | `p3_prepare_classification.json`; DECISIONS W1 |
| Causality: 0.0 change, 12 positions per model | `test_causality.json` |
| Trigram: discounts 0.9/0.8/0.1; 0.84 s fit, 22.80 s tuning; own test perplexity 116.54 on 1,390,673 targets | `ngram.json` |
| Table 3: perplexity 111.31, 109.37, 80.45, 156.94; params 6,445,342, 7,364,303, 8,033,999, 4,231,115; train 0.84, 3,388, 3,850, 2,246 s; GPU 229,944 / 164,100 / 286,163; CPU 411,334 / 4,291 / 4,027 / 4,960; GPU std 1,200 / 2,997 / 7,950; 981,353 targets | `part1_costs.csv` (also `tables/part1_table.csv`, `benchmark.json`) |
| Part II LSTM train times 3,394 s and 3,398 s; best epoch 20 for all; best validation loss 4.399 / 4.411 / 4.513 | `p2_lstm_w2v.json`, `p2_lstm_glove.json`, `train_lstm.json` |
| Per-epoch validation/training losses quoted (epochs 1, 5, 9, 10, 16, 20) and last-epoch drops 0.0004 / 0.0068 / 0.0048 and 0.005 to 0.007 | `epoch_losses` in `train_rnn.json`, `train_lstm.json`, `train_transformer.json`, `p2_lstm_w2v.json`, `p2_lstm_glove.json` |
| Generation counts (eos 1/0/1/1 of 3; mean new tokens 41.0, 50.0, 35.3, 48.7); all quoted text | `generations_*.json` metrics; `part1_generations.md`; `part2_generations.md` |
| Skip-gram settings; 273.8 s train, 274.5 s total; coverage 100%; fill std 0.279 | `word2vec_cost.json` |
| GloVe coverage 99.56% / 97.52% (99.57% / 99.95% without `<unk>`/`<eos>`); 19,914 of 20,002; 88 rows; fill std 0.503 | `glove_coverage.json` |
| Trainable parameters 6,033,699 vs 8,033,999; frozen check max change 0.0 | `p2_lstm_*.json` details; DECISIONS T2 |
| Table 6 and 5.2: C 0.01 / 0.1; validation 0.8154 / 0.7624; test accuracy 0.8192 / 0.7744; macro-F1 0.8192 / 0.7744; 904 and 1,128 errors; validation macro-F1 by C | `p3_classify.json`, `part3.csv` |
| Perplexity vs accuracy ranks | `part3_ranking.csv` |
| Error properties (203.8 vs 195.1; 45.0% vs 37.0%; 2.34% vs 2.02%; 87.7% vs 87.4%; Transformer equivalents) | `part3_error_stats.csv` |
| Table 7 excerpts and labels (indices 39703, 37791, 10082, 48419) | `part3_error_sample_p3_feat_lstm_mean.csv`, `part3_error_sample_p3_feat_transformer_mean.csv` |
| Hyperparameters (Appendix A) | `all_runs.csv`, configs |
| Package versions (Appendix C) | `README.md` (local runs only; Colab versions are not recorded anywhere I read) |
| **Derived:** 28.92, 76.49, 1.94 perplexity gaps; 0.98 and 8.86 (1.2%, 10.9%) Part II gaps; 456 s / 12% LSTM time spread, 462 s RNN-LSTM gap; CPU "within about 23%"; accuracy gap 0.0448 and validation gap 0.0530; train-minus-validation gaps (−0.047, −0.269, 0.216, 0.179, +0.349, +0.036); 0.44% (88/20,002) filled rows; binomial SE about 0.006 (formula, not in a file) | computed by the assistant; recheck |
| **Derived from errors CSVs:** 563 both, 341 LSTM only, 565 Transformer only; error directions 469/435 and 536/592; "3 of 4 Table 7 reviews missed by both" | `part3_errors_p3_feat_lstm_mean.csv`, `part3_errors_p3_feat_transformer_mean.csv` |

## 4. Rubric requirements the draft does not yet meet

- **Reproducibility (brief: README with data access, dependencies, commands, configs, seeds):** `README.md` is out of date (item 0.4). Colab package versions and the final GitHub link are missing. Not changed, per your instruction.
- **Your own analysis.** The brief says the analysis "should reflect your own understanding". The draft is the assistant's; you must verify and rewrite it, and the tool-use statement must say truthfully how much you did.
- **Appendix B** is a placeholder; paste the generation files. **Appendix C** is an outline. **Lesson learned** (Section 7) is blank for you.
- **Name, student number, date, repository link, references** need your input and confirmation.
- **Part I:** parameter count, training time and budget were not matched across architectures (stated, not fixed). Only 3 prompts and 1 sample each for generation. No larger context-length experiment, so context length is not separated from architecture.
- **Part II:** "training stability" evidence is per-epoch only (DECISIONS V1); all runs stopped at the epoch cap, so convergence is compared at a fixed budget. The random run was on an older commit and the frozen runs record `git_dirty: true`. "Representative generated text" rests on 3 prompts.
- **Part III:** only two representations (LSTM vs Transformer) with different feature sizes; one pooling; no Part II model was classified, so Part II cannot be tied to Part III by data. "Same reviews missed" is shown as counts only.
- **Limitations:** the `RESULTS.md` "test set used only for final evaluation" check shows FAIL because `src/benchmark.py` reads the test split (DECISIONS J4); decide how to word this.
- **Seeds:** every run uses seed 0, no multi-seed variance anywhere.
- **Pending decision:** the figure and table choices of `src/make_figures.py` are still marked NEEDS REVIEW (not yet in `DECISIONS.md`).
- **Page budget:** about 2.6 pages over (section 1).

## 5. The 5 questions a marker is most likely to ask you

1. **"Why is your Transformer the worst model, and is that an architecture result?"** Know: 156.94 vs LSTM 80.45; 4.23M parameters; settings copied from the LSTM and untuned; positions 64 to 255 never updated (trained on 64-token windows, scored at 256); validation loss flat after epoch 5 while training loss falls. You cannot say which cause dominates.
2. **"Does your result show that lower perplexity gives better classification?"** Know: only two models, one seed, feature size 256 vs 100, different layers and C; the RNN and the Part II models were not classified. Agreement of two rankings is weak evidence.
3. **"Do pretrained GloVe embeddings help or hurt, and why is GloVe worse than your own Skip-gram?"** Know: 90.28 vs 81.42 vs 80.45; GloVe differs in data, domain, tokenisation and vector scale all at once; 99.56% coverage; no run isolates one factor.
4. **"How do you know there is no test-set leakage and how were hyperparameters chosen?"** Know: split made once before the vocabulary; vocabulary from training only; discounts, C and early stopping chosen on validation; the one FAIL in `RESULTS.md` and why (benchmark reads the test split to time scoring).
5. **"Are your differences real given one seed and the 20-epoch cap?"** Know: 80.45 vs 81.42 and RNN 109.37 vs trigram 111.31 are small; the identical LSTM took 3,850 s in Part I and 3,394 s in Part II, so timing gaps of about 12% are not architecture; all runs were still improving at epoch 20.
