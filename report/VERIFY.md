# VERIFY v2: what to check in `report/REPORT_DRAFT.md` before you rewrite it

v1 written 2026-10-15; v2 revises it after your review and your Part III extension (commit `d0aacb1` plus the new uncommitted result files). Section numbers refer to `REPORT_DRAFT.md`.

## 0. State of the inputs

1. `report/template.docx` (0 bytes) is no longer in `report/`. The draft follows `~/Downloads/DSA4213_Assignment1_Report_Template.docx`. Confirm it is the template you meant.
2. All six *My notes* blocks in `RESULTS.md` were empty when I last read them (v1). I did not re-check after your extension; if you have written notes since, the draft does not build on them.
3. `README.md` was updated by your commit (status says all three parts are done; Part III commands include the five-way, PCA, McNemar and window-rescoring steps). The v1 complaint about it being out of date no longer applies; confirm the commands reproduce the numbers.
4. Uncommitted: `results/p3_classify_five.json`, `p3_feat_{rnn,lstm_w2v,lstm_glove}_mean.json`, `p3_rank_table.json`, `p3_rescore_windows.json`, `part3_five*.csv`, `part3_lm_vs_classifier.csv`, `part3_window_perplexity.csv`. The draft cites them. The window-rescoring record shows `git_dirty: true` (RESULTS_LOG).
5. **`src/make_figures.py` and the tables in `results/tables/` do not yet cover** the five-way Table 6, the 64-window perplexity column of Table 3, or the PCA row. Table 6 and the new Table 3 column in the draft were assembled by hand from `part3_five.csv`, `part3_lm_vs_classifier.csv`, `part3_pca.csv` and `part3_window_perplexity.csv`; check them against those files. I did not change code, as you asked in the earlier task and since the request here was a draft review.

## 1. Page-count estimate (main body only)

Rough model: 600 prose words per page, 0.035 page per table row, 0.3 per figure; tags and source brackets not counted. Paste into the Word template to measure for real.

| Section | Budget | v1 estimate | v2 estimate | Status |
|---|---|---|---|---|
| 1 Introduction | 0.3 | 0.19 | 0.21 | within |
| 2 Data | 0.5 | 0.46 | 0.41 | within |
| 3 Part I | 2.0 | 3.63 | **3.08** | **over by about 1.1** |
| 4 Part II | 1.5 | 1.74 | 1.46 | within |
| 5 Part III | 1.0 | 1.86 | **2.03** | **over by about 1.0** (grew with the extension) |
| 6 Discussion | 0.4 | 0.33 | 0.29 | within |
| 7 Limitations | 0.3 | 0.35 | 0.30 | within |
| **Total** | 6.0 | 8.6 | **about 7.8** | **still over by about 1.8** |

The draft is still too long. Candidates to cut: Section 3.3 (merge the cost, timing and convergence bullets; drop Table 2 or fold it into Table 3), Section 3.4 (move the commentary to Appendix B), Section 5.2 (drop the standard-error and Validation-vs-C bullets), Section 5.3 (keep one of the two bullets and Table 7 at two rows), Section 6 (it repeats Table 6 numbers). Decide what to keep; the 5.4 hypotheses are the most valuable text.

## 2. How each of the nine review points was handled

| # | Review point | Handling in v2 | Checked against |
|---|---|---|---|
| 1 | "Overfits" contradicts best epoch 20 | Reworded to "widening gap, slowing improvement, not a rising validation loss" (3.3) | `train_transformer.json`: validation falls every epoch; train minus validation negative from epoch 3 |
| 2 | Train > validation gap explained wrongly | Added the dropout explanation (train loss averaged with dropout on) and made the Transformer's negative gap the notable fact (3.3). The reason is cited from **code** (`src/train.py:194`, `src/evaluate.py:112`), not from a results file | code lines read |
| 3 | "Scales may differ" vs 0.503 / 0.279 | Now an [OBS] (ratio about 1.8, not 2x) and scale is listed as alternative (b) for GloVe's result (4.3) | `glove_coverage.json`, `word2vec_cost.json` `fill_std` |
| 4 | "Trainability costs little" says the opposite | Reworded to "freezing costs little"; the head-start observation (lowest validation loss at epochs 1, 5, 10, random lowest at 20 by 0.012) is now interpreted as [HYP] with alternatives (4.3) | epoch_losses of the three runs |
| 5 | Trigram training time misleading | Table 3 and Table 2 now show 23.6 s (0.84 + 22.80) with the CPU/GPU caveat | `ngram.json` |
| 6 | Muddled trigram alternative | Removed the "evaluation setups differ" clause; the alternative is now "RNN undertrained at the cap" (3.5) | all four models were scored on the same 981,353 targets |
| 7 | C is not a confound | Dropped item (d); 5.1 says choosing C per representation is the same procedure for each | DECISIONS Y1 |
| 8 | Circular introduction | Replaced with the pretraining motivation (1) | n/a |
| 9 | Length | Cut about a third of the v1 prose, but the Part III extension added text; still about 1.8 pages over (section 1) | n/a |

Two places where the draft goes slightly beyond the review: (a) the review says the Transformer's position problem "could explain both its poor perplexity and its poor classification". The new window rescoring supports the perplexity half as an [OBS] (156.94 at 256 tokens, 101.00 in 64-token windows). The classification half is **untested**, because Transformer features were extracted only from 256-token inputs; the draft therefore keeps it as [HYP]. (b) The window numbers also change the Transformer's train/validation-gap reading (3.3): at 64 tokens its test cross-entropy is below its final training loss by 0.180, like the LSTM (0.158) and RNN (0.211). Those three differences are derived by me and use test numbers, not validation.

## 3. Every [HYP] and [CHECK] item

**[HYP]**

| # | Location | Claim | Alternative(s) named | Separable? |
|---|---|---|---|---|
| H1 | 1 | Transformer confounded by length mismatch, so answer is "only loosely" | n/a (framing) | n/a |
| H2 | 3.3, last bullet | Transformer's widening train/validation gap mostly reflects 256-token scoring, not overfitting | genuine overfitting hidden by test-based window numbers | no |
| H3 | 3.4 | LSTM's lower perplexity not visible in 3 samples | perplexity gaps this size barely change fluency | no |
| H4 | 3.5 b1 | LSTM beats RNN via gating | parameters; RNN limited by lr/clipping | no |
| H5 | 3.5 b2 | Transformer perplexity inflated by untrained positions 64 to 255 | attention never saw >64 tokens; fewest parameters; untuned; 20 epochs | partly: window drop is consistent with it, but cannot isolate the cause |
| H6 | 3.5 b3 | Trigram near RNN because short context carries most information | RNN undertrained | no |
| H7 | 4.3 | Own-frozen embeddings give a head start that random embeddings close | seed noise on 0.012; longer budget might change it | no |
| H8 | 4.3 | GloVe worst: domain/tokenisation, scale (1.8x), 88 filled rows, extra data | each other | no (all change together) |
| H9 | 5.4 b3 | Frozen Skip-gram helps features (word geometry from same reviews) despite slightly worse perplexity | seed noise; C differs (0.01 vs 1); feature scale (checked: similar) | no |
| H10 | 5.4 b4 | Transformer's last place confounded by 256-token features | 100-d size (PCA test says size alone is not enough), parameters, no tuning | no |
| H11 | 6 | Perplexity is a coarse guide | single-seed noise; Transformer confound | no |

**[CHECK]**

| # | Location | What to verify |
|---|---|---|
| C1 | 3.4 | fluency and sentiment-consistency judgement; "didn ' t" occurs four times (count it) |
| C2 | 4.3, last bullet | "no clear quality difference" among the three LSTM settings |
| C3 | 5.3, Table 7 pattern column and the "15 random errors" bullet | patterns and "about 8 of 15" etc. come from the assistant's manual reading (`results/part3_error_notes.md`); re-read the 30 samples. Only the LSTM and Transformer were error-sampled, not the other three representations |
| C4 | 7 | wording on the test-set FAIL and on the window rescoring being a diagnostic on the test split (the rescoring was not used to choose anything) |

## 4. Every number and its source

All files under `results/` unless noted. "Derived" = computed by me from results numbers.

| Number(s) | Source |
|---|---|
| 50,000 reviews, 25,000 / 25,000 | `inspect_data.json` |
| Table 1 | `data.json`; `p3_prepare_classification.json` (label counts) |
| Causality 0.0, 12 positions | `test_causality.json` |
| Trigram: discounts, 0.84 s, 22.80 s, own perplexity 116.54 on 1,390,673 targets | `ngram.json` |
| Table 3: 111.31 / 109.37 / 80.45 / 156.94; parameters; train times; GPU and CPU tokens/s; 981,353 targets | `part1_costs.csv` |
| Table 3: 64-window perplexities 111.11 / 83.06 / 101.00; window cross-entropies 4.711 / 4.420 / 4.615 | `part3_window_perplexity.csv`, `p3_rescore_windows.json` |
| Final-epoch training loss 4.922 / 4.578 / 4.795; per-epoch losses quoted; last-epoch drops | `epoch_losses` in `train_rnn.json`, `train_lstm.json`, `train_transformer.json`, `p2_lstm_w2v.json`, `p2_lstm_glove.json` |
| Generation counts, quoted text | `generations_*.json`; `part1_generations.md`; `part2_generations.md` |
| Skip-gram settings, 273.8 s, fill std 0.279 | `word2vec_cost.json` |
| GloVe coverage 99.56% / 97.52% (99.57% / 99.95%), 19,914, 88 rows, fill std 0.503 | `glove_coverage.json` |
| Frozen-table check 0.0; trainable parameters 6,033,699 | `p2_lstm_*.json` |
| Table 6: LM perplexities, features, best C, accuracy = macro-F1, errors; accuracy and perplexity ranks | `part3_five.csv`, `part3_lm_vs_classifier.csv` |
| Spearman rho −0.7, p = 0.188 | `p3_rank_table.json` |
| McNemar 3,531 / 565 / 341 / 563, chi-square 54.89, p = 1.3e-13 | `p3_mcnemar.json` |
| PCA row 0.8128, 936 errors, 97.9% variance | `part3_pca.csv`, `p3_classify_pca.json` |
| Validation macro-F1 by C (LSTM 0.806, 0.815) | `p3_classify.json` |
| Feature standard deviations 0.130 / 0.133 / 0.129 | `p3_feat_lstm_mean.json`, `p3_feat_lstm_w2v_mean.json`, `p3_feat_lstm_glove_mean.json` (train feature std) |
| Error properties (203.8 vs 195.1, 45.0% vs 37.0%, etc.) | `part3_error_stats.csv` |
| Table 7 excerpts | `part3_error_sample_p3_feat_lstm_mean.csv`, `part3_error_sample_p3_feat_transformer_mean.csv` |
| Error directions 469/435 and 536/592 | counted from `part3_errors_p3_feat_*.csv` |
| Package versions | `README.md` (local only; Colab versions unrecorded) |
| **Derived:** +1.75, +2.61, −55.94 (window vs 256); 0.98 (1.2%) and 8.86; 0.012; ratio 1.8; 456 s / 12%; 462 s; "about 23%"; accuracy gaps 0.0400, 0.0084, 0.0038, 0.0410, 0.0384; train-minus-validation gaps (−0.047, −0.269, 0.216, 0.179); train-minus-window-test differences 0.180 / 0.158 / 0.211; 23.6 s; 0.44% (88/20,002); standard error about 0.006 (formula, not in a file) | assistant arithmetic; recheck |

## 5. Rubric requirements the draft does not yet meet

- **Length:** about 1.8 pages over (section 1).
- **Your own analysis.** The brief says the analysis "should reflect your own understanding"; the draft is the assistant's. The tool-use statement must say truthfully how much you did.
- **Placeholders:** name, student number, date, repository link, lesson learned, Appendix B text, Colab package versions, exact assistant model names.
- **Figures and tables out of sync with the new results:** Table 6 (five-way plus PCA) and the 64-window column are not yet in `src/make_figures.py` or `results/tables/`; Figure 1 and Figure 2 are unchanged and show only the original runs. Figure 2 shows training and validation loss (template asks for validation only).
- **Part III error analysis** covers two of the five representations; "do the representations fail on the same reviews" is answered only for LSTM vs Transformer (563 both wrong).
- **Headline perplexity (your decision, not mine):** Table 3 shows both the 256-token and 64-window perplexities. Which one is the headline number, and whether Section 3.2 should treat 64 tokens as the "common evaluation setup", changes the Part I ranking and is a design choice for you; DECISIONS AA2 only records how the window score is defined.
- **Seeds:** all runs use seed 0; no run-to-run variance anywhere. This weakens every difference of about 1 perplexity point or about 0.01 accuracy.
- **Part II:** stability evidence is per-epoch only; all runs hit the 20-epoch cap (DECISIONS V1).
- **Test-set check:** `RESULTS.md` FAIL for "test set used only for final evaluation" (benchmark reads the test split); decide how to word it.
- **Pending decision:** `make_figures.py` layout choices are still NEEDS REVIEW.
- **Untested confounds that a marker may raise:** Transformer features in 64-token windows; positions trained to 256; a rescaled-GloVe run; multiple seeds.

## 6. What changed between v1 and v2 (summary)

- New content: five-way Table 6 with PCA row; McNemar; Spearman; 64-token-window perplexity in Part I; Part II to Part III link in 5.4.
- Reworded: Transformer overfitting; train/validation gaps and dropout; scale; "freezing costs little"; trigram time; trigram alternative; C confound dropped; introduction.
- Removed or shortened: v1's Table 7 fourth row, the generation prompt-1 and prompt-3 prose, most of the Discussion, Appendix D contents, several repeated numbers (fewer decimals).
- The v1 conclusion ("rankings agree, can't say why") is replaced by "only loosely, with a Transformer confound".

## 7. The 5 questions a marker is most likely to ask you

1. **"Your Transformer was trained on 64 tokens and scored on 256. Doesn't that invalidate its perplexity and its Part III features?"** Know: 156.94 at 256 tokens vs 101.00 in 64-token windows; positions 64 to 255 never updated; Transformer features were never extracted at 64 tokens, so the classification effect is untested.
2. **"Does lower perplexity give better classification?"** Know: Spearman rho −0.7, p = 0.19 with n = 5; the best classifier is the second-best language model (frozen Skip-gram LSTM, 0.8592 vs 0.8192 for the best LM); the worst two are the same on both lists.
3. **"Why does a frozen Skip-gram LSTM classify better than the trainable-embedding LSTM when its perplexity is worse?"** Know: gap 0.0400 vs a standard error about 0.006; one seed; no McNemar run for this pair; C differs (1 vs 0.01); your hypothesis (word geometry from the same reviews) is untested.
4. **"Why is GloVe worse than your own Skip-gram, and does that show pretrained embeddings don't help?"** Know: 90.28 vs 81.42; GloVe differs in data, domain, tokenisation and vector scale (0.503 vs 0.279) at once; 99.56% coverage; no run isolates one factor; GloVe still classified better than the random LSTM (0.8276 vs 0.8192, a gap within noise).
5. **"Are these differences real given one seed and the 20-epoch cap?"** Know: 80.45 vs 81.42 and RNN 109.37 vs trigram 111.31 are small; identical LSTM training took 3,850 s vs 3,394 s; all runs were still improving at epoch 20; McNemar covers one pair; and how hyperparameters (discounts, C, early stopping) were chosen on validation only, with the benchmark FAIL explained.

## 8. v3 (second review, 2026-10-15): what changed and what to check

Body text no longer carries `[results/...]` or `DECISIONS` references and the `[OBS]`/`[HYP]` tags are gone; the sources are in section 4 above and the new ones are below. `[CHECK]` marks remain on purpose.

| Review point | Handling in v3 | Source / how checked |
|---|---|---|
| 1 Discussion contradicted by GloVe | Section 6 and 5.4 rebuilt around "across architectures vs within one architecture". The McNemar test showed GloVe vs random is NOT significant (p = 0.114), so the draft says a ten-point perplexity penalty did not lower accuracy, not that it raised it | `p3_mcnemar_glove.json`, `p3_mcnemar_w2v.json` |
| 2 Confounded Transformer, ρ, independence | 5.4 says the bottom-two order flips under window perplexity but the accuracy gap (0.0038) is within noise; 5.2 gives p = 0.19 (t-approximation) and exact permutation p = 0.23; notes the five points are not independent | permutation p computed by me over all 120 rankings (scratch script, not saved in the repo) |
| 2b "ρ drops to about 0.6" | **Not reproduced and not in the draft.** A valid 5-point window ρ needs 64-token perplexity for the two frozen LSTMs, which do not exist. Mixing 256 and 64 values would be invalid. For the three Part I models only, ρ is 0.5 in magnitude (n = 3, uninformative). A Colab run of `rescore_windows.py` on the two frozen LSTMs would settle it | derived |
| 3 No significance test for Skip-gram | Table 7 adds Skip-gram vs random and GloVe vs random | `results/p3_mcnemar_{w2v,glove}.json`; configs `configs/p3_mcnemar_{w2v,glove}.json` |
| 4 C alternative | Moved into 5.4. The argument is: random LSTM validation macro-F1 ≤ 0.815 for every C, below the Skip-gram LSTM's 0.8486 at its best C. I could not check the Skip-gram LSTM across C (no per-C grid for it in `p3_classify_five.json`), so the draft does not claim it is flat | `p3_classify.json` (random LSTM), `p3_classify_five.json` (best-C validation) |
| Parameter breakdown | New Table 2 and text in 3.5. Embedding 2,000,300; RNN core 223,232; LSTM core 892,928 (4.0x); output 5,140,771 (256→20,003) vs 2,020,303 (100→20,003); Transformer positions 25,600, core 184,912 (2 blocks of 92,356 + final norm 200). Each row sums to the stored total | derived from the architecture; the sums reproduce 7,364,303 / 8,033,999 / 4,231,115 exactly (check against `models.py`) |
| Truncation and pooling | 5.3 now uses 45.0% vs 37.0% truncated and the end-verdict pattern as a testable explanation (tail truncation or last-state pooling, not run) | `part3_error_stats.csv`, Table 8 |
| Negation statistic | Dropped from the body (87.7% vs 87.4% showed it does not discriminate) | n/a |
| GloVe tokenisation | Made concrete in 4.3(a). **Stated from general knowledge of GloVe 6B's Penn Treebank-style tokenisation, not checked in the file; marked [CHECK]**. Look up `n't`, `do`, `don`, `t` in `glove.6B.100d.txt` | needs you |
| Smaller points | Early-stopping loss at 256 tokens (3.3); window values absent for frozen LSTMs (Table 5 note); window rescoring on test, say so (Section 7); "Partly." now prose; rounding uses table values (+1.74, 0.97) | derived from table values |

**Still to do (yours):** length (not measured; v3 removed tags and the old Table 2 but added Tables 2 and 7, so the draft is probably still over 6 pages; paste into the template and measure); Appendix B; name and number; lesson; tool-use statement; verify every `[CHECK]`; read the 30 misclassified reviews and the generated text yourself. `src/make_figures.py` still does not produce Table 2 (parameter breakdown) or Table 7 (McNemar), and the five-way table is still hand-assembled.
**Optional runs from the review not done:** Transformer features in 64-token windows (Colab); mean-of-Skip-gram-vectors baseline (local, but needs new code and a design choice).
