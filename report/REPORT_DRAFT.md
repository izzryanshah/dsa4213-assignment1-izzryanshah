# DRAFT v2: not your text yet. Verify every line, rewrite in your own words, then delete this box.

**Status:** assistant-drafted (v1 on 2026-10-15; this v2 revises it after your review and your Part III extension). Written at your explicit request as an exception to the CLAUDE.md rule against report text (DECISIONS Z2). Structure and budgets follow `~/Downloads/DSA4213_Assignment1_Report_Template.docx`. Your *My notes* blocks in `RESULTS.md` were empty, so every analytical point is the assistant's.

**Tags.** `[OBS]` = shown directly by a results file (or, where marked "code", by the code). `[HYP]` = possible explanation; each names an alternative and says whether the experiments can separate them. `[CHECK]` = a judgement about text or errors you must verify by reading. Sources are in brackets. `VERIFY.md` lists everything to check.

**What changed from v1 (full list in VERIFY.md section 6):** Part III now uses five representations (RNN, LSTM, two frozen-embedding LSTMs, Transformer), PCA, McNemar and the Spearman test; Part I now includes the 64-token-window perplexities, which change how the Transformer result is read; the nine review points are applied; the text is cut by about a third.

---

# Does Better Next-Word Prediction Lead to Better Sentiment Representations? Evidence from IMDB Reviews

DSA4213 Assignment 1: Language Models, Embeddings, and Representations
[Your name] · [Student number] · [Date]
Code: [GitHub repository link; the Colab notebook uses https://github.com/izzryanshah/dsa4213-assignment1-izzryanshah.git, confirm it is the right one and accessible to the instructor]

## 1. Introduction and comparison question (≈ 0.3 page)

This report asks: **does a language model that predicts the next word better (lower test perplexity) also give better features for classifying review sentiment?** Next-word prediction is the standard way to pretrain language models, and IMDB lets us test whether it yields useful representations because the language-model text and the sentiment labels come from the same reviews. [OBS] Across five frozen representations, lower perplexity went with higher accuracy only loosely (Spearman rho −0.7, p = 0.19, n = 5), and the most accurate classifier did not come from the lowest-perplexity model (Section 5.2) [results/p3_rank_table.json]. [HYP] The Transformer's results are confounded by a train/test length mismatch (Sections 3.5 and 5.4), so we read the answer as "only loosely", with that caveat.

## 2. Data and preprocessing (≈ 0.5 page)

[OBS] We use the IMDB Dataset of 50K Movie Reviews (Kaggle copy of Maas et al., 2011): 25,000 positive and 25,000 negative reviews [results/inspect_data.json]. HTML tags are replaced by a space and text is lowercased; tokenisation is word-level with the regex `\w+|[^\s\w]` (punctuation marks are tokens; "don't" becomes `don`, `'`, `t`) (DECISIONS A5, A6). No subsampling. Reviews are split 80/10/10, stratified by label, seed 0, once, before the vocabulary is built. The vocabulary is the 20,000 most frequent training words plus `<pad>`, `<unk>`, `<eos>` (20,003 entries), from the training split only [results/data.json]. Every review ends with `<eos>`. Parts I to III share the same splits, so no classification test review was used to train a language model (the classifier does train on the same training reviews the language models saw).

**Table 1: Dataset splits after preprocessing.**

| Split | Reviews | Tokens (incl. `<eos>`) | `<unk>` rate | Positive / negative |
|---|---|---|---|---|
| Train | 40,000 | 11,067,213 | 2.07% | 20,000 / 20,000 |
| Validation | 5,000 | 1,388,412 | 2.23% | 2,500 / 2,500 |
| Test | 5,000 | 1,395,673 | 2.24% | 2,500 / 2,500 |

[results/data.json; results/p3_prepare_classification.json]

## 3. Part I: Language model comparison (≈ 2 pages)

### 3.1 Models and setup

All models share the vocabulary and test targets. The loss ignores `<pad>`; a perturbation test found no change in any output before the perturbed position for the RNN, LSTM and Transformer [results/test_causality.json]. The trigram model is interpolated Kneser-Ney (discounts 0.9, 0.8, 0.1 chosen on validation) [results/ngram.json]. The neural models use a random trainable 100-d embedding: RNN (tanh) and LSTM with 2 layers, hidden 256, dropout 0.3; Transformer with 2 pre-norm causal blocks, 4 heads, feed-forward width 256, learned positions up to 256 (DECISIONS L1, M1, M3). All three train on non-overlapping 64-token windows with Adam (lr 0.001), batch 64, gradient clipping 1.0, at most 20 epochs, patience 3, seed 0 [results/train_*.json]. We matched the recipe, not parameter count or training time. The Transformer's settings were copied from the LSTM, not tuned (DECISIONS M4).

**Table 2: Model configurations (full settings in Appendix A).**

| Model | Key architecture | Context | Embedding | Training budget |
|---|---|---|---|---|
| Trigram (KN) | interpolated KN | 2 words | n/a | 23.6 s (CPU) |
| RNN | 2 layers, 256, tanh | 64 train / 256 test | Random, 100 | 20 epochs (cap) |
| LSTM | 2 layers, 256 | 64 train / 256 test | Random, 100 | 20 epochs (cap) |
| Transformer | 2 blocks, 4 heads, FF 256 | 64 train / 256 test | Random, 100 | 20 epochs (cap) |

[results/ngram.json; results/train_*.json]

### 3.2 Evaluation and timing procedure

The metric is test perplexity (exp of mean natural-log cross-entropy per scored token). Main setup: each test review cut to its first 256 tokens, all four models on the same 981,353 targets, batch 64 [results/part1_costs.csv]. A second setup (a diagnostic added in Part III) scores the same targets with each review fed in consecutive 64-token pieces, the length used in training, with fresh recurrent state per piece [results/p3_rescore_windows.json; DECISIONS AA2]. Neural models were trained and timed on a Colab Tesla T4; the trigram model on a local Apple M2 CPU, so training times are not comparable across hardware. Throughput is the full scoring pass (batch 64, 256 tokens) after 2 warm-up passes, mean of 3 repeats, on the T4 and on one CPU thread; the trigram on CPU only [results/benchmark.json; DECISIONS O1, O2].

### 3.3 Results

**Table 3: Prediction quality and cost.**

| Model | Test ppl (256 tokens) | Test ppl (64-token windows) | Parameters | Train time (s) | GPU tokens/s | CPU tokens/s |
|---|---|---|---|---|---|---|
| Trigram (KN) | 111.31 | n/a | 6,445,342 (counts) | 23.6 (CPU) | n/a | 411,334 |
| RNN | 109.37 | 111.11 | 7,364,303 | 3,388 | 229,944 | 4,291 |
| LSTM | **80.45** | **83.06** | 8,033,999 | 3,850 | 164,100 | 4,027 |
| Transformer | 156.94 | 101.00 | 4,231,115 | 2,246 | 286,163 | 4,960 |

[results/part1_costs.csv; results/part3_window_perplexity.csv. Trigram time = 0.84 s fit + 22.80 s discount tuning [results/ngram.json]. The trigram's own end-of-run perplexity is 116.54 on 1,390,673 full-review targets; the table uses the matched-token value. Parameters for the trigram are stored counts, not weights.]

**Figure 1: Training and validation loss per epoch** [results/figures/part1_loss_curves.pdf; solid = training, dashed = validation].

**Observations.**

- [OBS] At 256 tokens the order is LSTM 80.45, RNN 109.37, trigram 111.31, Transformer 156.94. Scored in 64-token windows, the Transformer improves by 55.94 (to 101.00) while the RNN and LSTM worsen slightly (+1.75, +2.61), so the Transformer then ranks between the LSTM (83.06) and the RNN (111.11) [results/part3_window_perplexity.csv]. The LSTM is best in both setups.
- [OBS] Cost: the Transformer trains fastest and has the highest GPU throughput with the fewest parameters; the LSTM has the lowest GPU throughput. On one CPU thread the neural models are within about 23% of each other while the trigram scores 411,334 tokens/s [Table 3].
- [OBS] Timing caveat: the identical LSTM took 3,850 s in Part I and 3,394 s and 3,398 s in the two Part II runs (same 20 epochs, same GPU model) [results/train_lstm.json, p2_lstm_*.json]. The 456 s (12%) spread within one architecture is as large as the RNN-vs-LSTM gap (462 s), so that gap should not be read as architectural.
- [OBS] Convergence: all three hit the 20-epoch cap with best epoch 20, so none stopped early, and validation loss was still falling in the last epoch (by 0.0004 RNN, 0.0068 LSTM, 0.0048 Transformer) [results/train_*.json].
- [OBS] Training vs validation loss: for the RNN and LSTM, training loss lies above validation loss at all 20 epochs (gap at epoch 20: 0.216, 0.179). Training loss is averaged over the epoch with dropout 0.3 switched on, while validation loss is computed with dropout off (code: `src/train.py` line 194 `model.train()`, `src/evaluate.py` line 112 `model.eval()`), which is the usual reason for this sign. For the Transformer, training loss falls below validation loss from epoch 3 (−0.047) and the gap widens to −0.269 at epoch 20, even though its training loss also has dropout on; its validation loss improves slowly (5.135 at epoch 5, 5.064 at epoch 20) [results/train_transformer.json]. This is a widening gap and slowing improvement, not a rising validation loss, so we do not call it overfitting.
- [OBS] The Transformer's validation loss is scored on 256-token reviews, the setup in which Table 3 shows it is penalised. Its test cross-entropy in 64-token windows is 4.615, below its final training loss of 4.795 (difference 0.180), similar to the LSTM (4.420 vs 4.578, 0.158) and RNN (4.711 vs 4.922, 0.211) [results/p3_rescore_windows.json; results/train_*.json; differences derived]. At training length the Transformer shows the same train/evaluation gap as the other two.
- [HYP] So the Transformer's widening train/validation gap may mostly reflect scoring at 256 tokens rather than overfitting. The alternative is genuine overfitting that the window numbers hide; they cannot rule it out because they are test, not validation, numbers and mix several effects.

### 3.4 Generated text

Decoding, identical for all four models: temperature 1.0, top-k 40, 50 new tokens maximum, seed 0, one sample per prompt, `<pad>`/`<unk>` never sampled, stop at `<eos>`; three prompts [results/part1_generations.md; results/generations_*.json].

**Table 4: Continuations of the prompt "i was really disappointed by the acting and"** (copied from results/part1_generations.md; "..." marks my cuts).

| Model | Continuation (excerpt) |
|---|---|
| Trigram (KN) | "timing of the story lines for their lives in the same age as his wife and kid , and just plain sucks ." (`<eos>` after 23 tokens) |
| RNN | "directing , and it takes me out of some of the reviews and you see it if you ' re interested out with the most disgusting thing with a camera ..." (50 tokens) |
| LSTM | "costumes and the cast didn ' t take me to get bored , but i had nothing to do with any actor as the other reviewer who didn ' t go in there ..." (50 tokens) |
| Transformer | "directing this was so thin to make this film . i guess i think this was because to be sure why did they are in order to give it up ... and the dialogue is awful ." (`<eos>` after 46 tokens) |

- [OBS] `<eos>` was produced in 1 of 3 generations for the trigram, LSTM and Transformer and in 0 of 3 for the RNN; mean new tokens 41.0, 50.0, 35.3, 48.7 [results/generations_*.json]. On prompt 1 the LSTM stopped after 6 tokens ("this movie was one of the best movies i have ever seen .") while the others ran to 50 tokens [results/part1_generations.md]. In the LSTM continuation above, "didn ' t" occurs four times.
- [CHECK] My reading is that short stretches of every model look like plausible review phrases but all four drift off topic or become ungrammatical within about 20 tokens, with no clear fluency ranking from three prompts; and that the negative prompt gets a clearly negative ending only from the trigram ("just plain sucks") and Transformer ("the dialogue is awful"). Verify against Appendix B.
- [HYP] The LSTM's lower perplexity does not show up as visibly better text. One possible reason is that three single samples cannot show a difference that an average over about one million tokens detects; an alternative is that perplexity gaps of this size barely change local fluency. These samples cannot separate the two.

### 3.5 Interpretation

Observed: the LSTM predicts best in both setups; the Transformer is cheapest to train and fastest on GPU, and its perplexity depends strongly on the scoring length (Section 3.3).

- [HYP] The LSTM beats the RNN because gating helps it keep information over longer spans. Alternatives: it has more parameters (8.03M vs 7.36M), and the RNN may be limited by the shared learning rate and clipping. We varied architecture only, with one seed and unmatched parameter counts, so these cannot be separated.
- [HYP] The Transformer's 256-token perplexity (156.94) is inflated by a length mismatch: it trained on 64-token windows, so its learned position embeddings for positions 64 to 255 were never updated (DECISIONS M2). [OBS] The drop to 101.00 when scoring in windows is consistent with this. The window result cannot separate "untrained position vectors" from the more general alternative that attention never saw contexts longer than 64 tokens; separating them needs a Transformer with positions trained to 256 or sinusoidal positions, which we did not run. Other factors (fewest parameters, untuned LSTM settings, 20 epochs) remain; at 64 tokens the Transformer (101.00) is still about 18 above the LSTM (83.06), and we cannot say whether that gap is architecture or tuning.
- [HYP] The trigram is close to the RNN (111.31 vs 109.37 at 256 tokens) possibly because much of next-word prediction in reviews depends on the previous two words. An alternative is that the RNN is undertrained at the 20-epoch cap. We have no context-length experiment, so we cannot tell.
- [OBS] Supported trade-off: the best predictor (LSTM) is the slowest on GPU; the Transformer is cheapest and fastest. Budgets, parameter counts and tuning were not matched, so these are trade-offs of these configurations, not of the architectures.

## 4. Part II: Embedding variants (≈ 1.5 pages)

### 4.1 Setup

We chose the LSTM (best Part I perplexity; DECISIONS R1). The embedding dimension is 100 to match GloVe 100d. Own embeddings: Skip-gram (gensim) trained on the training split only, window 5, min_count 1, 10 epochs, 5 negative samples, 1 worker, seed 0 [results/word2vec_cost.json]. Public embeddings: `glove.6B.100d` (Wikipedia + Gigaword, 6B tokens, lowercased), matched by exact string (DECISIONS S1). Words without a vector get random normal values with the standard deviation of the real vectors in that table; `<pad>` is zeros (DECISIONS R2). GloVe has a vector for 19,914 of 20,002 non-pad types (99.56%) and covers 97.52% of training tokens when `<unk>` and `<eos>` count as missing (99.57% and 99.95% without them); 88 rows were filled randomly [results/glove_coverage.json]. The Skip-gram table covers all 20,002 types [results/word2vec_cost.json]. In both frozen runs the whole 20,003-row table is non-trainable and was checked to be exactly unchanged after training (max change 0.0) [results/p2_lstm_*.json]. Everything else is identical across the three runs (LSTM 2 layers, hidden 256, dropout 0.3, context 64, lr 0.001, batch 64, 20 epochs, seed 0); the random setting reuses the Part I LSTM.

### 4.2 Results

**Table 5: Embedding settings compared on the LSTM.**

| Setting | Trainable? | Extra data? | Test ppl | Best epoch | Embedding cost |
|---|---|---|---|---|---|
| Random | Yes | No | 80.45 | 20 | n/a |
| Own Skip-gram | No (frozen) | No | 81.42 | 20 | 273.8 s (local CPU) |
| GloVe 100d | No (frozen) | Yes (6B tokens) | 90.28 | 20 | n/a (pretrained) |

[results/train_lstm.json; p2_lstm_w2v.json; p2_lstm_glove.json; word2vec_cost.json. Trainable parameters 8,033,999 / 6,033,699 / 6,033,699; LSTM training time 3,850 / 3,394 / 3,398 s.]

**Figure 2: Training and validation loss for the three embedding settings** [results/figures/part2_loss_curves.pdf; the template asks for validation curves only, this also shows training loss].

### 4.3 Interpretation

- [OBS] Random 80.45 < own frozen 81.42 < GloVe frozen 90.28 test perplexity. Freezing our own embeddings costs 0.98 (1.2%); switching to GloVe costs a further 8.86.
- [OBS] Convergence: all three runs hit the 20-epoch cap and were still improving, so they are compared at a fixed budget (DECISIONS V1); validation loss fell every epoch, with no spikes (per-epoch records only). Validation loss at epochs 1, 5, 10, 20: random 5.425, 4.697, 4.525, 4.399; own frozen 5.304, 4.670, 4.518, 4.411; GloVe 5.503, 4.800, 4.640, 4.513 [results/train_lstm.json, p2_lstm_*.json]. The own-frozen run is lowest at epochs 1, 5 and 10; the random run is lowest at epoch 20 (by 0.012). GloVe is highest at all four.
- [HYP] The own-frozen embeddings give a head start that trainable random embeddings close by epoch 20, which would explain why freezing costs so little. Alternatives: the 0.012 difference at epoch 20 is within what another seed might give (one seed per run); and the random run might overtake with a longer budget, which we did not test. The experiments cannot say whether the gap would keep closing.
- [OBS] The two frozen tables differ in scale: the standard deviation of the real vectors is 0.503 for GloVe and 0.279 for Skip-gram [results/glove_coverage.json, word2vec_cost.json fill_std], a factor of about 1.8.
- [HYP] GloVe is worst for several reasons we cannot separate. (a) Domain and tokenisation: GloVe was trained on other text with other tokenisation, while our tokens split contractions and keep punctuation. (b) Vector scale: the 1.8x difference above changes the size of the LSTM's inputs. (c) The 88 randomly filled rows (0.44% of types, including `<unk>` and `<eos>`). (d) The extra pretraining data itself. Own-frozen vs GloVe changes all four at once, so the 8.86 gap cannot be attributed to extra data. A rescaled-GloVe run would test (b); we did not run it.
- [CHECK] Generated text (Appendix B): on prompt 1 all three LSTM settings produced the same continuation ("movies i have ever seen ." ending after 6 tokens) [results/part2_generations.md]. On prompt 2 the three differ ("costumes and the cast ...", "directing , and it took me too long ...", "lighting and a lot more than so much that ..."). I see no clear quality difference; this rests on 3 prompts, 1 sample each.

## 5. Part III: Downstream sentiment classification (≈ 1 page)

### 5.1 Setup

Task: binary sentiment on the 40,000/5,000/5,000 splits (Table 1). We compare five frozen representations: the Part I LSTM, Part I RNN and Part I Transformer (random trainable embeddings), and the Part II LSTMs with frozen own Skip-gram and frozen GloVe embeddings. Each review is cut to its first 256 tokens (head), `<eos>` dropped, right-padded (37.0% of train and 38.2% of test reviews truncated) [results/p3_prepare_classification.json; DECISIONS W1]. The review vector is the mean of the model's final hidden states over real (non-pad) positions: the last recurrent layer's output (256-d) for the RNN and LSTMs, the final-LayerNorm output (100-d) for the Transformer (DECISIONS X1). Weights are checked unchanged [results/p3_feat_*.json `weights_unchanged`]. The classifier is scikit-learn logistic regression on standardised features, C chosen per representation from {0.001, ..., 100} by validation macro-F1, then scored once on test; same procedure and seed for all five (DECISIONS Y1) [results/p3_classify_five.json]. Choosing C separately per representation is the same procedure for each, so C is not treated as a confound. Four of the five representations have 256 features and the same layer type; the Transformer has 100.

### 5.2 Results

**Table 6: Sentiment classification on the IMDB test split (5,000 reviews), ordered by LM perplexity.**

| Representation | LM test ppl | Features | Best C | Test accuracy = macro-F1 | Errors |
|---|---|---|---|---|---|
| LSTM (random embeddings) | 80.45 | 256 | 0.01 | 0.8192 | 904 |
| LSTM, frozen Skip-gram | 81.42 | 256 | 1 | **0.8592** | 704 |
| LSTM, frozen GloVe | 90.28 | 256 | 1 | 0.8276 | 862 |
| RNN | 109.37 | 256 | 0.1 | 0.7782 | 1,109 |
| Transformer | 156.94 | 100 | 0.1 | 0.7744 | 1,128 |
| LSTM (random), PCA to 100 | 80.45 | 100 | 0.01 | 0.8128 | 936 |

[results/part3_five.csv; results/part3_lm_vs_classifier.csv; results/part3_pca.csv. Accuracy and macro-F1 are equal to four decimals (balanced classes). PCA fitted on the training features only, 97.9% of variance kept [results/p3_classify_pca.json].]

- [OBS] Accuracy rank: Skip-gram LSTM, GloVe LSTM, LSTM, RNN, Transformer. Perplexity rank: LSTM, Skip-gram LSTM, GloVe LSTM, RNN, Transformer [results/part3_lm_vs_classifier.csv]. Spearman rho between perplexity and accuracy is −0.7 (two-sided p = 0.19, n = 5; the same for macro-F1) [results/p3_rank_table.json].
- [OBS] One standard error for an accuracy near 0.8 on 5,000 reviews is about 0.006 (formula, derived). Gaps: Skip-gram vs random LSTM 0.0400; GloVe vs random LSTM 0.0084; RNN vs Transformer 0.0038; LSTM vs RNN 0.0410.
- [OBS] LSTM vs Transformer: both right 3,531, only LSTM right 565, only Transformer right 341, both wrong 563; McNemar chi-square 54.9, p = 1.3e-13 [results/p3_mcnemar.json]. No McNemar test was run for the other pairs.
- [OBS] Reducing the LSTM features to 100 dimensions with PCA gives 0.8128 (vs 0.8192 unreduced), still 0.0384 above the 100-d Transformer features [results/part3_pca.csv]. Validation macro-F1 is nearly flat across C (LSTM: 0.806 at C = 0.001, 0.815 for C ≥ 0.01) [results/p3_classify.json].

### 5.3 Error analysis

LSTM features misclassify 904 of 5,000 test reviews and Transformer features 1,128 [Table 6]; 563 are missed by both [results/p3_mcnemar.json]. Directions are roughly balanced (LSTM: 469 negative predicted positive, 435 the reverse; Transformer 536 and 592) [counted from results/part3_errors_p3_feat_*.csv].

**Table 7: Representative misclassified reviews** (excerpts from results/part3_error_sample_p3_feat_*_mean.csv; the pattern column is a judgement, [CHECK]).

| Review (excerpt; index) | True → predicted | Pattern [CHECK] |
|---|---|---|
| "This movie definitely had some bad sound/editing/lighting/acting/etc. etc. problems. However ... it is really good." (39703, LSTM sample) | positive → negative | negative words, then a positive verdict |
| "Unashamedly ambitious sci-fi ... Unfortunately it's just not that good. It all starts well enough ..." (37791, LSTM sample) | negative → positive | praise inside a negative review |
| "I've seen every film Lou Ferrigno has made ... yet this film is the most painful thing I've ever seen." (10082, Transformer sample) | negative → positive | ironic build-up before the verdict |

- [OBS] Properties of errors vs correct reviews [results/part3_error_stats.csv]: mean tokens kept 203.8 vs 195.1 (LSTM), 202.0 vs 195.1 (Transformer); share truncated 45.0% vs 37.0% and 41.8% vs 37.5%; mean `<unk>` rate 2.34% vs 2.02% and 2.29% vs 2.02%; share with a negation word 87.7% vs 87.4% and 89.3% vs 87.0%. Negation words are about as common in correct reviews as in errors. Errors are slightly longer, more often truncated and slightly higher in `<unk>`; differences are small and untested. The error samples cover only the LSTM and Transformer, not the other three representations.
- [CHECK] In the 15 random errors per classifier read by the assistant, the LSTM errors were mostly positive reviews with complaints (about 8 of 15) or negative reviews that praise first (about 4 of 15); the Transformer errors included sarcasm in negative reviews (about 4 of 15) and positive reviews dwelling on faults (about 5 of 15) [results/part3_error_notes.md, assistant-drafted manual counts]. Re-read the 30 reviews before keeping any count.

### 5.4 Does language-model ranking predict downstream ranking?

- [OBS] Partly. The two worst classifiers are the two worst language models (RNN, Transformer), and the top three on perplexity are the top three on accuracy. Within those three, the order differs: the lowest-perplexity model (LSTM, 80.45) is third on accuracy, and the frozen Skip-gram LSTM (perplexity 0.98 higher) is first, 0.0400 above it. With five points the rank correlation of −0.7 is not significant (p = 0.19).
- [OBS] Part II connects to Part III here: within the matched LSTM family (same architecture, layer and 256 features), the embedding setting changed accuracy in a different order from perplexity: perplexity random < Skip-gram < GloVe, accuracy Skip-gram > GloVe > random (Tables 5 and 6). The 0.0400 gap is large against a standard error of about 0.006; the 0.0084 GloVe gap is not.
- [HYP] Why a slightly worse language model gives better features: the frozen Skip-gram table was learned from the same training reviews by a method that groups words by context, so the LSTM's hidden states may start from word geometry that is more useful for sentiment, while a trainable embedding is tuned only for next-word prediction. Alternatives: single-seed noise on a 0.98 perplexity difference; classifier sensitivity (C was 0.01 for the random LSTM but 1 for both frozen models); and feature scale (feature standard deviations were 0.130 random LSTM vs 0.133 and 0.129 frozen, so not an obvious scale effect [results/p3_feat_*.json]). Our runs cannot separate these; no multi-seed or trainable-vs-frozen classification control exists.
- [HYP] The Transformer's last place is confounded. Its features are extracted from 256-token inputs, the same setting in which its perplexity is inflated by untrained positions (Section 3.5). [OBS] We did not extract Transformer features in 64-token windows, so the effect on classification was not measured. If the length mismatch also lowers its features, its low accuracy (and part of the perplexity-accuracy agreement) would reflect that rather than the architecture. Alternatives: its 100-d feature size (the PCA result above shows the LSTM keeps a 0.0384 lead at 100 dimensions, so size alone is not enough, though PCA is not the same as a native 100-d vector), its fewer parameters, or no tuning. These cannot be separated with the saved runs.

## 6. Discussion: answering the comparison question (≈ 0.4 page)

[OBS] Better next-word prediction and better classification went together only loosely: the two worst models on perplexity were the two worst classifiers, but the best classifier (frozen Skip-gram LSTM, 0.8592) came from the second-best language model (81.42), and the best language model (LSTM, 80.45) was third (0.8192) [Table 6]. [HYP] Perplexity therefore seems a coarse guide: it separates models that differ a lot (80 to 90 vs 109 to 157) but not models that differ by about 1, where the embedding setup mattered more. Alternatives: single-seed noise, and the Transformer confound, which means the 80-to-157 gap includes a length artefact. [OBS] Part I shows the Transformer's perplexity is 156.94 or 101.00 depending on scoring length, so even the language-model ranking is setup-dependent. Part II gives the one clean contrast in our data (same architecture and features): freezing costs perplexity (81.42, 90.28 vs 80.45) yet changes accuracy in a different order. We did not run multiple seeds or Transformer features at training length, which would test these claims.

## 7. Limitations and lessons learned (≈ 0.3 page)

- One seed (0) per run; none of the gaps (80.45 vs 81.42; 109.37 vs 111.31; 0.8192 vs 0.8276) has a measured seed variance.
- All neural runs stopped at the 20-epoch cap while still improving; budgets and parameter counts were not matched; the Transformer was untuned and trained on 64-token windows but scored on up to 256 tokens (DECISIONS V1, M2, I4).
- Part III: one task, mean pooling, one classifier family; Transformer features at 256 tokens only; error samples for two of five representations; McNemar for one pair only.
- The Part II frozen runs record `git_dirty: true` and the Part I runs used an older commit; no rerun (DECISIONS V1). The `RESULTS.md` check "test set used only for final evaluation" shows FAIL because `src/benchmark.py` reads the test split for timing (DECISIONS J4); discounts, C and early stopping used validation data only. The 64-window rescoring is a diagnostic on the test split and was not used to choose anything. [CHECK] confirm this wording.
- Lesson: [your lesson, e.g. check that training and evaluation lengths match before comparing architectures].

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. ACL 2011.
- IMDB Dataset of 50K Movie Reviews (Kaggle): https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014 (`glove.6B.100d`, https://nlp.stanford.edu/projects/glove/).
- gensim (Řehůřek & Sojka, 2010); word2vec Skip-gram (Mikolov et al., 2013); Kneser & Ney (1995); PyTorch (Paszke et al., 2019); scikit-learn (Pedregosa et al., 2011); McNemar (1947). [Confirm each and add any course notebooks or external code you adapted.]

## Tool use and AI assistance

[Draft, true only after you verify it.] I used Claude Code (Anthropic's coding assistant) throughout. It wrote and tested the code in this repository (data pipeline, n-gram, RNN/LSTM/Transformer models, training, benchmark, generation, word2vec and GloVe handling, feature extraction, classifier, PCA, McNemar, window rescoring, rank table, tables and figures, and the `RESULTS.md`/`MY_TASKS.md` generators), following project rules I set in `CLAUDE.md`; every design choice is logged in `DECISIONS.md` and each change in `TOOL_USE.md`. I ran or approved the Colab GPU runs. The assistant also drafted the text and analysis of this report (including the [OBS]/[HYP] split and the error-pattern notes in `results/part3_error_notes.md`) from the result files, and revised it after my review. I then [verified every number against the results files, read the generated text and misclassified reviews myself, and rewrote and edited the text; state truthfully how much you changed]. [Replace with your own account of what you understand and can explain.] External material: IMDB data (Kaggle; Maas et al.), GloVe `glove.6B.100d`, gensim, PyTorch, scikit-learn [add exact assistant model names and versions].

## Appendix A: Hyperparameters

| Run | Layers | Hidden | Context (train) | Batch | LR | Dropout | Epochs (cap) | Seed | Hardware |
|---|---|---|---|---|---|---|---|---|---|
| ngram | n/a | n/a | 2 words | 32 (scoring) | n/a | n/a | n/a | 0 | CPU: Apple M2 |
| train_rnn | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| train_lstm | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| train_transformer | 2 blocks, 4 heads, FF 256 | 256 (FF) | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p2_lstm_w2v | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p2_lstm_glove | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p3 classifier (5 + PCA) | logistic regression, C grid 0.001 to 100 | n/a | 256 tokens | n/a | n/a | n/a | n/a | 0 | CPU: Apple M2 |

Also: Adam, gradient clip 1.0, patience 3, embedding dim 100, vocabulary 20,000 (+3 special) [results/all_runs.csv; configs/].

## Appendix B: Full generated samples

Decoding: temperature 1.0, top-k 40, 50 new tokens, seed 0, 1 sample, `<pad>`/`<unk>` banned, stop at `<eos>`. [Paste the full text of results/part1_generations.md and results/part2_generations.md here, grouped by prompt.]

## Appendix C: Reproducibility

[Repository link; hardware: Colab Tesla T4 for neural training, benchmark, generation, feature extraction and window rescoring; Apple M2 CPU for data, n-gram, word2vec, classifier, PCA, McNemar, rank table and figures; Python 3.10.11, torch 2.14.1, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 (README.md; Colab versions not recorded, add them). Commands: README.md section 4 and notebooks/run_colab.ipynb. Tables and figures: `python3 src/make_figures.py --config configs/make_figures.json` (this script does not yet include the five-way table, the window perplexities or the PCA row; see VERIFY.md).]

## Appendix D: Additional figures

[None yet. Optional: results/figures/val_loss_all_runs.png.]
