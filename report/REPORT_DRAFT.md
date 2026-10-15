# DRAFT v3: not your text yet. Verify every line, rewrite in your own words, then delete this box.

**Status:** assistant-drafted (v1 and v2 on 2026-10-15; this v3 applies the second review you pasted on 2026-10-15). Written at your explicit request as an exception to the CLAUDE.md rule against report text (DECISIONS Z2). Every analytical point is still the assistant's; you must read the samples and the 30 misclassified reviews yourself and keep only claims you can defend.

**What changed in v3 (details and the source of every number: `VERIFY.md` sections 4 and 8):** the Discussion is rebuilt around "across architectures vs within one architecture"; McNemar tests for Skip-gram vs random and GloVe vs random (GloVe is not significant, so claims about GloVe were softened); a parameter breakdown; Spearman reported with the exact permutation p and the scoring caveat; truncation and pooling as an error explanation; negation statistic dropped; the C-sensitivity alternative largely dismissed with validation evidence; `[OBS]`/`[HYP]` tags and `[results/...]`/`DECISIONS` references removed from the body (the observation/explanation distinction is kept in the wording). `[CHECK]` marks remain: they are judgements you must verify, then delete.

---

# Does Better Next-Word Prediction Lead to Better Sentiment Representations? Evidence from IMDB Reviews

DSA4213 Assignment 1: Language Models, Embeddings, and Representations
[Your name] · [Student number] · [Date]
Code: [GitHub repository link; the Colab notebook uses https://github.com/izzryanshah/dsa4213-assignment1-izzryanshah.git, confirm it is the right one and accessible to the instructor]

## 1. Introduction and comparison question

This report asks: **does a language model that predicts the next word better (lower test perplexity) also give better features for classifying review sentiment?** Next-word prediction is the standard way to pretrain language models, and IMDB lets us test whether it yields useful representations because the language-model text and the sentiment labels come from the same reviews. We find that perplexity tracked downstream accuracy across architectures, in that the two worst language models were the two worst classifiers, but not across embedding settings within one architecture (Sections 5.2 and 6). Across all five representations the rank correlation is −0.7 (n = 5, not significant). The Transformer's perplexity depends on how it is scored, which limits the architecture comparison (Sections 3.5 and 5.4).

## 2. Data and preprocessing

We use the IMDB Dataset of 50K Movie Reviews (Kaggle copy of Maas et al., 2011): 25,000 positive and 25,000 negative reviews. HTML tags are replaced by a space and text is lowercased; tokenisation is word-level with the regex `\w+|[^\s\w]` (punctuation marks are tokens; "don't" becomes `don`, `'`, `t`). No subsampling. Reviews are split 80/10/10, stratified by label, seed 0, once, before the vocabulary is built. The vocabulary is the 20,000 most frequent training words plus `<pad>`, `<unk>`, `<eos>` (20,003 entries), from the training split only. Every review ends with `<eos>`. Parts I to III share the same splits, so no classification test review was used to train a language model (the classifier does train on the same training reviews the language models saw).

**Table 1: Dataset splits after preprocessing.**

| Split | Reviews | Tokens (incl. `<eos>`) | `<unk>` rate | Positive / negative |
|---|---|---|---|---|
| Train | 40,000 | 11,067,213 | 2.07% | 20,000 / 20,000 |
| Validation | 5,000 | 1,388,412 | 2.23% | 2,500 / 2,500 |
| Test | 5,000 | 1,395,673 | 2.24% | 2,500 / 2,500 |

## 3. Part I: Language model comparison

### 3.1 Models and setup

All models share the vocabulary and test targets. The loss ignores `<pad>`, and a perturbation test found no change in any output before the perturbed position for the RNN, LSTM and Transformer. The trigram model is interpolated Kneser-Ney (discounts 0.9, 0.8, 0.1 chosen on validation). The neural models use a random trainable 100-d embedding. The RNN (tanh) and LSTM have 2 layers, hidden size 256, dropout 0.3. The Transformer has 2 pre-norm causal blocks, 4 heads, feed-forward width 256, and learned positions up to 256; its model width is 100, tied to the embedding size. All three train on non-overlapping 64-token windows with Adam (lr 0.001), batch 64, gradient clipping 1.0, at most 20 epochs, patience 3, seed 0. We matched the recipe, not parameter count or training time. The Transformer's settings were copied from the LSTM, not tuned. Full settings are in Appendix A.

**Table 2: Where the parameters are** (derived from the architectures; each row sums to the model total).

| Model | Embedding | Positions | Recurrent / attention core | Output layer | Total |
|---|---|---|---|---|---|
| RNN | 2,000,300 | n/a | 223,232 | 5,140,771 (256 → 20,003) | 7,364,303 |
| LSTM | 2,000,300 | n/a | 892,928 | 5,140,771 (256 → 20,003) | 8,033,999 |
| Transformer | 2,000,300 | 25,600 | 184,912 (2 blocks + final norm) | 2,020,303 (100 → 20,003) | 4,231,115 |

### 3.2 Evaluation and timing procedure

The metric is test perplexity (exp of mean natural-log cross-entropy per scored token). Main setup: each test review is cut to its first 256 tokens, and all four models are scored on the same 981,353 targets, batch 64. A second setup, added as a diagnostic in Part III, scores the same targets with each review fed in consecutive 64-token pieces (the training length) with fresh recurrent state per piece. Neural models were trained and timed on a Colab Tesla T4 and the trigram on a local Apple M2 CPU, so training times are not comparable across hardware. Throughput is a full scoring pass (batch 64, 256 tokens) after 2 warm-up passes, mean of 3 repeats, on the T4 and on one CPU thread; the trigram is measured on CPU only.

### 3.3 Results

**Table 3: Prediction quality and cost.**

| Model | Test ppl (256 tokens) | Test ppl (64-token windows) | Parameters | Train time (s) | GPU tokens/s | CPU tokens/s |
|---|---|---|---|---|---|---|
| Trigram (KN) | 111.31 | n/a | 6,445,342 (counts) | 23.6 (CPU) | n/a | 411,334 |
| RNN | 109.37 | 111.11 | 7,364,303 | 3,388 | 229,944 | 4,291 |
| LSTM | **80.45** | **83.06** | 8,033,999 | 3,850 | 164,100 | 4,027 |
| Transformer | 156.94 | 101.00 | 4,231,115 | 2,246 | 286,163 | 4,960 |

Trigram training time is 0.84 s fitting plus 22.80 s tuning the discounts. Its own end-of-run perplexity is 116.54 on 1,390,673 full-review targets; the table uses the matched-token value. Trigram parameters are stored counts, not weights.

**Figure 1: Training and validation loss per epoch** (solid = training, dashed = validation).

We observe the following.

- At 256 tokens the order is LSTM 80.45, RNN 109.37, trigram 111.31, Transformer 156.94. Scored in 64-token windows, the Transformer improves by 55.94 (to 101.00) while the RNN and LSTM worsen slightly (+1.74 and +2.61, from the table values), so the Transformer then ranks between the LSTM (83.06) and the RNN (111.11). The LSTM is best in both setups.
- The Transformer trains fastest and has the highest GPU throughput; the LSTM has the lowest GPU throughput. On one CPU thread the neural models are within about 23% of each other. Timing is noisy: the identical LSTM took 3,850 s in Part I and 3,394 s and 3,398 s in the two Part II runs (same 20 epochs, same GPU model). That 456 s spread is as large as the RNN-vs-LSTM gap (462 s), so the gap should not be read as architectural.
- All three neural models reached the 20-epoch cap with best epoch 20, and validation loss was still falling in the last epoch (by 0.0004 RNN, 0.0068 LSTM, 0.0048 Transformer). The Transformer's early-stopping and model-selection loss was computed at 256 tokens, the penalised setting; this did not matter here because every run used all 20 epochs.
- For the RNN and LSTM, training loss lies above validation loss at every epoch (gap at epoch 20: 0.216, 0.179), which is the usual sign when training loss is averaged with dropout 0.3 on and validation loss is computed with dropout off. The Transformer's training loss falls below its validation loss from epoch 3 and the gap widens to −0.269 by epoch 20, while validation loss improves slowly (5.135 at epoch 5, 5.064 at epoch 20). This is a widening gap, not a rising validation loss, so we do not call it overfitting.
- Because that validation loss is scored at 256 tokens, the widening gap may mostly reflect scoring length. In 64-token windows the Transformer's test cross-entropy (4.615) is 0.180 below its final training loss (4.795), similar to the LSTM (0.158) and RNN (0.211). The alternative, genuine overfitting hidden by the window numbers, cannot be ruled out, because those numbers are test numbers and mix several effects.

### 3.4 Generated text

Decoding, identical for all four models: temperature 1.0, top-k 40, at most 50 new tokens, seed 0, one sample per prompt, `<pad>` and `<unk>` never sampled, stop at `<eos>`; three prompts. Full text is in Appendix B.

**Table 4: Continuations of the prompt "i was really disappointed by the acting and"** ("..." marks cuts).

| Model | Continuation (excerpt) |
|---|---|
| Trigram (KN) | "timing of the story lines for their lives in the same age as his wife and kid , and just plain sucks ." (`<eos>` after 23 tokens) |
| RNN | "directing , and it takes me out of some of the reviews and you see it if you ' re interested out with the most disgusting thing with a camera ..." (50 tokens) |
| LSTM | "costumes and the cast didn ' t take me to get bored , but i had nothing to do with any actor as the other reviewer who didn ' t go in there ..." (50 tokens) |
| Transformer | "directing this was so thin to make this film . i guess i think this was because to be sure why did they are in order to give it up ... and the dialogue is awful ." (`<eos>` after 46 tokens) |

`<eos>` was produced in 1 of 3 generations for the trigram, LSTM and Transformer and in 0 of 3 for the RNN; mean new tokens 41.0, 50.0, 35.3, 48.7. On prompt 1 the LSTM stopped after 6 tokens while the others ran to 50. [CHECK] My reading is that short stretches of every model look like plausible review phrases but all four drift off topic or become ungrammatical within about 20 tokens, with no clear fluency ranking from three prompts, and that only the trigram ("just plain sucks") and Transformer ("the dialogue is awful") end the negative prompt with a clearly negative phrase. The LSTM's lower perplexity does not show up as visibly better text. One possible reason is that three single samples cannot show a difference that an average over about one million tokens detects; another is that perplexity gaps of this size barely change local fluency. These samples cannot separate the two.

### 3.5 Interpretation

The LSTM predicts best in both setups. The Transformer is cheapest to train and fastest on GPU, and its perplexity depends strongly on the scoring length.

- The LSTM beats the RNN possibly because gating helps it keep information over longer spans. The parameter breakdown (Table 2) makes the main alternative stronger than the totals (8.03M vs 7.36M) suggest: the two share the same embedding and output layer (7.14M parameters), so the LSTM's recurrent core is about four times the RNN's (892,928 vs 223,232). We varied architecture only, with one seed, so gating cannot be separated from extra capacity (or from the shared learning rate and clipping limiting the RNN).
- The Transformer's 256-token perplexity (156.94) is probably inflated by a length mismatch: it trained on 64-token windows, so its learned position embeddings for positions 64 to 255 were never updated. The drop to 101.00 when scoring in windows is consistent with this, but cannot separate untrained position vectors from the more general alternative that attention never saw contexts longer than 64 tokens; that would need positions trained to 256 or sinusoidal positions, which we did not run.
- The Transformer's small total (4.23M) comes almost entirely from its width being 100, tied to the embedding size, versus 256 hidden units in the recurrent models: its output layer has 2.02M parameters against 5.14M, and its attention core is only 185k. This width difference is a capacity confound for every Transformer-vs-recurrent comparison, and it is also why its Part III features are 100-dimensional. At 64 tokens the Transformer (101.00) is still about 18 above the LSTM (83.06), and we cannot say whether that gap is architecture, capacity or tuning.
- The trigram is close to the RNN (111.31 vs 109.37) possibly because much of next-word prediction in reviews depends on the previous two words. The alternative is that the RNN is undertrained at the 20-epoch cap. We have no context-length experiment to tell them apart.
- The best predictor (LSTM) is the slowest on GPU and the Transformer is cheapest and fastest. Budgets, parameter counts and tuning were not matched, so these are trade-offs of these configurations, not of the architectures.

## 4. Part II: Embedding variants

### 4.1 Setup

We chose the LSTM (best Part I perplexity). The embedding dimension is 100 to match GloVe 100d. Own embeddings: Skip-gram (gensim) trained on the training split only, window 5, min_count 1, 10 epochs, 5 negative samples, 1 worker, seed 0. Public embeddings: `glove.6B.100d` (Wikipedia + Gigaword, 6B tokens, lowercased), matched by exact string. Words without a vector get random normal values with the standard deviation of the real vectors in that table; `<pad>` is zeros. GloVe has a vector for 19,914 of 20,002 non-pad types (99.56%) and covers 97.52% of training tokens when `<unk>` and `<eos>` count as missing; 88 rows were filled randomly. The Skip-gram table covers all 20,002 types. In both frozen runs the whole 20,003-row table is non-trainable and was checked to be exactly unchanged after training. Everything else is identical across the three runs (LSTM 2 layers, hidden 256, dropout 0.3, context 64, lr 0.001, batch 64, 20 epochs, seed 0); the random setting reuses the Part I LSTM.

### 4.2 Results

**Table 5: Embedding settings compared on the LSTM.**

| Setting | Trainable? | Extra data? | Test ppl (256 tokens) | Best epoch | Embedding cost |
|---|---|---|---|---|---|
| Random | Yes | No | 80.45 | 20 | n/a |
| Own Skip-gram | No (frozen) | No | 81.42 | 20 | 273.8 s (local CPU) |
| GloVe 100d | No (frozen) | Yes (6B tokens) | 90.28 | 20 | n/a (pretrained) |

Trainable parameters 8,033,999 / 6,033,699 / 6,033,699; LSTM training time 3,850 / 3,394 / 3,398 s. Perplexity in 64-token windows was computed only for the Part I models, not for the two frozen LSTMs.

**Figure 2: Training and validation loss for the three embedding settings.**

### 4.3 Interpretation

- Random 80.45 < own frozen 81.42 < GloVe frozen 90.28 test perplexity. Freezing our own embeddings costs 0.97 (1.2%); switching to GloVe costs a further 8.86.
- All three runs hit the 20-epoch cap and were still improving, so they are compared at a fixed budget; validation loss fell every epoch with no spikes. Validation loss at epochs 1, 5, 10, 20: random 5.425, 4.697, 4.525, 4.399; own frozen 5.304, 4.670, 4.518, 4.411; GloVe 5.503, 4.800, 4.640, 4.513. The own-frozen run is lowest at epochs 1, 5 and 10; the random run is lowest at epoch 20 (by 0.012). GloVe is highest at all four. One explanation is that the own-frozen embeddings give a head start that trainable random embeddings close by epoch 20, which would explain why freezing costs so little. Alternatives: the 0.012 difference is within what another seed might give, and the random run might overtake with a longer budget. We cannot say whether the gap would keep closing.
- The two frozen tables differ in scale: the standard deviation of the real vectors is 0.503 for GloVe and 0.279 for Skip-gram, a factor of about 1.8.
- GloVe is worst for several reasons we cannot separate. (a) Domain and tokenisation: GloVe was trained on other text, and its tokenisation differs from ours. GloVe 6B is, to our understanding, split in the Penn Treebank style ("don't" as `do` + `n't`), while our regex produces `don`, `'`, `t`, so `don` and `t` receive GloVe vectors that mean something else; the 99.56% coverage therefore overstates how well the table fits our tokens [CHECK: confirm in the GloVe file]. (b) Vector scale: the 1.8x difference changes the size of the LSTM's inputs. (c) The 88 randomly filled rows (0.44% of types). (d) The extra pretraining data itself. Own-frozen vs GloVe changes all of these at once, so the 8.86 gap cannot be attributed to extra data. A rescaled-GloVe run would test (b); we did not run it.
- [CHECK] On prompt 1 all three LSTM settings produced the same continuation (ending after 6 tokens); on prompt 2 they differ. I see no clear quality difference, on 3 prompts with 1 sample each.

## 5. Part III: Downstream sentiment classification

### 5.1 Setup

Task: binary sentiment on the 40,000/5,000/5,000 splits (Table 1). We compare five frozen representations: the Part I LSTM, RNN and Transformer (random trainable embeddings), and the Part II LSTMs with frozen own Skip-gram and frozen GloVe embeddings. Each review is cut to its first 256 tokens (head), `<eos>` dropped, right-padded (37.0% of train and 38.2% of test reviews are truncated). The review vector is the mean of the model's final hidden states over real (non-pad) positions: the last recurrent layer's output (256-d) for the RNN and LSTMs, the final-LayerNorm output (100-d) for the Transformer. Model weights were checked unchanged. The classifier is scikit-learn logistic regression on standardised features, with C chosen per representation from {0.001, ..., 100} by validation macro-F1, then scored once on test; the procedure and seed are the same for all five. Because every representation gets the same C search, C is not treated as a confound.

### 5.2 Results

**Table 6: Sentiment classification on the IMDB test split (5,000 reviews), ordered by LM perplexity (256 tokens).**

| Representation | LM test ppl | Features | Best C | Test accuracy = macro-F1 | Errors |
|---|---|---|---|---|---|
| LSTM (random embeddings) | 80.45 | 256 | 0.01 | 0.8192 | 904 |
| LSTM, frozen Skip-gram | 81.42 | 256 | 1 | **0.8592** | 704 |
| LSTM, frozen GloVe | 90.28 | 256 | 1 | 0.8276 | 862 |
| RNN | 109.37 | 256 | 0.1 | 0.7782 | 1,109 |
| Transformer | 156.94 | 100 | 0.1 | 0.7744 | 1,128 |
| LSTM (random), PCA to 100 | 80.45 | 100 | 0.01 | 0.8128 | 936 |

Accuracy and macro-F1 are equal to four decimals (balanced classes). PCA was fitted on the training features only and keeps 97.9% of the variance.

**Table 7: Paired tests on the same 5,000 test reviews** (McNemar, continuity-corrected chi-square with 1 df; "only A" = reviews A gets right and B wrong).

| A vs B | Both right | Only A | Only B | Both wrong | Chi-square | p |
|---|---|---|---|---|---|---|
| LSTM vs Transformer | 3,531 | 565 | 341 | 563 | 54.9 | 1.3e-13 |
| Skip-gram LSTM vs random LSTM | 3,861 | 435 | 235 | 469 | 59.1 | 1.5e-14 |
| GloVe LSTM vs random LSTM | 3,781 | 357 | 315 | 547 | 2.50 | 0.114 |

- Accuracy rank: Skip-gram LSTM, GloVe LSTM, LSTM, RNN, Transformer. Perplexity rank: LSTM, Skip-gram LSTM, GloVe LSTM, RNN, Transformer. Spearman rho between 256-token perplexity and accuracy is −0.7 (n = 5). The reported two-sided p = 0.19 uses a t-approximation; the exact permutation p is 0.23. The five points are not independent, since three are the same LSTM architecture. Neither p-value changes the conclusion that five points cannot show a significant correlation.
- The Skip-gram LSTM beats the random LSTM significantly (+0.0400, McNemar p = 1.5e-14); the GloVe LSTM does not (+0.0084, p = 0.114). The McNemar test is the appropriate comparison for these pairs because both classifiers are scored on the same reviews; no McNemar test was run for the RNN vs Transformer pair or the other pairs.
- RNN vs Transformer differ by 0.0038 in accuracy, which is small compared with one standard error of about 0.006 for a single accuracy near 0.8 on 5,000 reviews (this is a rough guide; we did not run the paired test for this pair).
- Reducing the LSTM features to 100 dimensions with PCA gives 0.8128 (vs 0.8192), still 0.0384 above the 100-d Transformer features.

### 5.3 Error analysis

LSTM features misclassify 904 of 5,000 test reviews and Transformer features 1,128; 563 are missed by both. Directions are roughly balanced (LSTM: 469 negative predicted positive, 435 the reverse; Transformer 536 and 592).

**Table 8: Representative misclassified reviews** (excerpts; the pattern column is a judgement, [CHECK]).

| Review (excerpt; index) | True → predicted | Pattern [CHECK] |
|---|---|---|
| "This movie definitely had some bad sound/editing/lighting/acting/etc. etc. problems. However ... it is really good." (39703, LSTM sample) | positive → negative | negative words, then a positive verdict at the end |
| "Unashamedly ambitious sci-fi ... Unfortunately it's just not that good. It all starts well enough ..." (37791, LSTM sample) | negative → positive | praise inside a negative review |
| "I've seen every film Lou Ferrigno has made ... yet this film is the most painful thing I've ever seen." (10082, Transformer sample) | negative → positive | ironic build-up before the verdict at the end |

- Errors are more often truncated than correct reviews: 45.0% vs 37.0% for the LSTM features and 41.8% vs 37.5% for the Transformer; they are also slightly longer (203.8 vs 195.1 tokens kept for the LSTM) and slightly higher in `<unk>` rate (2.34% vs 2.02%). Two of the three examples in Table 8 put the verdict at the end. One explanation is that head-256 truncation sometimes cuts the final verdict and mean pooling dilutes a verdict that is kept. This is testable by truncating from the tail or by pooling the last hidden state; we did not run either. These differences are small and untested.
- [CHECK] In the 15 random errors per classifier read by the assistant, the LSTM errors were mostly positive reviews with complaints (about 8 of 15) or negative reviews that praise first (about 4 of 15); the Transformer errors included sarcasm in negative reviews (about 4 of 15) and positive reviews dwelling on faults (about 5 of 15). Re-read the 30 reviews before keeping any count. The error samples cover only the LSTM and Transformer.

### 5.4 Does language-model ranking predict downstream ranking?

- **Across architectures, partly.** The two worst classifiers are the two worst language models (RNN and Transformer), and the top three on perplexity are the top three on accuracy. This holds as a set. The order inside it depends on the scoring choice: the Transformer's 256-token perplexity (156.94) puts it last, but its 64-token value (101.00) places it ahead of the RNN (111.11), while it still classifies slightly worse (0.7744 vs 0.7782, a gap within noise). We could not recompute the five-point correlation with window perplexities because the two frozen LSTMs have no window value.
- **Within one architecture, no.** The three LSTM runs have the same architecture, layer and 256 features, and perplexity (random < Skip-gram < GloVe) and accuracy (Skip-gram > GloVe > random) are ordered differently. The Skip-gram LSTM has perplexity 0.97 higher than the random LSTM yet is 0.0400 more accurate (significant). The GloVe LSTM has perplexity about 10 higher (90.28 vs 80.45) and is 0.0084 more accurate, which is not significant (p = 0.114); so for GloVe we can say that a ten-point perplexity penalty did not lower accuracy, not that it raised it.
- One explanation for the Skip-gram result is that its frozen table was learned from the same training reviews by a method that groups words by context, so the LSTM's hidden states may start from word geometry that is more useful for sentiment, while a trainable embedding is tuned only for next-word prediction. The alternatives we can address: classifier sensitivity to C is unlikely to explain it, because the random LSTM's validation macro-F1 is nearly flat across C (0.806 at C = 0.001, 0.815 for C ≥ 0.01), below the Skip-gram LSTM's validation score of 0.8486 at its best C; and feature scale is not an obvious cause (training-feature standard deviations 0.130 random, 0.133 Skip-gram, 0.129 GloVe). We cannot rule out single-seed noise (all runs use seed 0), and we have no run that separates the contribution of the embedding table itself from the language model on top of it.
- The Transformer's last place is confounded. Its features come from 256-token inputs, the same setting in which its perplexity is inflated by untrained positions (Section 3.5), and we did not extract Transformer features in 64-token windows, so the effect on classification was not measured. Other differences: its 100-d width (the PCA result shows the LSTM keeps a 0.0384 lead at 100 dimensions, so size alone is not enough, though PCA is not a native 100-d vector), its smaller capacity (Table 2), and no tuning. These cannot be separated with the saved runs.

## 6. Discussion: answering the comparison question

Perplexity tracked downstream accuracy across architectures but not across embedding settings within one architecture. Across architectures, the two language models with the worst perplexity (RNN 109.37, Transformer 156.94) gave the two worst classifiers (0.7782, 0.7744), and the LSTM family (perplexity 80 to 90) gave the three best; the order inside each group is not settled, because the Transformer's perplexity changes from 156.94 to 101.00 with the scoring length and its accuracy gap to the RNN (0.0038) is within noise. Within the LSTM architecture, perplexity ranked the embedding settings random < Skip-gram < GloVe, but accuracy ranked them Skip-gram > GloVe > random: a perplexity difference of 0.97 coincided with a significant 0.0400 accuracy difference, and a difference of about ten perplexity points coincided with no significant accuracy difference (0.0084, McNemar p = 0.114). So perplexity separates models that differ a lot in how they use context, but does not order models that differ in what their word representations contain, and the embedding choice changed downstream accuracy more than it changed perplexity. These conclusions rest on one seed per run, on five representations of which three share an architecture, and on a Transformer whose perplexity and features were both produced at a length it was not trained on.

## 7. Limitations and lessons learned

- One seed (0) per run; none of the gaps (80.45 vs 81.42; 109.37 vs 111.31; 0.8192 vs 0.8276) has a measured seed variance.
- All neural runs stopped at the 20-epoch cap while still improving; budgets and parameter counts were not matched; the Transformer was untuned and trained on 64-token windows but scored on up to 256 tokens, with a width (100) different from the recurrent models (256).
- Part III: one task, mean pooling, one classifier family; Transformer features at 256 tokens only; error samples for two of five representations; McNemar for three pairs, not all ten.
- The 64-token-window rescoring is a diagnostic computed on the test split; running it on validation would have been cleaner. It was not used to choose anything, but it does shape how we read the Transformer. The Part II frozen runs were made from an uncommitted working tree and the Part I runs from an older commit; none was rerun. One automatic check in the repository reports a test-set use failure because the benchmark script reads the test split for timing; discounts, C and early stopping used validation data only. [CHECK] confirm this wording.
- Lesson: [your lesson, e.g. check that training and evaluation lengths match before comparing architectures].

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. ACL 2011.
- IMDB Dataset of 50K Movie Reviews (Kaggle): https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014 (`glove.6B.100d`, https://nlp.stanford.edu/projects/glove/).
- gensim (Řehůřek & Sojka, 2010); word2vec Skip-gram (Mikolov et al., 2013); Kneser & Ney (1995); PyTorch (Paszke et al., 2019); scikit-learn (Pedregosa et al., 2011); McNemar (1947). [Confirm each and add any course notebooks or external code you adapted.]

## Tool use and AI assistance

[Draft, true only after you verify it.] I used Claude Code (Anthropic's coding assistant) throughout. It wrote and tested the code in this repository (data pipeline, n-gram, RNN/LSTM/Transformer models, training, benchmark, generation, word2vec and GloVe handling, feature extraction, classifier, PCA, McNemar, window rescoring, rank table, tables and figures, and the result-summary generators), following project rules I set in `CLAUDE.md`; every design choice is logged in `DECISIONS.md` and each change in `TOOL_USE.md`. I ran or approved the Colab GPU runs. The assistant also drafted the text and analysis of this report from the result files, and revised it after two rounds of review. I then [verified every number against the results files, read the generated text and misclassified reviews myself, and rewrote and edited the text; state truthfully how much you changed]. [Replace with your own account of what you understand and can explain.] External material: IMDB data (Kaggle; Maas et al.), GloVe `glove.6B.100d`, gensim, PyTorch, scikit-learn [add exact assistant model names and versions].

## Appendix A: Hyperparameters

| Run | Layers | Hidden | Context (train) | Batch | LR | Dropout | Epochs (cap) | Seed | Hardware |
|---|---|---|---|---|---|---|---|---|---|
| ngram | n/a | n/a | 2 words | 32 (scoring) | n/a | n/a | n/a | 0 | CPU: Apple M2 |
| train_rnn | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| train_lstm | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| train_transformer | 2 blocks, 4 heads, FF 256, width 100 | 256 (FF) | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p2_lstm_w2v | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p2_lstm_glove | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p3 classifier (5 + PCA) | logistic regression, C grid 0.001 to 100 | n/a | 256 tokens | n/a | n/a | n/a | n/a | 0 | CPU: Apple M2 |

Also: Adam, gradient clip 1.0, patience 3, embedding dim 100, vocabulary 20,000 (+3 special).

## Appendix B: Full generated samples

Decoding: temperature 1.0, top-k 40, 50 new tokens, seed 0, 1 sample, `<pad>`/`<unk>` banned, stop at `<eos>`. [Paste the full text of results/part1_generations.md and results/part2_generations.md here, grouped by prompt. The brief requires this.]

## Appendix C: Reproducibility

[Repository link. Every number in this report comes from a `results/*.json` or `.csv` file in the repository, and every design choice and its reason is logged in `DECISIONS.md` in the same repository. Hardware: Colab Tesla T4 for neural training, benchmark, generation, feature extraction and window rescoring; Apple M2 CPU for data, n-gram, word2vec, classifier, PCA, McNemar, rank table and figures; Python 3.10.11, torch 2.14.1, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 (Colab versions not recorded, add them). Commands: README.md section 4 and notebooks/run_colab.ipynb.]

## Appendix D: Additional figures

[None yet. Optional: results/figures/val_loss_all_runs.png.]
