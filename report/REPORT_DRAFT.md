# DRAFT: not your text yet. Verify every line, rewrite in your own words, then delete this box.

**Status:** assistant-drafted on 2026-10-15 at your explicit request (an exception to the CLAUDE.md rule against writing report text). Structure and page budgets follow `~/Downloads/DSA4213_Assignment1_Report_Template.docx` (`report/template.docx` is a 0-byte file, so I could not read it). Your *My notes* blocks in `RESULTS.md` are all empty, so there was nothing of yours to build on; every analytical point below is the assistant's.

**Tags.** `[OBS]` = observation shown directly by a results file. `[HYP]` = possible explanation; each names an alternative and says whether the experiments can tell them apart. `[CHECK]` = a judgement about text or errors that you must verify by reading the examples yourself. Source files are in brackets after each number. Delete the tags and the source brackets in the final PDF as you see fit. `VERIFY.md` lists everything to check.

---

# Does Better Next-Word Prediction Lead to Better Sentiment Representations? Evidence from IMDB Reviews

DSA4213 Assignment 1: Language Models, Embeddings, and Representations
[Your name] · [Student number] · [Date: submission due 16 Oct 2026]
Code: [GitHub repository link; the Colab notebook uses https://github.com/izzryanshah/dsa4213-assignment1-izzryanshah.git, confirm it is the right one and accessible to the instructor]

## 1. Introduction and comparison question (≈ 0.3 page)

This report asks: **does a language model that predicts the next word better (lower test perplexity) also give better features for classifying review sentiment?** On IMDB reviews this matters because sentiment is the task that the language models' hidden states are later used for, so perplexity is only useful here if it tracks downstream quality. [OBS] Of the two models we classified, the LSTM has the lower perplexity (80.45 vs 156.94) and the higher test accuracy (0.8192 vs 0.7744) [results/part3_ranking.csv]. [OBS] This is two models, one seed, with feature size and layer also differing, so it is agreement in ranking, not evidence that perplexity causes better classification (Sections 5.4 and 6).

## 2. Data and preprocessing (≈ 0.5 page)

[OBS] We use the IMDB Dataset of 50K Movie Reviews (Kaggle copy of Maas et al., 2011): 50,000 reviews, 25,000 positive and 25,000 negative [results/inspect_data.json]. HTML tags such as `<br />` are replaced by a space and text is lowercased (DECISIONS A5). Tokenisation is word-level with the regex `\w+|[^\s\w]`, so each punctuation mark is its own token and "don't" becomes `don`, `'`, `t` (DECISIONS A6). All 50,000 reviews are kept (`subsample_size` null) [results/data.json config]. The reviews are split 80/10/10 into train/validation/test, stratified by label, seed 0, once, before the vocabulary is built (DECISIONS A10). The vocabulary is the 20,000 most frequent training words plus `<pad>`, `<unk>`, `<eos>` (20,003 entries), built from the training split only [results/data.json]. Every review ends with `<eos>`. Parts I to III use the same splits, so no classification test review was used to train a language model. The language models were trained on the same training reviews that the classifier is later trained on (the test reviews were never used for either).

**Table 1: Dataset splits after preprocessing.**

| Split | Reviews | Tokens (incl. `<eos>`) | `<unk>` rate | Positive / negative |
|---|---|---|---|---|
| Train | 40,000 | 11,067,213 | 2.07% | 20,000 / 20,000 |
| Validation | 5,000 | 1,388,412 | 2.23% | 2,500 / 2,500 |
| Test | 5,000 | 1,395,673 | 2.24% | 2,500 / 2,500 |

[Sources: results/data.json (reviews, tokens, `<unk>` rate); results/p3_prepare_classification.json (label counts).]

## 3. Part I: Language model comparison (≈ 2 pages)

### 3.1 Models and setup

All four models share the vocabulary and the test targets. The loss ignores `<pad>`, and each target is predicted only from earlier tokens: a perturbation test changed no output before the perturbed position for the RNN, LSTM and Transformer (max change 0.0 at 12 positions each) [results/test_causality.json]. The trigram model is interpolated Kneser-Ney with one discount per order (0.9, 0.8, 0.1 chosen on validation) [results/ngram.json]. The neural models use a random trainable 100-d embedding. The RNN and LSTM have 2 layers, hidden size 256, dropout 0.3 (RNN with tanh). The Transformer has 2 pre-norm causal blocks, 4 heads, feed-forward width 256, learned positions up to 256 [DECISIONS L1, L2, M1, M3]. All three train on non-overlapping 64-token windows with Adam (lr 0.001), batch size 64, gradient clipping 1.0, at most 20 epochs, early stopping with patience 3, seed 0 [results/train_rnn.json, train_lstm.json, train_transformer.json]. We matched the training recipe, not the parameter count or the training time: parameters are 7.36M (RNN), 8.03M (LSTM) and 4.23M (Transformer) [results/part1_costs.csv]. The Transformer's settings were copied from the LSTM rather than tuned (DECISIONS M4).

**Table 2: Model configurations (full settings in Appendix A).**

| Model | Key architecture | Context length | Embedding | Training budget |
|---|---|---|---|---|
| Trigram (KN) | interpolated Kneser-Ney, discounts 0.9/0.8/0.1 | 2 previous words | n/a | 0.84 s fit + 22.80 s discount tuning, local CPU |
| RNN | 2 layers, hidden 256, tanh | 64 in training, 256 at test | Random, dim 100 | 20 epochs (cap reached), 3,388 s |
| LSTM | 2 layers, hidden 256 | 64 in training, 256 at test | Random, dim 100 | 20 epochs (cap reached), 3,850 s |
| Transformer | 2 blocks, 4 heads, FF 256, learned positions | 64 in training, 256 at test | Random, dim 100 | 20 epochs (cap reached), 2,246 s |

[Sources: results/ngram.json; results/train_rnn.json; results/train_lstm.json; results/train_transformer.json.]

### 3.2 Evaluation and timing procedure

The metric is test perplexity, exp of the mean natural-log cross-entropy per scored token. All four models are scored on the same 981,353 target tokens: each test review cut to its first 256 tokens, batch size 64 [results/part1_costs.csv; DECISIONS I4, O1]. Neural models were trained and timed on a Colab Tesla T4 GPU; the trigram model was fitted on a local Apple M2 CPU [results/train_*.json hardware; results/ngram.json]. Training time is wall-clock seconds for the whole run. Throughput is the timed full `evaluate()` pass over the test targets (batch 64, context 256), after 2 warm-up passes, averaged over 3 repeats, on the T4 GPU and on one CPU thread; the trigram model was timed on CPU only [results/benchmark.json; DECISIONS O1, O2]. Re-scoring in the benchmark matched each saved perplexity within 0.1% [results/benchmark.json config `perplexity_rel_tolerance`].

### 3.3 Results

**Table 3: Prediction quality and cost.**

| Model | Test perplexity | Parameters | Training time (s) | GPU tokens/s | CPU tokens/s (1 thread) |
|---|---|---|---|---|---|
| Trigram (KN) | 111.31 | 6,445,342 (stored counts, not weights) | 0.84 (CPU) | n/a | 411,334 |
| RNN | 109.37 | 7,364,303 | 3,388 | 229,944 | 4,291 |
| LSTM | **80.45** | 8,033,999 | 3,850 | 164,100 | 4,027 |
| Transformer | 156.94 | 4,231,115 | 2,246 | 286,163 | 4,960 |

[Source: results/tables/part1_table.csv, built from results/part1_costs.csv. The trigram model's own end-of-run test perplexity is 116.54 on 1,390,673 full-review targets [results/ngram.json]; Table 3 uses the matched-token value. GPU throughput standard deviations over 3 repeats: RNN 1,200; LSTM 2,997; Transformer 7,950 [results/part1_costs.csv].]

**Figure 1: Training and validation loss per epoch** [results/figures/part1_loss_curves.pdf; solid = training, dashed = validation; same y-axis in all panels].

**Observations.**

- [OBS] Perplexity order, best to worst: LSTM 80.45, RNN 109.37, trigram 111.31, Transformer 156.94 [results/part1_costs.csv]. The LSTM is 28.92 below the RNN and 76.49 below the Transformer. The RNN is 1.94 below the trigram model, a small margin from one seed.
- [OBS] Cost: the Transformer trains fastest (2,246 s vs 3,388 s RNN and 3,850 s LSTM) and has the highest GPU throughput (286,163 vs 229,944 RNN and 164,100 LSTM tokens/s), with the fewest parameters [results/part1_costs.csv]. On one CPU thread the three neural models are within about 23% of each other (4,027 to 4,960 tokens/s) while the trigram model scores 411,334 tokens/s [results/part1_costs.csv].
- [OBS] Timing caveat: the identical LSTM architecture took 3,850 s in Part I but 3,394 s and 3,398 s in the two Part II runs (same 20 epochs, same GPU model) [results/train_lstm.json, p2_lstm_w2v.json, p2_lstm_glove.json]. A difference of about 450 s (12%) between runs of one architecture means the RNN-vs-LSTM training-time gap (462 s) is no larger than this run-to-run spread, so it should not be read as an architectural difference. [HYP] The spread may come from the Colab machine or session (different runs, not the same session). An alternative is the code version (the Part I LSTM ran on an older commit; DECISIONS V1). These cannot be separated from the saved records.
- [OBS] Convergence: all three neural models hit the 20-epoch cap with best epoch 20, so none triggered early stopping [results/train_*.json]. Over the last epoch the validation loss still fell by 0.0004 (RNN), 0.0068 (LSTM) and 0.0048 (Transformer) [results/train_*.json epoch_losses]. The RNN curve is nearly flat by epoch 10 (validation 4.745 at epoch 10, 4.705 at epoch 20); the LSTM is still improving at epoch 20 (4.525 at epoch 10, 4.399 at epoch 20) [same files].
- [OBS] Overfitting signal: the Transformer's training loss falls below its validation loss from epoch 3 on (train minus validation = −0.047 at epoch 3 and −0.269 at epoch 20, negative at every epoch from 3 to 20; +0.349 at epoch 1 and +0.036 at epoch 2), and its validation loss moves little after epoch 5 (5.135 at epoch 5, 5.064 at epoch 20) while training loss continues to fall (5.007 to 4.795) [results/train_transformer.json]. For the RNN and the LSTM the training loss stays above the validation loss at every one of the 20 epochs (gap at epoch 20: 0.216 and 0.179) [results/train_rnn.json, train_lstm.json]. The LSTM's validation loss fell every epoch; the RNN's rose slightly in two epochs (by 0.0095 at epoch 9 and 0.0024 at epoch 16) but never for 3 epochs in a row, so early stopping (patience 3) did not trigger [results/train_rnn.json, train_lstm.json].

### 3.4 Generated text

Decoding, identical for all four models: temperature 1.0, top-k 40, 50 new tokens maximum, seed 0, one sample per prompt, `<pad>` and `<unk>` never sampled, stop at `<eos>` [results/part1_generations.md; results/generations_*.json]. Three prompts only.

**Table 4: Continuations of the prompt "i was really disappointed by the acting and"** (negative opening; copied from results/part1_generations.md).

| Model | Continuation (excerpt) |
|---|---|
| Trigram (KN) | "timing of the story lines for their lives in the same age as his wife and kid , and just plain sucks ." (ended with `<eos>` after 23 tokens) |
| RNN | "directing , and it takes me out of some of the reviews and you see it if you ' re interested out with the most disgusting thing with a camera - - - up - the characters that do not seem like they do with some , which it would" (50 tokens) |
| LSTM | "costumes and the cast didn ' t take me to get bored , but i had nothing to do with any actor as the other reviewer who didn ' t go in there on this . she was funny , but she didn ' t come , then i didn" (50 tokens) |
| Transformer | "directing this was so thin to make this film . i guess i think this was because to be sure why did they are in order to give it up in the whole cast and the dialogue is awful . but what could do with !" (ended with `<eos>` after 46 tokens) |

Prompt 1 ("this movie was one of the best"): the LSTM continued "movies i have ever seen ." and ended after 6 tokens; the RNN gave "movies ever made . it is the story of life " , there are no surprises it ( the word ' are the most of the " i guess ..." (50 tokens); the Transformer "films ever made . it was the only thing about as a kid who had read to say ..." (50 tokens); the trigram "way to end . it is the very end when she was always going to happen ..." (50 tokens) [results/part1_generations.md]. Prompt 3 ("the story follows a young man who"): the full continuations are in Appendix B.

- [OBS] Stopping: with one sample per prompt, `<eos>` was produced in 1 of 3 generations for the trigram, LSTM and Transformer and in 0 of 3 for the RNN [results/generations_*.json metrics `num_ended_with_eos`]. The mean number of new tokens was 41.0 (trigram), 50.0 (RNN), 35.3 (LSTM), 48.7 (Transformer) [same files].
- [OBS] Repetition in the LSTM continuation of prompt 2: the contraction pieces "didn ' t" occur four times in the 50 tokens (the last one cut off by the length limit) [results/part1_generations.md].
- [CHECK] Judgement on local fluency: in the excerpts above, short stretches of every model read as plausible review phrases ("movies i have ever seen .", "and the dialogue is awful ."), but I judge all four continuations to drift off topic or become ungrammatical within about 20 tokens, with no clear fluency ranking visible from three prompts. Verify this yourself by reading Appendix B.
- [CHECK] Judgement on sentiment consistency: the trigram continuation of the negative prompt ends "just plain sucks ." and the Transformer's contains "the dialogue is awful", while the RNN's and LSTM's excerpts contain mixed signals ("she was funny , but she didn ' t come"). Verify on all three prompts; with one sample per prompt this is anecdote, not a measured difference.
- [HYP] The lower perplexity of the LSTM does not show up as visibly better text in these samples. One possible reason is that three single samples at temperature 1.0 are too few to show a difference that perplexity (an average over ~1M tokens) detects. An alternative is that perplexity differences of this size do not change local fluency much. These samples cannot separate the two; a larger blind rating of many samples would be needed.

### 3.5 Interpretation

Observed first: the LSTM predicts best; the Transformer predicts worst while being cheapest to train and fastest on GPU; the trigram model is close to the RNN; the RNN is nearly flat by epoch 10 (Section 3.3).

- [HYP] Why the LSTM beats the RNN. One possible reason is that gating lets the LSTM keep information over longer spans than the tanh RNN. Alternatives: the LSTM has more parameters (8.03M vs 7.36M), and the RNN may be limited by the shared learning rate and gradient clipping rather than by its architecture. Our experiments cannot tell these apart: we varied architecture only, with one seed and a fixed training recipe, and did not match parameter counts.
- [HYP] Why the Transformer is worst. One possible reason is that its learned position embeddings for positions 64 to 255 were never updated in training (trained on 64-token windows, scored on up to 256 tokens; DECISIONS M2), which could hurt scoring on 256-token inputs. Other alternatives: it has the fewest parameters (4.23M); its settings were copied from the LSTM and not tuned (no warm-up schedule, learning rate 0.001); it overfits, because its validation loss stops improving while its training loss keeps falling; or 20 epochs is not enough for it. The saved results cannot separate these: we have no run that scores the Transformer at 64 tokens, no run with trained positions up to 256, and no tuning runs.
- [HYP] Why a trigram model is competitive with the RNN. One possible reason is that a large share of next-word prediction in review text depends on the previous two words, which the trigram model captures exactly. An alternative is that the RNN was undertrained or that the evaluation setups differ slightly (the trigram model conditions only on two words but is fitted on all 11M training tokens in one pass). We have no experiment varying context length, so we cannot tell.
- [OBS] Trade-off supported by the results: the lowest perplexity (LSTM) costs the most training time and has the lowest GPU throughput of the three neural models; the Transformer is cheapest but has the worst perplexity (Table 3). Because training budgets, parameter counts and hyperparameters were not matched, these are trade-offs of these specific configurations, not of the architectures.

## 4. Part II: Embedding variants (≈ 1.5 pages)

### 4.1 Setup

We chose the LSTM, which had the lowest Part I perplexity (DECISIONS R1). The embedding dimension is 100 so that public GloVe 100d vectors fit without projection. Own embeddings: Skip-gram trained with gensim on the training split only (each review one sentence), window 5, min_count 1, 10 epochs, 5 negative samples, sampling 1e-3, 1 worker, seed 0 [results/word2vec_cost.json]. Public embeddings: `glove.6B.100d` (Wikipedia + Gigaword, 6B tokens, 400k lowercased words), matched to our tokens by exact string [DECISIONS S1]. Words without a vector get random normal values with the standard deviation of the real vectors in that table; `<pad>` is zeros [DECISIONS R2]. GloVe has a vector for 19,914 of the 20,002 non-pad vocabulary types (99.56%) and covers 97.52% of training tokens when `<unk>` and `<eos>` count as missing; excluding those two, 99.57% of types and 99.95% of tokens are covered; 88 rows were filled randomly (fill std 0.503) [results/glove_coverage.json]. The Skip-gram table covers all 20,002 types (fill std 0.279 is recorded but no row needed it) [results/word2vec_cost.json]. In both frozen runs the whole 20,003-row table has `requires_grad` off and the table is checked to be exactly unchanged after training (max absolute change 0.0) [results/p2_lstm_w2v.json, p2_lstm_glove.json details]. Everything else is identical across the three runs: LSTM with 2 layers, hidden 256, dropout 0.3, context 64, lr 0.001, batch 64, 20 epochs, seed 0 [results/train_lstm.json; configs]. The random-embedding setting reuses the Part I LSTM run.

### 4.2 Results

**Table 5: Embedding settings compared on the LSTM.**

| Setting | Trainable? | Extra data? | Test perplexity | Best epoch | Embedding cost |
|---|---|---|---|---|---|
| Random | Yes | No | 80.45 | 20 | n/a |
| Own Skip-gram | No (frozen) | No | 81.42 | 20 | 273.8 s train (local CPU, Apple M2; 274.5 s incl. vocabulary build) |
| GloVe 100d | No (frozen) | Yes (6B-token Wikipedia + Gigaword) | 90.28 | 20 | n/a (pretrained) |

[Sources: results/train_lstm.json; results/p2_lstm_w2v.json; results/p2_lstm_glove.json; results/word2vec_cost.json. Trainable parameters 8,033,999 (random) vs 6,033,699 (both frozen); LSTM training time 3,850 s / 3,394 s / 3,398 s; best validation loss 4.399 / 4.411 / 4.513.]

GloVe covers 99.56% of vocabulary types and 97.52% of training tokens; missing words were initialised with random normal values (std 0.503), `<pad>` with zeros [results/glove_coverage.json].

**Figure 2: Training and validation loss for the three embedding settings** [results/figures/part2_loss_curves.pdf. Note: the template asks for validation curves only; this figure also shows training loss, one panel per setting.]

### 4.3 Interpretation

- [OBS] Random 80.45 < own frozen 81.42 < GloVe frozen 90.28 test perplexity [Table 5]. Random vs own frozen differs by 0.98 (1.2%); own frozen vs GloVe by 8.86 (10.9%).
- [OBS] Convergence and stability: all three runs hit the 20-epoch cap with best epoch 20 and validation loss still falling by 0.005 to 0.007 in the last epoch [results/train_lstm.json, p2_lstm_w2v.json, p2_lstm_glove.json]. The comparison is therefore at a fixed 20-epoch budget, not at convergence (DECISIONS V1). At epoch 5 the validation losses are 4.697 (random), 4.670 (own frozen) and 4.800 (GloVe); at epoch 1 they are 5.425, 5.304 and 5.503 [same files]. The own-frozen run has the lowest validation loss at epochs 1, 5 and 10 (5.304, 4.670, 4.518 vs random 5.425, 4.697, 4.525), and the random run at epoch 20 (4.399 vs 4.411). The GloVe run is highest at all four epochs (5.503, 4.800, 4.640, 4.513). Per-epoch losses show no loss spikes; stability was only recorded per epoch, not per batch [DECISIONS V1].
- [HYP] Trainability costs little here. One possible reason is that Skip-gram vectors from our own 11M-token training set are already good input features, so letting the LSTM adapt them adds little. Alternatives: the 0.98 gap is within what a different seed might produce (single seed per run); the random run ran on an older commit than the frozen runs (DECISIONS V1); and with 20 epochs and all runs still improving, a longer budget might change the gap. The experiments can isolate trainability (same data) but cannot say whether 0.98 is larger than seed variation.
- [HYP] Why GloVe is worst. One possible reason is domain and tokenisation mismatch: GloVe was trained on Wikipedia/Gigaword text with its own tokenisation, while our tokens split contractions ("don ' t") and keep punctuation as tokens, and review slang may be rare in its corpus. Alternatives: the scale of the vectors (randomly filled rows have std 0.503 vs 0.279 for our fill rule, and the real vector scales may differ between tables); the 88 randomly filled rows (0.44% of types) including `<unk>` and `<eos>`; or something about the extra pretraining data itself. The design separates "extra data" only in combination with these differences: own-frozen vs GloVe changes data, tokenisation, domain and scale at once, so the experiments cannot attribute the 8.86 gap to extra data alone.
- [CHECK] Generated text (Appendix B): for prompt 1 all three LSTM settings produced the same continuation, "movies i have ever seen ." ending after 6 tokens [results/part2_generations.md]. For prompt 2 the three continuations differ ("costumes and the cast didn ' t take me...", "directing , and it took me too long to see this series...", "lighting and a lot more than so much that , for instance ..."). I see no clear quality difference among the three; this rests on 3 prompts and 1 sample each, so it is anecdotal. Verify by reading Appendix B.

## 5. Part III: Downstream sentiment classification (≈ 1 page)

### 5.1 Setup

Task: binary sentiment (positive/negative) on the same 40,000/5,000/5,000 splits (Table 1), so no classification test review was used to train a language model. We compare two representations: the Part I LSTM and the Part I Transformer (both with random trainable embeddings, trained as in Part I), frozen (weights checked unchanged) [results/p3_feat_lstm_mean.json, p3_feat_transformer_mean.json details `weights_unchanged`]. Each review is cut to its first 256 tokens, `<eos>` dropped, padded on the right (37.0% of train, 38.2% of test reviews are truncated) [results/p3_prepare_classification.json; DECISIONS W1, W2]. The review vector is the mean of the model's final hidden states over real (non-pad) positions: the last recurrent layer's output for the LSTM (256-d) and the output of the final LayerNorm for the Transformer (100-d) [DECISIONS X1, X2]. Both models are causal, so padding on the right cannot change real positions (checked in tests/test_extract_features.py). The classifier is scikit-learn logistic regression (L2) on standardised features (training mean/std), C chosen from {0.001, 0.01, 0.1, 1, 10, 100} by validation macro-F1, then evaluated once on the test split; same procedure and seed for both [DECISIONS Y1; results/p3_classify.json]. Metrics: accuracy and macro-F1 (classes are balanced, so they coincide to the reported precision).

### 5.2 Results

**Table 6: Sentiment classification on the IMDB test split (5,000 reviews).**

| Representation | LM test perplexity | Features | Best C | Val. accuracy | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|---|
| LSTM, mean-pooled | 80.45 | 256 | 0.01 | 0.8154 | 0.8192 | 0.8192 |
| Transformer, mean-pooled | 156.94 | 100 | 0.1 | 0.7624 | 0.7744 | 0.7744 |

[Sources: results/part3.csv; results/p3_classify.json; results/part3_ranking.csv.]

- [OBS] The LSTM representation is higher on test accuracy by 0.0448 (0.8192 vs 0.7744) and on validation by 0.0530 (0.8154 vs 0.7624) [results/p3_classify.json]. Classifier validation performance is almost flat across C for the LSTM (macro-F1 0.806 at C=0.001, 0.815 for C ≥ 0.01) [results/p3_classify.json details], so the choice of C matters little. All fits converged.

### 5.3 Error analysis

LSTM features misclassify 904 of 5,000 test reviews and Transformer features 1,128 [results/p3_classify.json]. 563 reviews are missed by both, 341 only by the LSTM classifier and 565 only by the Transformer classifier [counted from results/part3_errors_p3_feat_*.csv]. The direction of errors is roughly balanced: LSTM 469 negative reviews predicted positive and 435 positive predicted negative; Transformer 536 and 592 [counted from results/part3_errors_p3_feat_*.csv].

**Table 7: Representative misclassified reviews** (excerpts copied from results/part3_error_sample_p3_feat_*_mean.csv; the pattern column is a judgement, [CHECK]).

| Review (excerpt; original index) | True | Predicted | Pattern [CHECK] | Missed by both? |
|---|---|---|---|---|
| "This movie definitely had some bad sound/editing/lighting/acting/etc. etc. problems. However, this movie has many positive things about it. ... For what it is, just that, it is really good." (39703, LSTM sample) | positive | negative | negative-sounding words, then a positive verdict | yes |
| "Unashamedly ambitious sci-fi ... Unfortunately it's just not that good. It all starts well enough ... Nice ending though." (37791, LSTM sample) | negative | positive | praise mixed into a negative review | yes |
| "I've seen every film Lou Ferrigno has made. I've seen Batman & Robin...twice. ... yet this film is the most painful thing I've ever seen." (10082, Transformer sample) | negative | positive | ironic build-up before the verdict | yes |
| "The acting was pretty bad, but what can you expect from musicians ... Despite that, it was still a very entertaining film!" (48419, Transformer sample) | positive | negative | complaint first, positive verdict | no: only the Transformer classifier missed it |

- [OBS] Counted properties of errors vs correct reviews [results/part3_error_stats.csv]: mean tokens kept 203.8 vs 195.1 (LSTM) and 202.0 vs 195.1 (Transformer); share truncated at 256 tokens 45.0% vs 37.0% (LSTM) and 41.8% vs 37.5% (Transformer); mean `<unk>` rate 2.34% vs 2.02% (LSTM) and 2.29% vs 2.02% (Transformer); share with at least one negation word 87.7% vs 87.4% (LSTM) and 89.3% vs 87.0% (Transformer). Negation words are about as common in correct reviews as in errors, so a negation count alone does not separate them. Errors are slightly longer, more often truncated and slightly higher in `<unk>`; the differences are small and were not tested for significance.
- [CHECK] Patterns, from 15 random errors per classifier read by the assistant (so tendencies in 30 reviews, not rates): in the LSTM sample, positive reviews that spend words on faults and are predicted negative (about 8 of 15) and negative reviews that praise first (about 4 of 15); in the Transformer sample, sarcasm or deadpan mockery in negative reviews predicted positive (about 4 of 15) and positive reviews dwelling on faults predicted negative (about 5 of 15) [results/part3_error_notes.md, assistant-drafted; counts are manual and a review can fit several patterns]. Read the 30 reviews yourself before keeping any of these counts.
- [HYP] Sarcasm may be harder for the Transformer features. One possible reason is that its mean-pooled vectors lose word-order cues that irony depends on; alternatives are that the pattern is sampling noise in a 15-review sample, that the 100-d vector carries less information than the 256-d one, or that the two classifiers simply differ in C. The samples cannot tell these apart.

### 5.4 Does language-model ranking predict downstream ranking?

- [OBS] The rankings agree: perplexity LSTM 80.45 (rank 1) and Transformer 156.94 (rank 2); accuracy LSTM 0.8192 (rank 1) and Transformer 0.7744 (rank 2) [results/part3_ranking.csv].
- [OBS] Only two models were classified. The Part I RNN (perplexity 109.37, between the two) and the Part II frozen-embedding LSTMs (81.42, 90.28) were not, so with two positions the agreement could occur by chance (one of two possible orderings).
- [OBS] The accuracy gap of 0.0448 on 5,000 reviews is large relative to the binomial standard error of one accuracy near 0.8 (about 0.006 by the formula sqrt(p(1−p)/n); derived, not in a results file), and the classifiers' discordant errors are 341 vs 565. One seed and one pooling, so run-to-run variation was not measured.
- [HYP] Perplexity may predict downstream quality because a model that predicts the next word well must encode topic and sentiment cues in its hidden state. Alternatives that our runs cannot separate from this: (a) feature size (256 vs 100 columns: more features, more capacity for the classifier); (b) which layer the vector comes from (last LSTM layer vs Transformer after final LayerNorm); (c) the Transformer's untrained positions 64 to 255 when features are extracted from 256-token inputs (DECISIONS M2); (d) the classifier's C differing (0.01 vs 0.1). Only a design that holds these fixed while varying perplexity (for example several LSTMs of different perplexity with the same feature size) could separate them.
- Do the Part II findings help explain Part III? [OBS] No Part II model was classified; Part III used the random-embedding Part I models. [OBS] Part II shows that, within the LSTM, the embedding source changed perplexity from 80.45 to 90.28 [Table 5]. This tells us nothing directly about classification. Whether frozen GloVe or Skip-gram LSTMs would classify worse in proportion to their higher perplexity is untested here; it is the experiment that would extend the answer.

## 6. Discussion: answering the comparison question (≈ 0.4 page)

[OBS] In the one comparison we ran, better next-word prediction and better classification went together: the LSTM beats the Transformer on perplexity (80.45 vs 156.94) and on accuracy (0.8192 vs 0.7744) and macro-F1 (0.8192 vs 0.7744) [results/part3_ranking.csv]. [OBS] Part I shows that perplexity ranking is not the same as cost ranking: the best-predicting LSTM is the slowest on GPU and the Transformer the fastest. [OBS] Part II shows that a fixed embedding table can cost perplexity (81.42 and 90.28 vs 80.45), and that this cost depends strongly on where the vectors come from, with the public vectors costing most; this cost was measured on one LSTM seed at a fixed 20-epoch budget.

[HYP] A cautious answer is: perplexity and downstream accuracy agreed here, but with two models, one seed, and feature size and layer differing, our data cannot show that lower perplexity causes better features. The design cannot rule out feature size, the Transformer's untrained long positions or classifier regularisation as the reason (Sections 3.5, 5.4). [HYP] A third model (the Part I RNN, perplexity between the two) would show whether accuracy follows perplexity rank for three points; we did not run it.

## 7. Limitations and lessons learned (≈ 0.3 page)

- One seed (0) per run; no run-to-run variance for any number [all results files]. Differences such as 80.45 vs 81.42 (Part II) or 109.37 vs 111.31 (RNN vs trigram) may be within seed variation.
- All neural runs stopped at the 20-epoch cap while still improving, so comparisons are at a fixed budget, not at convergence (DECISIONS V1). Training budgets and parameter counts were not matched across architectures, and the Transformer was not tuned.
- Training windows are 64 tokens but scoring and feature extraction use up to 256 tokens (DECISIONS M2, I4, W1).
- Part I and Part II used one corpus and one test split; Part III compares only two language models on one task (IMDB sentiment) with one pooling method and one classifier family.
- The Part II frozen runs record `git_dirty: true` and the Part I LSTM ran on an older commit; no rerun was done (DECISIONS V1). The `RESULTS.md` check "test set used only for final evaluation" shows FAIL because `src/benchmark.py` reads the test split to time scoring (DECISIONS J4); the tuning of discounts, C and early stopping used validation data only [results/ngram.json; results/p3_classify.json]. [CHECK] confirm this wording of the FAIL.
- Lesson (to confirm in your own words): [your lesson, e.g. about matching budgets before comparing architectures].

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. ACL 2011.
- IMDB Dataset of 50K Movie Reviews (Kaggle): https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014 (`glove.6B.100d`, https://nlp.stanford.edu/projects/glove/).
- Řehůřek & Sojka (2010), gensim; Mikolov et al. (2013), word2vec (Skip-gram); Kneser & Ney (1995); PyTorch (Paszke et al., 2019); scikit-learn (Pedregosa et al., 2011). [Confirm each citation and add the course notebooks or external code you adapted, if any.]

## Tool use and AI assistance

[Draft, true only after you verify it.] I used Claude Code (Anthropic's coding assistant) throughout this assignment. It wrote and tested the code in this repository (data pipeline, n-gram, RNN/LSTM/Transformer models, training, benchmark, generation, word2vec and GloVe handling, feature extraction, classifier, error-review material, `RESULTS.md`/`MY_TASKS.md` generation, and the report tables and figures), following project rules I set in `CLAUDE.md`; every design choice and its status is logged in `DECISIONS.md` and each change in `TOOL_USE.md`. I ran or approved the Colab GPU runs. The assistant also drafted the text and analysis of this report, including the [OBS]/[HYP] split and the error-pattern notes, from the result files; I then [verified every number against the results files, read the generated text and the misclassified reviews myself, and rewrote and edited the text. State here truthfully how much you changed.] [Replace with your own account of what you understand and can explain.] The error-pattern notes in `results/part3_error_notes.md` were also assistant-drafted. External material used: IMDB data (Kaggle/Maas et al.), GloVe `glove.6B.100d`, gensim, PyTorch, scikit-learn [add exact model names/versions of the assistant used].

## Appendix A: Hyperparameters

| Run | Layers | Hidden | Context (train) | Batch | LR | Dropout | Epochs (cap) | Seed | Hardware |
|---|---|---|---|---|---|---|---|---|---|
| ngram (trigram KN) | n/a | n/a | 2 words | 32 (scoring) | n/a | n/a | n/a | 0 | CPU: Apple M2 |
| train_rnn | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| train_lstm | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| train_transformer | 2 blocks, 4 heads, FF 256 | 256 (FF) | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p2_lstm_w2v | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p2_lstm_glove | 2 | 256 | 64 | 64 | 0.001 | 0.3 | 20 | 0 | Tesla T4 |
| p3_classify | logistic regression, C grid 0.001 to 100 | n/a | 256 tokens | n/a | n/a | n/a | n/a | 0 | CPU: Apple M2 |

Also: Adam, gradient clip 1.0, patience 3, embedding dim 100, vocabulary 20,000 (+3 special) [results/all_runs.csv; configs/].

## Appendix B: Full generated samples

Decoding: temperature 1.0, top-k 40, 50 new tokens, seed 0, 1 sample, `<pad>`/`<unk>` banned, stop at `<eos>`. [Paste the full text of results/part1_generations.md (all four models, three prompts) and results/part2_generations.md (three LSTM embedding settings, three prompts) here, grouped by prompt.]

## Appendix C: Reproducibility

[Repository link; hardware: Colab Tesla T4 for neural training, benchmark, generation and feature extraction, Apple M2 CPU for data, n-gram, word2vec, classifier and figures; Python 3.10.11, torch 2.14.1, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 (README.md; Colab package versions not recorded, add them). Commands are in `notebooks/run_colab.ipynb` and `README.md`. Tables and figures: `python3 src/make_figures.py --config configs/make_figures.json`. NOTE: `README.md` is out of date (see VERIFY.md).]

## Appendix D: Additional figures

[None yet. Optional: the validation-loss comparison figure `results/figures/val_loss_all_runs.png`.]
