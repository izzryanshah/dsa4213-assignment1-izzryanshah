**Part II: LSTM with three embedding settings.**

| Embedding | Test perplexity | Best val. loss | Trainable params | Train time (s) | Coverage, types (%) | Coverage, tokens (%) | word2vec time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| random (learned) | 80.45 | 4.399 | 8,033,999 | 3850 | -- | -- | -- |
| own Skip-gram (frozen) | 81.42 | 4.411 | 6,033,699 | 3394 | 100.00 | 100.00 | 273.8 |
| public GloVe 100d (frozen) | 90.28 | 4.513 | 6,033,699 | 3398 | 99.56 | 97.52 | -- |

- Same LSTM, data, seed and settings in all rows; only the embedding table differs.
- Coverage: share of vocabulary types / training tokens that have a real vector before missing rows are filled. word2vec time: training on the local CPU (Apple M2), not the GPU used for the language-model runs.
- Total parameters including the frozen table: random (learned): 8,033,999; own Skip-gram (frozen): 8,033,999; public GloVe 100d (frozen): 8,033,999.
