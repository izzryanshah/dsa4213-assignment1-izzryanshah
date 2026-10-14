**Part I: language models on the IMDB test set.**

| Model | Test perplexity | Parameters | Train time (s) | GPU tokens/s | CPU tokens/s |
|---|---:|---:|---:|---:|---:|
| ngram | 111.31 | 6,445,342* | 0.84 | -- | 411,334 |
| rnn | 109.37 | 7,364,303 | 3388 | 229,944 | 4,291 |
| lstm | 80.45 | 8,033,999 | 3850 | 164,100 | 4,027 |
| transformer | 156.94 | 4,231,115 | 2246 | 286,163 | 4,960 |

- * stored n-gram counts, not weights (not comparable with the neural models).
- Test perplexity: all models on the same 981,353 test target tokens. Throughput: mean of repeated passes, tokens per second (GPU: Tesla T4; CPU: one thread). Train time: n-gram on local CPU, neural models on the Colab GPU.
