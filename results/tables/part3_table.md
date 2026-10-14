**Part III: classifier on frozen language-model features.**

| Representation | Features | Best C | Val. accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| p3_feat_lstm_mean | 256 | 0.01 | 0.8154 | 0.8192 | 0.8192 |
| p3_feat_transformer_mean | 100 | 0.1 | 0.7624 | 0.7744 | 0.7744 |

- Logistic regression on frozen features; C chosen on validation macro-F1; 5,000 test reviews.
