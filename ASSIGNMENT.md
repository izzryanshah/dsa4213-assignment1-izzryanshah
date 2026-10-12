# DSA4213 Assignment 1: Language Models, Embeddings, and Representations

Source: `Assignment1_2026.pdf` (Natural Language Processing for Data Science, Doudou Zhou, Semester 1, AY2026/2027; last updated 2026/09/08). This file restates the brief's requirements and scope; the PDF is the authoritative text.

## Overview
- Individual assignment, worth **30%** of the course grade.
- Covers language modeling, word embeddings, RNNs, LSTMs, Transformers, and learned representations.
- You compare model families on a common corpus, examine the effect of embedding choices, and evaluate whether the learned representations help a downstream task.
- Part III reuses a trained language model from Part I or II and may use a different dataset.

## Due date
- **October 16, 2026, 11:59 pm (Singapore time).**
- Late submission policy and submission platform: to be announced separately.

## General requirements
- Choose your own English text corpus for Parts I and II.
- For Part III, choose a suitable labeled dataset for the downstream task and use representations from a language model trained in Part I or II.
- Train the required language models yourself, from scratch. Part II explicitly allows public pretrained word embeddings.
- Allowed libraries: PyTorch, TensorFlow, HuggingFace, gensim, and other standard libraries. Course notebooks may be adapted with attribution.
- Choose your own dataset size and model size, keeping the scale computationally reasonable. A manageable subset of the data is allowed.
- Comparisons should be reasonable and clearly explained.

## Part I: Language Model Training and Comparison (40 points)

Using the **same training corpus**, train and compare these next-token language models:
1. n-gram language model with smoothing or interpolation
2. RNN language model
3. LSTM language model
4. Transformer language model with causal self-attention

Constraints:
- The three neural models use **randomly initialized, trainable input embeddings**.
- Select architectures, hyperparameters, context lengths and decoding settings as appropriate, and **document the choices**.
- **All models must predict each target using only preceding tokens.**

The comparison must include:
| Item | Required content |
|---|---|
| Prediction quality | Test perplexity or average next-token cross-entropy on the common evaluation setup. Padding tokens excluded from the loss. |
| Training and inference cost | Training time, model size, and inference time or throughput. State the hardware, training budget and timing procedure; use a common inference workload. |
| Learning behavior | Training and validation loss curves for the neural models; discuss convergence or overfitting. |
| Generated text | Continuations from the same small set of prompts, with matched decoding settings and generation lengths. Discuss concrete differences such as repetition, local fluency and consistency. |

Also: explain the main tradeoffs supported by the results. Separate observed differences from possible explanations involving architecture, parameter count, context length or training budget.

## Part II: Embedding Variants and Ablation (30 points)

Select **one** neural architecture from Part I and compare three input-embedding settings:
1. Randomly initialized embeddings learned jointly with the language model.
2. Embeddings trained by yourself on the training corpus using one method (Skip-gram, CBOW or GloVe), then **kept fixed** during language-model training.
3. Public pretrained word embeddings (such as GloVe), **kept fixed** during language-model training.

Constraints:
- Keep the embedding dimension, remaining architecture, data and training procedure the same across the three settings.
- Choose the dimension with the public embeddings in mind.
- Document vocabulary coverage and how missing words are initialized.
- In the fixed settings, keep the **entire input-embedding table** fixed.

Compare: test prediction quality, convergence, training stability, and representative generated text. Report the cost of training your own embeddings separately from language-model training. Interpret the comparison in light of both embedding trainability and the additional training data used by public embeddings.

Runs:
- Setting 1: reuse the Part I model if its embedding dimension, architecture, data and training procedure match this comparison.
- Train **two new language models**: one with your own frozen embeddings, one with public frozen embeddings. Their remaining parameters are trained from scratch.
- Training your own word embeddings is a separate step from these two runs.
- Testing every embedding setting on every architecture is optional.

## Part III: Downstream Representation Analysis (20 points)

Use the models from Parts I–II for a simple supervised task, such as sentiment or topic classification.

- Compare **at least two** representation choices on the same labeled data. Examples: LSTM vs Transformer; two embedding variants; two pooling methods on one trained model.
- Describe how each input becomes a fixed-length representation (for example a final hidden state, or an average of token states excluding padding).
- Use the same simple classifier family and evaluation procedure for both choices.
- Freezing the language models and training only the classifiers is sufficient. Fine-tuning is optional and must be applied consistently across the comparison.
- Report a suitable held-out task metric (accuracy or macro-F1) and examine representative errors.
- Explain which representations work better in the experiment.
- If different language models are compared, discuss whether their language-model performance ranking agrees with their downstream ranking.

## Experimental focus
Organize the discussion around a clear comparison question, such as: how model families differ on your corpus, whether fixed embeddings help, or whether better next-token prediction leads to better classification. The required experiments may be sufficient to answer the question; additional models or large hyperparameter searches are not required.

## Submission
- Submit a **PDF report** and your **code**, either as a GitHub link accessible to the instructor or as a zip file.
- Report length: **at most 6 pages**, excluding appendices and references. Keep the main results and analysis in the main report.
- The report should focus on results and comparisons; unnecessary background is not rewarded.

The report must clearly state:
- the dataset used and preprocessing decisions;
- the models and variants compared;
- the evaluation criteria;
- the main quantitative and qualitative observations;
- the most important limitations, failure cases and lessons learned.

The code must include a `README.md` with: data access and preprocessing instructions, dependencies, commands, configurations, and the random seeds needed to reproduce the main results.

Do not submit API keys, private credentials, or data you do not have permission to share.

## Grading
Marks reflect correct implementation, fair comparisons, supporting evidence and clear interpretation. Strong submissions explain specific successes and failures, connect results across the three parts, and distinguish observations from hypotheses.

Reduce marks: missing required comparisons, data leakage, unsupported conclusions, missing reproducibility information. A larger model or a better raw score alone does not earn higher marks.

| Component | Points |
|---|---|
| Part I: language model implementation, comparison, and interpretation | 40 |
| Part II: embedding variants, ablation design, and analysis | 30 |
| Part III: downstream task and representation analysis | 20 |
| Report clarity, reproducibility, and responsible tool/data use | 10 |
| **Total** | **100** |

## Academic integrity and tool use
- Allowed: standard libraries, tutorials, documentation, pretrained embeddings, LLMs and coding assistants.
- Required: clearly cite external code, datasets, pretrained models and substantial assistance from tools.
- Grading is based on problem formulation, implementation, experimental evidence, evaluation quality, interpretation, reproducibility, and the ability to explain the results. The submitted analysis and interpretation should reflect your own understanding.
