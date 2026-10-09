# DSA4213 Assignment 1 — NLP for Data Science

## Dataset

This project uses the **IMDB Dataset of 50K Movie Reviews** (50,000 reviews, labelled `positive`/`negative`).

The dataset is downloaded **manually** from Kaggle, not programmatically, and is **not committed to this repository** — `data/` is listed in `.gitignore` so the raw CSV never ends up in git history.

### How to get it

1. Download `IMDB Dataset.csv` from Kaggle: <PASTE THE EXACT KAGGLE DATASET URL YOU USED HERE> (requires a free Kaggle account).
2. Place the file at `data/IMDB Dataset.csv` in this repository.
3. Run `python src/inspect_data.py` to sanity-check it loaded correctly.

### Citation

This Kaggle dataset is itself built from the IMDB movie review data originally released by Maas et al. (2011). Cite both the Kaggle page and the original paper:

- Kaggle dataset page: <https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download>
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
