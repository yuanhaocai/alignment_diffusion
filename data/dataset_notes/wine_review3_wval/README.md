# wine_review3_wval

Constructed directly from `winemag-data-130k-v2.csv` by
`data/preprocessing/prepare_wine_reviews.py`. This is the only Wine Reviews
dataset used by the workflows.

Download the CSV from the
[Wine Reviews dataset page](https://www.kaggle.com/datasets/zynicide/wine-reviews).

- train: 54,087 rows
- validation: 7,726 rows
- test: 15,454 rows
- numerical predictors: 1
- categorical predictors: 2
- continuous response: review score
- text: review description

The initial 80/20 training/test split uses `random_state=42`. The validation
split then takes one eighth of the training pool with `random_state=0` while
preserving every validation categorical level in the remaining training set.
