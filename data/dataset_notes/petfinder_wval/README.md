# petfinder_wval

Constructed directly from `petfinder_processed/train.csv` and the labeled
`petfinder_processed/dev.csv` by
`data/preprocessing/prepare_petfinder.py`.

Both files are included in the
[AutoGluon PetFinder archive](https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip).

- train: 9,697 rows
- validation: 1,385 rows
- test: 2,748 rows
- numerical predictors: 3
- categorical predictors: 14
- response: five-class `AdoptionSpeed`
- text: cleaned English `Description`

The validation split uses `random_state=0` and contains one eighth of the
filtered training pool while preserving every validation categorical level in
the remaining training set.
