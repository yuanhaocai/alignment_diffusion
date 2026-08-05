# shopee

Constructed by `data/preprocessing/prepare_shopee.py` from the
[AutoGluon Shopee archive](https://automl-mm-bench.s3.amazonaws.com/vision_datasets/shopee.zip).

- training pool: 800 rows
- train: 640 rows by default
- validation: 160 rows by default
- test: 80 rows
- response: four product categories
- condition: 512-dimensional embeddings from the supervised AutoGluon
  ResNet-18 classifier

The preparation script makes a stratified 80/20 train/validation split from
the 800-row training pool before fitting the supervised feature extractor.
AutoGluon may additionally use its own 20% holdout within the 640-row training
split. Diffusion selection uses the 160-row validation split; the 80-row test
split is final-only.
