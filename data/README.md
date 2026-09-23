# Processed data contract

The repository does not distribute observation-level data. Place one directory
per dataset under a user-selected data root. The tabular/text datasets use:

```text
DATA_ROOT/
  dataset_name/
    X_num_train.npy
    X_num_val.npy
    X_num_test.npy
    X_cat_train.npy
    X_cat_val.npy
    X_cat_test.npy
    y_train.npy
    y_val.npy
    y_test.npy
    text_train.json
    text_val.json
    text_test.json
```

Numerical arrays have shape `(n, p_num)`, categorical arrays have shape
`(n, p_cat)`, and response arrays have shape `(n,)` or `(n, 1)`. Text files are
JSON lists whose row ordering must match the NumPy arrays.

Shopee has the same train/validation/test response and placeholder arrays,
plus `shopee_image_embd/{train,val,test}/`.

## Dataset directories

- `petfinder_wval`: 9,697 train, 1,385 validation, and 2,748 test rows; three
  numerical predictors, fourteen categorical predictors, five response
  classes, and one text field.
- `wine_review3_wval`: 54,087 train, 7,726 validation, and 15,454 test rows.
- `mimic_adm_pt_disch_discharge_censored_wval`: leakage-controlled,
  section-censored discharge-summary text with train/validation/test splits.
- `shopee`: 640 train, 160 validation, and 80 test rows by default. The first
  two splits come from the paper's 800-row training pool. Image embeddings are
  stored separately and indexed in the same row order.

The point-prediction workflows use the validation-set versions of Petfinder
and Wine Reviews. Revised Wine prediction intervals additionally use the
original `wine_review3` non-test pool (61,813 rows), then create a new
46,361/7,726/7,726 train/validation/calibration split in the run directory.
The original 15,454-row test is copied byte-for-byte. Calibration files use
the `_cal.npy` / `text_cal.json` suffixes. Do not use `wine_review3_wval/train`
as the original pool or reuse its validation set for calibration.

See the top-level README's complete full and tabular-only PI commands. The
PI CLIP cache retains all 77 token states (`--keep-padding`); it must not be
substituted with masked-padding point-prediction embeddings.

## Embedding layout

```text
EMBEDDING_ROOT/
  dataset_name_text_clip_embd/
    train/0.npy
    val/0.npy
    test/0.npy

PRETRAINED_ROOT/
  dataset_name/
    tabular_vae/
      train_z_dat.npy
      val_z_dat.npy
      test_z_dat.npy
    response_vae/
      train_z_y.npy
      val_z_y.npy
      test_z_y.npy
```

Continuous-response datasets use `response_standard_scaler/y_*_scaled.npy`
instead of a categorical response VAE. The commands in the top-level README
construct all these files.

## Additional dataset notes

Dataset-specific columns, filtering rules, and split details are documented in
`dataset_notes/`. Download and preprocessing commands are kept in the
top-level `README.md`.
