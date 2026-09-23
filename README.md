# Alignment and conditional-diffusion experiments

This repository contains the minimal code and configurations for the
Petfinder, Wine Reviews, MIMIC-IV, Shopee, and Wine prediction-interval
experiments in the paper.

## Repository layout

- `src/tabsynfnn/`: VAE and predictive conditional-diffusion implementation.
- `src/alignment/`: multimodal alignment implementation and paper baselines.
- `scripts/`: data preparation, embedding, alignment transformation, diffusion
  training and sampling, baseline, and evaluation commands.
- `configs/`: runnable JSON workflows and the CoDSA candidate manifest.
- `data/preprocessing/`: dataset construction and train/validation splitting.
- `data/dataset_notes/`: schemas for the processed datasets used by the code.

Prepare raw data and model artifacts locally before running the workflows. The
scripts below create embeddings from processed data and trained models.

## Installation

Use Python 3.10 and create one virtual environment named
`alignment-diffusion`. For example, from the repository root:

```bash
python3.10 -m venv alignment-diffusion
source alignment-diffusion/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

All dependencies for the included preprocessing, CLIP, alignment, diffusion,
prediction-interval, and Shopee/AutoGluon workflows are declared in
`pyproject.toml`. GPU training additionally requires a working NVIDIA driver
compatible with the installed PyTorch build.

LANISTR, LimiX, and CoDSA are external baseline repositories and are not
included in this source tree. Their links are provided under "Baselines"
below.

## Data preparation

All active tabular/text workflows use a validation split:

- `petfinder_wval`
- `wine_review3_wval`
- `mimic_adm_pt_disch_discharge_censored_wval`
- `shopee` for the separate image experiment

The revised Wine prediction-interval experiment starts from the original
61,813-row non-test pool and creates separate training, validation, and
calibration sets. It preserves the original 15,454-row test set; see
[Prediction intervals](#prediction-intervals-wine-reviews).

### Petfinder

Download and unzip the
[PetFinder archive](https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip).
It creates `petfinder_processed/`; this workflow uses `train.csv` as the
training pool and the labeled `dev.csv` as the test set:

```bash
curl -L https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip \
  -o /path/to/raw/petfinder_kaggle.zip
unzip /path/to/raw/petfinder_kaggle.zip -d /path/to/raw/petfinder

python data/preprocessing/prepare_petfinder.py \
  --train-csv /path/to/raw/petfinder/petfinder_processed/train.csv \
  --test-csv /path/to/raw/petfinder/petfinder_processed/dev.csv \
  --output-dir /path/to/data/petfinder_wval
```

### MIMIC-IV

Obtain credentialed access to [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/)
and [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/).
Only these downloaded files are needed:

- `mimiciv/2.2/hosp/admissions.csv.gz`
- `mimiciv/2.2/hosp/patients.csv.gz`
- `mimic-iv-note/2.2/note/discharge.csv.gz`

Decompress or copy them into the following layout:

```text
/path/to/raw/
  mimiciv/
    admissions.csv
    patients.csv
  mimic-iv-note/
    discharge.csv
```

Then construct the processed dataset:

```bash
python data/preprocessing/build_leakage_controlled_dataset.py \
  --data-root /path/to/data \
  --mimic-dir /path/to/raw/mimiciv \
  --discharge-csv /path/to/raw/mimic-iv-note/discharge.csv \
  --new-data-name mimic_adm_pt_disch_discharge_censored_wval
```

Dataset layouts and feature schemas are described in `data/README.md`.

### Wine Reviews

Download `winemag-data-130k-v2.csv` from the
[Wine Reviews dataset page](https://www.kaggle.com/datasets/zynicide/wine-reviews),
or use the configured Kaggle command-line client. Then convert it directly to
`wine_review3_wval`:

```bash
kaggle datasets download -d zynicide/wine-reviews -p /path/to/raw/wine_reviews
unzip /path/to/raw/wine_reviews/wine-reviews.zip -d /path/to/raw/wine_reviews

python data/preprocessing/prepare_wine_reviews.py \
  --raw-csv /path/to/raw/wine_reviews/winemag-data-130k-v2.csv \
  --output-dir /path/to/data/wine_review3_wval
```

### Shopee

Download and unzip the
[Shopee archive](https://automl-mm-bench.s3.amazonaws.com/vision_datasets/shopee.zip).
Use its `train.csv`, `test.csv`, `train/`, and `test/` files. Shopee uses
512-dimensional embeddings extracted by the supervised AutoGluon ResNet-18
classifier used for the image baseline:

```bash
curl -L https://automl-mm-bench.s3.amazonaws.com/vision_datasets/shopee.zip \
  -o /path/to/raw/shopee.zip
unzip /path/to/raw/shopee.zip -d /path/to/raw/shopee

python data/preprocessing/prepare_shopee.py \
  --train-csv /path/to/raw/shopee/train.csv \
  --test-csv /path/to/raw/shopee/test.csv \
  --image-root /path/to/raw/shopee \
  --output-dir /path/to/data/shopee \
  --predictor-dir /path/to/artifacts/shopee_autogluon
```

See `data/README.md` for processed array shapes, directory names, and embedding
layouts.

## Creating embeddings

### Text

The text encoder uses `openai/clip-vit-large-patch14`, maximum length 77, and
writes one `(1, 77, 768)` array per row. Padding rows are zeroed after encoding
for masked mean pooling in the alignment and diffusion models.

```bash
python scripts/extract_clip_text_embeddings.py \
  --input-json /path/to/data/wine_review3_wval/text_train.json \
  --output-dir /path/to/embeddings/wine_review3_wval_text_clip_embd/train
```

Run it once for each of `train`, `val`, and `test`.

### Tabular predictor and categorical response

```bash
python scripts/train_vae.py \
  --data-dir /path/to/data/petfinder_wval \
  --output-dir /path/to/pretrained/petfinder_wval/tabular_vae \
  --target dat --task-type multiclass \
  --latent-dim 768 --d-token 7 --gpu 0

python scripts/encode_vae_latents.py \
  --data-dir /path/to/data/petfinder_wval \
  --vae-dir /path/to/pretrained/petfinder_wval/tabular_vae \
  --target dat --task-type multiclass --device cuda:0

python scripts/train_vae.py \
  --data-dir /path/to/data/petfinder_wval \
  --output-dir /path/to/pretrained/petfinder_wval/response_vae \
  --target y --task-type multiclass --d-token 5 --gpu 0

python scripts/encode_vae_latents.py \
  --data-dir /path/to/data/petfinder_wval \
  --vae-dir /path/to/pretrained/petfinder_wval/response_vae \
  --target y --task-type multiclass --device cuda:0
```

Continuous responses, including Wine Reviews, use a training-only scaler:

```bash
python scripts/scale_continuous_response.py \
  --data-dir /path/to/data/wine_review3_wval \
  --output-dir /path/to/pretrained/wine_review3_wval/response_standard_scaler
```

## Running the proposed method

Each workflow configuration listed below defines an ordered sequence of
stages. List the stages before running a workflow:

```bash
python scripts/run_experiment.py configs/wine_full.json --list
```

The available stage names are:

| Configuration | Stages, in execution order |
|---|---|
| `wine_full.json` | `alignment_train`, `alignment_transform`, `diffusion_train`, `sample_validation`, `evaluate_validation`, `sample_test`, `evaluate_test` |
| `petfinder_full.json` | `alignment_train`, `alignment_transform`, `diffusion_train`, `sample_validation`, `evaluate_validation`, `sample_test`, `evaluate_test` |
| `mimic_full.json` | `alignment_train`, `alignment_transform`, `diffusion_train`, `sample_validation`, `sample_test`, `evaluate_validation_threshold` |
| `shopee_image_only.json` | `diffusion_train`, `sample_validation`, `evaluate_validation`, `sample_test`, `evaluate_test` |
| `wine_prediction_interval.json` | `prepare_interval_data`, `train_interval_model`, `sample_calibration_and_test`, `evaluate_interval` |

The main stages have the following roles:

- `alignment_train` fits the alignment module on the training split and selects
  its checkpoint using validation loss.
- `alignment_transform` uses the selected checkpoint to create aligned
  conditioning embeddings for the train, validation, and test splits.
- `diffusion_train` fits the conditional diffusion model on the training split.
- `sample_validation` and `evaluate_validation` generate and evaluate
  validation predictions.
- `sample_test` and `evaluate_test` generate and evaluate final test
  predictions.
- `evaluate_validation_threshold` selects the MIMIC-IV classification threshold
  by validation F1 and applies that same threshold to the test predictions.
- `prepare_interval_data` makes the independent calibration split and fits
  training-only preprocessing; `train_interval_model` retrains the VAE and
  the selected variant and selects its checkpoint on validation.
- `sample_calibration_and_test` samples the frozen model on calibration and
  test with separate seeds. `evaluate_interval` applies corrected residual
  ranks and scores the empirical residual predictive distribution.
- `mlp_train_evaluate` trains and evaluates the no-diffusion MLP ablation.

### Ablation variants

The Petfinder, Wine Reviews, and MIMIC-IV full configurations also provide all
four ablations through `--variant`:

| Variant | Change from the full method |
|---|---|
| `tabular-only` | Conditions diffusion on raw tabular VAE embeddings; disables text and alignment |
| `no-diffusion` | Trains alignment, then fits the paper's two-hidden-layer MLP instead of diffusion |
| `no-response-awareness` | Sets the supervised alignment-loss weight to zero and the alignment-loss weight to one |
| `no-alignment` | Conditions diffusion directly on raw tabular VAE and CLIP text embeddings |

The no-diffusion MLP reserves 15% of training for early stopping. For MIMIC-IV,
the MLP and diffusion workflows select the binary probability threshold on
validation and apply it unchanged to test.

For example, run the Wine Reviews tabular-only workflow with:

```bash
python scripts/run_experiment.py configs/wine_full.json \
  --variant tabular-only \
  --set data_root=/path/to/data \
  --set embedding_root=/path/to/embeddings \
  --set pretrained_root=/path/to/pretrained \
  --set artifacts_root=/path/to/output
```

Replace `wine_full.json` with `petfinder_full.json` or `mimic_full.json` to run
the same ablation on another tabular--text dataset. Replace `tabular-only` with
any other variant in the table. Each variant writes to its own output directory,
such as `wine_tabular_only`, `wine_no_diffusion`,
`wine_no_response_awareness`, or `wine_no_alignment`.

List the stages for a particular variant before running it:

```bash
python scripts/run_experiment.py configs/mimic_full.json \
  --variant no-diffusion --list
```

Use `--dry-run` to inspect the fully expanded commands and paths without
starting any training:

```bash
python scripts/run_experiment.py configs/wine_full.json --dry-run \
  --set data_root=/path/to/data \
  --set embedding_root=/path/to/embeddings \
  --set pretrained_root=/path/to/pretrained \
  --set artifacts_root=/path/to/output
```

Remove `--dry-run` to run every stage in the displayed order. To run only part
of a workflow, repeat `--only` with the desired stage names:

```bash
python scripts/run_experiment.py configs/wine_full.json \
  --set data_root=/path/to/data \
  --set embedding_root=/path/to/embeddings \
  --set pretrained_root=/path/to/pretrained \
  --set artifacts_root=/path/to/output \
  --only alignment_train \
  --only alignment_transform
```

`--only` does not infer or run prerequisites. For example, running
`sample_validation` alone requires the corresponding trained diffusion model
to exist already. Selected stages always run in their order in the
configuration file, not in the order of the `--only` arguments.

Across the provided workflows, fitting stages use `train`, selection decisions
use `val`, and `test` is evaluated only after those decisions are fixed. This
is why validation and test sampling are separate stages. In the MIMIC-IV
workflow, `evaluate_validation_threshold` reads both sets of predictions,
chooses the threshold using validation F1, and reports test metrics without
changing the threshold.

The categorical diffusion sampler decodes generated response embeddings using
the nearest training-class response prototype.

## Prediction intervals: Wine Reviews

The PI workflow retrains with separate training, validation, and calibration
sets, preserving the original test set. It uses M=200 and alpha=0.05.

Start from the original 61,813-row `wine_review3` training pool, not
`wine_review3_wval/train`. If needed, prepare it from the downloaded CSV:

```bash
python data/preprocessing/prepare_wine_reviews.py \
  --raw-csv /path/to/raw/wine_reviews/winemag-data-130k-v2.csv \
  --output-dir /path/to/data/wine_review3 --pool-only
```

Reuse the original PI CLIP embeddings, or extract them with `--keep-padding`
in a separate directory from the point-prediction embeddings:

```bash
for split in train test
do
  python scripts/extract_clip_text_embeddings.py \
    --input-json /path/to/data/wine_review3/text_${split}.json \
    --output-dir /path/to/embeddings/wine_review3_text_clip_embd/${split} \
    --keep-padding
done
```

Run full and tabular-only sequentially, sharing the split and newly trained
VAE. Each command includes training, checkpoint selection, sampling, and
evaluation; completed shared stages are skipped:

```bash
for variant in full tabular-only
do
  python scripts/run_experiment.py configs/wine_prediction_interval.json \
    --variant "$variant" \
    --set source_dir=/path/to/data/wine_review3 \
    --set text_dir=/path/to/embeddings/wine_review3_text_clip_embd \
    --set run_root=/path/to/output/wine_prediction_interval \
    --set gpu=0
done
```

To run one variant, use only `full` or `tabular-only` in the loop. Test files
default to those in `source_dir`; use `--set test_source_dir=/path/to/existing/test_dataset`
if stored separately. Add `--dry-run` to inspect commands or
`--only evaluate_interval` to evaluate completed sample banks.

Results are written to `run_root/results/summary.json` and
`run_root/results/table3_new_rows.tex`. See the
[experiment notes](docs/wine_prediction_intervals.md) for the method,
configuration, recorded results, and additional evaluation commands.

## Baselines

### Classical tabular baselines

Classical tabular baselines use the same processed arrays and validation split:

```bash
python scripts/run_tabular_baselines.py \
  --data-dir /path/to/data/wine_review3_wval \
  --task regression \
  --models linear random_forest xgboost mdn quantile split_conformal \
  --output /path/to/output/wine_baselines.json
```

The split-conformal implementation reserves 15% of the training data for
calibration.

### External baselines

Third-party baseline code is not vendored.

| Component | Upstream repository |
|---|---|
| LANISTR | https://github.com/google-research/lanistr |
| LimiX | https://github.com/limix-ldm-ai/LimiX |
| CoDSA | https://github.com/shakayoyo/CoDSA |

AutoGluon is installed with this project because it is used directly by
`data/preprocessing/prepare_shopee.py`.

### CoDSA result selection

Use the [original CoDSA repository](https://github.com/shakayoyo/CoDSA) to
generate synthetic data and fit candidate classifiers. Its classification
implementation is provided as experiment notebooks under
`Simulation/classification/`.

The optional `scripts/select_codsa_candidate.py` begins after those candidate
classifiers have been fitted. For each CoDSA `(r, alpha, m)` and classifier
combination, save its positive-class probabilities on the unchanged validation
and test predictors. For a scikit-learn classifier, for example:

```python
import numpy as np

np.save("validation_proba.npy", classifier.predict_proba(X_val)[:, 1])
np.save("test_proba.npy", classifier.predict_proba(X_test)[:, 1])
```

Copy `configs/mimic_codsa_candidates.json`, add one entry per candidate, and
replace `validation_proba` and `test_proba` with the saved file paths. Each
probability file must contain one row per corresponding label in `y_val.npy` or
`y_test.npy`. Relative paths are resolved from the manifest's directory. This
manifest is input to the CoDSA selector; it is not a `run_experiment.py`
workflow.

Run the selector with:

```bash
python scripts/select_codsa_candidate.py \
  --manifest /path/to/mimic_codsa_candidates.json \
  --data-dir /path/to/data/mimic_adm_pt_disch_discharge_censored_wval \
  --output /path/to/output/codsa_selection.json
```

The helper selects both the candidate and classification threshold using
validation F1. Only after selection does it load the selected candidate's test
probabilities and calculate the test metrics. It does not generate CoDSA data
or train classifiers.

## License and attribution

Original code in this repository is released under the MIT License.
