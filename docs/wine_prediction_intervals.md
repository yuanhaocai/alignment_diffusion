# Revised Wine prediction intervals: protocol and recorded experiment

The complete commands for full and tabular-only runs are in the top-level
[README](../README.md#prediction-intervals-wine-reviews). The checked-in
[reference JSON](../results/wine_pi_20260921/reference.json) contains the
measured September 2026 server results, not results from running a new
experiment during this release update. No observation-level data, model
weights, or generated sample banks are included.

## Construction

Fit and select the prediction procedure using training and a separate
validation split. Fix M=200 and the sampling algorithm before calibration.
For each calibration observation form

\[
\widehat\mu_{M,i}=\frac{1}{M}\sum_{r=1}^M\widetilde Y_i^{(r)},
\qquad R_i=Y_i-\widehat\mu_{M,i}.
\]

With n calibration residuals sorted in increasing order, use one-based ranks

\[
k_L=\lfloor(n+1)\alpha/2\rfloor,\qquad
k_U=\lceil(n+1)(1-\alpha/2)\rceil.
\]

The interval is `[test_mean + R_(k_L), test_mean + R_(k_U)]`. Sentinels
`R_(0)=-infinity` and `R_(n+1)=+infinity` are retained rather than clipping
ranks. For n=7,726 and alpha=.05 the ranks are 193 and 7,534. Each test row
gets a different MC center and the same signed residual offsets.

The empirical predictive distribution for CRPS is the set of the test MC
mean plus **all calibration residuals**. Changing only the interval's two
endpoint ranks does not change CRPS. This is not raw diffusion-draw CRPS,
absolute-residual calibration, or a bootstrap-refitting procedure.

Under the marginal proposition's independence and identical-distribution
assumptions, the rank argument gives coverage at least
`(7534 - 193) / 7727 = 0.9500452957`, averaging over calibration, the future
observation, and their simulation randomness. This does not guarantee 95%
coverage in every realized test set or conditional on every covariate value.
The result permits ties. It does not by itself guarantee narrower intervals
or better CRPS.

## What changed from the historical experiment

The earlier pipeline used interpolated empirical quantiles and the old
validation residuals. Its copied VAE had seen the calibration covariates;
full alignment also used that calibration set for checkpoint selection.
Its category-constrained split was not an unconditional random partition.
An endpoint-only recalculation could not resolve these dependencies.

The new pipeline starts with the original 61,813 non-test rows, preserves
their ordering, and makes exactly one permutation with NumPy
`default_rng(20260921)`: first 7,726 calibration, next 7,726 validation,
remaining 46,361 training. It uses no stratification, category constraints,
or choice among split seeds. The 15,454-row original test is kept unchanged.

Training alone fits missing-value imputation, the quantile-normal numeric
transform, category vocabulary with a reserved unknown level, response
scaler, VAE, alignment, and diffusion. Validation is used for the VAE learning
rate scheduler and alignment/diffusion checkpoint selection. There is no
final train+validation refit. Both variants share the split and newly trained
VAE. Calibration and test outcomes are not loaded by the sampling dataset.

The old response SD, **2.9503095034391293**, is used only to report comparable
Width, CRPS, Winkler, and center RMSE. Internal response scaling is newly
fitted on training. Raw Width/CRPS are also saved.

## Recorded fitting settings

Machine-readable settings and seeds are in
[`configs/wine_pi_protocol.json`](../configs/wine_pi_protocol.json).

| Stage | Epochs | Batch size | Learning rate | Other settings | Seed |
|---|---:|---:|---:|---|---:|
| Tabular VAE | 3000 | 4096 | 0.001749103698339596 | latent_dim=768, d_token=8 | 2026092101 |
| Alignment | 100 | 64 | .001 | residual scale=.1; contrastive/supervised weights=1/1 | 2026092102 |
| Full diffusion | 1000 | 2048 | .0002 | dim_t=1024 | 2026092103 |
| Tabular-only diffusion | 1000 | 512 | .0001 | dim_t=1536 | 2026092104 |

The VAE keeps its minimum-training-reconstruction checkpoint, then re-encodes
all splits from that checkpoint. Alignment chooses minimum validation loss
(recorded best: epoch 99, zero-based). Diffusion candidates are the
minimum-training-loss `model.pt` plus epochs 250/500/750/1000; selection
minimizes validation RMSE of M=50 MC means with common seeds across candidates.
Full selected epoch 1000 (raw validation RMSE=1.5405187242); tabular-only
selected `model.pt` from epoch 922 (RMSE=2.2811054900). Both trained all 1000
epochs. Final calibration and test use M=200 and 100 sampling steps.

| Variant | Validation base seed | Calibration base seed | Test base seed |
|---|---:|---:|---:|
| Full | 210001000 | 210002000 | 210003000 |
| Tabular-only | 220001000 | 220002000 | 220003000 |

The recorded run finished at 2026-09-21 14:27:38 UTC-05:00 (2026-09-22
03:27:38 Beijing), taking about 3 h 29 min with concurrent GPU stages.
The portable runner uses one chosen GPU sequentially; its elapsed time can
therefore differ. A run's `status/master.json` identifies the last completed
phase; the final results require the evaluate phase, not only prepare/train.

## PI-specific model implementation

The server experiment's fitting code differs from this release's main
point-prediction profile. To reproduce the measured PI experiment faithfully,
`scripts/wine_pi/alignment/` preserves its alignment implementation rather
than silently substituting the generic `src/alignment/` model:

- Its supervised response head takes the **tabular projection**. The separate
  text response head exists but its loss weight is zero. Text still participates
  in the contrastive alignment loss. This is not the normalized joint head used
  by the generic point-prediction configuration.
- The frozen CLIP encoder is `openai/clip-vit-large-patch14`, retaining all 77
  token states. The PI path averages all projected token states and disables
  tabular/text/projected condition normalization. Do not substitute masked
  padding or the point-prediction normalization defaults.
- The dedicated dataset adapter caches full tokens for alignment and the exact
  projected token mean for diffusion. Token-cosine computation is rewritten as
  equivalent matrix multiplication; value/gradient and model-output checks are
  included. No text encoder is trained in this experiment.
- The VAE adapter passes training at index 0 and validation at index 2 of the
  release interface, leaving its test slot empty. It never supplies the real
  test split to VAE fitting or the scheduler.

The training/evaluation loop, model, and saving helpers were adapted from the
source snapshot saved with the completed server run. Imports are namespaced
under the PI workflow to avoid affecting ordinary experiments. The existing
release VAE and continuous-response diffusion implementations are reused with
explicit settings. The server and release versions were compared before this
port; categorical decoding changes are irrelevant to continuous Wine responses.

## Recorded results

| Model | Procedure | Coverage | Width | CRPS |
|---|---|---:|---:|---:|
| Full | Historical fit, interpolated endpoints | .952310082 | 2.020698223 | .267194671 |
| Full | Historical fit, corrected endpoints only | .952698331 | 2.024751841 | .267194671 |
| Full | New fit, interpolated endpoints | .953539537 | 2.165088174 | .283751467 |
| Full | New fit, corrected endpoints | **.953798369** | **2.167872142** | **.283751467** |
| Tabular-only | Historical fit, interpolated endpoints | .948039343 | 2.962012294 | .426035457 |
| Tabular-only | Historical fit, corrected endpoints only | .948104051 | 2.964298550 | .426035457 |
| Tabular-only | New fit, interpolated endpoints | .948362883 | 2.950699834 | .425724580 |
| Tabular-only | New fit, corrected endpoints | **.948492300** | **2.952668409** | **.425724580** |

Compared with the historical paper row, full Width increases 7.28% and CRPS
6.20%; tabular-only changes by -0.315% and -0.073%. Correcting endpoints on
the new full model increases Width only 0.129%, so most of the difference is
associated with the refitting/splitting pipeline. Full center RMSE also rises
from .486203 to .515715. Training counts, preprocessing, latent models,
initialization, model selection, and calibration all changed together; this
single rerun cannot identify their separate causal contributions.

Full still has 26.58% lower Width and 33.35% lower CRPS than the newly fitted
tabular-only model. Other baseline methods were **not** retrained in this
four-split experiment. Tabular-only is not the no-alignment ablation that
retains text.

## Provenance and limitations

The reference JSON records all four test hashes, selected-model hashes,
sampling-bank hashes, selection scores, and the fixed protocol. The workflow
saves its own source snapshots, input-path record, split indices, hashes,
training logs, and per-observation outputs under the user-chosen run directory.
New evaluation does not require access to historical private sample banks.

Historical model development reused these data and the original test set.
The new pipeline isolates calibration within this fitting run; it cannot
erase historical selection dependencies. The stronger conditional-coverage
theory additionally assumes common continuous additive noise independent of
covariates and suitable generation accuracy. Wine ratings are discretized;
aggregate coverage near 95% does not establish those assumptions. Pointwise
uncertainty also does not cover repeated training/split/calibration seeds.

The PI models and the old point-prediction table use different fitting splits
and model profiles. Manuscript descriptions should distinguish them rather
than claiming that both tables use identical fitted models.

## Release verification

The port passed 18 unit tests, including corrected ranks, infinite small-sample
endpoints, exact CRPS with ties, training-only preprocessing, disjoint splits,
and test immutability. Both variants' recorded metrics were independently
recomputed from the archived per-observation means/residuals to absolute
tolerance `1e-10`.

A separate CUDA smoke test used synthetic data, two VAE/diffusion epochs,
one alignment epoch, and two draws. It exercised both models, validation
selection, feature transformation, calibration/test generation, and interval
evaluation. Held-out response files were temporarily unavailable during
sampling, verifying that generation does not read them. This is an interface
check, not a repeat of the full scientific experiment.

```bash
PYTHONPATH=src:scripts python -m unittest discover -s tests -v
python scripts/validate_release.py

# Optional, on a CUDA machine, with a new empty output path:
PYTHONPATH=src:scripts python tests/smoke_wine_pi_gpu.py \
  --output-dir /path/to/fresh/synthetic_pi_smoke
```

## Additional evaluation commands

Add `--dry-run` to the JSON workflow command in the README to inspect every command
without creating files or training. `--list` lists its four stages.
`--only evaluate_interval` runs only evaluation after sampling has completed;
it does not run prerequisites. Repeating a complete command skips completed
stages, while changed inputs, protocol, or source code require a new run
directory. Training uses CUDA; data preparation and interval evaluation use
the CPU. The portable commands run on one GPU; the recorded server experiment
used multiple GPUs for independent stages.

For direct evaluation of existing independent-calibration sample banks:

```bash
python scripts/observed_residual_pi.py \
  --data-dir /path/to/output/wine_prediction_interval/data \
  --cal-samples /path/to/output/wine_prediction_interval/full/sampling_cal_m200/y_test_all.npy \
  --test-samples /path/to/output/wine_prediction_interval/full/sampling_test_m200/y_test_all.npy \
  --m 200 --alpha 0.05 --metric-scale 2.9503095034391293 \
  --output /path/to/output/wine_prediction_interval/results/full_direct.json
```

Replace **both** `full` path components with `tabular_only` to evaluate that
variant. Both banks have shape `(200, n_split)` and contain responses in the
original rating scale. Despite its legacy filename, the bank under
`sampling_cal_m200/` contains calibration samples, not test samples.
`--endpoint-rule linear` performs an endpoint-only sensitivity comparison.
Small calibration sets may yield unbounded corrected intervals; JSON encodes
these bounds/widths as `"-Infinity"` / `"Infinity"`.
