# mimic_adm_pt_disch_discharge_censored_wval

Constructed by
`data/preprocessing/build_leakage_controlled_dataset.py`.

Required files from [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/)
and [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/):

- MIMIC-IV `hosp/admissions.csv.gz`
- MIMIC-IV `hosp/patients.csv.gz`
- MIMIC-IV-Note `note/discharge.csv.gz`

- train: 224,155 rows
- validation: 32,022 rows
- test: 64,045 rows
- numerical predictor: `anchor_age`
- categorical predictors: `race`, `marital_status`, `admission_type`,
  `admission_location`, `insurance`, `language`, and `gender`
- response: binary `hospital_expire_flag`
- text: admission-oriented sections retained from discharge summaries, capped
  at 77 regex tokens

The builder excludes outcome-oriented sections and redacts lines containing
explicit mortality or disposition cues. This is section censoring, not strict
timestamp censoring; the limitation is recorded in each generated
`leakage_control_metadata.json`.
