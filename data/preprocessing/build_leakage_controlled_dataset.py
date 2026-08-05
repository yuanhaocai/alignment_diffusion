#!/usr/bin/env python
"""Build the leakage-controlled MIMIC-IV admission/text dataset.

The default workflow requires MIMIC-IV v2.2 ``hosp/admissions.csv`` and
``hosp/patients.csv`` plus MIMIC-IV-Note v2.2 ``note/discharge.csv``. It merges
the tables by admission, removes rows with missing model features, creates a
seeded 80/20 training-pool/test split, and reserves one eighth of the training
pool for validation.

The default ``discharge_admission_sections`` mode keeps admission-oriented
sections from discharge summaries, removes outcome-oriented sections and
explicit mortality/disposition cue lines, and applies a 77-token budget. The
optional ``timestamped_notes`` mode accepts timestamped non-discharge notes
and keeps notes available within the configured prediction window.

Outputs include train/validation/test NumPy arrays, aligned JSON text lists,
the split indices, and ``leakage_control_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CAT_COLS = [
    "race",
    "marital_status",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "gender",
]
NUM_COLS = ["anchor_age"]
Y_COLS = ["hospital_expire_flag"]
COHORT_FEATURE_COLS = [
    "race",
    "marital_status",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "hospital_expire_flag",
    "gender",
    "anchor_age",
]
TIME_COLS = ["admittime", "dischtime", "deathtime", "edregtime"]
NOTE_TIME_CANDIDATES = ["charttime", "chartdate", "storetime"]
NOTE_TYPE_CANDIDATES = ["note_type", "category", "description"]
POST_EVENT_NOTE_PATTERNS = [
    r"\bdischarge\b",
    r"\bdischarge\s+summary\b",
    r"^ds$",
]
DISCHARGE_ADMISSION_SECTION_PATTERNS = [
    r"chief complaint",
    r"history of present illness",
    r"past medical history",
    r"past surgical history",
    r"social history",
    r"family history",
    r"physical exam",
    r"physical examination",
    r"medications on admission",
    r"admission medications",
    r"preadmission medication",
    r"allergies",
]
DISCHARGE_POST_SECTION_PATTERNS = [
    r"discharge",
    r"brief hospital course",
    r"hospital course",
    r"pertinent results",
    r"laboratory data",
    r"procedures",
    r"operations",
    r"condition",
    r"diagnosis",
    r"disposition",
    r"followup",
    r"follow-up",
    r"instructions",
]
DISCHARGE_SECTION_PRIORITY_PATTERNS = [
    r"chief complaint",
    r"history of present illness",
    r"medications on admission",
    r"admission medications",
    r"preadmission medication",
    r"past medical history",
    r"past surgical history",
    r"physical exam",
    r"physical examination",
    r"allergies",
    r"social history",
    r"family history",
]
OUTCOME_CUE_PATTERN = re.compile(
    r"\b(expired|deceased|died|death|autopsy|comfort measures|cm[o]?|hospice|palliative)\b",
    flags=re.IGNORECASE,
)
TEXT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'/][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def clean_note_text(text: object) -> str:
    """Generic text cleanup that does not search for outcome-related words."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"(_{2,}|-{2,}|={2,})", " ", text)
    text = re.sub(r"(?im)^(name|unit no|date of birth|sex|attending):.*$", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def text_tokens(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return TEXT_TOKEN_PATTERN.findall(str(text))


def detokenize_text(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+'\s*", "'", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def clip_text_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text.strip()
    tokens = text_tokens(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return detokenize_text(tokens[:max_tokens])


def token_count(text: object) -> int:
    return len(text_tokens(text))


def normalise_key_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def make_feature_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(normalise_key_value(row[col]) for col in COHORT_FEATURE_COLS)


def read_admission_patient_table(mimic_dir: Path) -> pd.DataFrame:
    admissions_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "race",
        "marital_status",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "hospital_expire_flag",
        "edregtime",
    ]
    patient_cols = ["subject_id", "gender", "anchor_age"]

    admissions = pd.read_csv(mimic_dir / "admissions.csv", usecols=admissions_cols)
    patients = pd.read_csv(mimic_dir / "patients.csv", usecols=patient_cols)
    cohort = pd.merge(admissions, patients, on="subject_id", how="left", sort=False)
    for col in TIME_COLS:
        cohort[col] = pd.to_datetime(cohort[col], errors="coerce")
    return cohort


def recover_reference_cohort(mimic_dir: Path, cohort_template_csv: Path) -> pd.DataFrame:
    """Recover subject and admission IDs for a processed cohort template.

    The template contains model features without identifiers. A greedy
    subsequence match against the admissions/patients table recovers the
    identifiers without using discharge text.
    """
    admission_patient = read_admission_patient_table(mimic_dir)
    admission_patient = admission_patient.dropna(subset=COHORT_FEATURE_COLS).reset_index(drop=False)
    template = pd.read_csv(cohort_template_csv, usecols=COHORT_FEATURE_COLS, low_memory=False)

    source_keys = [make_feature_key(row) for _, row in admission_patient[COHORT_FEATURE_COLS].iterrows()]
    template_keys = [make_feature_key(row) for _, row in template.iterrows()]

    matched_positions: list[int] = []
    source_pos = 0
    for template_pos, key in enumerate(template_keys):
        while source_pos < len(source_keys) and source_keys[source_pos] != key:
            source_pos += 1
        if source_pos == len(source_keys):
            raise RuntimeError(
                "Could not recover cohort-template identifiers. "
                f"First unmatched template row: {template_pos}."
            )
        matched_positions.append(source_pos)
        source_pos += 1

    recovered = admission_patient.iloc[matched_positions].copy()
    recovered = recovered.drop(columns=["index"]).reset_index(drop=True)
    recovered["cohort_row_id"] = np.arange(len(recovered))
    return recovered


def read_raw_discharge_cohort(mimic_dir: Path, discharge_csv: Path) -> pd.DataFrame:
    discharge_cols = [
        "note_id",
        "subject_id",
        "hadm_id",
        "note_type",
        "note_seq",
        "charttime",
        "storetime",
        "text",
    ]
    discharge_notes = pd.read_csv(discharge_csv, usecols=discharge_cols, low_memory=False)
    discharge_notes = discharge_notes.rename(columns={"text": "raw_note_text"})
    for col in ["charttime", "storetime"]:
        discharge_notes[col] = pd.to_datetime(discharge_notes[col], errors="coerce")

    cohort = read_admission_patient_table(mimic_dir)
    cohort = pd.merge(cohort, discharge_notes, on=["subject_id", "hadm_id"], how="inner", sort=False)
    cohort = cohort.dropna(subset=COHORT_FEATURE_COLS + ["raw_note_text"]).reset_index(drop=True)
    cohort["cohort_row_id"] = np.arange(len(cohort))
    return cohort


def is_post_event_note_table(path: Path) -> bool:
    name = path.name.lower()
    return any(re.search(pattern, name) for pattern in POST_EVENT_NOTE_PATTERNS)


def choose_note_columns(path: Path) -> tuple[list[str], str, list[str]]:
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    if "subject_id" not in columns or "text" not in columns:
        raise ValueError(f"{path} must contain at least subject_id and text columns.")

    time_col = next((col for col in NOTE_TIME_CANDIDATES if col in columns), None)
    if time_col is None:
        raise ValueError(
            f"{path} has no timestamp column. Expected one of {NOTE_TIME_CANDIDATES}."
        )

    type_cols = [col for col in NOTE_TYPE_CANDIDATES if col in columns]
    usecols = ["subject_id", "text", time_col] + type_cols
    if "hadm_id" in columns:
        usecols.append("hadm_id")
    return usecols, time_col, type_cols


def note_type_mask(notes: pd.DataFrame, type_cols: list[str]) -> pd.Series:
    keep = pd.Series(True, index=notes.index)
    for col in type_cols:
        values = notes[col].fillna("").astype(str).str.lower()
        for pattern in POST_EVENT_NOTE_PATTERNS:
            keep &= ~values.str.contains(pattern, regex=True)
    return keep


def event_cutoff(cohort: pd.DataFrame, prediction_window_hours: float) -> pd.Series:
    prediction_cutoff = cohort["admittime"] + pd.to_timedelta(prediction_window_hours, unit="h")
    cutoffs = pd.concat(
        [
            prediction_cutoff.rename("prediction_cutoff"),
            cohort["dischtime"].rename("dischtime"),
            cohort["deathtime"].rename("deathtime"),
        ],
        axis=1,
    )
    return cutoffs.min(axis=1, skipna=True)


def lower_note_bound(cohort: pd.DataFrame, include_ed_notes: bool) -> pd.Series:
    if include_ed_notes:
        return cohort["edregtime"].fillna(cohort["admittime"])
    return cohort["admittime"]


def combine_notes(values: pd.Series, max_chars: int, max_tokens: int) -> str:
    cleaned = [clean_note_text(value) for value in values]
    cleaned = [value for value in cleaned if value]
    combined = "\n\n".join(cleaned)
    if max_chars > 0 and len(combined) > max_chars:
        combined = combined[:max_chars].rsplit(" ", 1)[0]
    return clip_text_to_token_budget(combined, max_tokens)


def normalise_heading(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip()


def heading_matches(heading: str, patterns: list[str]) -> bool:
    norm = normalise_heading(heading)
    return any(re.search(pattern, norm) for pattern in patterns)


def section_priority(heading: str) -> int:
    norm = normalise_heading(heading)
    for index, pattern in enumerate(DISCHARGE_SECTION_PRIORITY_PATTERNS):
        if re.search(pattern, norm):
            return index
    return len(DISCHARGE_SECTION_PRIORITY_PATTERNS)


def strip_after_discharge_marker(text: str) -> str:
    match = re.search(r"(?im)^\s*discharge\s*:\s*$", text)
    if match:
        return text[: match.start()]
    return text


def redact_outcome_cue_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if OUTCOME_CUE_PATTERN.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_discharge_admission_sections(
    text: object,
    max_chars: int,
    max_tokens: int,
    redact_outcome_cues: bool,
) -> str:
    """Keep only admission-oriented sections from a discharge summary.

    This is a section-censoring fallback, not true timestamp-based truncation.
    It drops discharge and hospital-course portions likely to encode the
    outcome.
    """
    if pd.isna(text):
        return ""
    source = strip_after_discharge_marker(str(text))
    heading_re = re.compile(r"(?im)^\s*([A-Za-z][A-Za-z0-9 /#'(),.\-]{1,80})\s*:\s*$")
    matches = list(heading_re.finditer(source))

    sections: list[tuple[int, int, str]] = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        if heading_matches(heading, DISCHARGE_POST_SECTION_PATTERNS):
            continue
        if not heading_matches(heading, DISCHARGE_ADMISSION_SECTION_PATTERNS):
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[start:end].strip()
        if body:
            sections.append((section_priority(heading), i, f"{heading}:\n{body}"))

    if sections:
        extracted = "\n\n".join(section for _, _, section in sorted(sections))
    else:
        extracted = source

    if redact_outcome_cues:
        extracted = redact_outcome_cue_lines(extracted)
    extracted = clean_note_text(extracted)
    if max_chars > 0 and len(extracted) > max_chars:
        extracted = extracted[:max_chars].rsplit(" ", 1)[0]
    return clip_text_to_token_budget(extracted, max_tokens)


def construct_text_from_discharge_sections(
    cohort: pd.DataFrame,
    max_chars_per_admission: int,
    max_text_tokens: int,
    redact_outcome_cues: bool,
) -> tuple[pd.Series, dict[str, object]]:
    if "raw_note_text" not in cohort.columns:
        raise ValueError("Discharge section-censoring mode requires raw_note_text in the cohort.")

    text_by_row = cohort["raw_note_text"].apply(
        lambda value: extract_discharge_admission_sections(
            value,
            max_chars=max_chars_per_admission,
            max_tokens=max_text_tokens,
            redact_outcome_cues=redact_outcome_cues,
        )
    )
    audit: dict[str, object] = {
        "text_source_mode": "discharge_admission_sections",
        "timestamp_truncation_available": False,
        "timestamp_limitation": (
            "MIMIC-IV-Note discharge summaries have note-level charttime/storetime, "
            "but the clinical statements inside the summary are not individually timestamped. "
            "This mode therefore performs section censoring rather than strict prediction-time truncation."
        ),
        "note_type_policy": "DS discharge summaries only; no discharge disposition/diagnosis/condition/instructions or hospital-course sections retained.",
        "kept_section_patterns": DISCHARGE_ADMISSION_SECTION_PATTERNS,
        "dropped_section_patterns": DISCHARGE_POST_SECTION_PATTERNS,
        "redact_outcome_cue_lines": redact_outcome_cues,
        "max_chars_per_admission": max_chars_per_admission,
        "max_text_tokens": max_text_tokens,
        "tokenizer": "regex_word_punctuation",
        "tokenizer_note": (
            "This preprocessing step enforces the token budget with a "
            "deterministic regex word/punctuation tokenizer."
        ),
        "raw_note_rows": int(len(cohort)),
        "admissions_with_eligible_text": int((text_by_row.fillna("").str.len() > 0).sum()),
        "eligible_note_rows": int(len(cohort)),
        "max_observed_text_tokens": int(text_by_row.fillna("").map(token_count).max()) if len(text_by_row) else 0,
    }
    if "note_type" in cohort.columns:
        audit["note_type_counts"] = {
            str(key): int(value) for key, value in cohort["note_type"].value_counts(dropna=False).items()
        }
    if "charttime" in cohort.columns:
        audit["note_charttime_min"] = str(cohort["charttime"].min())
        audit["note_charttime_max"] = str(cohort["charttime"].max())
    if "storetime" in cohort.columns:
        audit["note_storetime_min"] = str(cohort["storetime"].min())
        audit["note_storetime_max"] = str(cohort["storetime"].max())
    return text_by_row, audit


def construct_text(
    cohort: pd.DataFrame,
    note_paths: list[Path],
    prediction_window_hours: float,
    include_ed_notes: bool,
    max_chars_per_admission: int,
    max_text_tokens: int,
    allow_discharge_notes: bool,
) -> tuple[pd.Series, dict[str, object]]:
    text_by_row = pd.Series("", index=cohort.index, dtype=object)
    audit: dict[str, object] = {
        "prediction_window_hours": prediction_window_hours,
        "include_ed_notes": include_ed_notes,
        "max_chars_per_admission": max_chars_per_admission,
        "max_text_tokens": max_text_tokens,
        "tokenizer": "regex_word_punctuation",
        "tokenizer_note": (
            "This preprocessing step enforces the token budget with a "
            "deterministic regex word/punctuation tokenizer."
        ),
        "note_tables": [],
    }

    merge_cols = ["cohort_row_id", "subject_id", "hadm_id", "admittime", "edregtime", "dischtime", "deathtime"]
    cohort_for_notes = cohort[merge_cols].copy()
    cohort_for_notes["note_lower_bound"] = lower_note_bound(cohort, include_ed_notes)
    cohort_for_notes["note_cutoff"] = event_cutoff(cohort, prediction_window_hours)

    all_eligible: list[pd.DataFrame] = []
    for path in note_paths:
        table_audit: dict[str, object] = {"path": str(path)}
        if is_post_event_note_table(path) and not allow_discharge_notes:
            table_audit["status"] = "skipped_post_event_table"
            audit["note_tables"].append(table_audit)
            continue

        usecols, time_col, type_cols = choose_note_columns(path)
        notes = pd.read_csv(path, usecols=usecols, low_memory=False)
        table_audit["raw_rows"] = int(len(notes))
        table_audit["timestamp_column"] = time_col
        table_audit["note_type_columns"] = type_cols

        notes[time_col] = pd.to_datetime(notes[time_col], errors="coerce")
        notes = notes[notes[time_col].notna()].copy()
        table_audit["rows_with_timestamp"] = int(len(notes))

        if type_cols and not allow_discharge_notes:
            notes = notes[note_type_mask(notes, type_cols)].copy()
        table_audit["rows_after_type_filter"] = int(len(notes))

        if "hadm_id" in notes.columns:
            merged = pd.merge(notes, cohort_for_notes, on=["subject_id", "hadm_id"], how="inner", sort=False)
            table_audit["join_keys"] = ["subject_id", "hadm_id"]
        else:
            merged = pd.merge(notes, cohort_for_notes, on=["subject_id"], how="inner", sort=False)
            table_audit["join_keys"] = ["subject_id"]

        eligible = merged[
            (merged[time_col] >= merged["note_lower_bound"])
            & (merged[time_col] <= merged["note_cutoff"])
        ].copy()
        eligible = eligible.rename(columns={time_col: "note_time"})
        table_audit["joined_rows"] = int(len(merged))
        table_audit["eligible_rows"] = int(len(eligible))
        audit["note_tables"].append(table_audit)
        if not eligible.empty:
            all_eligible.append(eligible[["cohort_row_id", "note_time", "text"]])

    if all_eligible:
        eligible_notes = pd.concat(all_eligible, ignore_index=True)
        eligible_notes = eligible_notes.sort_values(["cohort_row_id", "note_time"])
        grouped = eligible_notes.groupby("cohort_row_id", sort=False)["text"].apply(
            lambda values: combine_notes(values, max_chars_per_admission, max_text_tokens)
        )
        text_by_row.loc[grouped.index.astype(int)] = grouped.values
        audit["admissions_with_eligible_text"] = int((text_by_row.str.len() > 0).sum())
        audit["eligible_note_rows"] = int(len(eligible_notes))
        audit["max_observed_text_tokens"] = int(text_by_row.fillna("").map(token_count).max()) if len(text_by_row) else 0
    else:
        audit["admissions_with_eligible_text"] = 0
        audit["eligible_note_rows"] = 0
        audit["max_observed_text_tokens"] = 0
    return text_by_row, audit


def make_splits(data_size: int, train_prop: float, test_seed: int, val_random_state: int):
    train_size = floor(data_size * train_prop)
    np.random.seed(test_seed)
    original_train_idx = np.random.choice(data_size, size=train_size, replace=False)
    test_idx = np.setdiff1d(np.arange(data_size), original_train_idx)

    train_positions = np.arange(len(original_train_idx))
    val_size = len(original_train_idx) // 8
    train_pos, val_pos = train_test_split(
        train_positions,
        test_size=val_size,
        random_state=val_random_state,
    )

    return {
        "train": original_train_idx[train_pos],
        "val": original_train_idx[val_pos],
        "test": test_idx,
        "original_train_idx": original_train_idx,
        "train_pos": train_pos,
        "val_pos": val_pos,
    }


def save_json_list(path: Path, values: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(values, f, ensure_ascii=False)


def save_dataset(cohort: pd.DataFrame, text: pd.Series, split_indices: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_cat = cohort[CAT_COLS].to_numpy().astype(str)
    x_num = cohort[NUM_COLS].to_numpy().astype(np.float32)
    y_dat = cohort[Y_COLS].to_numpy().astype(str)
    text_values = text.fillna("").astype(str).tolist()

    for split in ["train", "val", "test"]:
        idx = split_indices[split]
        np.save(output_dir / f"X_num_{split}.npy", x_num[idx])
        np.save(output_dir / f"X_cat_{split}.npy", x_cat[idx])
        np.save(output_dir / f"y_{split}.npy", y_dat[idx])
        save_json_list(output_dir / f"text_{split}.json", [text_values[int(i)] for i in idx])

    np.save(output_dir / "original_train_idx.npy", split_indices["original_train_idx"])
    np.save(output_dir / "train_idx.npy", split_indices["train_pos"])
    np.save(output_dir / "val_idx.npy", split_indices["val_pos"])
    np.save(output_dir / "train_source_idx.npy", split_indices["train"])
    np.save(output_dir / "val_source_idx.npy", split_indices["val"])
    np.save(output_dir / "test_source_idx.npy", split_indices["test"])


def discover_note_paths(project_root: Path, mimic_dir: Path) -> list[Path]:
    candidates = [
        mimic_dir / "radiology.csv",
        mimic_dir / "radiology.csv.gz",
        project_root / "mimic-iv-note" / "radiology.csv",
        project_root / "mimic-iv-note" / "radiology.csv.gz",
        mimic_dir.parent / "mimic-iv-note" / "radiology.csv",
        mimic_dir.parent / "mimic-iv-note" / "radiology.csv.gz",
    ]
    return [path for path in candidates if path.exists()]


def default_new_data_name(dataname: str, prediction_window_hours: float, text_source_mode: str) -> str:
    if text_source_mode == "discharge_admission_sections":
        return f"{dataname}_discharge_censored_wval"
    if float(prediction_window_hours).is_integer():
        window = f"{int(prediction_window_hours)}h"
    else:
        window = f"{str(prediction_window_hours).replace('.', 'p')}h"
    return f"{dataname}_lc_{window}_wval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="mimic_adm_pt_disch")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--new-data-name", default=None)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument(
        "--discharge-csv",
        type=Path,
        default=None,
        help="Raw MIMIC-IV-Note discharge.csv used for section-censored text mode.",
    )
    parser.add_argument(
        "--text-source-mode",
        choices=["auto", "timestamped_notes", "discharge_admission_sections"],
        default="auto",
        help=(
            "auto uses timestamped non-discharge notes when provided/discovered, otherwise "
            "falls back to section-censored discharge summaries if --discharge-csv exists."
        ),
    )
    parser.add_argument(
        "--cohort-template-csv",
        type=Path,
        default=None,
        help=(
            "Optional processed feature CSV used to recover cohort identifiers "
            "for timestamped-notes mode."
        ),
    )
    parser.add_argument(
        "--cohort-discharge-csv",
        type=Path,
        default=None,
        help="Optional raw discharge.csv used only for subject_id/hadm_id cohort membership, never for text.",
    )
    parser.add_argument(
        "--note-csv",
        type=Path,
        action="append",
        default=None,
        help="Timestamped note CSV to use for text, for example MIMIC-IV-Note radiology.csv. Can be repeated.",
    )
    parser.add_argument(
        "--prediction-window-hours",
        type=float,
        default=24.0,
        help="Use notes from ED/admission through admittime plus this many hours, capped at discharge/death.",
    )
    parser.add_argument("--train-prop", type=float, default=0.8)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--val-random-state", type=int, default=0)
    parser.add_argument("--max-chars-per-admission", type=int, default=6000)
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=77,
        help=(
            "Maximum regex word/punctuation tokens saved per sample. Default 77 matches the CLIP "
            "text context length; use 75 if your downstream tokenizer adds start/end tokens."
        ),
    )
    parser.add_argument("--exclude-ed-notes", action="store_true")
    parser.add_argument(
        "--keep-outcome-cue-lines",
        action="store_true",
        help="Do not drop lines containing explicit mortality/disposition cue words in discharge section-censored mode.",
    )
    parser.add_argument("--allow-discharge-notes", action="store_true")
    parser.add_argument(
        "--allow-empty-text",
        action="store_true",
        help="Permit writing a tabular-only leak-free dataset when no eligible note source is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimic_dir = args.mimic_dir.resolve()
    data_root = args.data_root.resolve()
    cohort_template_csv = (
        args.cohort_template_csv.resolve()
        if args.cohort_template_csv is not None
        else mimic_dir / "adm_pt_disch3.csv"
    )
    discharge_csv = (
        args.discharge_csv.resolve()
        if args.discharge_csv is not None
        else mimic_dir.parent / "mimic-iv-note" / "discharge.csv"
    )
    project_root = mimic_dir.parent

    note_paths = [path.resolve() for path in args.note_csv] if args.note_csv else discover_note_paths(project_root, mimic_dir)
    if args.text_source_mode == "auto":
        if note_paths:
            text_source_mode = "timestamped_notes"
        elif discharge_csv.exists():
            text_source_mode = "discharge_admission_sections"
        else:
            text_source_mode = "timestamped_notes"
    else:
        text_source_mode = args.text_source_mode

    new_data_name = args.new_data_name or default_new_data_name(
        args.dataname,
        args.prediction_window_hours,
        text_source_mode,
    )
    output_dir = data_root / new_data_name

    if text_source_mode == "discharge_admission_sections":
        if not discharge_csv.exists():
            raise FileNotFoundError(f"Could not find discharge CSV: {discharge_csv}")
        cohort = read_raw_discharge_cohort(mimic_dir, discharge_csv)
        cohort_source = str(discharge_csv)
        text, audit = construct_text_from_discharge_sections(
            cohort=cohort,
            max_chars_per_admission=args.max_chars_per_admission,
            max_text_tokens=args.max_text_tokens,
            redact_outcome_cues=not args.keep_outcome_cue_lines,
        )
    else:
        if not note_paths and not args.allow_empty_text:
            raise FileNotFoundError(
                "No temporally valid note source was found. Provide --note-csv /path/to/radiology.csv, "
                "or use --text-source-mode discharge_admission_sections with --discharge-csv."
            )
        usable_note_paths = [
            path for path in note_paths if args.allow_discharge_notes or not is_post_event_note_table(path)
        ]
        if note_paths and not usable_note_paths and not args.allow_empty_text:
            raise ValueError(
                "All provided note sources look like post-event discharge-note tables. "
                "Use --text-source-mode discharge_admission_sections for the section-censored fallback, "
                "or provide a timestamped non-discharge note source such as radiology.csv."
            )

        if args.cohort_discharge_csv is not None:
            cohort = read_raw_discharge_cohort(mimic_dir, args.cohort_discharge_csv.resolve())
            cohort_source = str(args.cohort_discharge_csv.resolve())
        else:
            cohort = recover_reference_cohort(mimic_dir, cohort_template_csv)
            cohort_source = str(cohort_template_csv)

        text, audit = construct_text(
            cohort=cohort,
            note_paths=note_paths,
            prediction_window_hours=args.prediction_window_hours,
            include_ed_notes=not args.exclude_ed_notes,
            max_chars_per_admission=args.max_chars_per_admission,
            max_text_tokens=args.max_text_tokens,
            allow_discharge_notes=args.allow_discharge_notes,
        )

    if audit["admissions_with_eligible_text"] == 0 and not args.allow_empty_text:
        raise RuntimeError(
            "No eligible text remained after note-type and timestamp censoring. "
            "Check the note source and censoring settings, or use --allow-empty-text for a tabular-only run."
        )
    if args.max_text_tokens > 0:
        max_observed_tokens = int(text.fillna("").map(token_count).max()) if len(text) else 0
        if max_observed_tokens > args.max_text_tokens:
            raise AssertionError(
                f"Text token budget exceeded: observed {max_observed_tokens}, "
                f"limit {args.max_text_tokens}."
            )

    split_indices = make_splits(
        data_size=len(cohort),
        train_prop=args.train_prop,
        test_seed=args.test_seed,
        val_random_state=args.val_random_state,
    )
    print(f"Saving leakage-controlled dataset to {output_dir.name}...")
    print(output_dir)
    save_dataset(cohort, text, split_indices, output_dir)

    metadata = {
        "dataset": output_dir.name,
        "source_dataname": args.dataname,
        "new_data_name": output_dir.name,
        "data_root": str(data_root),
        "cohort_source": cohort_source,
        "text_source_mode": text_source_mode,
        "data_size": int(len(cohort)),
        "splits": {split: int(len(split_indices[split])) for split in ["train", "val", "test"]},
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "y_cols": Y_COLS,
        "test_seed": args.test_seed,
        "val_random_state": args.val_random_state,
        "max_text_tokens": args.max_text_tokens,
        "text_tokenizer": "regex_word_punctuation",
        "test_split_method": "seeded_80_20_then_training_pool_validation",
        "excluded_post_event_note_patterns": POST_EVENT_NOTE_PATTERNS,
        "allow_discharge_notes": args.allow_discharge_notes,
        "text_audit": audit,
    }
    with (output_dir / "leakage_control_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
