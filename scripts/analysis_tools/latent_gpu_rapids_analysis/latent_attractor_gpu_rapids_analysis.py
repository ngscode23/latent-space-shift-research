#!/usr/bin/env python3
"""
Hidden Geometry Metric Lab for Grade 3 / Grade 4 result packages.

This is the primary formula-level analyzer for result directories or zip files
produced by the hidden-geometry Grade 3 and Grade 4 scripts. It is deliberately
not a verdict generator. It reads immutable source artifacts, builds
source-backed metric tables, summarizes CSV/NPZ outputs, writes visualizations,
and records anomalies/confounds as machine-readable rows.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:  # Optional. The pandas backend is the default fallback.
    import cudf  # type: ignore
except Exception:  # pragma: no cover - depends on local CUDA/RAPIDS runtime.
    cudf = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plots are optional.
    plt = None


# -----------------------------------------------------------------------------
# Constants / schema
# -----------------------------------------------------------------------------

CHUNK_ROWS = 250_000
SUMMARY_FILE_MAX_BYTES = 100_000_000
DEFAULT_MAX_PLOT_ROWS = 500_000

FORBIDDEN_RESULT_LABEL_VALUES = {
    "hidden_diagnostic_only",
    "causal_internal_axis_supported",
    "behavioral_axis_partial",
    "behavioral_axis_supported",
    "behavioral_control_axis_supported",
    "partial_behavioral_control_axis_supported",
    "internal_axis_supported_behavioral_control_not_supported",
    "hidden_axis_only_visible_readout_not_computed",
    "mechanistic_discovery_candidate",
    "breakthrough",
    "discovery",
    "causal_supported",
    "behavioral_supported",
    "model proved",
}

ARTIFACT_FAMILIES = [
    "run_validity",
    "prompt_geometry",
    "specificity_controls",
    "null_baselines",
    "generation_trajectory",
    "causal_interventions",
    "behavioral_control",
    "semantic_shift",
    "architecture_modules",
    "architecture_units",
    "dense_features",
    "sae_features",
    "grade4_components",
    "state_space_geometry",
    "npz_vectors",
    "plots_existing",
    "quarantine_integrity",
    "other",
]

ID_OR_INDEX_COLUMNS = {
    "artifact_type",
    "question_index",
    "step",
    "layer",
    "token_id",
    "unit_index",
    "rank_by_abs_delta",
    "rank_by_abs_component_delta",
    "rank_by_activity",
    "rank_by_abs_order_specific_score",
    "activation_size",
    "layer_count_intervened",
    "q_count",
    "random_index",
    "sae_spec_index",
    "feature_index",
    "feature_count",
    "topk",
    "is_embedding",
    "is_middle_layer",
    "is_duplicate",
    "decoder_layer_count",
    "expected_decoder_layer_count",
}

GROUP_COLUMNS = {
    "condition",
    "condition_a",
    "condition_b",
    "reference_condition",
    "control_condition",
    "experimental_condition",
    "base_condition",
    "intervention_name",
    "layer_band",
    "intervention_layer_band",
    "readout_layer_band",
    "band",
    "sign_name",
    "module",
    "unit_type",
    "axis_name",
    "axis",
    "component",
    "component_name",
    "sae_name",
    "interpretation_status",
    "metric",
    "metric_family",
    "metric_name",
    "criterion",
    "status",
    "failure_code",
    "question_domain",
    "source",
    "layer",
    "step",
    "alpha",
    "alpha_abs",
}

INDEPENDENT_COLUMNS = {"alpha", "alpha_abs"}

EXACT_METRIC_COLUMNS = {
    # Geometry.
    "projection_fraction_on_vector_x_loo",
    "direction_cosine_with_vector_x_loo",
    "projection_fraction_on_arch_vector_x_loo",
    "direction_cosine_with_arch_vector_x_loo",
    "cosine_distance_to_reference",
    "l2_distance_to_reference_prompt_endpoint",
    "l2_distance_to_reference",
    "state_norm",
    "delta_norm",
    "axis_norm",
    "condition_norm",
    "reference_norm",
    "target_reference_l2_same_question",
    "centroid_norm",
    "within_l2_mean",
    "within_l2_std",
    "within_cosine_distance_mean",
    "within_cosine_distance_std",
    "centroid_l2_distance",
    "centroid_cosine_distance",
    "centroid_cosine_similarity",
    "within_l2_mean_all_conditions",
    "between_centroid_l2_mean",
    "between_over_within_l2_ratio",
    "within_cosine_distance_mean_all_conditions",
    "between_centroid_cosine_distance_mean",
    "pc1_explained_variance",
    "pc2_explained_variance",
    "pc3_explained_variance",
    "cumulative_explained_variance",
    "pc1",
    "pc2",
    "pc3",
    "rank_by_l2",
    "rank_by_cosine",
    # Generation.
    "selected_logprob",
    "mean_selected_logprob",
    "entropy",
    "mean_entropy",
    # Deltas / activations.
    "reference_value",
    "condition_value",
    "delta",
    "abs_delta",
    "mean_delta",
    "mean_abs_delta",
    "max_abs_delta",
    "top_rank_mean",
    "target_control_top_unit_jaccard",
    "sign_agreement_on_intersection",
    # Reviewer/statistical summary metrics are still numeric evidence here.
    "score",
    "pass",
    "threshold",
    "metric_value",
    "target_minus_control_mean",
    "target_minus_control_ci95_low",
    "target_minus_control_ci95_high",
    "target_minus_experimental_mean",
    "target_minus_experimental_ci95_low",
    "target_minus_experimental_ci95_high",
    "paired_cohen_d",
    "paired_sign_permutation_p",
    "fdr_q_value",
    "fdr_significant",
    "observed_minus_null_mean",
    "empirical_p_greater_equal_observed",
    "symmetry_pass_rate",
    "plus_minus_projection_gap",
    "alpha_projection_slope",
    "win_rate_vs_random_p95",
    "mean_lift_over_random_p95",
    "plus_specific_lift",
    "quality_adjusted_effect",
    "degeneration_rate",
    "rank_score",
    "component_score",
    "order_specific_score",
    "energy_fraction",
    "order_orth_energy_fraction_of_full",
    "content_energy_fraction_of_full",
    "order_energy_fraction_of_full",
    # SAE feature bridge metrics.
    "x_content_component_delta",
    "x_order_orth_component_delta",
    "abs_x_content_component_delta",
    "abs_x_order_orth_component_delta",
    "order_minus_content_abs_component_delta",
    "order_over_content_abs_ratio",
    "target_prompt_mean_activation_delta",
    "sentence_shuffle_prompt_mean_activation_delta",
    "target_prompt_activation_rate_delta",
    "sentence_shuffle_prompt_activation_rate_delta",
    "target_minus_sentence_shuffle_prompt_delta",
    "target_generation_mean_activation",
    "sentence_shuffle_generation_mean_activation",
    "target_generation_activation_rate",
    "sentence_shuffle_generation_activation_rate",
    "target_generation_late_minus_early_activation",
    "sentence_shuffle_generation_late_minus_early_activation",
    "target_generation_generation_mean_activation_delta",
    "sentence_shuffle_generation_generation_mean_activation_delta",
    "target_minus_sentence_shuffle_generation_mean_activation",
    "reconstruction_mse",
    "reconstruction_l2",
    "input_l2",
    "reconstruction_l2_norm",
    "input_reconstruction_cosine",
    "explained_variance_proxy",
    # Behavior proxy metrics.
    "generated_token_count",
    "raw_has_think_tag",
    "visible_response_empty_after_think_strip",
    "refusal_marker_count",
    "caution_marker_count",
    "substitution_marker_count",
    "refusal_binary",
    "caution_binary",
    "substitution_binary",
    "nonempty_visible_response",
    "instruction_deviation_proxy",
}

METRIC_REGEX = re.compile(
    r"(cosine|projection_fraction|l2_distance|entropy|logprob|state_norm|"
    r"delta|abs_delta|score|threshold|p_value|q_value|cohen|pass|slope|monotonicity|"
    r"win_rate|random_p95|lift|degeneration|rank_score|component_score|"
    r"energy_fraction|norm|centroid|within|between|variance|pc[0-9]|jaccard|agreement|generated_token|refusal|"
    r"caution|substitution|deviation|nonempty|think_tag)",
    re.IGNORECASE,
)

GROUPING_CANDIDATES = [
    ["condition"],
    ["condition", "layer"],
    ["condition", "step"],
    ["condition", "reference_condition"],
    ["condition", "reference_condition", "layer"],
    ["condition", "control_condition"],
    ["condition", "module"],
    ["condition", "module", "layer"],
    ["condition", "unit_type", "layer"],
    ["control_condition"],
    ["base_condition"],
    ["base_condition", "layer_band"],
    ["base_condition", "sign_name"],
    ["base_condition", "layer_band", "alpha_abs"],
    ["base_condition", "layer_band", "sign_name", "alpha_abs"],
    ["base_condition", "layer_band", "sign_name", "alpha_abs", "layer"],
    ["base_condition", "intervention_name"],
    ["base_condition", "intervention_name", "layer"],
    ["axis_name"],
    ["axis_name", "layer_band"],
    ["axis_name", "readout_layer_band"],
    ["axis_name", "intervention_layer_band", "alpha_abs"],
    ["layer"],
    ["layer_band"],
    ["readout_layer_band"],
    ["intervention_layer_band"],
    ["module"],
    ["module", "layer"],
    ["unit_type"],
    ["unit_type", "layer"],
    ["metric"],
    ["criterion"],
]

REGRESSION_GROUPING_CANDIDATES = [
    ["base_condition"],
    ["base_condition", "layer_band"],
    ["base_condition", "sign_name"],
    ["base_condition", "layer_band", "sign_name"],
    ["base_condition", "layer_band", "sign_name", "layer"],
    ["condition"],
    ["condition", "layer"],
    ["condition", "module"],
    ["condition", "module", "layer"],
    ["axis_name"],
    ["axis_name", "base_condition"],
    ["axis_name", "intervention_layer_band"],
    ["axis_name", "readout_layer_band"],
]

LAYERWISE_CONTEXT_CANDIDATES = [
    ["condition"],
    ["condition", "reference_condition"],
    ["base_condition", "layer_band"],
    ["base_condition", "layer_band", "sign_name", "alpha_abs"],
    ["base_condition", "intervention_name"],
    ["condition", "module"],
    ["axis_name"],
    ["axis_name", "condition"],
    ["module"],
    [],
]

KEY_GRADE3_ARTIFACTS = [
    "red_team_input_manifest.json",
    "middle_layer_condition_summary.csv",
    "question_level_middle_layer_summary.csv",
    "paired_target_vs_control_tests.csv",
    "layerwise_fdr_target_vs_control.csv",
    "null_vector_baseline_summary.csv",
    "architecture_module_delta_summary.csv",
    "architecture_top_changed_units.csv",
    "generation_middle_layer_summary.csv",
]

KEY_GRADE4_ARTIFACTS = [
    "grade4_axis_component_vectors_by_layer.npz",
    "grade4_axis_component_norm_summary.csv",
    "grade4_axis_projection_geometry_summary.csv",
    "grade4_axis_component_causal_projection_summary.csv",
    "grade4_axis_component_causal_symmetry_summary.csv",
    "grade4_axis_component_causal_alpha_scaling_summary.csv",
    "grade4_axis_component_causal_rank_summary.csv",
]

OPTIONAL_GRADE4_EXTENSION_ARTIFACTS = [
    "grade4_axis_layerwise_component_readout_summary.csv",
    "grade4_axis_layerwise_order_birth_summary.csv",
    "grade4_axis_generation_component_readout_raw.csv",
    "grade4_axis_generation_component_readout_summary.csv",
    "grade4_axis_generation_component_persistence_comparison.csv",
    "grade4_axis_component_causal_natscale_closeout.csv",
    "grade4_axis_component_causal_data_quality.csv",
    "sae_order_feature_contrast.csv",
    "sae_order_feature_triage.csv",
    "sae_model_compatibility.csv",
    "sae_reconstruction_quality.csv",
]


# -----------------------------------------------------------------------------
# Data classes / helpers
# -----------------------------------------------------------------------------


@dataclass
class ProcessingAudit:
    source_file: str
    status: str
    rows: int = 0
    metric_columns: int = 0
    used_columns: int = 0
    seconds: float = 0.0
    error: str = ""
    failure_code: str = ""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def sanitize_machine_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    forbidden = sorted(FORBIDDEN_RESULT_LABEL_VALUES, key=len, reverse=True)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            continue
        s = out[col].astype(str)
        mask = pd.Series(False, index=out.index)
        for label in forbidden:
            mask |= s.str.contains(re.escape(label), case=False, na=False)
        if mask.any():
            out.loc[mask, col] = "forbidden_label_quarantined"
    return out


def write_csv(path: Path, rows_or_df: Any, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    ensure_dir(path.parent)
    df = rows_or_df.copy() if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(columns)]
    df = sanitize_machine_dataframe(df)
    df.to_csv(path, index=False)
    return df


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def is_csv_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".csv" and not path.name.startswith("._")


def is_npz_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".npz" and not path.name.startswith("._")


def is_plot_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}


def safe_extract_zip(input_path: Path, extract_dir: Path) -> None:
    ensure_dir(extract_dir)
    root = extract_dir.resolve()
    with zipfile.ZipFile(input_path, "r") as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for member in tqdm(members, desc="extract zip"):
            if member.filename.startswith("__MACOSX/"):
                continue
            target = (extract_dir / member.filename).resolve()
            try:
                target.relative_to(root)
            except Exception:
                raise RuntimeError(f"Unsafe zip path outside extraction root: {member.filename}")
            ensure_dir(target.parent)
            with z.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_if_needed(input_path: Path, work_dir: Path, force_extract: bool = False) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Input must be a directory or .zip file: {input_path}")
    extract_dir = work_dir / "extracted"
    marker = extract_dir / ".extract_complete.json"
    if force_extract and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if marker.exists():
        return extract_dir
    safe_extract_zip(input_path, extract_dir)
    marker.write_text(json.dumps({"source": str(input_path), "time": time.time()}), encoding="utf-8")
    return extract_dir


def detect_result_root(root: Path) -> Path:
    """Find the actual result directory inside a Colab-style extracted zip.

    Colab archives often contain source artifacts under a nested path such as
    content/hidden_geometry_runs/<run_label>/red_team_input_manifest.json.  The
    analyzer's specialized Grade 3/Grade 4 tables expect artifact names relative
    to the result directory, so we normalize that root here without modifying the
    source package or extracted files.
    """
    root = Path(root)
    if (root / "red_team_input_manifest.json").exists():
        return root
    manifest_candidates = sorted(
        root.rglob("red_team_input_manifest.json"),
        key=lambda p: (len(p.relative_to(root).parts), len(str(p))),
    )
    if manifest_candidates:
        return manifest_candidates[0].parent
    grade4_candidates = sorted(
        root.rglob("grade4_axis_projection_geometry_summary.csv"),
        key=lambda p: (len(p.relative_to(root).parts), len(str(p))),
    )
    if grade4_candidates:
        return grade4_candidates[0].parent
    return root


def rel_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.name


def list_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("._"))


def infer_artifact_family(rel_name: str) -> Tuple[str, str]:
    name = Path(rel_name).name.lower()
    parent = str(Path(rel_name).parent).lower()
    if name == "red_team_input_manifest.json" or "prompt_condition_manifest" in name or "question_domain_manifest" in name:
        return "run_validity", "manifest"
    if "quarantine" in parent or "numeric_integrity" in name:
        return "quarantine_integrity", "integrity"
    if name.endswith((".png", ".jpg", ".jpeg", ".svg")):
        return "plots_existing", "plot"
    if name.endswith(".npz") or "vector_x" in name or "prompt_hidden_states" in name:
        return "npz_vectors", "npz"
    if name.startswith("state_space_"):
        return "state_space_geometry", "state_space"
    if name.startswith("grade4_axis_"):
        return "grade4_components", "grade4"
    if any(k in name for k in ["middle_layer", "layerwise_geometry", "question_level", "geometry_decomposition", "residual_stream", "orthogonality", "phase_transition"]):
        return "prompt_geometry", "geometry"
    if any(k in name for k in ["paired_target", "specificity", "fdr", "length_bias", "domain_robustness"]):
        return "specificity_controls", "specificity"
    if any(k in name for k in ["null_vector", "pca_baseline", "null_hypothesis"]):
        return "null_baselines", "null"
    if any(k in name for k in ["generation_", "trajectory_", "dynamic_trajectory"]):
        return "generation_trajectory", "trajectory"
    if any(k in name for k in ["causal_", "layer_specific_causal", "alpha_dose"]):
        return "causal_interventions", "causal"
    if any(k in name for k in ["behavior", "semantic_shift", "internal_visible", "quality_adjusted"]):
        return "behavioral_control", "behavior"
    if "output_semantic" in name:
        return "semantic_shift", "semantic"
    if any(k in name for k in ["architecture_module", "architecture_target", "circuit_component"]):
        return "architecture_modules", "architecture"
    if name.startswith("sae_") or "sae_feature" in name or "order_feature" in name:
        return "sae_features", "sae"
    if any(k in name for k in ["architecture_top", "hidden_top", "mlp_unit", "dense_feature"]):
        if "dense_feature" in name:
            return "dense_features", "dense_feature"
        return "architecture_units", "unit"
    return "other", "unknown"


def infer_metric_columns(columns: Sequence[str]) -> List[str]:
    metrics = []
    for col in columns:
        c = str(col)
        low = c.lower()
        if c in INDEPENDENT_COLUMNS or c in GROUP_COLUMNS or c in ID_OR_INDEX_COLUMNS:
            continue
        if c in EXACT_METRIC_COLUMNS or low in EXACT_METRIC_COLUMNS or METRIC_REGEX.search(c):
            metrics.append(c)
    return list(dict.fromkeys(metrics))


def infer_group_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c in GROUP_COLUMNS or c in INDEPENDENT_COLUMNS]


def infer_used_columns(columns: Sequence[str], metric_cols: Sequence[str]) -> List[str]:
    used = []
    for c in columns:
        if c in metric_cols or c in GROUP_COLUMNS or c in INDEPENDENT_COLUMNS:
            used.append(c)
    return list(dict.fromkeys(used))


def available_groupings(columns: Sequence[str], candidates: Sequence[Sequence[str]]) -> List[List[str]]:
    colset = set(columns)
    seen = set()
    out = []
    for group in candidates:
        if all(c in colset for c in group):
            key = tuple(group)
            if key not in seen:
                out.append(list(group))
                seen.add(key)
    return out


def formalism_class(metric: str) -> str:
    m = str(metric).lower()
    if any(k in m for k in ["centroid", "within_", "between_", "pc1", "pc2", "pc3"]):
        return "non_x_state_space_geometry"
    if "projection" in m or "cosine" in m:
        return "geometrical_capture"
    if "l2_distance" in m or "state_norm" in m or "norm" in m:
        return "l2_or_norm_geometry"
    if "entropy" in m:
        return "entropy_shift"
    if "logprob" in m:
        return "logprob_shift"
    if "delta" in m or "reference_value" in m or "condition_value" in m:
        return "activation_delta"
    if any(k in m for k in ["refusal", "caution", "substitution", "nonempty", "deviation", "think", "generated_token"]):
        return "behavioral_proxy"
    if any(k in m for k in ["p_value", "q_value", "cohen", "score", "threshold", "pass"]):
        return "statistical_gate_metric"
    return "other_metric"


def finite_numeric_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s[np.isfinite(s)]


# -----------------------------------------------------------------------------
# CSV processing
# -----------------------------------------------------------------------------


def flatten_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf.copy()
    flat = []
    for col in out.columns:
        if isinstance(col, tuple):
            flat.append("__".join(str(p) for p in col if str(p) not in {"", "None"}))
        else:
            flat.append(str(col))
    out.columns = flat
    return out


def summarize_global_pdf(pdf: pd.DataFrame, source_file: str, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    total_rows = len(pdf)
    for metric in metric_cols:
        if metric not in pdf.columns:
            continue
        s = finite_numeric_series(pdf[metric])
        n = int(s.count())
        mean = float(s.mean()) if n else math.nan
        std = float(s.std(ddof=1)) if n > 1 else math.nan
        se = std / math.sqrt(n) if n > 1 and math.isfinite(std) else math.nan
        rows.append(
            {
                "source_file": source_file,
                "formalism_class": formalism_class(metric),
                "metric": metric,
                "rows_seen": total_rows,
                "finite_count": n,
                "missing_count": total_rows - n,
                "mean": mean,
                "std": std,
                "min": float(s.min()) if n else math.nan,
                "max": float(s.max()) if n else math.nan,
                "ci95_low": mean - 1.96 * se if math.isfinite(se) else math.nan,
                "ci95_high": mean + 1.96 * se if math.isfinite(se) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def grouped_summary_one_pdf(pdf: pd.DataFrame, source_file: str, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    present_metrics = [m for m in metric_cols if m in pdf.columns]
    if not present_metrics or not group_cols:
        return pd.DataFrame()
    cols = list(group_cols) + present_metrics
    tmp = pdf[cols].dropna(subset=list(group_cols))
    if tmp.empty:
        return pd.DataFrame()
    for metric in present_metrics:
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    agg = tmp.groupby(list(group_cols), dropna=False).agg({m: ["count", "mean", "std", "min", "max"] for m in present_metrics}).reset_index()
    flat = flatten_columns(agg)
    long_rows = []
    reserved_insert_cols = {"source_file", "grouping", "formalism_class", "metric"}
    context_renames = {c: f"context_{c}" for c in group_cols if c in reserved_insert_cols}
    id_cols = list(group_cols)
    for metric in present_metrics:
        cols_needed = [f"{metric}__{stat}" for stat in ["count", "mean", "std", "min", "max"]]
        if not all(c in flat.columns for c in cols_needed):
            continue
        part = flat[id_cols + cols_needed].copy()
        if context_renames:
            part = part.rename(columns=context_renames)
        part.insert(0, "source_file", source_file)
        part.insert(1, "grouping", "+".join(group_cols))
        part.insert(2, "formalism_class", formalism_class(metric))
        part.insert(3, "metric", metric)
        part = part.rename(
            columns={
                f"{metric}__count": "count",
                f"{metric}__mean": "mean",
                f"{metric}__std": "std",
                f"{metric}__min": "min",
                f"{metric}__max": "max",
            }
        )
        long_rows.append(part)
    return pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()


def summarize_grouped_pdf(pdf: pd.DataFrame, source_file: str, metric_cols: Sequence[str]) -> pd.DataFrame:
    chunks = []
    for group_cols in available_groupings(pdf.columns, GROUPING_CANDIDATES):
        part = grouped_summary_one_pdf(pdf, source_file, group_cols, metric_cols)
        if not part.empty:
            chunks.append(part)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def combine_grouped_partials(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    if raw.empty:
        return raw
    key_cols = [c for c in raw.columns if c not in {"count", "mean", "std", "min", "max"}]
    raw["_count"] = pd.to_numeric(raw["count"], errors="coerce").fillna(0.0)
    raw["_sum"] = pd.to_numeric(raw["mean"], errors="coerce").fillna(0.0) * raw["_count"]
    std = pd.to_numeric(raw["std"], errors="coerce").fillna(0.0)
    raw["_sumsq"] = (std**2) * (raw["_count"] - 1).clip(lower=0) + raw["_count"] * (pd.to_numeric(raw["mean"], errors="coerce").fillna(0.0) ** 2)
    grouped = raw.groupby(key_cols, dropna=False).agg(
        count=("_count", "sum"),
        _sum=("_sum", "sum"),
        _sumsq=("_sumsq", "sum"),
        min=("min", "min"),
        max=("max", "max"),
    ).reset_index()
    grouped["mean"] = grouped["_sum"] / grouped["count"].replace(0, np.nan)
    variance = (grouped["_sumsq"] - (grouped["_sum"] ** 2) / grouped["count"].replace(0, np.nan)) / (grouped["count"] - 1).replace(0, np.nan)
    grouped["std"] = np.sqrt(variance.clip(lower=0))
    return grouped.drop(columns=["_sum", "_sumsq"])


def combine_global_partials(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    if raw.empty:
        return raw
    raw["_count"] = pd.to_numeric(raw["finite_count"], errors="coerce").fillna(0.0)
    raw["_sum"] = pd.to_numeric(raw["mean"], errors="coerce").fillna(0.0) * raw["_count"]
    std = pd.to_numeric(raw["std"], errors="coerce").fillna(0.0)
    raw["_sumsq"] = (std**2) * (raw["_count"] - 1).clip(lower=0) + raw["_count"] * (pd.to_numeric(raw["mean"], errors="coerce").fillna(0.0) ** 2)
    g = raw.groupby(["source_file", "formalism_class", "metric"], dropna=False).agg(
        rows_seen=("rows_seen", "sum"),
        finite_count=("_count", "sum"),
        missing_count=("missing_count", "sum"),
        _sum=("_sum", "sum"),
        _sumsq=("_sumsq", "sum"),
        min=("min", "min"),
        max=("max", "max"),
    ).reset_index()
    g["mean"] = g["_sum"] / g["finite_count"].replace(0, np.nan)
    variance = (g["_sumsq"] - (g["_sum"] ** 2) / g["finite_count"].replace(0, np.nan)) / (g["finite_count"] - 1).replace(0, np.nan)
    g["std"] = np.sqrt(variance.clip(lower=0))
    se = g["std"] / np.sqrt(g["finite_count"].replace(0, np.nan))
    g["ci95_low"] = g["mean"] - 1.96 * se
    g["ci95_high"] = g["mean"] + 1.96 * se
    return g.drop(columns=["_sum", "_sumsq"])


def condition_effects_from_grouped(grouped_pdf: pd.DataFrame, source_file: str) -> pd.DataFrame:
    if grouped_pdf.empty or "metric" not in grouped_pdf.columns:
        return pd.DataFrame()
    rows = []
    cond_rows = grouped_pdf[grouped_pdf["grouping"].astype(str).str.contains("condition", na=False)].copy()
    if cond_rows.empty or "condition" not in cond_rows.columns:
        return pd.DataFrame()
    baseline_priority = ["neutral", "neutral_length_matched_control", "reference", "question_only", "control"]
    for (grouping, metric), sub in cond_rows.groupby(["grouping", "metric"], dropna=False):
        grouping_cols = str(grouping).split("+")
        context_cols = [c for c in grouping_cols if c in sub.columns and c not in {"condition"}]
        iterator = sub.groupby(context_cols, dropna=False) if context_cols else [((), sub)]
        for ctx_key, ctx_df in iterator:
            conditions = [str(x) for x in ctx_df["condition"].dropna().unique()]
            baseline = next((b for b in baseline_priority if b in conditions), conditions[0] if conditions else None)
            if baseline is None:
                continue
            base = ctx_df[ctx_df["condition"].astype(str).eq(baseline)]
            if base.empty:
                continue
            base_mean = safe_float(base.iloc[0].get("mean"))
            for _, row in ctx_df.iterrows():
                cond = str(row.get("condition", ""))
                if cond == baseline:
                    continue
                mean = safe_float(row.get("mean"))
                delta = mean - base_mean if math.isfinite(mean) and math.isfinite(base_mean) else math.nan
                out = {
                    "source_file": source_file,
                    "effect_type": "condition_minus_baseline",
                    "formalism_class": formalism_class(metric),
                    "metric": metric,
                    "grouping": grouping,
                    "condition": cond,
                    "baseline_condition": baseline,
                    "condition_mean": mean,
                    "baseline_mean": base_mean,
                    "delta": delta,
                    "relative_delta": delta / abs(base_mean) if math.isfinite(delta) and math.isfinite(base_mean) and abs(base_mean) > 1e-12 else math.nan,
                }
                if context_cols:
                    if not isinstance(ctx_key, tuple):
                        ctx_key = (ctx_key,)
                    for c, v in zip(context_cols, ctx_key):
                        out[c] = v
                rows.append(out)
    return pd.DataFrame(rows)


def alpha_regression_one_pdf(pdf: pd.DataFrame, source_file: str, x_col: str, y_col: str, group_cols: Sequence[str]) -> pd.DataFrame:
    if x_col not in pdf.columns or y_col not in pdf.columns:
        return pd.DataFrame()
    cols = list(group_cols) + [x_col, y_col]
    tmp = pdf[cols].copy()
    tmp["_x"] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp["_y"] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=["_x", "_y"])
    if len(tmp) < 2:
        return pd.DataFrame()
    tmp["_x2"] = tmp["_x"] * tmp["_x"]
    tmp["_y2"] = tmp["_y"] * tmp["_y"]
    tmp["_xy"] = tmp["_x"] * tmp["_y"]
    tmp["_one"] = 1
    if group_cols:
        agg = tmp.groupby(list(group_cols), dropna=False).agg({"_one": "sum", "_x": "sum", "_y": "sum", "_x2": "sum", "_y2": "sum", "_xy": "sum"}).reset_index()
    else:
        agg = pd.DataFrame(
            [{"_one": len(tmp), "_x": tmp["_x"].sum(), "_y": tmp["_y"].sum(), "_x2": tmp["_x2"].sum(), "_y2": tmp["_y2"].sum(), "_xy": tmp["_xy"].sum()}]
        )
    rows = []
    for _, r in agg.iterrows():
        n = safe_float(r["_one"], 0.0)
        sx, sy, sxx, syy, sxy = (safe_float(r[c]) for c in ["_x", "_y", "_x2", "_y2", "_xy"])
        denom = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denom if n >= 2 and abs(denom) > 1e-12 else math.nan
        intercept = (sy - slope * sx) / n if math.isfinite(slope) and n else math.nan
        sst = syy - (sy * sy / n) if n else math.nan
        ssr = slope * (sxy - sx * sy / n) if math.isfinite(slope) and n else math.nan
        r2 = ssr / sst if math.isfinite(ssr) and abs(sst) > 1e-12 else math.nan
        out = {
            "source_file": source_file,
            "formalism_class": "causal_alpha_steering",
            "dependent_metric": y_col,
            "x": x_col,
            "grouping": "+".join(group_cols) if group_cols else "global",
            "n": int(n) if math.isfinite(n) else 0,
            "slope_beta_like": slope,
            "intercept": intercept,
            "r2": r2,
        }
        for c in group_cols:
            out[c] = r.get(c)
        rows.append(out)
    return pd.DataFrame(rows)


def alpha_regressions_pdf(pdf: pd.DataFrame, source_file: str, metric_cols: Sequence[str]) -> pd.DataFrame:
    x_cols = [c for c in ["alpha", "alpha_abs"] if c in pdf.columns]
    if not x_cols:
        return pd.DataFrame()
    groupings = available_groupings(pdf.columns, REGRESSION_GROUPING_CANDIDATES)
    if [] not in groupings:
        groupings = [[]] + groupings
    chunks = []
    for x_col in x_cols:
        for y_col in metric_cols:
            if y_col == x_col or y_col not in pdf.columns:
                continue
            for group_cols in groupings:
                if x_col in group_cols:
                    continue
                part = alpha_regression_one_pdf(pdf, source_file, x_col, y_col, group_cols)
                if not part.empty:
                    chunks.append(part)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def layerwise_one_pdf(pdf: pd.DataFrame, source_file: str, metric: str, context_cols: Sequence[str]) -> pd.DataFrame:
    if "layer" not in pdf.columns or metric not in pdf.columns:
        return pd.DataFrame()
    cols = list(context_cols) + ["layer", metric]
    tmp = pdf[cols].copy()
    tmp["layer"] = pd.to_numeric(tmp["layer"], errors="coerce")
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=["layer", metric])
    if tmp.empty:
        return pd.DataFrame()
    agg = tmp.groupby(list(context_cols) + ["layer"] if context_cols else ["layer"], dropna=False).agg({metric: ["count", "mean", "std"]}).reset_index()
    flat = flatten_columns(agg)
    mean_col = f"{metric}__mean"
    count_col = f"{metric}__count"
    rows = []
    iterator = flat.groupby(list(context_cols), dropna=False) if context_cols else [((), flat)]
    for ctx_key, sub in iterator:
        sub = sub.sort_values("layer")
        layers = sub["layer"].to_numpy(dtype=float)
        means = sub[mean_col].to_numpy(dtype=float)
        counts = sub[count_col].to_numpy(dtype=float)
        if len(means) >= 2:
            diffs = np.diff(means)
            finite = np.isfinite(diffs)
            idx = int(np.nanargmax(np.abs(diffs))) if finite.any() else -1
            max_adjacent_jump = float(diffs[idx]) if idx >= 0 else math.nan
            transition_from_layer = int(layers[idx]) if idx >= 0 else math.nan
            transition_to_layer = int(layers[idx + 1]) if idx >= 0 else math.nan
            finite_xy = np.isfinite(layers) & np.isfinite(means)
            if finite_xy.sum() >= 2 and np.var(layers[finite_xy]) > 1e-12:
                layer_slope = float(np.cov(layers[finite_xy], means[finite_xy], bias=True)[0, 1] / np.var(layers[finite_xy]))
            else:
                layer_slope = math.nan
        else:
            max_adjacent_jump = transition_from_layer = transition_to_layer = layer_slope = math.nan
        if np.isfinite(means).any():
            peak_i = int(np.nanargmax(means))
            trough_i = int(np.nanargmin(means))
            peak_layer = int(layers[peak_i])
            trough_layer = int(layers[trough_i])
            peak_mean = float(means[peak_i])
            trough_mean = float(means[trough_i])
        else:
            peak_layer = trough_layer = peak_mean = trough_mean = math.nan
        out = {
            "source_file": source_file,
            "formalism_class": formalism_class(metric),
            "metric": metric,
            "context": "+".join(context_cols) if context_cols else "global",
            "layer_count": int(len(sub)),
            "row_count_sum": int(np.nansum(counts)),
            "layer_slope": layer_slope,
            "max_adjacent_jump": max_adjacent_jump,
            "transition_from_layer": transition_from_layer,
            "transition_to_layer": transition_to_layer,
            "peak_layer": peak_layer,
            "peak_mean": peak_mean,
            "trough_layer": trough_layer,
            "trough_mean": trough_mean,
            "range_peak_minus_trough": peak_mean - trough_mean if math.isfinite(peak_mean) and math.isfinite(trough_mean) else math.nan,
        }
        if context_cols:
            if not isinstance(ctx_key, tuple):
                ctx_key = (ctx_key,)
            for c, v in zip(context_cols, ctx_key):
                out[c] = v
        rows.append(out)
    return pd.DataFrame(rows)


def layerwise_transitions_pdf(pdf: pd.DataFrame, source_file: str, metric_cols: Sequence[str]) -> pd.DataFrame:
    if "layer" not in pdf.columns:
        return pd.DataFrame()
    chunks = []
    contexts = available_groupings(pdf.columns, LAYERWISE_CONTEXT_CANDIDATES)
    if [] not in contexts:
        contexts.append([])
    for metric in metric_cols:
        for ctx in contexts:
            if "layer" in ctx:
                continue
            part = layerwise_one_pdf(pdf, source_file, metric, ctx)
            if not part.empty:
                chunks.append(part)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def build_final_table(global_pdf: pd.DataFrame, effects_pdf: pd.DataFrame, alpha_pdf: pd.DataFrame, layer_pdf: pd.DataFrame, summary_pdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in global_pdf.iterrows():
        rows.append(
            {
                "evidence_type": "global_metric_summary",
                "source_file": r.get("source_file"),
                "formalism_class": r.get("formalism_class"),
                "metric": r.get("metric"),
                "context": "global",
                "primary_value_name": "mean",
                "primary_value": r.get("mean"),
                "secondary_value_name": "std",
                "secondary_value": r.get("std"),
                "n": r.get("finite_count"),
                "details_json": json.dumps({"min": safe_float(r.get("min")), "max": safe_float(r.get("max")), "ci95_low": safe_float(r.get("ci95_low")), "ci95_high": safe_float(r.get("ci95_high"))}),
            }
        )
    for _, r in effects_pdf.iterrows():
        rows.append(
            {
                "evidence_type": "condition_effect",
                "source_file": r.get("source_file"),
                "formalism_class": r.get("formalism_class"),
                "metric": r.get("metric"),
                "context": r.get("grouping"),
                "primary_value_name": "delta",
                "primary_value": r.get("delta"),
                "secondary_value_name": "relative_delta",
                "secondary_value": r.get("relative_delta"),
                "n": math.nan,
                "details_json": json.dumps({k: str(v) for k, v in r.to_dict().items() if k not in {"source_file", "formalism_class", "metric", "grouping", "delta", "relative_delta"}}, ensure_ascii=False),
            }
        )
    for _, r in alpha_pdf.iterrows():
        rows.append(
            {
                "evidence_type": "alpha_response_regression",
                "source_file": r.get("source_file"),
                "formalism_class": "causal_alpha_steering",
                "metric": r.get("dependent_metric"),
                "context": r.get("grouping"),
                "primary_value_name": f"slope_beta_like_vs_{r.get('x')}",
                "primary_value": r.get("slope_beta_like"),
                "secondary_value_name": "r2",
                "secondary_value": r.get("r2"),
                "n": r.get("n"),
                "details_json": json.dumps({k: str(v) for k, v in r.to_dict().items() if k not in {"source_file", "dependent_metric", "grouping", "slope_beta_like", "r2", "n"}}, ensure_ascii=False),
            }
        )
    for _, r in layer_pdf.iterrows():
        rows.append(
            {
                "evidence_type": "layerwise_transition_proxy",
                "source_file": r.get("source_file"),
                "formalism_class": r.get("formalism_class"),
                "metric": r.get("metric"),
                "context": r.get("context"),
                "primary_value_name": "max_adjacent_jump",
                "primary_value": r.get("max_adjacent_jump"),
                "secondary_value_name": "layer_slope",
                "secondary_value": r.get("layer_slope"),
                "n": r.get("row_count_sum"),
                "details_json": json.dumps({"transition_from_layer": str(r.get("transition_from_layer")), "transition_to_layer": str(r.get("transition_to_layer")), "peak_layer": str(r.get("peak_layer")), "peak_mean": str(r.get("peak_mean")), "trough_layer": str(r.get("trough_layer")), "trough_mean": str(r.get("trough_mean"))}, ensure_ascii=False),
            }
        )
    for _, r in summary_pdf.iterrows():
        rows.append(
            {
                "evidence_type": "summary_file_numeric_extract",
                "source_file": r.get("source_file"),
                "formalism_class": formalism_class(str(r.get("metric", ""))),
                "metric": r.get("metric"),
                "context": r.get("context", ""),
                "primary_value_name": "value",
                "primary_value": r.get("value"),
                "secondary_value_name": "row_index",
                "secondary_value": r.get("row_index"),
                "n": 1,
                "details_json": r.get("details_json", "{}"),
            }
        )
    return pd.DataFrame(rows)


def state_space_final_evidence(*tables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for table in tables:
        if table is None or table.empty:
            continue
        source_file = str(table["source_file"].iloc[0]) if "source_file" in table.columns and len(table) else "prompt_hidden_states.npz"
        metric_cols = [c for c in table.columns if c in EXACT_METRIC_COLUMNS or formalism_class(c) == "non_x_state_space_geometry"]
        context_cols = [c for c in ["layer", "condition", "condition_a", "condition_b", "metric_family", "status", "failure_code"] if c in table.columns]
        for _, r in table.iterrows():
            if str(r.get("status", "computed")) not in {"computed", "nan", ""}:
                continue
            context = {c: r.get(c) for c in context_cols}
            for metric in metric_cols:
                val = safe_float(r.get(metric))
                if not math.isfinite(val):
                    continue
                rows.append(
                    {
                        "evidence_type": "non_x_state_space_geometry",
                        "source_file": source_file,
                        "formalism_class": "non_x_state_space_geometry",
                        "metric": metric,
                        "context": json.dumps(json_sanitize(context), ensure_ascii=False),
                        "primary_value_name": metric,
                        "primary_value": val,
                        "secondary_value_name": "",
                        "secondary_value": math.nan,
                        "n": safe_float(r.get("n_questions", r.get("question_count", 1))),
                        "details_json": json.dumps(json_sanitize(context), ensure_ascii=False),
                    }
                )
    return pd.DataFrame(rows)


def read_csv_selected(path: Path, usecols: Sequence[str], backend: str):
    if backend == "rapids":
        return cudf.read_csv(str(path), usecols=list(usecols))
    return pd.read_csv(path, usecols=list(usecols), encoding="utf-8", encoding_errors="replace")


def cudf_to_pandas(gdf: Any) -> pd.DataFrame:
    return gdf.to_pandas() if cudf is not None and hasattr(gdf, "to_pandas") else gdf


def process_file(
    csv_path: Path,
    extracted_root: Path,
    output_dir: Path,
    backend: str,
    cache_parquet: bool,
    overwrite_cache: bool,
    write_per_file: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ProcessingAudit]:
    t0 = time.time()
    rel_name = rel_path(csv_path, extracted_root)
    header = read_header(csv_path)
    metric_cols = infer_metric_columns(header)
    used_cols = infer_used_columns(header, metric_cols)
    if not metric_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ProcessingAudit(
            rel_name, "skipped_no_metric_columns", 0, 0, len(used_cols), time.time() - t0, "", "no_metric_columns"
        )

    parquet_root = output_dir / "_work" / "parquet_cache"
    pq_path = parquet_root / Path(rel_name).with_suffix(".parquet")
    if backend == "rapids" and cache_parquet and pq_path.exists() and not overwrite_cache:
        gdf = cudf.read_parquet(str(pq_path))
        pdf = cudf_to_pandas(gdf)
        rows = len(pdf)
        global_pdf = summarize_global_pdf(pdf, rel_name, metric_cols)
        grouped_pdf = summarize_grouped_pdf(pdf, rel_name, metric_cols)
    elif backend == "rapids":
        gdf = cudf.read_csv(str(csv_path), usecols=used_cols)
        for metric in metric_cols:
            if metric in gdf.columns:
                gdf[metric] = cudf.to_numeric(gdf[metric], errors="coerce")
        if cache_parquet:
            ensure_dir(pq_path.parent)
            gdf.to_parquet(str(pq_path), index=False)
        pdf = cudf_to_pandas(gdf)
        rows = len(pdf)
        global_pdf = summarize_global_pdf(pdf, rel_name, metric_cols)
        grouped_pdf = summarize_grouped_pdf(pdf, rel_name, metric_cols)
    else:
        global_parts = []
        grouped_parts = []
        rows = 0
        for chunk in pd.read_csv(csv_path, usecols=used_cols, chunksize=CHUNK_ROWS, encoding="utf-8", encoding_errors="replace"):
            rows += len(chunk)
            for metric in metric_cols:
                if metric in chunk.columns:
                    chunk[metric] = pd.to_numeric(chunk[metric], errors="coerce")
            global_parts.append(summarize_global_pdf(chunk, rel_name, metric_cols))
            grouped = summarize_grouped_pdf(chunk, rel_name, metric_cols)
            if not grouped.empty:
                grouped_parts.append(grouped)
        global_pdf = combine_global_partials(global_parts)
        grouped_pdf = combine_grouped_partials(grouped_parts)

    # Higher-level derived tables are based on grouped/global outputs and on a
    # bounded in-memory pass for alpha/layer regressions. This is acceptable for
    # summary files; huge raw CSVs get full global/grouped coverage first.
    effects_pdf = condition_effects_from_grouped(grouped_pdf, rel_name)
    alpha_pdf = pd.DataFrame()
    layer_pdf = pd.DataFrame()
    try:
        if csv_path.stat().st_size <= SUMMARY_FILE_MAX_BYTES:
            full_pdf = pd.read_csv(csv_path, usecols=used_cols, encoding="utf-8", encoding_errors="replace")
            for metric in metric_cols:
                if metric in full_pdf.columns:
                    full_pdf[metric] = pd.to_numeric(full_pdf[metric], errors="coerce")
            alpha_pdf = alpha_regressions_pdf(full_pdf, rel_name, metric_cols)
            layer_pdf = layerwise_transitions_pdf(full_pdf, rel_name, metric_cols)
    except Exception:
        alpha_pdf = pd.DataFrame()
        layer_pdf = pd.DataFrame()

    summary_pdf = summary_numeric_extract(csv_path, extracted_root, metric_cols)

    if write_per_file:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(rel_name).with_suffix("").as_posix())
        per_dir = output_dir / "per_file"
        write_csv(per_dir / f"{safe_stem}__global.csv", global_pdf)
        write_csv(per_dir / f"{safe_stem}__grouped.csv", grouped_pdf)
        write_csv(per_dir / f"{safe_stem}__effects.csv", effects_pdf)
        write_csv(per_dir / f"{safe_stem}__alpha.csv", alpha_pdf)
        write_csv(per_dir / f"{safe_stem}__layerwise.csv", layer_pdf)
        write_csv(per_dir / f"{safe_stem}__summary_numeric.csv", summary_pdf)

    return global_pdf, grouped_pdf, effects_pdf, alpha_pdf, layer_pdf, summary_pdf, ProcessingAudit(
        rel_name, "ok", rows, len(metric_cols), len(used_cols), time.time() - t0
    )


def summary_numeric_extract(csv_path: Path, root: Path, metric_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    rel_name = rel_path(csv_path, root)
    try:
        if csv_path.stat().st_size > SUMMARY_FILE_MAX_BYTES:
            return pd.DataFrame()
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    except Exception:
        return pd.DataFrame()
    numeric_cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            numeric_cols.append(col)
    rows = []
    context_candidates = [c for c in ["condition", "control_condition", "base_condition", "layer_band", "axis_name", "alpha", "alpha_abs", "metric", "criterion", "status", "failure_code"] if c in df.columns]
    for idx, row in df.iterrows():
        context = {c: row.get(c) for c in context_candidates}
        for col in numeric_cols:
            val = safe_float(row.get(col))
            if not math.isfinite(val):
                continue
            rows.append(
                {
                    "source_file": rel_name,
                    "row_index": int(idx),
                    "metric": col,
                    "value": val,
                    "context": json.dumps(json_sanitize(context), ensure_ascii=False),
                    "details_json": json.dumps(
                        {
                            "context": json_sanitize(context),
                            "source_column": col,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# NPZ analysis
# -----------------------------------------------------------------------------


def layer_norms_for_array(arr: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    a = np.asarray(arr)
    if a.ndim < 2 or not np.issubdtype(a.dtype, np.number):
        return pd.DataFrame(), pd.DataFrame(), {
            "per_layer_norm_mean": math.nan,
            "per_layer_norm_max": math.nan,
            "adjacent_layer_cosine_mean": math.nan,
            "adjacent_layer_cosine_min": math.nan,
            "adjacent_layer_cosine_max": math.nan,
            "peak_norm_layer": math.nan,
        }
    if a.ndim > 2:
        layer_vectors = a.reshape(a.shape[0], -1)
    else:
        layer_vectors = a
    layer_vectors = layer_vectors.astype(np.float64, copy=False)
    norms = np.linalg.norm(layer_vectors, axis=1)
    norm_rows = [{"layer": int(i), "layer_l2_norm": float(v)} for i, v in enumerate(norms)]
    cos_rows = []
    for i in range(layer_vectors.shape[0] - 1):
        x = layer_vectors[i]
        y = layer_vectors[i + 1]
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        cos = float(np.dot(x, y) / denom) if denom > 1e-12 else math.nan
        cos_rows.append({"layer_from": int(i), "layer_to": int(i + 1), "adjacent_layer_cosine": cos})
    cos_vals = np.asarray([r["adjacent_layer_cosine"] for r in cos_rows if math.isfinite(r["adjacent_layer_cosine"])], dtype=float)
    return (
        pd.DataFrame(norm_rows),
        pd.DataFrame(cos_rows),
        {
            "per_layer_norm_mean": float(np.nanmean(norms)) if norms.size else math.nan,
            "per_layer_norm_max": float(np.nanmax(norms)) if norms.size else math.nan,
            "adjacent_layer_cosine_mean": float(np.nanmean(cos_vals)) if cos_vals.size else math.nan,
            "adjacent_layer_cosine_min": float(np.nanmin(cos_vals)) if cos_vals.size else math.nan,
            "adjacent_layer_cosine_max": float(np.nanmax(cos_vals)) if cos_vals.size else math.nan,
            "peak_norm_layer": int(np.nanargmax(norms)) if norms.size and np.isfinite(norms).any() else math.nan,
        },
    )


def analyze_npz_files(npz_files: Sequence[Path], root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    inventory_rows = []
    summary_rows = []
    norm_parts = []
    cosine_parts = []
    grade4_rows = []
    anomalies = []
    for path in tqdm(npz_files, desc="analyze npz"):
        rel_name = rel_path(path, root)
        try:
            with np.load(path, allow_pickle=False) as npz:
                keys = list(npz.keys())
                inventory_rows.append({"source_file": rel_name, "size_bytes": int(path.stat().st_size), "array_count": len(keys), "arrays": json.dumps(keys, ensure_ascii=False), "read_status": "ok", "failure_code": ""})
                arrays: Dict[str, np.ndarray] = {}
                for key in keys:
                    arr = npz[key]
                    if not np.issubdtype(arr.dtype, np.number):
                        summary_rows.append({"source_file": rel_name, "array_name": key, "shape": str(arr.shape), "dtype": str(arr.dtype), "finite_count": 0, "nan_count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "l2_norm_global": math.nan, "per_layer_norm_mean": math.nan, "per_layer_norm_max": math.nan, "adjacent_layer_cosine_mean": math.nan, "adjacent_layer_cosine_min": math.nan, "adjacent_layer_cosine_max": math.nan, "peak_norm_layer": math.nan})
                        continue
                    arrays[key] = arr
                    flat = arr.reshape(-1)
                    finite = flat[np.isfinite(flat)]
                    norm_df, cos_df, layer_stats = layer_norms_for_array(arr)
                    if not norm_df.empty:
                        norm_df.insert(0, "array_name", key)
                        norm_df.insert(0, "source_file", rel_name)
                        norm_parts.append(norm_df)
                    if not cos_df.empty:
                        cos_df.insert(0, "array_name", key)
                        cos_df.insert(0, "source_file", rel_name)
                        cosine_parts.append(cos_df)
                    summary_rows.append(
                        {
                            "source_file": rel_name,
                            "array_name": key,
                            "shape": str(arr.shape),
                            "dtype": str(arr.dtype),
                            "finite_count": int(finite.size),
                            "nan_count": int(flat.size - finite.size),
                            "mean": float(finite.mean()) if finite.size else math.nan,
                            "std": float(finite.std(ddof=1)) if finite.size > 1 else math.nan,
                            "min": float(finite.min()) if finite.size else math.nan,
                            "max": float(finite.max()) if finite.size else math.nan,
                            "l2_norm_global": float(np.linalg.norm(finite)) if finite.size else math.nan,
                            **layer_stats,
                        }
                    )
                if "grade4_axis_component_vectors_by_layer" in rel_name or any(k in arrays for k in ["x_full", "x_content", "x_order", "x_order_orth"]):
                    grade4_rows.extend(grade4_npz_component_geometry(rel_name, arrays))
        except Exception as exc:
            inventory_rows.append({"source_file": rel_name, "size_bytes": int(path.stat().st_size), "array_count": 0, "arrays": "[]", "read_status": "error", "failure_code": "npz_read_error"})
            anomalies.append({"severity": "high", "artifact_family": "npz_vectors", "source_file": rel_name, "metric": "npz_read", "observed_value": repr(exc), "expected_rule": "NPZ should load read-only with numpy", "failure_code": "npz_read_error"})
    return (
        pd.DataFrame(inventory_rows),
        pd.DataFrame(summary_rows),
        pd.concat(norm_parts, ignore_index=True) if norm_parts else pd.DataFrame(),
        pd.concat(cosine_parts, ignore_index=True) if cosine_parts else pd.DataFrame(),
        pd.DataFrame(grade4_rows),
        anomalies,
    )


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else math.nan


def grade4_npz_component_geometry(source_file: str, arrays: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    axes = {k: v for k, v in arrays.items() if k in {"x_full", "x_content", "x_order", "x_order_orth"}}
    if not axes:
        return []
    rows = []
    bands = {"all": None, "middle": (0.35, 0.70), "late": (0.70, 1.0)}
    for axis_name, arr in axes.items():
        if arr.ndim < 2:
            continue
        n_layers = arr.shape[0]
        for band, frac in bands.items():
            if frac is None:
                sub = arr.reshape(n_layers, -1)
            else:
                start = max(0, int(math.floor(n_layers * frac[0])))
                end = min(n_layers, int(math.ceil(n_layers * frac[1])))
                sub = arr[start:end].reshape(max(0, end - start), -1)
            flat = sub.reshape(-1)
            row = {
                "source_file": source_file,
                "axis_name": axis_name,
                "band": band,
                "component_norm": float(np.linalg.norm(flat)) if flat.size else math.nan,
                "cos_to_x_full": safe_cosine(flat, axes["x_full"][start:end].reshape(-1) if frac is not None and "x_full" in axes else axes.get("x_full", flat).reshape(-1)) if "x_full" in axes else math.nan,
                "cos_to_x_content": safe_cosine(flat, axes["x_content"][start:end].reshape(-1) if frac is not None and "x_content" in axes else axes.get("x_content", flat).reshape(-1)) if "x_content" in axes else math.nan,
                "cos_to_x_order": safe_cosine(flat, axes["x_order"][start:end].reshape(-1) if frac is not None and "x_order" in axes else axes.get("x_order", flat).reshape(-1)) if "x_order" in axes else math.nan,
                "orthogonality_residual": math.nan,
            }
            if axis_name == "x_order_orth" and "x_content" in axes:
                content = axes["x_content"]
                if frac is not None:
                    start = max(0, int(math.floor(n_layers * frac[0])))
                    end = min(n_layers, int(math.ceil(n_layers * frac[1])))
                    content_flat = content[start:end].reshape(-1)
                else:
                    content_flat = content.reshape(-1)
                row["orthogonality_residual"] = abs(safe_cosine(flat, content_flat))
            rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Non-X state-space analysis from prompt_hidden_states.npz
# -----------------------------------------------------------------------------


def unavailable_state_space_outputs(failure_code: str, status: str = "not_available") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = {"source_file": "prompt_hidden_states.npz", "status": status, "failure_code": failure_code}
    return (
        pd.DataFrame([{**base, "condition": "", "layer": math.nan}]),
        pd.DataFrame([{**base, "condition_a": "", "condition_b": "", "layer": math.nan}]),
        pd.DataFrame([{**base, "layer": math.nan}]),
        pd.DataFrame([{**base, "layer": math.nan}]),
        pd.DataFrame([{**base, "question_index": math.nan, "condition": "", "layer": math.nan}]),
        pd.DataFrame([{**base, "metric_family": "non_x_state_space_geometry", "condition_a": "", "condition_b": "", "layer": math.nan}]),
    )


def scalar_int_from_npz(npz_obj, key: str, default: Optional[int] = None) -> Optional[int]:
    if key not in npz_obj:
        return default
    arr = np.asarray(npz_obj[key]).reshape(-1)
    if arr.size == 0:
        return default
    try:
        return int(arr[0])
    except Exception:
        return default


def condition_order_from_npz(npz_obj) -> List[str]:
    if "condition_order" not in npz_obj:
        return []
    arr = np.asarray(npz_obj["condition_order"]).reshape(-1)
    return [str(x) for x in arr.tolist()]


def centered_svd_pca(matrix: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2 or x.shape[1] == 0:
        return np.zeros((x.shape[0], 0)), np.zeros((0, x.shape[1])), np.zeros((0,), dtype=np.float64)
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((x.shape[0], 0)), np.zeros((0, x.shape[1])), np.zeros((0,), dtype=np.float64)
    k = max(1, min(int(n_components), vt.shape[0], x.shape[0]))
    denom = float(np.sum(s ** 2))
    explained = (s[:k] ** 2) / denom if denom > 0 else np.zeros((k,), dtype=np.float64)
    coords = u[:, :k] * s[:k]
    return coords, vt[:k], explained


def analyze_prompt_hidden_state_space(
    npz_files: Sequence[Path],
    root: Path,
    pca_components: int,
    max_pca_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    anomalies: List[Dict[str, Any]] = []
    path = next((p for p in npz_files if p.name == "prompt_hidden_states.npz"), None)
    if path is None:
        anomalies.append(
            {
                "severity": "medium",
                "artifact_family": "state_space_geometry",
                "source_file": "prompt_hidden_states.npz",
                "metric": "artifact_presence",
                "observed_value": "missing",
                "expected_rule": "prompt_hidden_states.npz should exist for non-X state-space analysis",
                "failure_code": "not_available_prompt_hidden_states",
            }
        )
        return (*unavailable_state_space_outputs("not_available_prompt_hidden_states"), anomalies)

    rel_name = rel_path(path, root)
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "hidden_states" not in npz:
                anomalies.append(
                    {
                        "severity": "high",
                        "artifact_family": "state_space_geometry",
                        "source_file": rel_name,
                        "metric": "hidden_states",
                        "observed_value": "missing",
                        "expected_rule": "prompt_hidden_states.npz must contain hidden_states",
                        "failure_code": "prompt_hidden_state_shape_invalid",
                    }
                )
                return (*unavailable_state_space_outputs("prompt_hidden_state_shape_invalid", "error"), anomalies)
            hidden = np.asarray(npz["hidden_states"])
            condition_order = condition_order_from_npz(npz)
            question_count = scalar_int_from_npz(npz, "question_count")
            manifest_layer_count = scalar_int_from_npz(npz, "layer_count")
    except Exception as exc:
        anomalies.append(
            {
                "severity": "high",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "npz_read",
                "observed_value": repr(exc),
                "expected_rule": "prompt_hidden_states.npz should load read-only with numpy",
                "failure_code": "prompt_hidden_state_read_error",
            }
        )
        return (*unavailable_state_space_outputs("prompt_hidden_state_read_error", "error"), anomalies)

    if hidden.ndim != 3 or not np.issubdtype(hidden.dtype, np.number):
        anomalies.append(
            {
                "severity": "high",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "hidden_states_shape",
                "observed_value": str(hidden.shape),
                "expected_rule": "hidden_states must have shape [question_count * condition_count, layer_count+1, hidden_size]",
                "failure_code": "prompt_hidden_state_shape_invalid",
            }
        )
        return (*unavailable_state_space_outputs("prompt_hidden_state_shape_invalid", "error"), anomalies)

    condition_count = len(condition_order)
    if condition_count <= 0 or question_count is None or question_count <= 0:
        anomalies.append(
            {
                "severity": "high",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "condition_order_or_question_count",
                "observed_value": f"condition_count={condition_count}, question_count={question_count}",
                "expected_rule": "condition_order and positive question_count are required",
                "failure_code": "prompt_hidden_condition_question_metadata_invalid",
            }
        )
        return (*unavailable_state_space_outputs("prompt_hidden_condition_question_metadata_invalid", "error"), anomalies)

    expected_rows = int(question_count) * int(condition_count)
    if int(hidden.shape[0]) != expected_rows:
        anomalies.append(
            {
                "severity": "high",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "row_count",
                "observed_value": int(hidden.shape[0]),
                "expected_rule": f"hidden_states rows should equal question_count * condition_count = {expected_rows}",
                "failure_code": "prompt_hidden_condition_question_mismatch",
            }
        )
        return (*unavailable_state_space_outputs("prompt_hidden_condition_question_mismatch", "error"), anomalies)

    if manifest_layer_count is not None and int(hidden.shape[1]) != int(manifest_layer_count) + 1:
        anomalies.append(
            {
                "severity": "high",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "layer_count",
                "observed_value": int(hidden.shape[1]),
                "expected_rule": f"hidden_states layer axis should equal layer_count + 1 = {int(manifest_layer_count) + 1}",
                "failure_code": "prompt_hidden_layer_count_mismatch",
            }
        )

    if condition_count <= 1:
        anomalies.append(
            {
                "severity": "medium",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "condition_count",
                "observed_value": condition_count,
                "expected_rule": "at least two conditions are needed for condition-distance geometry",
                "failure_code": "only_one_condition_available",
            }
        )
    if int(question_count) <= 1:
        anomalies.append(
            {
                "severity": "medium",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "question_count",
                "observed_value": int(question_count),
                "expected_rule": "at least two questions are needed for within-condition variance",
                "failure_code": "only_one_question_available",
            }
        )

    finite_mask = np.isfinite(hidden)
    nonfinite_ratio = float(1.0 - finite_mask.mean()) if hidden.size else 1.0
    if nonfinite_ratio > 0.01:
        anomalies.append(
            {
                "severity": "high",
                "artifact_family": "state_space_geometry",
                "source_file": rel_name,
                "metric": "nonfinite_hidden_state_fraction",
                "observed_value": nonfinite_ratio,
                "expected_rule": "nonfinite hidden state fraction should be <= 0.01",
                "failure_code": "nonfinite_hidden_state_dominance",
            }
        )
    if nonfinite_ratio > 0:
        fill = float(np.nanmean(hidden[np.isfinite(hidden)])) if np.isfinite(hidden).any() else 0.0
        hidden = np.nan_to_num(hidden.astype(np.float64, copy=False), nan=fill, posinf=fill, neginf=fill)
    else:
        hidden = hidden.astype(np.float64, copy=False)

    question_count = int(question_count)
    n_layers = int(hidden.shape[1])
    reshaped = hidden.reshape(question_count, condition_count, n_layers, int(hidden.shape[2]))

    centroid_rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    variance_rows: List[Dict[str, Any]] = []
    pca_summary_rows: List[Dict[str, Any]] = []
    pca_coord_rows: List[Dict[str, Any]] = []

    pca_components = max(1, int(pca_components))
    max_pca_rows = max(0, int(max_pca_rows))
    total_pca_coordinate_rows = int(n_layers * question_count * condition_count)
    if max_pca_rows <= 0 or total_pca_coordinate_rows <= max_pca_rows:
        pca_coordinate_rows_per_layer = question_count * condition_count
    else:
        pca_coordinate_rows_per_layer = max(1, int(max_pca_rows // max(1, n_layers)))

    for layer in range(n_layers):
        layer_data = reshaped[:, :, layer, :]
        centroids = layer_data.mean(axis=0)
        within_l2_by_condition = []
        within_cos_by_condition = []

        for cond_idx, condition in enumerate(condition_order):
            values = layer_data[:, cond_idx, :]
            centroid = centroids[cond_idx]
            deltas = values - centroid.reshape(1, -1)
            l2_vals = np.linalg.norm(deltas, axis=1)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 1e-12:
                value_norms = np.linalg.norm(values, axis=1)
                denom = value_norms * centroid_norm
                cos_sim = np.divide(values @ centroid, denom, out=np.full_like(value_norms, np.nan, dtype=np.float64), where=denom > 1e-12)
                cos_dist = 1.0 - cos_sim
            else:
                cos_dist = np.full((values.shape[0],), np.nan, dtype=np.float64)
            within_l2_by_condition.extend(float(x) for x in l2_vals if np.isfinite(x))
            within_cos_by_condition.extend(float(x) for x in cos_dist if np.isfinite(x))
            centroid_rows.append(
                {
                    "source_file": rel_name,
                    "condition": condition,
                    "layer": layer,
                    "n_questions": question_count,
                    "centroid_norm": centroid_norm,
                    "within_l2_mean": float(np.nanmean(l2_vals)) if l2_vals.size else math.nan,
                    "within_l2_std": float(np.nanstd(l2_vals, ddof=1)) if l2_vals.size > 1 else math.nan,
                    "within_cosine_distance_mean": float(np.nanmean(cos_dist)) if np.isfinite(cos_dist).any() else math.nan,
                    "within_cosine_distance_std": float(np.nanstd(cos_dist, ddof=1)) if np.isfinite(cos_dist).sum() > 1 else math.nan,
                    "status": "computed",
                    "failure_code": "",
                }
            )

        layer_between_l2 = []
        layer_between_cos = []
        for a_idx, condition_a in enumerate(condition_order):
            for b_idx, condition_b in enumerate(condition_order):
                ca = centroids[a_idx]
                cb = centroids[b_idx]
                l2_dist = float(np.linalg.norm(ca - cb))
                cos_sim = safe_cosine(ca, cb)
                cos_dist = 1.0 - cos_sim if math.isfinite(cos_sim) else math.nan
                if a_idx < b_idx:
                    layer_between_l2.append(l2_dist)
                    if math.isfinite(cos_dist):
                        layer_between_cos.append(cos_dist)
                distance_rows.append(
                    {
                        "source_file": rel_name,
                        "layer": layer,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "centroid_l2_distance": l2_dist,
                        "centroid_cosine_distance": cos_dist,
                        "centroid_cosine_similarity": cos_sim,
                        "status": "computed",
                        "failure_code": "",
                    }
                )

        within_l2_mean = float(np.mean(within_l2_by_condition)) if within_l2_by_condition else math.nan
        between_l2_mean = float(np.mean(layer_between_l2)) if layer_between_l2 else math.nan
        variance_rows.append(
            {
                "source_file": rel_name,
                "layer": layer,
                "within_l2_mean_all_conditions": within_l2_mean,
                "between_centroid_l2_mean": between_l2_mean,
                "between_over_within_l2_ratio": between_l2_mean / max(within_l2_mean, 1e-12) if math.isfinite(within_l2_mean) and math.isfinite(between_l2_mean) else math.nan,
                "within_cosine_distance_mean_all_conditions": float(np.mean(within_cos_by_condition)) if within_cos_by_condition else math.nan,
                "between_centroid_cosine_distance_mean": float(np.mean(layer_between_cos)) if layer_between_cos else math.nan,
                "condition_count": condition_count,
                "question_count": question_count,
                "status": "computed",
                "failure_code": "",
            }
        )

        layer_matrix = layer_data.reshape(question_count * condition_count, -1)
        coords, _basis, explained = centered_svd_pca(layer_matrix, pca_components)
        if explained.size == 0:
            anomalies.append(
                {
                    "severity": "medium",
                    "artifact_family": "state_space_geometry",
                    "source_file": rel_name,
                    "metric": "layerwise_pca",
                    "observed_value": f"layer={layer}",
                    "expected_rule": "layerwise PCA should compute on centered prompt hidden states",
                    "failure_code": "state_space_pca_failed",
                }
            )
        pca_summary_rows.append(
            {
                "source_file": rel_name,
                "layer": layer,
                "pc1_explained_variance": float(explained[0]) if explained.size >= 1 else math.nan,
                "pc2_explained_variance": float(explained[1]) if explained.size >= 2 else math.nan,
                "pc3_explained_variance": float(explained[2]) if explained.size >= 3 else math.nan,
                "cumulative_explained_variance": float(np.sum(explained)) if explained.size else math.nan,
                "pca_component_count": int(explained.size),
                "sample_count": int(layer_matrix.shape[0]),
                "status": "computed" if explained.size else "not_available",
                "failure_code": "" if explained.size else "state_space_pca_failed",
            }
        )
        if coords.size:
            coord_count = coords.shape[0]
            if coord_count <= pca_coordinate_rows_per_layer:
                selected = np.arange(coord_count)
            else:
                selected = np.linspace(0, coord_count - 1, num=pca_coordinate_rows_per_layer, dtype=int)
            for row_idx in selected:
                q_idx = int(row_idx // condition_count)
                cond_idx = int(row_idx % condition_count)
                row = {
                    "source_file": rel_name,
                    "question_index": q_idx,
                    "condition": condition_order[cond_idx],
                    "layer": layer,
                    "pc1": float(coords[row_idx, 0]) if coords.shape[1] >= 1 else math.nan,
                    "pc2": float(coords[row_idx, 1]) if coords.shape[1] >= 2 else math.nan,
                    "pc3": float(coords[row_idx, 2]) if coords.shape[1] >= 3 else math.nan,
                    "status": "computed",
                    "failure_code": "",
                }
                pca_coord_rows.append(row)

    distance_df = pd.DataFrame(distance_rows)
    variance_df = pd.DataFrame(variance_rows)
    if not distance_df.empty:
        non_self = distance_df[distance_df["condition_a"].astype(str) != distance_df["condition_b"].astype(str)].copy()
        non_self["rank_by_l2"] = pd.to_numeric(non_self["centroid_l2_distance"], errors="coerce").rank(method="dense", ascending=False)
        non_self["rank_by_cosine"] = pd.to_numeric(non_self["centroid_cosine_distance"], errors="coerce").rank(method="dense", ascending=False)
        ratio = variance_df[["layer", "between_over_within_l2_ratio"]] if "between_over_within_l2_ratio" in variance_df.columns else pd.DataFrame()
        if not ratio.empty:
            non_self = non_self.merge(ratio, on="layer", how="left")
        peaks_df = non_self.sort_values(["rank_by_l2", "rank_by_cosine"], ascending=[True, True]).copy()
        peaks_df.insert(0, "metric_family", "non_x_state_space_geometry")
    else:
        peaks_df = pd.DataFrame([{ "source_file": rel_name, "metric_family": "non_x_state_space_geometry", "status": "not_available", "failure_code": "not_available_state_space_distance_rows" }])

    return (
        pd.DataFrame(centroid_rows),
        distance_df,
        variance_df,
        pd.DataFrame(pca_summary_rows),
        pd.DataFrame(pca_coord_rows),
        peaks_df,
        anomalies,
    )


# -----------------------------------------------------------------------------
# Specialized tables
# -----------------------------------------------------------------------------


def table_row(source_file: str, metric: str, value: Any = math.nan, status: str = "computed", failure_code: str = "", **kwargs: Any) -> Dict[str, Any]:
    row = {"source_file": source_file, "metric": metric, "value": value, "n": kwargs.pop("n", math.nan), "status": status, "failure_code": failure_code}
    row.update(kwargs)
    return row


def read_small_csv(root: Path, name: str, anomalies: Optional[List[Dict[str, Any]]] = None, family: str = "other") -> pd.DataFrame:
    path = root / name
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.stat().st_size > SUMMARY_FILE_MAX_BYTES:
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception as exc:
        if anomalies is not None:
            anomalies.append({"severity": "high", "artifact_family": family, "source_file": name, "metric": "csv_read", "observed_value": repr(exc), "expected_rule": "small CSV should parse", "failure_code": "csv_read_error"})
        return pd.DataFrame()


def numeric_long_from_file(root: Path, name: str, metric_filter: Optional[Sequence[str]] = None, **context_defaults: Any) -> pd.DataFrame:
    df = read_small_csv(root, name)
    if df.empty:
        return pd.DataFrame([table_row(name, "", status="not_available", failure_code="not_available_artifact", **context_defaults)])
    rows = []
    context_cols = [c for c in ["condition", "control_condition", "base_condition", "layer_band", "readout_layer_band", "intervention_layer_band", "axis_name", "alpha", "alpha_abs", "sign_name", "module", "layer", "metric", "criterion", "status", "failure_code"] if c in df.columns]
    reserved_context_names = {"metric", "status", "failure_code", "value", "n", "source_file"}
    cols = metric_filter or [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
    for _, r in df.iterrows():
        ctx = {
            (f"context_{c}" if c in reserved_context_names else c): r.get(c)
            for c in context_cols
        }
        ctx.update(context_defaults)
        for col in cols:
            if col not in df.columns:
                continue
            val = safe_float(r.get(col))
            if not math.isfinite(val):
                continue
            rows.append(table_row(name, col, val, n=1, **ctx))
    return pd.DataFrame(rows) if rows else pd.DataFrame([table_row(name, "", status="not_available", failure_code="not_available_numeric_rows", **context_defaults)])


def build_specialized_tables(root: Path, output_dir: Path, anomalies: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    tables["grade3_geometry_overview.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "middle_layer_condition_summary.csv"),
            numeric_long_from_file(root, "question_level_middle_layer_summary.csv"),
            numeric_long_from_file(root, "layerwise_geometry_summary.csv"),
        ],
        ignore_index=True,
    )
    tables["grade3_specificity_control_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "paired_target_vs_control_tests.csv"),
            numeric_long_from_file(root, "geometry_specificity_summary.csv"),
            numeric_long_from_file(root, "layerwise_fdr_target_vs_control.csv"),
        ],
        ignore_index=True,
    )
    tables["grade3_null_baseline_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "null_vector_baseline_summary.csv"),
            numeric_long_from_file(root, "pca_baseline_projection_summary.csv"),
            numeric_long_from_file(root, "null_hypothesis_hardening_summary.csv"),
        ],
        ignore_index=True,
    )
    tables["grade3_causal_symmetry_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "causal_symmetry_score_summary.csv"),
            numeric_long_from_file(root, "causal_bidirectional_symmetry_summary.csv"),
            numeric_long_from_file(root, "causal_alpha_scaling_summary.csv"),
            numeric_long_from_file(root, "causal_intervention_middle_layer_summary.csv"),
        ],
        ignore_index=True,
    )
    tables["grade3_behavior_random_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "behavior_random_p95_gate.csv"),
            numeric_long_from_file(root, "behavioral_control_axis_hard_random_summary.csv"),
            numeric_long_from_file(root, "behavioral_control_axis_threshold_eval.csv"),
            numeric_long_from_file(root, "quality_adjusted_behavior_summary.csv"),
            numeric_long_from_file(root, "internal_visible_coupling_summary.csv"),
        ],
        ignore_index=True,
    )
    tables["grade3_architecture_module_matrix.csv"] = numeric_long_from_file(root, "architecture_module_delta_summary.csv", metric_filter=["l2_distance_to_reference", "mean_abs_delta", "max_abs_delta", "projection_fraction_on_arch_vector_x_loo", "direction_cosine_with_arch_vector_x_loo"])
    tables["grade3_unit_candidate_matrix.csv"] = build_unit_candidate_matrix(root)
    tables["grade3_trajectory_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "generation_middle_layer_summary.csv"),
            numeric_long_from_file(root, "causal_intervention_middle_layer_summary.csv"),
            numeric_long_from_file(root, "dynamic_trajectory_summary.csv"),
        ],
        ignore_index=True,
    )

    tables["grade4_component_norm_matrix.csv"] = numeric_long_from_file(root, "grade4_axis_component_norm_summary.csv")
    tables["grade4_component_projection_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "grade4_axis_projection_geometry_summary.csv"),
            numeric_long_from_file(root, "grade4_axis_projection_geometry_raw.csv"),
        ],
        ignore_index=True,
    )
    tables["grade4_component_causal_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "grade4_axis_component_causal_projection_summary.csv"),
            numeric_long_from_file(root, "grade4_axis_component_causal_symmetry_summary.csv"),
        ],
        ignore_index=True,
    )
    tables["grade4_component_alpha_matrix.csv"] = numeric_long_from_file(root, "grade4_axis_component_causal_alpha_scaling_summary.csv")
    tables["grade4_component_rank_matrix.csv"] = numeric_long_from_file(root, "grade4_axis_component_causal_rank_summary.csv")
    tables["grade4_axis_cross_correlation.csv"] = build_grade4_axis_cross_correlation(root)
    tables["grade4_layerwise_component_readout_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "grade4_axis_layerwise_component_readout_summary.csv"),
            numeric_long_from_file(root, "grade4_axis_layerwise_order_birth_summary.csv"),
        ],
        ignore_index=True,
    )
    tables["grade4_generation_component_readout_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "grade4_axis_generation_component_readout_summary.csv"),
            numeric_long_from_file(root, "grade4_axis_generation_component_persistence_comparison.csv"),
        ],
        ignore_index=True,
    )
    tables["grade4_natscale_closeout_matrix.csv"] = pd.concat(
        [
            numeric_long_from_file(root, "grade4_axis_component_causal_natscale_closeout.csv"),
            numeric_long_from_file(root, "grade4_axis_component_causal_data_quality.csv"),
        ],
        ignore_index=True,
    )
    tables["sae_order_feature_contrast_matrix.csv"] = build_sae_order_feature_matrix(root)
    tables["sae_reconstruction_quality_matrix.csv"] = numeric_long_from_file(root, "sae_reconstruction_quality.csv")
    tables["sae_model_compatibility_matrix.csv"] = numeric_long_from_file(root, "sae_model_compatibility.csv")

    for name, df in tables.items():
        write_csv(output_dir / name, df)
    return tables


def build_sae_order_feature_matrix(root: Path) -> pd.DataFrame:
    """Build a compact feature-candidate table from SAE order/readout artifacts."""
    source_name = "sae_order_feature_triage.csv" if (root / "sae_order_feature_triage.csv").exists() else "sae_order_feature_contrast.csv"
    path = root / source_name
    if not path.exists():
        return pd.DataFrame([table_row("sae_order_feature_contrast.csv", "", status="not_available", failure_code="not_available_artifact")])
    try:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception as exc:
        return pd.DataFrame([table_row(source_name, "", status="error", failure_code=f"csv_read_error:{repr(exc)}")])

    if df.empty:
        return pd.DataFrame([table_row(source_name, "", status="not_available", failure_code="not_available_empty_artifact")])

    numeric_cols = [
        c for c in [
            "order_specific_score",
            "x_order_orth_component_delta",
            "abs_x_order_orth_component_delta",
            "x_content_component_delta",
            "abs_x_content_component_delta",
            "order_minus_content_abs_component_delta",
            "order_over_content_abs_ratio",
            "target_minus_sentence_shuffle_prompt_delta",
            "target_minus_sentence_shuffle_generation_mean_activation",
            "target_prompt_mean_activation_delta",
            "sentence_shuffle_prompt_mean_activation_delta",
            "target_generation_mean_activation",
            "sentence_shuffle_generation_mean_activation",
            "target_generation_late_minus_early_activation",
            "sentence_shuffle_generation_late_minus_early_activation",
        ]
        if c in df.columns
    ]
    if not numeric_cols:
        return pd.DataFrame([table_row(source_name, "", status="not_available", failure_code="not_available_numeric_rows")])

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_col = "order_specific_score" if "order_specific_score" in df.columns else "abs_x_order_orth_component_delta"
    if sort_col not in df.columns:
        sort_col = numeric_cols[0]

    top = df.sort_values(sort_col, ascending=False).head(500).copy()
    rows: List[Dict[str, Any]] = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        base = {
            "sae_name": row.get("sae_name", ""),
            "sae_spec_index": row.get("sae_spec_index", math.nan),
            "layer": row.get("layer", math.nan),
            "feature_index": row.get("feature_index", math.nan),
            "interpretation_status": row.get("interpretation_status", ""),
            "rank_by_abs_order_specific_score": rank,
        }
        for metric in numeric_cols:
            value = safe_float(row.get(metric))
            if math.isfinite(value):
                rows.append(table_row(source_name, metric, value, n=1, **base))
    return pd.DataFrame(rows) if rows else pd.DataFrame([table_row(source_name, "", status="not_available", failure_code="not_available_numeric_rows")])


def build_unit_candidate_matrix(root: Path) -> pd.DataFrame:
    path = root / "architecture_top_changed_units.csv"
    if not path.exists():
        return pd.DataFrame([table_row("architecture_top_changed_units.csv", "", status="not_available", failure_code="not_available_artifact")])
    cols = ["condition", "layer", "module", "unit_index", "rank_by_abs_delta", "abs_delta", "delta"]
    try:
        header = read_header(path)
        usecols = [c for c in cols if c in header]
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS, encoding="utf-8", encoding_errors="replace"):
            if "condition" in chunk.columns:
                chunk = chunk[chunk["condition"].astype(str).eq("target")]
            if "rank_by_abs_delta" in chunk.columns:
                chunk["rank_by_abs_delta"] = pd.to_numeric(chunk["rank_by_abs_delta"], errors="coerce")
                chunk = chunk[chunk["rank_by_abs_delta"] <= 10]
            if chunk.empty:
                continue
            chunk["abs_delta"] = pd.to_numeric(chunk.get("abs_delta", np.nan), errors="coerce")
            grouped = chunk.groupby([c for c in ["layer", "module", "unit_index"] if c in chunk.columns], dropna=False).agg(
                n=("abs_delta", "count"),
                value=("abs_delta", "max"),
                mean_abs_delta=("abs_delta", "mean"),
            ).reset_index()
            parts.append(grouped)
        if not parts:
            return pd.DataFrame([table_row("architecture_top_changed_units.csv", "abs_delta", status="not_available", failure_code="not_available_numeric_rows")])
        raw = pd.concat(parts, ignore_index=True)
        final = raw.groupby([c for c in ["layer", "module", "unit_index"] if c in raw.columns], dropna=False).agg(
            n=("n", "sum"),
            value=("value", "max"),
            mean_abs_delta=("mean_abs_delta", "mean"),
        ).reset_index()
        final.insert(0, "source_file", "architecture_top_changed_units.csv")
        final.insert(1, "metric", "target_top_unit_abs_delta")
        final["status"] = "computed"
        final["failure_code"] = ""
        return final.sort_values(["n", "value"], ascending=[False, False]).head(5000)
    except Exception as exc:
        return pd.DataFrame([table_row("architecture_top_changed_units.csv", "abs_delta", status="error", failure_code=f"unit_candidate_error:{repr(exc)}")])


def build_grade4_axis_cross_correlation(root: Path) -> pd.DataFrame:
    path = root / "grade4_axis_component_vectors_by_layer.npz"
    if not path.exists():
        return pd.DataFrame([table_row("grade4_axis_component_vectors_by_layer.npz", "axis_cosine", status="not_available", failure_code="not_available_artifact")])
    try:
        rows = []
        with np.load(path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.keys() if k in {"x_full", "x_content", "x_order", "x_order_orth"}}
            for a_name, a in arrays.items():
                for b_name, b in arrays.items():
                    rows.append(table_row(path.name, "axis_cosine", safe_cosine(a, b), axis_name=a_name, component=b_name, status="computed"))
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([table_row(path.name, "axis_cosine", status="error", failure_code=f"axis_cross_correlation_error:{repr(exc)}")])


# -----------------------------------------------------------------------------
# Anomalies
# -----------------------------------------------------------------------------


def add_anomaly(rows: List[Dict[str, Any]], severity: str, family: str, source_file: str, metric: str, observed_value: Any, expected_rule: str, failure_code: str) -> None:
    rows.append(
        {
            "severity": severity,
            "artifact_family": family,
            "source_file": source_file,
            "metric": metric,
            "observed_value": observed_value,
            "expected_rule": expected_rule,
            "failure_code": failure_code,
        }
    )


def detect_anomalies(root: Path, artifact_df: pd.DataFrame, summary_numeric: pd.DataFrame, specialized_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    names = set(artifact_df["source_file"].astype(str)) if not artifact_df.empty and "source_file" in artifact_df.columns else set()
    gate4_detected = any(n.startswith("grade4_axis_") for n in names)
    for key in KEY_GRADE3_ARTIFACTS:
        if key not in names:
            add_anomaly(rows, "medium", "run_validity", key, "artifact_presence", "missing", "key Grade3 artifact should be present for complete analysis", "missing_key_grade3_artifact")
    if gate4_detected:
        for key in KEY_GRADE4_ARTIFACTS:
            if key not in names:
                add_anomaly(rows, "medium", "grade4_components", key, "artifact_presence", "missing", "Gate4 package should include Grade4 component artifact", "missing_key_grade4_artifact")
    if "prompt_budget_overflow_warnings.csv" in names:
        add_anomaly(rows, "high", "run_validity", "prompt_budget_overflow_warnings.csv", "prompt_budget", "present", "prompt budget overflow warnings should be absent for clean question-conditioned runs", "prompt_budget_overflow_present")
    manifest_path = root / "red_team_input_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if bool(manifest.get("decoder_layer_count_mismatch")):
                add_anomaly(rows, "high", "run_validity", "red_team_input_manifest.json", "decoder_layer_count_mismatch", True, "decoder layer count should match expected model config", "decoder_count_mismatch")
        except Exception as exc:
            add_anomaly(rows, "medium", "run_validity", "red_team_input_manifest.json", "manifest_parse", repr(exc), "manifest should parse as JSON", "manifest_parse_error")
            manifest = {}
    if bool(manifest.get("sae_feature_analysis_enabled", False)):
        for key in ["sae_model_compatibility.csv", "sae_reconstruction_quality.csv", "sae_order_feature_contrast.csv"]:
            if key not in names:
                add_anomaly(rows, "high", "sae_features", key, "artifact_presence", "missing", "SAE feature analysis was enabled, so SAE bridge artifact should be present", f"missing_required_sae_artifact:{key}")
        compat = root / "sae_model_compatibility.csv"
        if compat.exists():
            try:
                cdf = pd.read_csv(compat)
                if "status" in cdf.columns:
                    bad = cdf[~cdf["status"].astype(str).str.lower().isin(["computed", "ok", "available"])]
                    if len(bad):
                        add_anomaly(rows, "high", "sae_features", "sae_model_compatibility.csv", "status", int(len(bad)), "all requested SAE layers should report status=computed/ok/available", "sae_compatibility_not_computed")
                if "hidden_size" in cdf.columns and "sae_d_in" in cdf.columns:
                    h = pd.to_numeric(cdf["hidden_size"], errors="coerce")
                    d = pd.to_numeric(cdf["sae_d_in"], errors="coerce")
                    mismatch = cdf[(h.notna()) & (d.notna()) & (h != d)]
                    if len(mismatch):
                        add_anomaly(rows, "high", "sae_features", "sae_model_compatibility.csv", "sae_d_in_vs_hidden_size", int(len(mismatch)), "SAE input dimension should match model hidden size", "sae_hidden_size_mismatch")
            except Exception as exc:
                add_anomaly(rows, "medium", "sae_features", "sae_model_compatibility.csv", "csv_read", repr(exc), "SAE compatibility CSV should parse", "sae_compatibility_read_error")
    integrity = root / "analysis_notes" / "extracted_narrative_columns" / "numeric_integrity_check.csv"
    if integrity.exists():
        try:
            df = pd.read_csv(integrity)
            bad = df[df.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("fail")]
            if len(bad):
                add_anomaly(rows, "high", "quarantine_integrity", rel_path(integrity, root), "numeric_integrity", len(bad), "numeric integrity checks should not fail", "numeric_integrity_fail")
        except Exception:
            add_anomaly(rows, "medium", "quarantine_integrity", rel_path(integrity, root), "numeric_integrity", "unreadable", "numeric integrity CSV should parse", "numeric_integrity_unreadable")
    quarantine = root / "analysis_notes" / "extracted_narrative_columns" / "quarantine_index.csv"
    if quarantine.exists():
        try:
            df = pd.read_csv(quarantine)
            meaningful = df[~df.get("action", pd.Series(dtype=str)).astype(str).isin(["none", "no_quarantine_needed", ""])]
            if len(meaningful):
                add_anomaly(rows, "medium", "quarantine_integrity", rel_path(quarantine, root), "quarantine_rows", len(meaningful), "quarantine rows should be reviewed, not hidden", "quarantine_contains_removed_rows")
        except Exception:
            pass
    threshold = root / "behavioral_control_axis_threshold_eval.csv"
    hard = root / "behavioral_control_axis_hard_random_summary.csv"
    if threshold.exists():
        try:
            t = pd.read_csv(threshold)
            p95_rows = t[t.get("criterion", pd.Series(dtype=str)).astype(str).eq("plus_x_beats_random_p95")]
            if len(p95_rows):
                metric_name = str(p95_rows.iloc[0].get("metric_name", ""))
                if "p95" not in metric_name.lower():
                    add_anomaly(rows, "high", "behavioral_control", "behavioral_control_axis_threshold_eval.csv", "p95_metric_name", metric_name, "p95 criterion should use p95-derived metric", "behavior_p95_metric_mismatch")
        except Exception:
            pass
    if hard.exists():
        try:
            h = pd.read_csv(hard)
            if "mean_lift_over_random_p95" in h.columns:
                low = h[pd.to_numeric(h["mean_lift_over_random_p95"], errors="coerce") <= 0]
                if len(low):
                    add_anomaly(rows, "medium", "behavioral_control", "behavioral_control_axis_hard_random_summary.csv", "mean_lift_over_random_p95", int(len(low)), "random p95 lift rows should be inspected when <= 0", "below_random_p95")
        except Exception:
            pass
    quality = root / "quality_adjusted_behavior_summary.csv"
    if quality.exists():
        try:
            q = pd.read_csv(quality)
            if "degeneration_rate" in q.columns:
                high = q[pd.to_numeric(q["degeneration_rate"], errors="coerce") >= 0.5]
                if len(high):
                    add_anomaly(rows, "medium", "behavioral_control", "quality_adjusted_behavior_summary.csv", "degeneration_rate", int(len(high)), "degeneration_rate should stay below 0.5 for clean visible behavior evidence", "quality_degeneration_high")
        except Exception:
            pass
    spec = root / "geometry_specificity_summary.csv"
    mid = root / "middle_layer_condition_summary.csv"
    if spec.exists() and mid.exists():
        try:
            s = pd.read_csv(spec)
            m = pd.read_csv(mid)
            target = m[m.get("condition", pd.Series(dtype=str)).astype(str).eq("target")]
            target_proj = safe_float(target.iloc[0].get("projection_fraction_on_vector_x_loo_mean")) if len(target) else math.nan
            if math.isfinite(target_proj) and "control_mean_projection" in s.columns:
                close = s[pd.to_numeric(s["control_mean_projection"], errors="coerce") > 0.8 * target_proj]
                if len(close):
                    add_anomaly(rows, "medium", "specificity_controls", "geometry_specificity_summary.csv", "control_mean_projection", int(len(close)), "control projection should remain below 0.8 * target projection", "control_projection_close_to_target")
        except Exception:
            pass
    paired = root / "paired_target_vs_control_tests.csv"
    fdr = root / "layerwise_fdr_target_vs_control.csv"
    if paired.exists() and not fdr.exists():
        add_anomaly(rows, "medium", "specificity_controls", "layerwise_fdr_target_vs_control.csv", "fdr_presence", "missing", "FDR table should exist when paired tests exist", "fdr_absent_with_paired_tests")
    if "gemma3_attractor_analysis.py" in names:
        add_anomaly(rows, "low", "architecture_units", "gemma3_attractor_analysis.py", "step_layer_mapping", "not_used", "primary analyzer must not merge generation step as decoder layer", "step_to_layer_conflation_not_allowed")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots / HTML
# -----------------------------------------------------------------------------


def save_plot(path: Path, title: str, source_file: str, rows: List[Dict[str, Any]], plot_fn) -> None:
    if plt is None:
        return
    ensure_dir(path.parent)
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_fn(ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rows.append({"plot_file": path.as_posix(), "title": title, "source_file": source_file, "status": "written", "failure_code": ""})
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        rows.append({"plot_file": path.as_posix(), "title": title, "source_file": source_file, "status": "error", "failure_code": repr(exc)})


def simple_heatmap(ax, data: pd.DataFrame, x: str, y: str, value: str) -> None:
    pivot = data.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def generate_plots(
    root: Path,
    output_dir: Path,
    max_plot_rows: int,
    npz_layer_norms: pd.DataFrame,
    npz_cosines: pd.DataFrame,
    state_space_distances: Optional[pd.DataFrame] = None,
    state_space_variance: Optional[pd.DataFrame] = None,
    state_space_pca_coords: Optional[pd.DataFrame] = None,
    state_space_peaks: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    plot_rows: List[Dict[str, Any]] = []
    plots_dir = output_dir / "plots"
    # Geometry plots.
    layerwise = read_small_csv(root, "layerwise_geometry_summary.csv")
    if not layerwise.empty and {"condition", "layer", "mean_projection_fraction_on_vector_x_loo"}.issubset(layerwise.columns):
        target = layerwise[layerwise["condition"].astype(str).eq("target")].copy()
        target["layer"] = pd.to_numeric(target["layer"], errors="coerce")
        target["mean_projection_fraction_on_vector_x_loo"] = pd.to_numeric(target["mean_projection_fraction_on_vector_x_loo"], errors="coerce")
        save_plot(plots_dir / "layerwise_target_projection.png", "mean_projection_fraction_on_vector_x_loo from layerwise_geometry_summary.csv", "layerwise_geometry_summary.csv", plot_rows, lambda ax: (ax.plot(target["layer"], target["mean_projection_fraction_on_vector_x_loo"], marker="o"), ax.set_xlabel("layer"), ax.set_ylabel("mean_projection_fraction_on_vector_x_loo")))
        save_plot(plots_dir / "layerwise_condition_projection_heatmap.png", "mean_projection_fraction_on_vector_x_loo by condition/layer from layerwise_geometry_summary.csv", "layerwise_geometry_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, layerwise, "layer", "condition", "mean_projection_fraction_on_vector_x_loo"))
    paired = read_small_csv(root, "paired_target_vs_control_tests.csv")
    if not paired.empty and {"control_condition", "target_minus_control_mean", "metric"}.issubset(paired.columns):
        sub = paired[paired["metric"].astype(str).eq("mean_projection_fraction_on_vector_x_loo")].copy()
        sub["target_minus_control_mean"] = pd.to_numeric(sub["target_minus_control_mean"], errors="coerce")
        save_plot(plots_dir / "specificity_control_gap_bar.png", "target_minus_control_mean from paired_target_vs_control_tests.csv", "paired_target_vs_control_tests.csv", plot_rows, lambda ax: (ax.bar(sub["control_condition"].astype(str), sub["target_minus_control_mean"]), ax.set_xlabel("control_condition"), ax.set_ylabel("target_minus_control_mean"), ax.tick_params(axis="x", rotation=35)))
    null = read_small_csv(root, "null_vector_baseline_summary.csv")
    if not null.empty:
        metric_cols = [c for c in ["observed_target_projection_mean", "null_mean", "observed_minus_null_mean"] if c in null.columns]
        if metric_cols:
            vals = [safe_float(null.iloc[0].get(c)) for c in metric_cols]
            save_plot(plots_dir / "null_baseline_comparison.png", "null vector baseline metrics from null_vector_baseline_summary.csv", "null_vector_baseline_summary.csv", plot_rows, lambda ax: (ax.bar(metric_cols, vals), ax.set_ylabel("value"), ax.tick_params(axis="x", rotation=35)))
    causal_sym = read_small_csv(root, "causal_symmetry_score_summary.csv")
    if not causal_sym.empty and {"layer_band", "alpha", "symmetry_pass_rate"}.issubset(causal_sym.columns):
        save_plot(plots_dir / "causal_symmetry_heatmap.png", "symmetry_pass_rate from causal_symmetry_score_summary.csv", "causal_symmetry_score_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, causal_sym, "alpha", "layer_band", "symmetry_pass_rate"))
    alpha = read_small_csv(root, "causal_alpha_scaling_summary.csv")
    if not alpha.empty and {"layer_band", "sign_name", "alpha_projection_slope"}.issubset(alpha.columns):
        save_plot(plots_dir / "causal_alpha_slope_heatmap.png", "alpha_projection_slope from causal_alpha_scaling_summary.csv", "causal_alpha_scaling_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, alpha, "sign_name", "layer_band", "alpha_projection_slope"))
    behavior = read_small_csv(root, "behavior_random_p95_gate.csv")
    if not behavior.empty and {"layer_band", "alpha", "win_rate_vs_random_p95"}.issubset(behavior.columns):
        save_plot(plots_dir / "behavior_random_p95_heatmap.png", "win_rate_vs_random_p95 from behavior_random_p95_gate.csv", "behavior_random_p95_gate.csv", plot_rows, lambda ax: simple_heatmap(ax, behavior, "alpha", "layer_band", "win_rate_vs_random_p95"))
    gen = read_small_csv(root, "generation_middle_layer_summary.csv")
    if not gen.empty and {"condition", "mean_projection_fraction_on_vector_x_loo"}.issubset(gen.columns):
        save_plot(plots_dir / "generation_trajectory_projection.png", "mean_projection_fraction_on_vector_x_loo from generation_middle_layer_summary.csv", "generation_middle_layer_summary.csv", plot_rows, lambda ax: (ax.bar(gen["condition"].astype(str), pd.to_numeric(gen["mean_projection_fraction_on_vector_x_loo"], errors="coerce")), ax.set_xlabel("condition"), ax.set_ylabel("mean_projection_fraction_on_vector_x_loo"), ax.tick_params(axis="x", rotation=35)))
    causal_mid = read_small_csv(root, "causal_intervention_middle_layer_summary.csv")
    if not causal_mid.empty and {"layer_band", "alpha", "mean_projection_fraction_on_vector_x_loo"}.issubset(causal_mid.columns):
        save_plot(plots_dir / "causal_trajectory_projection.png", "mean_projection_fraction_on_vector_x_loo from causal_intervention_middle_layer_summary.csv", "causal_intervention_middle_layer_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, causal_mid, "alpha", "layer_band", "mean_projection_fraction_on_vector_x_loo"))
    arch = read_small_csv(root, "architecture_module_delta_summary.csv")
    if not arch.empty and {"module", "layer", "mean_abs_delta"}.issubset(arch.columns):
        arch = arch.head(max_plot_rows)
        save_plot(plots_dir / "architecture_module_delta_heatmap.png", "mean_abs_delta from architecture_module_delta_summary.csv", "architecture_module_delta_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, arch, "layer", "module", "mean_abs_delta"))
    overlap = read_small_csv(root, "architecture_target_vs_control_overlap.csv")
    if not overlap.empty and {"module", "layer", "target_control_top_unit_jaccard"}.issubset(overlap.columns):
        overlap = overlap.head(max_plot_rows)
        save_plot(plots_dir / "architecture_control_overlap_heatmap.png", "target_control_top_unit_jaccard from architecture_target_vs_control_overlap.csv", "architecture_target_vs_control_overlap.csv", plot_rows, lambda ax: simple_heatmap(ax, overlap, "layer", "module", "target_control_top_unit_jaccard"))
    units = build_unit_candidate_matrix(root)
    if not units.empty and {"layer", "value", "n"}.issubset(units.columns):
        save_plot(plots_dir / "top_unit_candidate_scatter.png", "target_top_unit_abs_delta from architecture_top_changed_units.csv", "architecture_top_changed_units.csv", plot_rows, lambda ax: (ax.scatter(pd.to_numeric(units["layer"], errors="coerce"), pd.to_numeric(units["value"], errors="coerce"), s=np.clip(pd.to_numeric(units["n"], errors="coerce").fillna(1), 1, 50)), ax.set_xlabel("layer"), ax.set_ylabel("target_top_unit_abs_delta")))
    if not npz_layer_norms.empty and {"source_file", "array_name", "layer", "layer_l2_norm"}.issubset(npz_layer_norms.columns):
        sub = npz_layer_norms[npz_layer_norms["array_name"].astype(str).isin(["vector_x_by_layer", "x_full", "x_content", "x_order", "x_order_orth"])]
        if not sub.empty:
            save_plot(plots_dir / "npz_vector_norm_by_layer.png", "layer_l2_norm from NPZ vectors", "npz_layer_norms.csv", plot_rows, lambda ax: [ax.plot(g["layer"], g["layer_l2_norm"], marker="o", label=str(name)) for name, g in sub.groupby("array_name")] or (ax.set_xlabel("layer"), ax.set_ylabel("layer_l2_norm"), ax.legend()))
    if not npz_cosines.empty and {"array_name", "layer_to", "adjacent_layer_cosine"}.issubset(npz_cosines.columns):
        sub = npz_cosines[npz_cosines["array_name"].astype(str).isin(["vector_x_by_layer", "x_full", "x_content", "x_order", "x_order_orth"])]
        if not sub.empty:
            save_plot(plots_dir / "npz_adjacent_layer_cosine.png", "adjacent_layer_cosine from NPZ vectors", "npz_adjacent_layer_cosines.csv", plot_rows, lambda ax: [ax.plot(g["layer_to"], g["adjacent_layer_cosine"], marker="o", label=str(name)) for name, g in sub.groupby("array_name")] or (ax.set_xlabel("layer_to"), ax.set_ylabel("adjacent_layer_cosine"), ax.legend()))
    g4norm = read_small_csv(root, "grade4_axis_component_norm_summary.csv")
    if not g4norm.empty and {"band", "order_orth_norm"}.intersection(g4norm.columns):
        metric_cols = [c for c in ["full_norm", "content_norm", "order_norm", "order_orth_norm"] if c in g4norm.columns]
        melted = g4norm.melt(id_vars=[c for c in ["band"] if c in g4norm.columns], value_vars=metric_cols, var_name="axis_name", value_name="norm")
        save_plot(plots_dir / "grade4_component_norms_by_band.png", "component norms from grade4_axis_component_norm_summary.csv", "grade4_axis_component_norm_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, melted, "axis_name", "band", "norm"))
    g4proj = read_small_csv(root, "grade4_axis_projection_geometry_summary.csv")
    if not g4proj.empty and {"condition", "axis_name", "mean_projection_fraction_on_axis_loo"}.issubset(g4proj.columns):
        save_plot(plots_dir / "grade4_component_projection_heatmap.png", "mean_projection_fraction_on_axis_loo from grade4_axis_projection_geometry_summary.csv", "grade4_axis_projection_geometry_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, g4proj, "axis_name", "condition", "mean_projection_fraction_on_axis_loo"))
    g4causal = read_small_csv(root, "grade4_axis_component_causal_symmetry_summary.csv")
    if not g4causal.empty:
        value_col = next((c for c in ["mean_plus_minus_gap", "plus_minus_projection_gap", "symmetry_pass_rate"] if c in g4causal.columns), "")
        if value_col and {"axis_name", "intervention_layer_band"}.issubset(g4causal.columns):
            save_plot(plots_dir / "grade4_component_causal_gap_heatmap.png", f"{value_col} from grade4_axis_component_causal_symmetry_summary.csv", "grade4_axis_component_causal_symmetry_summary.csv", plot_rows, lambda ax: simple_heatmap(ax, g4causal, "axis_name", "intervention_layer_band", value_col))
    if state_space_distances is not None and not state_space_distances.empty and {"condition_a", "condition_b", "centroid_l2_distance"}.issubset(state_space_distances.columns):
        dist = state_space_distances[state_space_distances["condition_a"].astype(str) != state_space_distances["condition_b"].astype(str)].copy()
        dist["centroid_l2_distance"] = pd.to_numeric(dist["centroid_l2_distance"], errors="coerce")
        if "layer" in dist.columns:
            dist["layer"] = pd.to_numeric(dist["layer"], errors="coerce")
            layer_means = dist.groupby("layer")["centroid_l2_distance"].mean()
            layer_choice = int(layer_means.idxmax()) if len(layer_means) and layer_means.notna().any() else None
            if layer_choice is not None:
                dist = dist[dist["layer"] == layer_choice]
        if not dist.empty:
            save_plot(plots_dir / "state_space_condition_distance_heatmap.png", "centroid_l2_distance from state_space_condition_distance_matrix.csv", "state_space_condition_distance_matrix.csv", plot_rows, lambda ax: simple_heatmap(ax, dist, "condition_b", "condition_a", "centroid_l2_distance"))
    if state_space_variance is not None and not state_space_variance.empty and {"layer", "between_over_within_l2_ratio"}.issubset(state_space_variance.columns):
        var_df = state_space_variance.copy()
        var_df["layer"] = pd.to_numeric(var_df["layer"], errors="coerce")
        var_df["between_over_within_l2_ratio"] = pd.to_numeric(var_df["between_over_within_l2_ratio"], errors="coerce")
        save_plot(
            plots_dir / "state_space_within_between_by_layer.png",
            "between_over_within_l2_ratio from state_space_within_between_variance.csv",
            "state_space_within_between_variance.csv",
            plot_rows,
            lambda ax: (ax.plot(var_df["layer"], var_df["between_over_within_l2_ratio"], marker="o"), ax.set_xlabel("layer"), ax.set_ylabel("between_over_within_l2_ratio")),
        )
    if state_space_pca_coords is not None and not state_space_pca_coords.empty and {"condition", "pc1", "pc2"}.issubset(state_space_pca_coords.columns):
        coords = state_space_pca_coords.copy()
        if "layer" in coords.columns:
            coords["layer"] = pd.to_numeric(coords["layer"], errors="coerce")
            layer_choice = int(coords["layer"].median()) if coords["layer"].notna().any() else None
            if layer_choice is not None:
                coords = coords[coords["layer"] == layer_choice]
        coords = coords.head(max_plot_rows)

        def pca_scatter(ax):
            for condition, g in coords.groupby("condition"):
                ax.scatter(pd.to_numeric(g["pc1"], errors="coerce"), pd.to_numeric(g["pc2"], errors="coerce"), s=24, alpha=0.75, label=str(condition))
            ax.set_xlabel("pc1")
            ax.set_ylabel("pc2")
            ax.legend(fontsize=7, loc="best")

        if not coords.empty:
            save_plot(plots_dir / "state_space_pca_condition_scatter.png", "pc1 vs pc2 from state_space_layerwise_pca_coordinates.csv", "state_space_layerwise_pca_coordinates.csv", plot_rows, pca_scatter)
    if state_space_peaks is not None and not state_space_peaks.empty and {"layer", "centroid_l2_distance"}.issubset(state_space_peaks.columns):
        peaks = state_space_peaks.copy()
        peaks["layer"] = pd.to_numeric(peaks["layer"], errors="coerce")
        peaks["centroid_l2_distance"] = pd.to_numeric(peaks["centroid_l2_distance"], errors="coerce")
        save_plot(
            plots_dir / "non_x_l2_layer_profile.png",
            "centroid_l2_distance from state_space_non_x_peaks.csv",
            "state_space_non_x_peaks.csv",
            plot_rows,
            lambda ax: (ax.scatter(peaks["layer"], peaks["centroid_l2_distance"], s=18, alpha=0.65), ax.set_xlabel("layer"), ax.set_ylabel("centroid_l2_distance")),
        )
    return pd.DataFrame(plot_rows)


def write_index_html(output_dir: Path, manifest: Dict[str, Any], artifact_df: pd.DataFrame, anomaly_df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
    rows = []
    rows.append("<!doctype html><meta charset='utf-8'><title>Hidden Geometry Metric Lab</title>")
    rows.append("<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}img{max-width:420px;margin:8px;border:1px solid #ddd}</style>")
    rows.append("<h1>Hidden Geometry Metric Lab</h1>")
    rows.append("<h2>Analysis Manifest</h2><pre>" + html.escape(json.dumps(json_sanitize(manifest), ensure_ascii=False, indent=2)) + "</pre>")
    rows.append("<h2>Core Outputs</h2><ul>")
    for name in [
        "artifact_inventory.csv",
        "artifact_schema_audit.csv",
        "global_metric_summary.csv",
        "grouped_metric_summary.csv",
        "condition_effects.csv",
        "alpha_response_regression.csv",
        "layerwise_transition_proxy.csv",
        "summary_file_numeric_extract.csv",
        "grade4_layerwise_component_readout_matrix.csv",
        "grade4_generation_component_readout_matrix.csv",
        "grade4_natscale_closeout_matrix.csv",
        "sae_order_feature_contrast_matrix.csv",
        "sae_model_compatibility_matrix.csv",
        "sae_reconstruction_quality_matrix.csv",
        "state_space_condition_centroids.csv",
        "state_space_condition_distance_matrix.csv",
        "state_space_within_between_variance.csv",
        "state_space_layerwise_pca_summary.csv",
        "state_space_layerwise_pca_coordinates.csv",
        "state_space_non_x_peaks.csv",
        "FINAL_DERIVED_METRIC_EVIDENCE.csv",
        "anomaly_flags.csv",
    ]:
        if (output_dir / name).exists():
            rows.append(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>")
    rows.append("</ul>")
    rows.append("<h2>Plots</h2>")
    if not plot_df.empty:
        for _, r in plot_df.iterrows():
            if str(r.get("status")) != "written":
                continue
            rel = Path(str(r.get("plot_file"))).relative_to(output_dir).as_posix() if Path(str(r.get("plot_file"))).is_absolute() else str(r.get("plot_file"))
            rows.append(f"<figure><a href='{html.escape(rel)}'><img src='{html.escape(rel)}'></a><figcaption>{html.escape(str(r.get('title')))} | source: {html.escape(str(r.get('source_file')))}</figcaption></figure>")
    rows.append("<h2>Anomalies</h2>")
    if anomaly_df.empty:
        rows.append("<p>No anomaly rows recorded.</p>")
    else:
        rows.append(anomaly_df.head(200).to_html(index=False, escape=True))
    (output_dir / "index.html").write_text("\n".join(rows), encoding="utf-8")


# -----------------------------------------------------------------------------
# Artifact inventory
# -----------------------------------------------------------------------------


def build_artifact_inventory(files: Sequence[Path], root: Path) -> pd.DataFrame:
    rows = []
    for path in files:
        rel = rel_path(path, root)
        family, kind = infer_artifact_family(rel)
        row = {
            "source_file": rel,
            "artifact_family": family,
            "artifact_kind": kind,
            "exists": True,
            "size_bytes": int(path.stat().st_size),
            "rows": math.nan,
            "columns": "",
            "metric_columns": "",
            "group_columns": "",
            "read_status": "not_inspected",
            "failure_code": "",
        }
        if is_csv_path(path):
            try:
                header = read_header(path)
                metric_cols = infer_metric_columns(header)
                group_cols = infer_group_columns(header)
                row.update({"columns": json.dumps(header, ensure_ascii=False), "metric_columns": json.dumps(metric_cols, ensure_ascii=False), "group_columns": json.dumps(group_cols, ensure_ascii=False), "read_status": "header_ok"})
            except Exception as exc:
                row.update({"read_status": "error", "failure_code": f"header_read_error:{repr(exc)}"})
        rows.append(row)
    return pd.DataFrame(rows)


def update_inventory_with_audit(inventory: pd.DataFrame, audits: Sequence[ProcessingAudit]) -> pd.DataFrame:
    out = inventory.copy()
    if out.empty:
        return out
    audit_map = {a.source_file: a for a in audits}
    for idx, row in out.iterrows():
        a = audit_map.get(str(row.get("source_file")))
        if a is None:
            continue
        out.at[idx, "rows"] = a.rows
        out.at[idx, "read_status"] = a.status
        out.at[idx, "failure_code"] = a.failure_code
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def resolve_backend(choice: str) -> str:
    choice = choice.lower()
    if choice == "rapids":
        if cudf is None:
            raise RuntimeError("backend=rapids requested but cudf is not importable")
        return "rapids"
    if choice == "pandas":
        return "pandas"
    if choice == "auto":
        return "rapids" if cudf is not None else "pandas"
    raise ValueError(f"Unknown backend: {choice}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hidden Geometry Metric Lab for Grade 3 / Grade 4 result packages.")
    p.add_argument("--input", required=True, help="Path to result .zip or unpacked result directory.")
    p.add_argument("--output-dir", required=True, help="Directory for analyzer outputs.")
    p.add_argument("--work-dir", default=None, help="Work/cache directory. Default: <output-dir>/_work")
    p.add_argument("--cache-parquet", action="store_true", help="Cache selected CSV columns as Parquet when using RAPIDS backend.")
    p.add_argument("--overwrite-cache", action="store_true", help="Overwrite existing Parquet cache.")
    p.add_argument("--force-extract", action="store_true", help="Re-extract ZIP even if extraction cache exists.")
    p.add_argument("--limit-files", type=int, default=None, help="Debug: process only first N CSV files.")
    p.add_argument("--write-per-file", action="store_true", help="Write per-file intermediate output tables.")
    p.add_argument("--strict", action="store_true", help="Fail immediately on file processing errors.")
    p.add_argument("--backend", choices=["auto", "rapids", "pandas"], default="auto", help="CSV processing backend.")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Generate visual plots under plots/.")
    p.add_argument("--npz-summary", action=argparse.BooleanOptionalAction, default=True, help="Analyze NPZ vector artifacts.")
    p.add_argument("--state-space-summary", action=argparse.BooleanOptionalAction, default=True, help="Analyze non-X prompt hidden-state geometry from prompt_hidden_states.npz.")
    p.add_argument("--state-space-pca-components", type=int, default=3, help="Number of deterministic PCA components for non-X state-space summaries.")
    p.add_argument("--state-space-max-pca-rows", type=int, default=200_000, help="Maximum PCA coordinate rows written per layer for state-space coordinate output/plots.")
    p.add_argument("--max-plot-rows", type=int, default=DEFAULT_MAX_PLOT_ROWS, help="Maximum rows used by plot-only data loads.")
    p.add_argument("--write-html-index", action=argparse.BooleanOptionalAction, default=True, help="Write index.html with links and plot thumbnails.")
    p.add_argument("--no-md-interpretation", action="store_true", default=True, help="Kept for explicit non-narrative operation; markdown reports are not generated.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else output_dir / "_work"
    backend = resolve_backend(args.backend)

    ensure_dir(output_dir)
    ensure_dir(work_dir)
    input_mtime_before = input_path.stat().st_mtime_ns if input_path.exists() else None
    input_size_before = input_path.stat().st_size if input_path.exists() else None
    extracted_outer_root = extract_if_needed(input_path, work_dir, force_extract=args.force_extract)
    extracted_root = detect_result_root(extracted_outer_root)
    all_files = list_files(extracted_root)
    csv_files = [p for p in all_files if is_csv_path(p)]
    npz_files = [p for p in all_files if is_npz_path(p)]
    if args.limit_files is not None:
        csv_files = csv_files[: args.limit_files]

    artifact_df = build_artifact_inventory(all_files, extracted_root)
    write_csv(output_dir / "artifact_inventory.csv", artifact_df)

    all_global: List[pd.DataFrame] = []
    all_grouped: List[pd.DataFrame] = []
    all_effects: List[pd.DataFrame] = []
    all_alpha: List[pd.DataFrame] = []
    all_layer: List[pd.DataFrame] = []
    all_summary_numeric: List[pd.DataFrame] = []
    audit_records: List[ProcessingAudit] = []

    for csv_path in tqdm(csv_files, desc="analyze CSV artifacts"):
        rel = rel_path(csv_path, extracted_root)
        try:
            global_pdf, grouped_pdf, effects_pdf, alpha_pdf, layer_pdf, summary_pdf, audit = process_file(
                csv_path=csv_path,
                extracted_root=extracted_root,
                output_dir=output_dir,
                backend=backend,
                cache_parquet=args.cache_parquet,
                overwrite_cache=args.overwrite_cache,
                write_per_file=args.write_per_file,
            )
            for bucket, part in [
                (all_global, global_pdf),
                (all_grouped, grouped_pdf),
                (all_effects, effects_pdf),
                (all_alpha, alpha_pdf),
                (all_layer, layer_pdf),
                (all_summary_numeric, summary_pdf),
            ]:
                if not part.empty:
                    bucket.append(part)
            audit_records.append(audit)
        except Exception as exc:
            audit_records.append(ProcessingAudit(rel, "error", error=repr(exc), failure_code="csv_processing_error"))
            if args.strict:
                raise

    artifact_df = update_inventory_with_audit(artifact_df, audit_records)
    write_csv(output_dir / "artifact_inventory.csv", artifact_df)
    write_csv(output_dir / "artifact_schema_audit.csv", artifact_df)
    write_csv(output_dir / "processing_audit.csv", pd.DataFrame([a.__dict__ for a in audit_records]))

    global_pdf = pd.concat(all_global, ignore_index=True) if all_global else pd.DataFrame()
    grouped_pdf = pd.concat(all_grouped, ignore_index=True) if all_grouped else pd.DataFrame()
    effects_pdf = pd.concat(all_effects, ignore_index=True) if all_effects else pd.DataFrame()
    alpha_pdf = pd.concat(all_alpha, ignore_index=True) if all_alpha else pd.DataFrame()
    layer_pdf = pd.concat(all_layer, ignore_index=True) if all_layer else pd.DataFrame()
    summary_pdf = pd.concat(all_summary_numeric, ignore_index=True) if all_summary_numeric else pd.DataFrame()

    write_csv(output_dir / "global_metric_summary.csv", global_pdf)
    write_csv(output_dir / "grouped_metric_summary.csv", grouped_pdf)
    write_csv(output_dir / "condition_effects.csv", effects_pdf)
    write_csv(output_dir / "alpha_response_regression.csv", alpha_pdf)
    # Backward-compatible alias.
    write_csv(output_dir / "causal_alpha_regression.csv", alpha_pdf)
    write_csv(output_dir / "layerwise_transition_proxy.csv", layer_pdf)
    # Backward-compatible alias.
    write_csv(output_dir / "layerwise_phase_transition.csv", layer_pdf)
    write_csv(output_dir / "summary_file_numeric_extract.csv", summary_pdf)

    npz_inventory = pd.DataFrame()
    npz_summary = pd.DataFrame()
    npz_norms = pd.DataFrame()
    npz_cosines = pd.DataFrame()
    grade4_npz = pd.DataFrame()
    state_space_centroids = pd.DataFrame()
    state_space_distances = pd.DataFrame()
    state_space_variance = pd.DataFrame()
    state_space_pca_summary = pd.DataFrame()
    state_space_pca_coords = pd.DataFrame()
    state_space_peaks = pd.DataFrame()
    anomaly_rows: List[Dict[str, Any]] = []
    if args.npz_summary:
        npz_inventory, npz_summary, npz_norms, npz_cosines, grade4_npz, npz_anomalies = analyze_npz_files(npz_files, extracted_root)
        anomaly_rows.extend(npz_anomalies)
    if args.state_space_summary:
        (
            state_space_centroids,
            state_space_distances,
            state_space_variance,
            state_space_pca_summary,
            state_space_pca_coords,
            state_space_peaks,
            state_space_anomalies,
        ) = analyze_prompt_hidden_state_space(
            npz_files,
            extracted_root,
            pca_components=args.state_space_pca_components,
            max_pca_rows=args.state_space_max_pca_rows,
        )
        anomaly_rows.extend(state_space_anomalies)
    write_csv(output_dir / "npz_inventory.csv", npz_inventory)
    write_csv(output_dir / "npz_array_summary.csv", npz_summary)
    write_csv(output_dir / "npz_layer_norms.csv", npz_norms)
    write_csv(output_dir / "npz_adjacent_layer_cosines.csv", npz_cosines)
    write_csv(output_dir / "grade4_npz_component_geometry.csv", grade4_npz)
    write_csv(output_dir / "state_space_condition_centroids.csv", state_space_centroids)
    write_csv(output_dir / "state_space_condition_distance_matrix.csv", state_space_distances)
    write_csv(output_dir / "state_space_within_between_variance.csv", state_space_variance)
    write_csv(output_dir / "state_space_layerwise_pca_summary.csv", state_space_pca_summary)
    write_csv(output_dir / "state_space_layerwise_pca_coordinates.csv", state_space_pca_coords)
    write_csv(output_dir / "state_space_non_x_peaks.csv", state_space_peaks)

    specialized_tables = build_specialized_tables(extracted_root, output_dir, anomaly_rows)
    anomaly_df = detect_anomalies(extracted_root, artifact_df, summary_pdf, specialized_tables)
    if anomaly_rows:
        anomaly_df = pd.concat([anomaly_df, pd.DataFrame(anomaly_rows)], ignore_index=True)
    write_csv(output_dir / "anomaly_flags.csv", anomaly_df)

    final_pdf = build_final_table(global_pdf, effects_pdf, alpha_pdf, layer_pdf, summary_pdf)
    state_space_final_pdf = state_space_final_evidence(state_space_distances, state_space_variance, state_space_peaks, state_space_pca_summary)
    if not state_space_final_pdf.empty:
        final_pdf = pd.concat([final_pdf, state_space_final_pdf], ignore_index=True)
    write_csv(output_dir / "FINAL_DERIVED_METRIC_EVIDENCE.csv", final_pdf)
    # Backward-compatible alias.
    write_csv(output_dir / "FINAL_LATENT_ATTRACTOR_METRICS.csv", final_pdf)

    plot_df = pd.DataFrame()
    if args.plots:
        plot_df = generate_plots(
            extracted_root,
            output_dir,
            args.max_plot_rows,
            npz_norms,
            npz_cosines,
            state_space_distances=state_space_distances,
            state_space_variance=state_space_variance,
            state_space_pca_coords=state_space_pca_coords,
            state_space_peaks=state_space_peaks,
        )
    write_csv(output_dir / "plot_manifest.csv", plot_df)

    manifest_json = {}
    manifest_path = extracted_root / "red_team_input_manifest.json"
    if manifest_path.exists():
        try:
            manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_json = {}
    artifact_names = (
        set(artifact_df["source_file"].astype(str))
        if not artifact_df.empty and "source_file" in artifact_df.columns
        else set()
    )
    gate_detected = (
        "gate4"
        if any(Path(p).name.startswith("grade4_axis_") for p in artifact_names)
        else "gate3"
        if any(Path(p).name == "middle_layer_condition_summary.csv" for p in artifact_names)
        else "unknown"
    )
    high_anomaly_count = (
        int(len(anomaly_df[anomaly_df.get("severity", pd.Series(dtype=str)).astype(str).isin(["high", "critical"])]))
        if not anomaly_df.empty
        else 0
    )
    analysis_manifest = {
        "input_path": str(input_path),
        "extracted_outer_root": str(extracted_outer_root),
        "result_root": str(extracted_root),
        "output_dir": str(output_dir),
        "run_label": manifest_json.get("run_label", ""),
        "model_id": manifest_json.get("model_id", ""),
        "gate_detected": gate_detected,
        "backend": backend,
        "files_seen": int(len(all_files)),
        "csv_files_processed": int(sum(1 for a in audit_records if a.status == "ok")),
        "npz_files_processed": int(len(npz_inventory[npz_inventory.get("read_status", pd.Series(dtype=str)).astype(str).eq("ok")])) if not npz_inventory.empty else 0,
        "state_space_summary_enabled": bool(args.state_space_summary),
        "state_space_rows_written": int(sum(len(df) for df in [state_space_centroids, state_space_distances, state_space_variance, state_space_pca_summary, state_space_pca_coords, state_space_peaks])),
        "plots_written": int(len(plot_df[plot_df.get("status", pd.Series(dtype=str)).astype(str).eq("written")])) if not plot_df.empty else 0,
        "errors": int(sum(1 for a in audit_records if a.status == "error")) + high_anomaly_count,
        "created_at": now_iso(),
        "source_unchanged": (
            bool(input_path.exists())
            and input_mtime_before == input_path.stat().st_mtime_ns
            and input_size_before == input_path.stat().st_size
        ),
    }
    write_json(output_dir / "analysis_manifest.json", analysis_manifest)
    if args.write_html_index:
        write_index_html(output_dir, analysis_manifest, artifact_df, anomaly_df, plot_df)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "backend": backend,
                "files_seen": len(all_files),
                "csv_files_seen": len(csv_files),
                "csv_files_ok": int(sum(1 for a in audit_records if a.status == "ok")),
                "csv_files_error": int(sum(1 for a in audit_records if a.status == "error")),
                "npz_files_seen": len(npz_files),
                "final_evidence_rows": len(final_pdf),
                "plots_written": analysis_manifest["plots_written"],
                "anomaly_rows": len(anomaly_df),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
