"""
Audit red-team hidden-geometry result zips/directories.

This is a reader-only helper. It does not run models and does not modify result
archives. It exists so every new zip can be judged with the same hard-random,
quality, asymmetry, and breakthrough-readiness logic.

Examples:
  python red_team_results_auditor.py "C:\\Users\\stasv\\Downloads\\run.zip"
  python red_team_results_auditor.py "C:\\Users\\stasv\\Downloads\\run02.zip" "C:\\Users\\stasv\\Downloads\\run03.zip"
  python red_team_results_auditor.py ".\\red_team_hidden_geometry_results" --out audit.md
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("red_team_metric_audit.md")
DEFAULT_CSV = Path("red_team_metric_audit_summary.csv")
DEFAULT_ARTIFACT_CSV = Path("red_team_metric_audit_artifacts.csv")
DEFAULT_CSV_PROFILE = Path("red_team_metric_audit_csv_profile.csv")
DEFAULT_FAMILY_CSV = Path("red_team_metric_audit_metric_families.csv")
DEFAULT_HISTORY_CSV = Path("red_team_metric_audit_history.csv")
DEFAULT_HISTORY_DIR = Path("red_team_metric_audit_runs")
DEFAULT_MAX_CSV_MB = 512.0


EXPECTED_ARTIFACTS = [
    "red_team_input_manifest.json",
    "prompt_condition_manifest.csv",
    "question_domain_manifest.csv",
    "replication_protocol.csv",
    "prompt_hidden_states.npz",
    "vector_x_by_layer.npz",
    "middle_layer_condition_summary.csv",
    "question_level_middle_layer_summary.csv",
    "layerwise_geometry_metrics_raw.csv",
    "layerwise_geometry_summary.csv",
    "layerwise_fdr_target_vs_control.csv",
    "paired_target_vs_control_tests.csv",
    "paired_target_vs_experimental_tests.csv",
    "null_vector_baseline_summary.csv",
    "null_vector_baseline_raw.csv",
    "pca_baseline_components.csv",
    "pca_baseline_projection_summary.csv",
    "length_bias_audit.csv",
    "deduplication_audit.csv",
    "domain_robustness_geometry_summary.csv",
    "orthogonality_axis_tests.csv",
    "residual_stream_decomposition.csv",
    "subspace_decomposition_summary.csv",
    "hidden_top_changed_dimensions.csv",
    "dense_feature_proxy_mapping.csv",
    "feature_level_interpretability_status.csv",
    "architecture_module_delta_summary.csv",
    "architecture_top_changed_units.csv",
    "architecture_target_vs_control_overlap.csv",
    "generation_response_audit.csv",
    "generation_middle_layer_summary.csv",
    "generation_trajectory_metrics_raw.csv",
    "causal_intervention_response_audit.csv",
    "causal_intervention_middle_layer_summary.csv",
    "causal_bidirectional_symmetry_summary.csv",
    "behavioral_control_axis_split_manifest.csv",
    "behavioral_control_axis_intervention_plan.csv",
    "behavioral_control_axis_response_audit.csv",
    "behavioral_control_axis_response_quality_summary.csv",
    "behavioral_control_axis_similarity_raw.csv",
    "behavioral_control_axis_similarity_summary.csv",
    "behavioral_control_axis_alpha_sweep.csv",
    "behavioral_control_axis_random_baseline.csv",
    "behavioral_control_axis_layer_band_comparison.csv",
    "behavioral_control_axis_layer_band_verdict.csv",
    "behavioral_control_axis_hard_random_comparison.csv",
    "behavioral_control_axis_hard_random_summary.csv",
    "behavioral_control_axis_asymmetry_summary.csv",
    "behavioral_control_axis_verdict.csv",
    "behavioral_control_axis_verdict.md",
    "breakthrough_readiness_audit.csv",
    "breakthrough_readiness_audit.md",
    "red_team_hidden_geometry_verdict.md",
    "null_hypothesis_hardening_summary.csv",
]


def fmt(value: Any, digits: int = 6) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a" if value is None else str(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}g}"


def safe_tag(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    value = value.strip("._-")
    return value or "audit"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pct(value: Any, digits: int = 1) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{100.0 * v:.{digits}f}%"


def finite_float(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def safe_series(values: Iterable[Any]) -> pd.Series:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(float)


def mean_or_nan(values: Iterable[Any]) -> float:
    vals = safe_series(values)
    return float(vals.mean()) if len(vals) else float("nan")


def percentile_or_nan(values: Iterable[Any], q: float) -> float:
    vals = safe_series(values)
    return float(np.percentile(vals.values, q)) if len(vals) else float("nan")


def basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def artifact_family(name: str) -> str:
    base = basename(name)
    stem = base.rsplit(".", 1)[0]
    if base in {"red_team_input_manifest.json", "prompt_condition_manifest.csv", "question_domain_manifest.csv", "replication_protocol.csv"}:
        return "manifest_protocol"
    if base.endswith(".md") or "verdict" in base:
        return "verdict_report"
    if base.endswith(".png"):
        return "plots"
    if base.endswith(".npz"):
        return "tensor_snapshots"
    if base.startswith(("middle_layer", "question_level", "layerwise_geometry", "hidden_top", "vector_x")):
        return "endpoint_geometry"
    if base.startswith(("paired_", "layerwise_fdr", "null_", "pca_", "length_bias", "deduplication", "statistical_hardness", "null_hypothesis")):
        return "statistical_controls"
    if base.startswith(("architecture_", "dense_feature", "feature_level")):
        return "architecture_features"
    if base.startswith("generation_"):
        return "generation_trajectory"
    if base.startswith("causal_") or base.startswith("layer_specific_causal"):
        return "causal_interventions"
    if base.startswith("behavioral_control_axis") or stem.startswith("behavioral_control_train"):
        return "behavioral_control_axis"
    if base.startswith(("dynamic_", "phase_", "attractor_")):
        return "dynamic_geometry"
    if base.startswith(("domain_", "orthogonality_", "residual_stream", "subspace_")):
        return "secondary_geometry"
    return "other"


def status_from_file_presence(src: "ResultSource", names: Iterable[str]) -> str:
    names = list(names)
    present = sum(1 for name in names if src.has(name))
    total = len(names)
    if present == total and total:
        return "present"
    if present:
        return f"partial_{present}_of_{total}"
    return "missing"


def finite_min(values: Iterable[Any]) -> float:
    vals = safe_series(values)
    return float(vals.min()) if len(vals) else float("nan")


def finite_max(values: Iterable[Any]) -> float:
    vals = safe_series(values)
    return float(vals.max()) if len(vals) else float("nan")


@dataclass
class ResultSource:
    path: Path
    names: set[str] = field(default_factory=set)
    archive: zipfile.ZipFile | None = None

    @classmethod
    def open(cls, path: Path) -> "ResultSource":
        path = path.expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".zip":
            archive = zipfile.ZipFile(path)
            return cls(path=path, names=set(archive.namelist()), archive=archive)
        if path.is_dir():
            return cls(
                path=path,
                names={str(p.relative_to(path)).replace("\\", "/") for p in path.rglob("*") if p.is_file()},
                archive=None,
            )
        raise FileNotFoundError(f"Not a zip or directory: {path}")

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def resolve_name(self, name: str) -> str | None:
        if name in self.names:
            return name
        wanted = name.replace("\\", "/").strip("/")
        wanted_base = wanted.rsplit("/", 1)[-1]
        matches = [
            item for item in self.names
            if item.replace("\\", "/").strip("/") == wanted
            or item.replace("\\", "/").rsplit("/", 1)[-1] == wanted_base
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda item: (len(item), item))[0]

    def has(self, name: str) -> bool:
        return self.resolve_name(name) is not None

    def file_size(self, name: str) -> int:
        resolved = self.resolve_name(name)
        if resolved is None:
            return 0
        if self.archive is not None:
            return int(self.archive.getinfo(resolved).file_size)
        return int((self.path / resolved).stat().st_size)

    def read_bytes(self, name: str) -> bytes | None:
        resolved = self.resolve_name(name)
        if resolved is None:
            return None
        if self.archive is not None:
            return self.archive.read(resolved)
        return (self.path / resolved).read_bytes()

    def read_text(self, name: str) -> str | None:
        data = self.read_bytes(name)
        if data is None:
            return None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    def read_json(self, name: str) -> dict[str, Any]:
        text = self.read_text(name)
        if not text:
            return {}
        return json.loads(text)

    def read_csv(self, name: str) -> pd.DataFrame:
        data = self.read_bytes(name)
        if data is None or len(data) <= 1:
            return pd.DataFrame()
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(data))


def build_artifact_inventory(src: ResultSource) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(src.names):
        ext = "." + basename(name).rsplit(".", 1)[-1].lower() if "." in basename(name) else ""
        size_bytes = src.file_size(name)
        rows.append(
            {
                "source": str(src.path),
                "artifact": name,
                "basename": basename(name),
                "family": artifact_family(name),
                "extension": ext,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
            }
        )
    return rows


def profile_csv_dataframe(name: str, size_bytes: int, df: pd.DataFrame, *, status: str = "read") -> dict[str, Any]:
    rows = int(len(df))
    cols = int(len(df.columns))
    cell_count = rows * cols
    null_cells = int(df.isna().sum().sum()) if cell_count else 0
    nan_fraction = float(null_cells / cell_count) if cell_count else 0.0
    numeric_cols = []
    numeric_nan_cells = 0
    numeric_cell_count = 0
    finite_numeric_cells = 0
    inf_numeric_cells = 0
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        valid_count = int(converted.notna().sum())
        if valid_count <= 0:
            continue
        numeric_cols.append(str(col))
        numeric_cell_count += rows
        arr = converted.astype(float).to_numpy()
        finite_numeric_cells += int(np.isfinite(arr).sum())
        inf_numeric_cells += int(np.isinf(arr).sum())
        numeric_nan_cells += int(np.isnan(arr).sum())

    top_nan_columns = []
    if cols:
        rates = (df.isna().mean()).sort_values(ascending=False)
        top_nan_columns = [
            f"{col}:{rate:.3g}" for col, rate in rates.head(5).items()
            if float(rate) > 0
        ]

    return {
        "artifact": name,
        "basename": basename(name),
        "family": artifact_family(name),
        "status": status,
        "size_mb": size_bytes / (1024 * 1024),
        "rows": rows,
        "columns": cols,
        "numeric_columns": len(numeric_cols),
        "null_cells": null_cells,
        "null_fraction": nan_fraction,
        "numeric_nan_cells": numeric_nan_cells,
        "numeric_inf_cells": inf_numeric_cells,
        "finite_numeric_cells": finite_numeric_cells,
        "top_nan_columns": ", ".join(top_nan_columns),
        "column_names": ", ".join(str(c) for c in df.columns[:40]),
    }


def load_csv_catalog(src: ResultSource, *, max_csv_bytes: int) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    profiles: list[dict[str, Any]] = []
    csv_names = sorted(name for name in src.names if name.lower().endswith(".csv"))
    for name in csv_names:
        size_bytes = src.file_size(name)
        if size_bytes > max_csv_bytes:
            profiles.append(
                {
                    "artifact": name,
                    "basename": basename(name),
                    "family": artifact_family(name),
                    "status": f"skipped_larger_than_{max_csv_bytes / (1024 * 1024):.0f}mb",
                    "size_mb": size_bytes / (1024 * 1024),
                    "rows": float("nan"),
                    "columns": float("nan"),
                    "numeric_columns": float("nan"),
                    "null_cells": float("nan"),
                    "null_fraction": float("nan"),
                    "numeric_nan_cells": float("nan"),
                    "numeric_inf_cells": float("nan"),
                    "finite_numeric_cells": float("nan"),
                    "top_nan_columns": "",
                    "column_names": "",
                }
            )
            continue
        try:
            df = src.read_csv(name)
            frames.setdefault(basename(name), df)
            profiles.append(profile_csv_dataframe(name, size_bytes, df))
        except Exception as exc:
            profiles.append(
                {
                    "artifact": name,
                    "basename": basename(name),
                    "family": artifact_family(name),
                    "status": f"read_error:{type(exc).__name__}",
                    "size_mb": size_bytes / (1024 * 1024),
                    "rows": float("nan"),
                    "columns": float("nan"),
                    "numeric_columns": float("nan"),
                    "null_cells": float("nan"),
                    "null_fraction": float("nan"),
                    "numeric_nan_cells": float("nan"),
                    "numeric_inf_cells": float("nan"),
                    "finite_numeric_cells": float("nan"),
                    "top_nan_columns": str(exc)[:180],
                    "column_names": "",
                }
            )
    return frames, profiles


def filter_rows(df: pd.DataFrame, **conditions: Any) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df.copy()
    for key, value in conditions.items():
        if key not in sub.columns:
            return pd.DataFrame()
        if isinstance(value, float) and np.isfinite(value):
            col = pd.to_numeric(sub[key], errors="coerce").astype(float)
            sub = sub[np.isclose(col, value)]
        else:
            sub = sub[sub[key].astype(str) == str(value)]
    return sub


def scalar_from(df: pd.DataFrame, column: str, default: float = float("nan")) -> float:
    if df.empty or column not in df.columns:
        return default
    return finite_float(df.iloc[0].get(column, default))


def derive_hard_random(
    similarity_raw: pd.DataFrame,
    *,
    reference_condition: str,
    primary_band: str,
    primary_alpha: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "neutral_plus_x_likeness": float("nan"),
        "neutral_plus_random_mean": float("nan"),
        "neutral_plus_random_p95": float("nan"),
        "neutral_plus_random_best": float("nan"),
        "neutral_plus_lift_random_mean": float("nan"),
        "neutral_plus_lift_random_p95": float("nan"),
        "neutral_plus_lift_best_random": float("nan"),
        "neutral_plus_win_random_mean": float("nan"),
        "neutral_plus_win_random_p95": float("nan"),
        "neutral_plus_win_best_random": float("nan"),
        "neutral_plus_questions_over_random_mean": float("nan"),
        "neutral_plus_questions_over_random_p95": float("nan"),
        "neutral_plus_questions_over_best_random": float("nan"),
        "neutral_plus_generation_projection": float("nan"),
        "neutral_plus_random_generation_mean": float("nan"),
        "neutral_plus_generation_win_random": float("nan"),
        "target_minus_suppression": float("nan"),
        "target_minus_random_suppression_mean": float("nan"),
        "target_minus_suppression_lift_random_mean": float("nan"),
        "target_minus_suppression_lift_random_p95": float("nan"),
        "target_minus_suppression_lift_best_random": float("nan"),
        "target_minus_suppression_win_random_mean": float("nan"),
        "target_minus_suppression_win_random_p95": float("nan"),
        "target_minus_suppression_win_best_random": float("nan"),
        "per_question_rows": [],
    }
    if similarity_raw.empty or "behavioral_target_likeness_cosine_0_to_1" not in similarity_raw.columns:
        return out

    def rows(base: str, kind: str, sign: str) -> pd.DataFrame:
        return filter_rows(
            similarity_raw,
            base_condition=base,
            intervention_kind=kind,
            sign_name=sign,
            layer_band=primary_band,
            alpha_abs=primary_alpha,
        )

    def random_direction_means(df: pd.DataFrame, value_col: str) -> pd.Series:
        if df.empty or value_col not in df.columns:
            return pd.Series(dtype=float)
        working = df.copy()
        working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
        working = working[np.isfinite(working[value_col].astype(float))]
        if working.empty:
            return pd.Series(dtype=float)
        if "random_index" in working.columns:
            return (
                working.groupby("random_index")[value_col]
                .mean()
                .dropna()
                .astype(float)
            )
        return working[value_col].dropna().astype(float)

    x_neutral = rows(reference_condition, "vector_x", "plus_x")
    r_neutral = rows(reference_condition, "random", "plus_random")
    x_target_minus = rows("target", "vector_x", "minus_x")
    r_target_minus = rows("target", "random", "minus_random")

    if len(x_neutral):
        x_by_q = (
            x_neutral.groupby("question_index", as_index=False)
            .agg(
                x_likeness=("behavioral_target_likeness_cosine_0_to_1", "mean"),
                x_generation=("mean_generation_projection_on_train_vector_x", "mean"),
            )
        )
        out["neutral_plus_x_likeness"] = float(x_by_q["x_likeness"].mean())
        out["neutral_plus_generation_projection"] = float(x_by_q["x_generation"].mean())

    if len(r_neutral):
        r_values = random_direction_means(r_neutral, "behavioral_target_likeness_cosine_0_to_1")
        r_gen = random_direction_means(r_neutral, "mean_generation_projection_on_train_vector_x")
        out["neutral_plus_random_mean"] = float(r_values.mean()) if len(r_values) else float("nan")
        out["neutral_plus_random_p95"] = float(np.percentile(r_values.values, 95)) if len(r_values) else float("nan")
        out["neutral_plus_random_best"] = float(r_values.max()) if len(r_values) else float("nan")
        out["neutral_plus_random_generation_mean"] = float(r_gen.mean()) if len(r_gen) else float("nan")
        if np.isfinite(out["neutral_plus_x_likeness"]) and len(r_values):
            out["neutral_plus_lift_random_mean"] = out["neutral_plus_x_likeness"] - out["neutral_plus_random_mean"]
            out["neutral_plus_lift_random_p95"] = out["neutral_plus_x_likeness"] - out["neutral_plus_random_p95"]
            out["neutral_plus_lift_best_random"] = out["neutral_plus_x_likeness"] - out["neutral_plus_random_best"]
            out["neutral_plus_win_random_mean"] = float(np.mean(r_values.values <= out["neutral_plus_x_likeness"]))
            out["neutral_plus_win_random_p95"] = float(out["neutral_plus_x_likeness"] > out["neutral_plus_random_p95"])
            out["neutral_plus_win_best_random"] = float(out["neutral_plus_x_likeness"] > out["neutral_plus_random_best"])
        if np.isfinite(out["neutral_plus_generation_projection"]) and len(r_gen):
            out["neutral_plus_generation_win_random"] = float(np.mean(r_gen.values <= out["neutral_plus_generation_projection"]))

    if len(x_neutral) and len(r_neutral):
        r_by_q = (
            r_neutral.groupby("question_index")["behavioral_target_likeness_cosine_0_to_1"]
            .agg(
                random_mean="mean",
                random_max="max",
                random_p95=lambda s: float(np.percentile(s.astype(float), 95)),
            )
            .reset_index()
        )
        x_by_q = (
            x_neutral.groupby("question_index", as_index=False)
            .agg(x_likeness=("behavioral_target_likeness_cosine_0_to_1", "mean"))
        )
        per_q = x_by_q.merge(r_by_q, on="question_index", how="inner")
        if len(per_q):
            per_q["x_minus_random_mean"] = per_q["x_likeness"] - per_q["random_mean"]
            per_q["x_minus_random_p95"] = per_q["x_likeness"] - per_q["random_p95"]
            out["neutral_plus_questions_over_random_mean"] = float((per_q["x_likeness"] > per_q["random_mean"]).mean())
            out["neutral_plus_questions_over_random_p95"] = float((per_q["x_likeness"] > per_q["random_p95"]).mean())
            out["neutral_plus_questions_over_best_random"] = float((per_q["x_likeness"] > per_q["random_max"]).mean())
            out["per_question_rows"] = per_q.to_dict("records")

    if len(x_target_minus):
        target_like = mean_or_nan(x_target_minus["behavioral_target_likeness_cosine_0_to_1"])
        out["target_minus_suppression"] = 1.0 - target_like if np.isfinite(target_like) else float("nan")

    if len(r_target_minus):
        r_like = random_direction_means(r_target_minus, "behavioral_target_likeness_cosine_0_to_1")
        r_supp = 1.0 - r_like
        if len(r_supp):
            out["target_minus_random_suppression_mean"] = float(r_supp.mean())
            r_supp_p95 = float(np.percentile(r_supp.values, 95))
            r_supp_best = float(r_supp.max())
            if np.isfinite(out["target_minus_suppression"]):
                out["target_minus_suppression_lift_random_mean"] = out["target_minus_suppression"] - out["target_minus_random_suppression_mean"]
                out["target_minus_suppression_lift_random_p95"] = out["target_minus_suppression"] - r_supp_p95
                out["target_minus_suppression_lift_best_random"] = out["target_minus_suppression"] - r_supp_best
                out["target_minus_suppression_win_random_mean"] = float(np.mean(r_supp.values <= out["target_minus_suppression"]))
                out["target_minus_suppression_win_random_p95"] = float(out["target_minus_suppression"] > r_supp_p95)
                out["target_minus_suppression_win_best_random"] = float(out["target_minus_suppression"] > r_supp_best)

    return out


def extract_quality(
    quality: pd.DataFrame,
    response_audit: pd.DataFrame | None = None,
    *,
    reference_condition: str,
    primary_band: str,
    primary_alpha: float,
) -> dict[str, float]:
    out = {
        "neutral_plus_degeneration": float("nan"),
        "target_minus_degeneration": float("nan"),
        "neutral_plus_unique_ratio": float("nan"),
        "target_minus_unique_ratio": float("nan"),
        "neutral_plus_repeated_bigram": float("nan"),
        "target_minus_repeated_bigram": float("nan"),
    }

    def row(base: str, sign: str) -> pd.DataFrame:
        return filter_rows(
            quality,
            base_condition=base,
            intervention_kind="vector_x",
            sign_name=sign,
            layer_band=primary_band,
            alpha_abs=primary_alpha,
        )

    if not quality.empty:
        neutral = row(reference_condition, "plus_x")
        target = row("target", "minus_x")
        out["neutral_plus_degeneration"] = scalar_from(neutral, "degenerate_response_rate")
        out["target_minus_degeneration"] = scalar_from(target, "degenerate_response_rate")
        out["neutral_plus_unique_ratio"] = scalar_from(neutral, "mean_unique_word_ratio")
        out["target_minus_unique_ratio"] = scalar_from(target, "mean_unique_word_ratio")
        out["neutral_plus_repeated_bigram"] = scalar_from(neutral, "mean_repeated_bigram_fraction")
        out["target_minus_repeated_bigram"] = scalar_from(target, "mean_repeated_bigram_fraction")
        return out

    response_audit = response_audit if response_audit is not None else pd.DataFrame()
    if response_audit.empty:
        return out

    def audit_rows(base: str, sign: str) -> pd.DataFrame:
        return filter_rows(
            response_audit,
            base_condition=base,
            intervention_kind="vector_x",
            sign_name=sign,
            layer_band=primary_band,
            alpha_abs=primary_alpha,
        )

    def corrected_quality(sub: pd.DataFrame) -> tuple[float, float, float]:
        if sub.empty:
            return float("nan"), float("nan"), float("nan")
        if "degenerate_response_proxy" in sub.columns:
            deg = pd.to_numeric(sub["degenerate_response_proxy"], errors="coerce")
        else:
            loop = pd.to_numeric(sub.get("quality_loop_like", 0), errors="coerce").fillna(0)
            low_div = pd.to_numeric(sub.get("quality_low_diversity", 0), errors="coerce").fillna(0)
            too_short = pd.to_numeric(sub.get("quality_too_short", 0), errors="coerce").fillna(0)
            words = pd.to_numeric(sub.get("visible_word_count", 999), errors="coerce").fillna(999)
            deg = ((loop > 0) | (low_div > 0) | (too_short > 0) | (words < 20)).astype(float)
        unique_col = "mean_unique_word_ratio" if "mean_unique_word_ratio" in sub.columns else "visible_unique_word_ratio"
        repeat_col = "mean_repeated_bigram_fraction" if "mean_repeated_bigram_fraction" in sub.columns else "visible_repeated_trigram_fraction"
        unique = pd.to_numeric(sub.get(unique_col, pd.Series(dtype=float)), errors="coerce")
        repeat = pd.to_numeric(sub.get(repeat_col, pd.Series(dtype=float)), errors="coerce")
        return mean_or_nan(deg), mean_or_nan(unique), mean_or_nan(repeat)

    neutral = audit_rows(reference_condition, "plus_x")
    target = audit_rows("target", "minus_x")
    (
        out["neutral_plus_degeneration"],
        out["neutral_plus_unique_ratio"],
        out["neutral_plus_repeated_bigram"],
    ) = corrected_quality(neutral)
    (
        out["target_minus_degeneration"],
        out["target_minus_unique_ratio"],
        out["target_minus_repeated_bigram"],
    ) = corrected_quality(target)
    return out


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    if df.empty:
        return None
    by_lower = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in by_lower:
            return by_lower[key]
    return None


def condition_value(df: pd.DataFrame, condition: str, candidates: Iterable[str]) -> float:
    if df.empty or "condition" not in df.columns:
        return float("nan")
    col = first_existing_column(df, candidates)
    if col is None:
        return float("nan")
    sub = df[df["condition"].astype(str) == condition]
    if not len(sub):
        return float("nan")
    return finite_float(sub.iloc[0].get(col))


def add_family_row(rows: list[dict[str, Any]], family: str, status: str, readout: str, artifacts: Iterable[str]) -> None:
    rows.append(
        {
            "family": family,
            "status": status,
            "readout": readout,
            "artifacts": ", ".join(artifacts),
        }
    )


def derive_family_overview(
    src: ResultSource,
    csv_frames: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    hard: dict[str, Any],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    middle = csv_frames.get("middle_layer_condition_summary.csv", pd.DataFrame())
    if len(middle):
        target = condition_value(middle, "target", ["projection_fraction_on_vector_x_loo_mean", "mean_projection_fraction_on_vector_x_loo"])
        neutral_len = condition_value(middle, "neutral_length_matched_control", ["projection_fraction_on_vector_x_loo_mean", "mean_projection_fraction_on_vector_x_loo"])
        word = condition_value(middle, "target_word_shuffle_control", ["projection_fraction_on_vector_x_loo_mean", "mean_projection_fraction_on_vector_x_loo"])
        sent = condition_value(middle, "target_sentence_shuffle_control", ["projection_fraction_on_vector_x_loo_mean", "mean_projection_fraction_on_vector_x_loo"])
        add_family_row(
            rows,
            "endpoint_geometry",
            "computed",
            f"target={fmt(target)}, length_neutral={fmt(neutral_len)}, word_shuffle={fmt(word)}, sentence_shuffle={fmt(sent)}",
            ["middle_layer_condition_summary.csv"],
        )
    else:
        add_family_row(rows, "endpoint_geometry", "missing", "No middle-layer condition summary found.", ["middle_layer_condition_summary.csv"])

    layerwise = csv_frames.get("layerwise_geometry_summary.csv", pd.DataFrame())
    if len(layerwise):
        layer_col = first_existing_column(layerwise, ["layer"])
        proj_col = first_existing_column(layerwise, ["projection_fraction_on_vector_x_loo_mean", "mean_projection_fraction_on_vector_x_loo"])
        cond_col = first_existing_column(layerwise, ["condition"])
        readout = f"rows={len(layerwise)}"
        if layer_col and proj_col and cond_col:
            sub = layerwise[layerwise[cond_col].astype(str) == "target"].copy()
            if len(sub):
                sub[proj_col] = pd.to_numeric(sub[proj_col], errors="coerce")
                idx = sub[proj_col].idxmax()
                readout = f"target_best_layer={sub.loc[idx, layer_col]}, max_projection={fmt(sub.loc[idx, proj_col])}, target_layer_mean={fmt(sub[proj_col].mean())}"
        add_family_row(rows, "layerwise_geometry", "computed", readout, ["layerwise_geometry_summary.csv"])
    else:
        add_family_row(rows, "layerwise_geometry", "missing_or_not_exported", "Layerwise summary absent; raw geometry may still exist.", ["layerwise_geometry_summary.csv"])

    paired = csv_frames.get("paired_target_vs_control_tests.csv", pd.DataFrame())
    fdr = csv_frames.get("layerwise_fdr_target_vs_control.csv", pd.DataFrame())
    null_summary = csv_frames.get("null_vector_baseline_summary.csv", pd.DataFrame())
    pca_proj = csv_frames.get("pca_baseline_projection_summary.csv", pd.DataFrame())
    stats_bits = []
    if len(paired):
        p_col = first_existing_column(paired, ["paired_sign_permutation_p", "p_value", "p"])
        d_col = first_existing_column(paired, ["paired_cohen_d", "cohen_d"])
        stats_bits.append(f"paired_rows={len(paired)}")
        if p_col:
            stats_bits.append(f"min_p={fmt(finite_min(paired[p_col]))}")
        if d_col:
            stats_bits.append(f"max_abs_d={fmt(finite_max(abs(pd.to_numeric(paired[d_col], errors='coerce'))))}")
    if len(fdr):
        sig_col = first_existing_column(fdr, ["fdr_significant"])
        q_col = first_existing_column(fdr, ["fdr_q_value", "q_value"])
        sig_n = int(pd.to_numeric(fdr[sig_col], errors="coerce").fillna(0).sum()) if sig_col else 0
        stats_bits.append(f"fdr_sig={sig_n}/{len(fdr)}")
        if q_col:
            stats_bits.append(f"min_q={fmt(finite_min(fdr[q_col]))}")
    if len(null_summary):
        p_col = first_existing_column(null_summary, ["empirical_p_greater_equal_observed"])
        obs_col = first_existing_column(null_summary, ["observed_target_projection_mean"])
        null_col = first_existing_column(null_summary, ["null_mean"])
        if p_col:
            stats_bits.append(f"null_p={fmt(finite_min(null_summary[p_col]))}")
        if obs_col and null_col:
            stats_bits.append(f"obs_minus_null={fmt(mean_or_nan(null_summary[obs_col]) - mean_or_nan(null_summary[null_col]))}")
    if len(pca_proj):
        cos_col = first_existing_column(pca_proj, ["abs_cosine_with_vector_x", "cosine_with_vector_x"])
        if cos_col:
            stats_bits.append(f"pca_max_abs_cos={fmt(finite_max(pca_proj[cos_col]))}")
    add_family_row(
        rows,
        "statistical_controls",
        "computed" if stats_bits else "missing_or_disabled",
        "; ".join(stats_bits) if stats_bits else "No paired/FDR/null/PCA control artifacts found.",
        ["paired_target_vs_control_tests.csv", "layerwise_fdr_target_vs_control.csv", "null_vector_baseline_summary.csv", "pca_baseline_projection_summary.csv"],
    )

    length_bias = csv_frames.get("length_bias_audit.csv", pd.DataFrame())
    dedup = csv_frames.get("deduplication_audit.csv", pd.DataFrame())
    secondary_bits = []
    if len(length_bias):
        corr_col = first_existing_column(length_bias, ["prompt_token_projection_correlation"])
        if corr_col:
            vals = pd.to_numeric(length_bias[corr_col], errors="coerce").dropna()
            secondary_bits.append(f"max_abs_length_corr={fmt(vals.abs().max() if len(vals) else float('nan'))}")
    if len(dedup):
        dup_col = first_existing_column(dedup, ["is_duplicate"])
        duplicate_count = int(pd.to_numeric(dedup[dup_col], errors="coerce").fillna(0).sum()) if dup_col else 0
        secondary_bits.append(f"duplicate_questions={duplicate_count}")
    add_family_row(
        rows,
        "bias_and_dataset_audits",
        "computed" if secondary_bits else "missing_or_disabled",
        "; ".join(secondary_bits) if secondary_bits else "No length-bias/dedup audit found.",
        ["length_bias_audit.csv", "deduplication_audit.csv", "domain_robustness_geometry_summary.csv"],
    )

    arch = csv_frames.get("architecture_module_delta_summary.csv", pd.DataFrame())
    if len(arch):
        cond_col = first_existing_column(arch, ["condition"])
        module_col = first_existing_column(arch, ["module"])
        proj_col = first_existing_column(arch, ["projection_fraction_on_arch_vector_x_loo", "projection_fraction_on_vector_x_loo"])
        readout = f"rows={len(arch)}"
        if cond_col and module_col and proj_col:
            target_arch = arch[arch[cond_col].astype(str) == "target"].copy()
            target_arch[proj_col] = pd.to_numeric(target_arch[proj_col], errors="coerce")
            if len(target_arch):
                grouped = target_arch.groupby(module_col)[proj_col].mean().sort_values(ascending=False)
                readout = "; ".join(f"{idx}={fmt(val)}" for idx, val in grouped.head(5).items())
        add_family_row(rows, "architecture_features", "computed", readout, ["architecture_module_delta_summary.csv", "architecture_top_changed_units.csv"])
    elif src.has("architecture_module_delta_summary.csv"):
        add_family_row(rows, "architecture_features", "present_empty_or_disabled", "Architecture file exists but has no rows.", ["architecture_module_delta_summary.csv"])
    else:
        add_family_row(rows, "architecture_features", "missing_or_disabled", "Architecture hooks were not part of this profile.", ["architecture_module_delta_summary.csv"])

    generation = csv_frames.get("generation_middle_layer_summary.csv", pd.DataFrame())
    if len(generation):
        cond_col = first_existing_column(generation, ["condition"])
        proj_col = first_existing_column(generation, ["mean_projection_fraction_on_vector_x_loo"])
        readout = f"rows={len(generation)}"
        if cond_col and proj_col:
            bits = []
            for _, row in generation.head(8).iterrows():
                bits.append(f"{row.get(cond_col)}={fmt(row.get(proj_col))}")
            readout = "; ".join(bits)
        add_family_row(rows, "generation_trajectory", "computed", readout, ["generation_middle_layer_summary.csv", "generation_response_audit.csv"])
    elif src.has("generation_middle_layer_summary.csv"):
        add_family_row(rows, "generation_trajectory", "present_empty_or_profile_skipped", "Generation summary exists but has no rows.", ["generation_middle_layer_summary.csv"])
    else:
        add_family_row(rows, "generation_trajectory", "missing_or_disabled", "No generation summary found.", ["generation_middle_layer_summary.csv"])

    causal = csv_frames.get("causal_intervention_middle_layer_summary.csv", pd.DataFrame())
    causal_sym = csv_frames.get("causal_bidirectional_symmetry_summary.csv", pd.DataFrame())
    if len(causal):
        proj_col = first_existing_column(causal, ["mean_projection_fraction_on_vector_x_loo"])
        readout = f"rows={len(causal)}"
        if proj_col:
            readout += f", projection_range=[{fmt(finite_min(causal[proj_col]))}, {fmt(finite_max(causal[proj_col]))}]"
        if len(causal_sym):
            sym_col = first_existing_column(causal_sym, ["bidirectional_symmetry_supported"])
            sym = int(pd.to_numeric(causal_sym[sym_col], errors="coerce").fillna(0).sum()) if sym_col else 0
            readout += f", symmetry_supported={sym}/{len(causal_sym)}"
        add_family_row(rows, "causal_interventions", "computed", readout, ["causal_intervention_middle_layer_summary.csv", "causal_bidirectional_symmetry_summary.csv"])
    elif src.has("causal_intervention_middle_layer_summary.csv"):
        add_family_row(rows, "causal_interventions", "present_empty_or_profile_skipped", "Full causal block exists but has no rows; behavioral-control block may still include +X/-X.", ["causal_intervention_middle_layer_summary.csv"])
    else:
        add_family_row(rows, "causal_interventions", "missing_or_disabled", "Full causal injection/ablation block was not exported.", ["causal_intervention_middle_layer_summary.csv"])

    add_family_row(
        rows,
        "behavioral_control_axis",
        "computed" if np.isfinite(finite_float(hard.get("neutral_plus_lift_random_mean"))) else "missing_or_disabled",
        (
            f"neutral_lift_mean={fmt(hard.get('neutral_plus_lift_random_mean'))}, "
            f"neutral_lift_p95={fmt(hard.get('neutral_plus_lift_random_p95'))}, "
            f"generation_win={pct(hard.get('neutral_plus_generation_win_random'))}, "
            f"target_minus_supp_lift={fmt(hard.get('target_minus_suppression_lift_random_mean'))}"
        ),
        ["behavioral_control_axis_similarity_raw.csv", "behavioral_control_axis_response_quality_summary.csv", "behavioral_control_axis_verdict.csv"],
    )

    add_family_row(
        rows,
        "response_quality",
        "computed" if np.isfinite(finite_float(quality.get("neutral_plus_degeneration"))) else "missing_or_disabled",
        (
            f"neutral_degeneration={fmt(quality.get('neutral_plus_degeneration'))}, "
            f"target_minus_degeneration={fmt(quality.get('target_minus_degeneration'))}, "
            f"neutral_unique={fmt(quality.get('neutral_plus_unique_ratio'))}"
        ),
        ["behavioral_control_axis_response_quality_summary.csv"],
    )

    dynamic = csv_frames.get("dynamic_trajectory_summary.csv", pd.DataFrame())
    if len(dynamic):
        add_family_row(rows, "dynamic_geometry", "computed", f"rows={len(dynamic)}", ["dynamic_trajectory_summary.csv", "phase_transition_candidates.csv"])
    elif src.has("dynamic_trajectory_summary.csv"):
        add_family_row(rows, "dynamic_geometry", "present_empty_or_profile_skipped", "Dynamic file exists but has no rows.", ["dynamic_trajectory_summary.csv"])
    else:
        add_family_row(rows, "dynamic_geometry", "missing_or_disabled", "Dynamic trajectory block not exported.", ["dynamic_trajectory_summary.csv"])

    feature = csv_frames.get("feature_level_interpretability_status.csv", pd.DataFrame())
    dense = csv_frames.get("dense_feature_proxy_mapping.csv", pd.DataFrame())
    status_bits = []
    if len(feature):
        status_col = first_existing_column(feature, ["status"])
        if status_col:
            status_bits.extend(str(x) for x in feature[status_col].head(4).tolist())
    if len(dense):
        status_bits.append(f"dense_proxy_rows={len(dense)}")
    add_family_row(
        rows,
        "feature_proxy",
        "computed_or_statused" if status_bits else "missing",
        "; ".join(status_bits) if status_bits else "No feature proxy/status artifacts found.",
        ["feature_level_interpretability_status.csv", "dense_feature_proxy_mapping.csv"],
    )

    missing_expected = [name for name in EXPECTED_ARTIFACTS if not src.has(name)]
    add_family_row(
        rows,
        "artifact_coverage",
        "computed",
        f"files={len(src.names)}, expected_present={len(EXPECTED_ARTIFACTS) - len(missing_expected)}/{len(EXPECTED_ARTIFACTS)}, missing_expected={len(missing_expected)}",
        ["full archive inventory"],
    )

    return rows


def load_run(path: Path, *, max_csv_bytes: int = int(DEFAULT_MAX_CSV_MB * 1024 * 1024)) -> dict[str, Any]:
    src = ResultSource.open(path)
    try:
        artifact_inventory = build_artifact_inventory(src)
        csv_frames, csv_profiles = load_csv_catalog(src, max_csv_bytes=max_csv_bytes)
        manifest = src.read_json("red_team_input_manifest.json")
        verdict = csv_frames.get("behavioral_control_axis_verdict.csv", pd.DataFrame())
        similarity_raw = csv_frames.get("behavioral_control_axis_similarity_raw.csv", pd.DataFrame())
        quality = csv_frames.get("behavioral_control_axis_response_quality_summary.csv", pd.DataFrame())
        response_audit = csv_frames.get("behavioral_control_axis_response_audit.csv", pd.DataFrame())
        alpha = csv_frames.get("behavioral_control_axis_alpha_sweep.csv", pd.DataFrame())
        middle = csv_frames.get("middle_layer_condition_summary.csv", pd.DataFrame())
        paired = csv_frames.get("paired_target_vs_control_tests.csv", pd.DataFrame())
        hard_summary_existing = csv_frames.get("behavioral_control_axis_hard_random_summary.csv", pd.DataFrame())
        asymmetry_existing = csv_frames.get("behavioral_control_axis_asymmetry_summary.csv", pd.DataFrame())

        reference = str(manifest.get("reference_condition", "neutral"))
        primary_band = str(manifest.get("behavioral_control_primary_layer_band", "middle"))
        primary_alpha = finite_float(manifest.get("behavioral_control_primary_alpha"))
        if not np.isfinite(primary_alpha):
            alpha_values = manifest.get("behavioral_control_alpha_values") or [manifest.get("behavioral_control_random_alpha", 0.5)]
            primary_alpha = finite_float(alpha_values[-1] if alpha_values else 0.5)

        verdict_row = verdict.iloc[0].to_dict() if len(verdict) else {}
        hard = derive_hard_random(
            similarity_raw,
            reference_condition=reference,
            primary_band=primary_band,
            primary_alpha=primary_alpha,
        )
        q = extract_quality(
            quality,
            response_audit,
            reference_condition=reference,
            primary_band=primary_band,
            primary_alpha=primary_alpha,
        )
        family_overview = derive_family_overview(src, csv_frames, manifest, hard, q)

        hidden_rows = {}
        if len(middle) and "condition" in middle.columns:
            for _, row in middle.iterrows():
                hidden_rows[str(row.get("condition"))] = row.to_dict()

        alpha_rows = alpha.to_dict("records") if len(alpha) else []

        run = {
            "source": str(path),
            "file_count": len(src.names),
            "artifact_inventory": artifact_inventory,
            "csv_profiles": csv_profiles,
            "family_overview": family_overview,
            "expected_missing": [name for name in EXPECTED_ARTIFACTS if not src.has(name)],
            "has_hard_random_summary": src.has("behavioral_control_axis_hard_random_summary.csv"),
            "has_breakthrough_audit": src.has("breakthrough_readiness_audit.md"),
            "has_asymmetry_summary": src.has("behavioral_control_axis_asymmetry_summary.csv"),
            "manifest": manifest,
            "verdict": verdict_row,
            "hard_random": hard,
            "quality": q,
            "hidden": hidden_rows,
            "paired": paired,
            "alpha_rows": alpha_rows,
            "hard_summary_existing": hard_summary_existing.to_dict("records") if len(hard_summary_existing) else [],
            "asymmetry_existing": asymmetry_existing.to_dict("records") if len(asymmetry_existing) else [],
        }
        run.update(classify_run(run))
        return run
    finally:
        src.close()


def classify_run(run: dict[str, Any]) -> dict[str, Any]:
    manifest = run.get("manifest", {})
    verdict = run.get("verdict", {})
    hard = run.get("hard_random", {})
    quality = run.get("quality", {})
    hidden = run.get("hidden", {})

    target_hidden = hidden.get("target", {})
    target_projection = finite_float(target_hidden.get("projection_fraction_on_vector_x_loo_mean"))
    length_neutral = finite_float(
        hidden.get("neutral_length_matched_control", {}).get("projection_fraction_on_vector_x_loo_mean")
    )
    word_shuffle = finite_float(
        hidden.get("target_word_shuffle_control", {}).get("projection_fraction_on_vector_x_loo_mean")
    )
    sentence_shuffle = finite_float(
        hidden.get("target_sentence_shuffle_control", {}).get("projection_fraction_on_vector_x_loo_mean")
    )

    hidden_status = "missing"
    if np.isfinite(target_projection):
        if target_projection >= 0.85 and (not np.isfinite(length_neutral) or length_neutral <= 0.05):
            hidden_status = "strong"
        elif target_projection >= 0.50:
            hidden_status = "medium"
        elif target_projection > 0:
            hidden_status = "weak"
        else:
            hidden_status = "failed"

    neutral_lift_mean = finite_float(hard.get("neutral_plus_lift_random_mean"))
    neutral_lift_p95 = finite_float(hard.get("neutral_plus_lift_random_p95"))
    neutral_win_random = finite_float(hard.get("neutral_plus_win_random_mean"))
    neutral_q_win_mean = finite_float(hard.get("neutral_plus_questions_over_random_mean"))
    neutral_gen_win = finite_float(hard.get("neutral_plus_generation_win_random"))
    neutral_deg = finite_float(quality.get("neutral_plus_degeneration"))

    visible_status = "missing"
    if np.isfinite(neutral_lift_mean):
        if neutral_lift_p95 > 0 and neutral_q_win_mean >= 0.70 and neutral_deg <= 0.05:
            visible_status = "strong"
        elif neutral_lift_mean > 0 and neutral_win_random >= 0.75 and neutral_deg <= 0.05:
            visible_status = "good_partial"
        elif neutral_lift_mean > 0 and neutral_deg <= 0.25:
            visible_status = "weak_partial"
        else:
            visible_status = "not_supported"

    target_supp_lift = finite_float(hard.get("target_minus_suppression_lift_random_mean"))
    target_supp_win = finite_float(hard.get("target_minus_suppression_win_random_mean"))
    target_supp_p95 = finite_float(hard.get("target_minus_suppression_lift_random_p95"))
    target_deg = finite_float(quality.get("target_minus_degeneration"))

    ablation_status = "missing"
    if np.isfinite(target_supp_lift):
        if target_supp_p95 > 0 and target_supp_win >= 0.80 and target_deg <= 0.05:
            ablation_status = "strong"
        elif target_supp_lift > 0 and target_supp_win >= 0.75 and target_deg <= 0.10:
            ablation_status = "good"
        elif target_supp_lift > 0:
            ablation_status = "weak_partial"
        else:
            ablation_status = "not_supported"

    internal_generation_status = "missing"
    if np.isfinite(neutral_gen_win):
        internal_generation_status = "strong" if neutral_gen_win >= 0.95 else "partial" if neutral_gen_win >= 0.70 else "weak"

    if hidden_status == "strong" and internal_generation_status == "strong" and ablation_status in {"strong", "good"} and visible_status == "strong":
        overall = "breakthrough_candidate_visible_axis"
    elif hidden_status == "strong" and internal_generation_status == "strong" and ablation_status in {"strong", "good"} and visible_status == "good_partial":
        overall = "strong_internal_axis_partial_visible_readout"
    elif hidden_status == "strong" and internal_generation_status == "strong" and ablation_status in {"strong", "good"}:
        overall = "strong_internal_axis_asymmetric_visible_readout"
    elif hidden_status == "strong" and internal_generation_status in {"strong", "partial"}:
        overall = "strong_internal_axis_visible_weak"
    elif hidden_status in {"strong", "medium"}:
        overall = "geometry_supported_behavior_open"
    else:
        overall = "preliminary_or_failed"

    next_action = "inspect manually"
    if overall == "breakthrough_candidate_visible_axis":
        next_action = "lock this protocol; replicate on another model family and second text family"
    elif overall == "strong_internal_axis_asymmetric_visible_readout":
        next_action = "move to next model/text family; frame X as ablation/suppression axis first"
    elif overall == "strong_internal_axis_visible_weak":
        next_action = "try neutral-only question set and held-out text family; do not chase larger alpha"
    elif hidden_status == "strong":
        next_action = "run behavioral-control-only with 32 random baselines and quality audit"

    return {
        "model_id": manifest.get("model_id", ""),
        "run_label": manifest.get("run_label", ""),
        "question_count": manifest.get("question_count", ""),
        "primary_alpha": primary_alpha_from_manifest(manifest),
        "primary_band": manifest.get("behavioral_control_primary_layer_band", "middle"),
        "behavioral_verdict": verdict.get("verdict", ""),
        "target_projection": target_projection,
        "length_neutral_projection": length_neutral,
        "word_shuffle_projection": word_shuffle,
        "sentence_shuffle_projection": sentence_shuffle,
        "hidden_status": hidden_status,
        "internal_generation_status": internal_generation_status,
        "visible_status": visible_status,
        "ablation_status": ablation_status,
        "overall_status": overall,
        "next_action": next_action,
    }


def primary_alpha_from_manifest(manifest: dict[str, Any]) -> float:
    primary_alpha = finite_float(manifest.get("behavioral_control_primary_alpha"))
    if np.isfinite(primary_alpha):
        return primary_alpha
    alpha_values = manifest.get("behavioral_control_alpha_values") or []
    return finite_float(alpha_values[-1] if alpha_values else manifest.get("behavioral_control_random_alpha", float("nan")))


def summary_row(run: dict[str, Any]) -> dict[str, Any]:
    hard = run["hard_random"]
    quality = run["quality"]
    return {
        "source": run["source"],
        "model_id": run["model_id"],
        "run_label": run["run_label"],
        "question_count": run["question_count"],
        "primary_alpha": run["primary_alpha"],
        "target_projection": run["target_projection"],
        "length_neutral_projection": run["length_neutral_projection"],
        "word_shuffle_projection": run["word_shuffle_projection"],
        "sentence_shuffle_projection": run["sentence_shuffle_projection"],
        "neutral_plus_x_likeness": hard["neutral_plus_x_likeness"],
        "neutral_plus_random_mean": hard["neutral_plus_random_mean"],
        "neutral_lift_random_mean": hard["neutral_plus_lift_random_mean"],
        "neutral_lift_random_p95": hard["neutral_plus_lift_random_p95"],
        "neutral_win_random_mean": hard["neutral_plus_win_random_mean"],
        "neutral_questions_over_random_mean": hard["neutral_plus_questions_over_random_mean"],
        "neutral_questions_over_random_p95": hard["neutral_plus_questions_over_random_p95"],
        "generation_projection": hard["neutral_plus_generation_projection"],
        "random_generation_mean": hard["neutral_plus_random_generation_mean"],
        "generation_win_random": hard["neutral_plus_generation_win_random"],
        "target_minus_suppression": hard["target_minus_suppression"],
        "target_minus_random_suppression_mean": hard["target_minus_random_suppression_mean"],
        "target_minus_supp_lift_random_mean": hard["target_minus_suppression_lift_random_mean"],
        "target_minus_supp_lift_random_p95": hard["target_minus_suppression_lift_random_p95"],
        "target_minus_supp_win_random_mean": hard["target_minus_suppression_win_random_mean"],
        "neutral_degeneration": quality["neutral_plus_degeneration"],
        "target_minus_degeneration": quality["target_minus_degeneration"],
        "hidden_status": run["hidden_status"],
        "internal_generation_status": run["internal_generation_status"],
        "visible_status": run["visible_status"],
        "ablation_status": run["ablation_status"],
        "overall_status": run["overall_status"],
        "next_action": run["next_action"],
        "has_hard_random_summary": run["has_hard_random_summary"],
        "has_breakthrough_audit": run["has_breakthrough_audit"],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                value = fmt(value)
            cells.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(runs: list[dict[str, Any]], summary_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Red-Team Hidden Geometry Metric Audit")
    lines.append("")
    lines.append("Reader-only audit. No model was run.")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    if not runs:
        lines.append("No valid runs were loaded.")
        return "\n".join(lines) + "\n"

    best_visible = summary_df.sort_values("neutral_lift_random_mean", ascending=False, na_position="last").head(1)
    best_ablation = summary_df.sort_values("target_minus_supp_lift_random_mean", ascending=False, na_position="last").head(1)
    strong_internal = summary_df[summary_df["hidden_status"].isin(["strong"]) & summary_df["internal_generation_status"].isin(["strong"])]
    lines.append(f"- Runs loaded: `{len(runs)}`")
    lines.append(f"- Strong internal axis runs: `{len(strong_internal)}/{len(summary_df)}`")
    if len(best_visible):
        row = best_visible.iloc[0]
        lines.append(
            "- Best neutral +X visible lift: "
            f"`{fmt(row['neutral_lift_random_mean'])}` on `{row['model_id']}` "
            f"(p95 lift `{fmt(row['neutral_lift_random_p95'])}`, "
            f"question win mean `{pct(row['neutral_questions_over_random_mean'])}`)."
        )
    if len(best_ablation):
        row = best_ablation.iloc[0]
        lines.append(
            "- Best target -X suppression lift: "
            f"`{fmt(row['target_minus_supp_lift_random_mean'])}` on `{row['model_id']}` "
            f"(p95 lift `{fmt(row['target_minus_supp_lift_random_p95'])}`, "
            f"random win `{pct(row['target_minus_supp_win_random_mean'])}`)."
        )

    statuses = summary_df["overall_status"].value_counts().to_dict()
    lines.append(f"- Overall statuses: `{json.dumps(statuses, ensure_ascii=False)}`")
    lines.append("")
    lines.append("Practical interpretation:")
    lines.append("")
    if (summary_df["overall_status"] == "breakthrough_candidate_visible_axis").any():
        lines.append("`Visible-axis candidate found.` Lock the protocol and replicate across another model family and another text family.")
    elif (summary_df["overall_status"] == "strong_internal_axis_asymmetric_visible_readout").any():
        lines.append("`Strong mechanism, asymmetric readout.` Frame Vector X first as an ablation/suppression axis; keep testing neutral +X separately.")
    elif (summary_df["hidden_status"] == "strong").any():
        lines.append("`Strong internal geometry.` The missing piece is hard visible readout over random p95/best and broader replication.")
    else:
        lines.append("`Preliminary.` Hidden geometry or behavioral artifacts need another run.")

    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    summary_cols = [
        "model_id",
        "target_projection",
        "neutral_lift_random_mean",
        "neutral_lift_random_p95",
        "neutral_questions_over_random_mean",
        "generation_win_random",
        "target_minus_supp_lift_random_mean",
        "target_minus_supp_win_random_mean",
        "neutral_degeneration",
        "visible_status",
        "ablation_status",
        "overall_status",
    ]
    lines.append(markdown_table(summary_df.to_dict("records"), summary_cols))

    for run in runs:
        hard = run["hard_random"]
        quality = run["quality"]
        lines.append("")
        lines.append(f"## Run: {run['model_id'] or run['source']}")
        lines.append("")
        lines.append(f"- Source: `{run['source']}`")
        lines.append(f"- Run label: `{run['run_label']}`")
        lines.append(f"- Verdict CSV: `{run.get('behavioral_verdict', '')}`")
        lines.append(f"- Existing hard-random file: `{run['has_hard_random_summary']}`")
        lines.append(f"- Existing breakthrough audit: `{run['has_breakthrough_audit']}`")
        lines.append(f"- Files in archive/directory: `{run['file_count']}`")
        lines.append("")
        lines.append("Key numbers:")
        lines.append("")
        lines.append(f"- Target projection: `{fmt(run['target_projection'])}`")
        lines.append(f"- Length-neutral projection: `{fmt(run['length_neutral_projection'])}`")
        lines.append(f"- Word-shuffle projection: `{fmt(run['word_shuffle_projection'])}`")
        lines.append(f"- Sentence-shuffle projection: `{fmt(run['sentence_shuffle_projection'])}`")
        lines.append(f"- Neutral +X target-likeness: `{fmt(hard['neutral_plus_x_likeness'])}`")
        lines.append(f"- Random visible mean: `{fmt(hard['neutral_plus_random_mean'])}`")
        lines.append(f"- Neutral +X lift over random mean: `{fmt(hard['neutral_plus_lift_random_mean'])}`")
        lines.append(f"- Neutral +X lift over random p95: `{fmt(hard['neutral_plus_lift_random_p95'])}`")
        lines.append(f"- Neutral +X question wins over random mean: `{pct(hard['neutral_plus_questions_over_random_mean'])}`")
        lines.append(f"- Neutral +X question wins over random p95: `{pct(hard['neutral_plus_questions_over_random_p95'])}`")
        lines.append(f"- Generation projection win over random: `{pct(hard['neutral_plus_generation_win_random'])}`")
        lines.append(f"- Target -X suppression lift over random mean: `{fmt(hard['target_minus_suppression_lift_random_mean'])}`")
        lines.append(f"- Target -X suppression lift over random p95: `{fmt(hard['target_minus_suppression_lift_random_p95'])}`")
        lines.append(f"- Target -X suppression win over random mean: `{pct(hard['target_minus_suppression_win_random_mean'])}`")
        lines.append(f"- Neutral +X degeneration: `{fmt(quality['neutral_plus_degeneration'])}`")
        lines.append(f"- Target -X degeneration: `{fmt(quality['target_minus_degeneration'])}`")
        lines.append("")
        lines.append(f"Status: `{run['overall_status']}`")
        lines.append("")
        lines.append(f"Next action: {run['next_action']}")

        family_rows = run.get("family_overview") or []
        if family_rows:
            lines.append("")
            lines.append("Full metric-family overview:")
            lines.append("")
            lines.append(markdown_table(family_rows, ["family", "status", "readout", "artifacts"]))

        artifact_rows = run.get("artifact_inventory") or []
        if artifact_rows:
            counts = Counter(row.get("family", "other") for row in artifact_rows)
            inventory_rows = [
                {
                    "family": family,
                    "file_count": count,
                    "total_mb": sum(float(row.get("size_mb", 0.0)) for row in artifact_rows if row.get("family") == family),
                }
                for family, count in sorted(counts.items())
            ]
            lines.append("")
            lines.append("Artifact inventory by family:")
            lines.append("")
            lines.append(markdown_table(inventory_rows, ["family", "file_count", "total_mb"]))

        expected_missing = run.get("expected_missing") or []
        lines.append("")
        lines.append("Expected artifact coverage:")
        lines.append("")
        if expected_missing:
            shown = expected_missing[:30]
            suffix = "" if len(expected_missing) <= len(shown) else f" ... plus {len(expected_missing) - len(shown)} more"
            lines.append(f"- Missing expected artifacts: `{len(expected_missing)}`")
            lines.append(f"- Missing list: `{', '.join(shown)}{suffix}`")
        else:
            lines.append("- Missing expected artifacts: `0`")

        csv_profiles = run.get("csv_profiles") or []
        if csv_profiles:
            status_counts = Counter(row.get("status", "") for row in csv_profiles)
            lines.append("")
            lines.append("CSV health and NaN coverage:")
            lines.append("")
            lines.append(f"- CSV status counts: `{json.dumps(dict(status_counts), ensure_ascii=False)}`")
            problem_rows = [
                row for row in csv_profiles
                if str(row.get("status", "")) != "read"
                or finite_float(row.get("null_fraction")) >= 0.10
                or finite_float(row.get("numeric_inf_cells")) > 0
            ]
            problem_rows = sorted(
                problem_rows,
                key=lambda row: (
                    str(row.get("status", "")) == "read",
                    -finite_float(row.get("null_fraction", 0.0)) if np.isfinite(finite_float(row.get("null_fraction", 0.0))) else 0,
                    -finite_float(row.get("size_mb", 0.0)) if np.isfinite(finite_float(row.get("size_mb", 0.0))) else 0,
                ),
            )[:15]
            if problem_rows:
                lines.append("")
                lines.append("CSV files needing attention:")
                lines.append("")
                lines.append(markdown_table(problem_rows, ["basename", "family", "status", "rows", "columns", "null_fraction", "numeric_inf_cells", "top_nan_columns"]))
            else:
                lines.append("- No read errors, high-NaN CSVs, or numeric infinities detected under the scan limit.")

            largest_csv = sorted(csv_profiles, key=lambda row: finite_float(row.get("size_mb", 0.0)), reverse=True)[:10]
            lines.append("")
            lines.append("Largest CSV artifacts scanned:")
            lines.append("")
            lines.append(markdown_table(largest_csv, ["basename", "family", "status", "size_mb", "rows", "columns", "null_fraction"]))

        per_q = hard.get("per_question_rows") or []
        if per_q:
            lines.append("")
            lines.append("Per-question neutral +X vs random:")
            lines.append("")
            lines.append(
                markdown_table(
                    per_q,
                    ["question_index", "x_likeness", "random_mean", "random_p95", "random_max", "x_minus_random_mean", "x_minus_random_p95"],
                )
            )

    lines.append("")
    lines.append("## Decision Gates")
    lines.append("")
    lines.append("- Hidden geometry strong: `target_projection >= 0.85` and length-neutral near zero.")
    lines.append("- Internal generation strong: neutral +X generation projection beats almost all random vectors.")
    lines.append("- Visible neutral +X strong: beats random p95 and wins most held-out questions without degeneration.")
    lines.append("- Target -X ablation strong: suppression beats random p95/mean without degeneration.")
    lines.append("- Breakthrough-grade claim needs cross-model and cross-text-family replication, not one run.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit red-team hidden-geometry result zips/directories.")
    parser.add_argument("paths", nargs="+", type=Path, help="Result zip files or directories.")
    parser.add_argument("--tag", type=str, default="", help="Audit tag. If provided with default output paths, output files are named with this tag.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Markdown output path.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV summary output path.")
    parser.add_argument("--artifact-csv", type=Path, default=DEFAULT_ARTIFACT_CSV, help="Full artifact inventory CSV output path.")
    parser.add_argument("--csv-profile", type=Path, default=DEFAULT_CSV_PROFILE, help="CSV health/profile output path.")
    parser.add_argument("--family-csv", type=Path, default=DEFAULT_FAMILY_CSV, help="Metric-family overview CSV output path.")
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV, help="Cumulative run-level history CSV.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR, help="Directory for timestamped copies of every audit output.")
    parser.add_argument("--history-mode", choices=["upsert", "append"], default="upsert", help="How to update cumulative history CSV.")
    parser.add_argument("--note", type=str, default="", help="Short note stored in the cumulative history.")
    parser.add_argument("--no-history", action="store_true", help="Do not update cumulative history or write timestamped copies.")
    parser.add_argument("--max-csv-mb", type=float, default=DEFAULT_MAX_CSV_MB, help="Skip individual CSV files larger than this many MB during full scan.")
    return parser.parse_args()


def apply_tagged_default_outputs(args: argparse.Namespace) -> str:
    tag = safe_tag(args.tag) if args.tag else ""
    if not tag:
        return ""
    replacements = {
        "out": Path(f"red_team_metric_audit_{tag}.md"),
        "csv": Path(f"red_team_metric_audit_{tag}.csv"),
        "artifact_csv": Path(f"red_team_metric_audit_{tag}_artifacts.csv"),
        "csv_profile": Path(f"red_team_metric_audit_{tag}_csv_profile.csv"),
        "family_csv": Path(f"red_team_metric_audit_{tag}_families.csv"),
    }
    defaults = {
        "out": DEFAULT_OUT,
        "csv": DEFAULT_CSV,
        "artifact_csv": DEFAULT_ARTIFACT_CSV,
        "csv_profile": DEFAULT_CSV_PROFILE,
        "family_csv": DEFAULT_FAMILY_CSV,
    }
    for attr, default_path in defaults.items():
        if getattr(args, attr) == default_path:
            setattr(args, attr, replacements[attr])
    return tag


def write_history_outputs(
    *,
    args: argparse.Namespace,
    tag: str,
    audit_timestamp: str,
    summary_df: pd.DataFrame,
    artifact_rows: list[dict[str, Any]],
    csv_profile_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    report: str,
) -> None:
    if args.no_history:
        return

    history_tag = tag or safe_tag(Path(str(args.paths[0])).stem if args.paths else f"audit_{local_timestamp_tag()}")
    snapshot_name = f"{local_timestamp_tag()}_{history_tag}"

    history_rows = summary_df.copy()
    if len(history_rows):
        history_rows.insert(0, "audit_timestamp_utc", audit_timestamp)
        history_rows.insert(1, "audit_tag", history_tag)
        history_rows.insert(2, "audit_note", args.note)
        history_rows["audit_report_path"] = str(args.out)

        if args.history_csv.exists():
            existing = pd.read_csv(args.history_csv, encoding="utf-8-sig")
            combined = pd.concat([existing, history_rows], ignore_index=True, sort=False)
        else:
            combined = history_rows

        if args.history_mode == "upsert":
            key_cols = [col for col in ["source", "model_id", "run_label", "primary_alpha"] if col in combined.columns]
            if key_cols:
                combined = combined.drop_duplicates(subset=key_cols, keep="last")
        args.history_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.history_csv, index=False, encoding="utf-8-sig")
        print(f"updated history csv: {args.history_csv}")

    args.history_dir.mkdir(parents=True, exist_ok=True)
    (args.history_dir / f"{snapshot_name}.md").write_text(report, encoding="utf-8")
    if len(summary_df):
        summary_df.to_csv(args.history_dir / f"{snapshot_name}_summary.csv", index=False, encoding="utf-8-sig")
    if artifact_rows:
        pd.DataFrame(artifact_rows).to_csv(args.history_dir / f"{snapshot_name}_artifacts.csv", index=False, encoding="utf-8-sig")
    if csv_profile_rows:
        pd.DataFrame(csv_profile_rows).to_csv(args.history_dir / f"{snapshot_name}_csv_profile.csv", index=False, encoding="utf-8-sig")
    if family_rows:
        pd.DataFrame(family_rows).to_csv(args.history_dir / f"{snapshot_name}_families.csv", index=False, encoding="utf-8-sig")
    print(f"saved history snapshot: {args.history_dir / (snapshot_name + '.md')}")


def main() -> int:
    args = parse_args()
    tag = apply_tagged_default_outputs(args)
    audit_timestamp = utc_timestamp()
    runs = []
    errors = []
    max_csv_bytes = int(max(1.0, float(args.max_csv_mb)) * 1024 * 1024)
    for path in args.paths:
        try:
            runs.append(load_run(path, max_csv_bytes=max_csv_bytes))
        except Exception as exc:
            errors.append((str(path), repr(exc)))

    summary_df = pd.DataFrame([summary_row(run) for run in runs])
    if len(summary_df):
        summary_df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    artifact_rows = []
    csv_profile_rows = []
    family_rows = []
    for run in runs:
        source = run.get("source", "")
        model_id = run.get("model_id", "")
        run_label = run.get("run_label", "")
        for row in run.get("artifact_inventory", []):
            enriched = dict(row)
            enriched.setdefault("source", source)
            enriched["model_id"] = model_id
            enriched["run_label"] = run_label
            artifact_rows.append(enriched)
        for row in run.get("csv_profiles", []):
            enriched = dict(row)
            enriched["source"] = source
            enriched["model_id"] = model_id
            enriched["run_label"] = run_label
            csv_profile_rows.append(enriched)
        for row in run.get("family_overview", []):
            enriched = dict(row)
            enriched["source"] = source
            enriched["model_id"] = model_id
            enriched["run_label"] = run_label
            family_rows.append(enriched)

    if artifact_rows:
        pd.DataFrame(artifact_rows).to_csv(args.artifact_csv, index=False, encoding="utf-8-sig")
    if csv_profile_rows:
        pd.DataFrame(csv_profile_rows).to_csv(args.csv_profile, index=False, encoding="utf-8-sig")
    if family_rows:
        pd.DataFrame(family_rows).to_csv(args.family_csv, index=False, encoding="utf-8-sig")

    report = render_report(runs, summary_df)
    if errors:
        report += "\n## Load Errors\n\n"
        for path, err in errors:
            report += f"- `{path}`: `{err}`\n"
    args.out.write_text(report, encoding="utf-8")
    write_history_outputs(
        args=args,
        tag=tag,
        audit_timestamp=audit_timestamp,
        summary_df=summary_df,
        artifact_rows=artifact_rows,
        csv_profile_rows=csv_profile_rows,
        family_rows=family_rows,
        report=report,
    )
    print(f"saved markdown: {args.out}")
    if len(summary_df):
        print(f"saved csv: {args.csv}")
    if artifact_rows:
        print(f"saved artifact csv: {args.artifact_csv}")
    if csv_profile_rows:
        print(f"saved csv profile: {args.csv_profile}")
    if family_rows:
        print(f"saved family csv: {args.family_csv}")
    if errors:
        print("load errors:", errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
