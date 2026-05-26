"""
Read-only analyzer for hidden-geometry result packages.

The analyzer accepts a finished result directory or zip archive, never modifies
the source package, and writes an external audit bundle under --out.

It is intentionally conservative: missing artifacts become not_available_*
statuses, machine CSV outputs contain numeric/status/source fields, and human
interpretation is limited to analysis_summary.md with explicit source files.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
}

DEFAULT_SCOREBOARD_COLUMNS = [
    "run_label",
    "model_id",
    "gate",
    "result_path",
    "valid_package",
    "decoder_ok",
    "decoder_layer_source",
    "decoder_layer_count",
    "expected_decoder_layer_count",
    "prompt_budget_ok",
    "numeric_integrity_ok",
    "geometry_pass",
    "specificity_pass",
    "strict_causal_symmetry_pass",
    "behavior_random_p95_pass",
    "gate4_component_available",
    "best_layer_band",
    "best_axis",
    "best_component",
    "best_alpha",
    "target_middle_projection",
    "target_direction_cosine",
    "observed_minus_random_null",
    "best_target_control_gap",
    "strict_causal_symmetry_score",
    "behavior_random_p95_score",
    "main_failure_code",
    "recommended_next_experiment",
]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("_") or "unnamed"


def is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "ok"}


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def write_csv(path: Path, rows_or_df: Any, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df.copy()
    else:
        df = pd.DataFrame(rows_or_df)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(columns)]
    df.to_csv(path, index=False)
    return df


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class PackageFile:
    name: str
    size: int


class ResultPackage:
    def __init__(self, source: Path):
        self.source = Path(source)
        self._zip: Optional[zipfile.ZipFile] = None
        self._names: Optional[List[str]] = None
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.source)
        elif not self.source.is_dir():
            raise FileNotFoundError(f"Result package not found or unsupported: {self.source}")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def names(self) -> List[str]:
        if self._names is not None:
            return self._names
        if self._zip is not None:
            self._names = [info.filename.replace("\\", "/") for info in self._zip.infolist() if not info.is_dir()]
        else:
            self._names = [
                str(path.relative_to(self.source)).replace("\\", "/")
                for path in self.source.rglob("*")
                if path.is_file()
            ]
        return self._names

    def exists(self, name: str) -> bool:
        return name.replace("\\", "/") in set(self.names())

    def inventory(self) -> List[PackageFile]:
        if self._zip is not None:
            return [
                PackageFile(info.filename.replace("\\", "/"), int(info.file_size))
                for info in self._zip.infolist()
                if not info.is_dir()
            ]
        return [
            PackageFile(str(path.relative_to(self.source)).replace("\\", "/"), int(path.stat().st_size))
            for path in self.source.rglob("*")
            if path.is_file()
        ]

    def read_bytes(self, name: str) -> bytes:
        name = name.replace("\\", "/")
        if self._zip is not None:
            return self._zip.read(name)
        return (self.source / name).read_bytes()

    def read_text(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8")

    def read_json(self, name: str) -> Dict[str, Any]:
        return json.loads(self.read_text(name))

    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        data = self.read_bytes(name)
        return pd.read_csv(io.BytesIO(data), **kwargs)


class Analyzer:
    def __init__(self, package: ResultPackage, out_dir: Path, run_label_arg: str = ""):
        self.package = package
        self.out_dir = Path(out_dir)
        self.run_label_arg = run_label_arg
        self.cache: Dict[str, Optional[pd.DataFrame]] = {}
        self.manifest: Dict[str, Any] = {}
        self.anomalies: List[Dict[str, Any]] = []
        self.top_peaks: List[Dict[str, Any]] = []

    def add_anomaly(
        self,
        severity: str,
        artifact: str,
        metric: str,
        observed_value: Any,
        expected_rule: str,
        failure_code: str,
        source_file: str,
    ) -> None:
        self.anomalies.append(
            {
                "severity": severity,
                "artifact": artifact,
                "metric": metric,
                "observed_value": observed_value,
                "expected_rule": expected_rule,
                "failure_code": failure_code,
                "source_file": source_file,
            }
        )

    def add_peak(self, category: str, metric: str, value: Any, source_file: str, detail: str = "") -> None:
        self.top_peaks.append(
            {
                "category": category,
                "metric": metric,
                "value": value,
                "source_file": source_file,
                "detail": detail,
            }
        )

    def has(self, name: str) -> bool:
        return self.package.exists(name)

    def csv(self, name: str, max_bytes: Optional[int] = None) -> Optional[pd.DataFrame]:
        if name in self.cache:
            return self.cache[name]
        if not self.has(name):
            self.cache[name] = None
            return None
        if max_bytes is not None:
            size = next((f.size for f in self.package.inventory() if f.name == name), 0)
            if size > max_bytes:
                self.cache[name] = None
                self.add_anomaly(
                    "info",
                    name,
                    "csv_read_skipped",
                    size,
                    f"file size must be <= {max_bytes} for lightweight analyzer read",
                    "not_available_large_csv_skipped",
                    name,
                )
                return None
        try:
            df = self.package.read_csv(name)
            self.cache[name] = df
            return df
        except Exception as exc:
            self.cache[name] = None
            self.add_anomaly("high", name, "csv_read", repr(exc), "CSV should be readable", "csv_read_failed", name)
            return None

    def load_manifest(self) -> None:
        if self.has("red_team_input_manifest.json"):
            try:
                self.manifest = self.package.read_json("red_team_input_manifest.json")
            except Exception as exc:
                self.add_anomaly(
                    "high",
                    "red_team_input_manifest.json",
                    "manifest_read",
                    repr(exc),
                    "manifest should be valid JSON",
                    "manifest_read_failed",
                    "red_team_input_manifest.json",
                )
                self.manifest = {}
        else:
            self.manifest = {}
            self.add_anomaly(
                "high",
                "red_team_input_manifest.json",
                "manifest_presence",
                "missing",
                "result package should include red_team_input_manifest.json",
                "not_available_manifest_missing",
                "red_team_input_manifest.json",
            )

    def validity_flags(self) -> Dict[str, Any]:
        names = set(self.package.names())
        decoder_count = self.manifest.get("decoder_layer_count")
        expected_count = self.manifest.get("expected_decoder_layer_count")
        decoder_source = str(self.manifest.get("decoder_layer_source", ""))
        decoder_mismatch = to_bool(self.manifest.get("decoder_layer_count_mismatch", False))
        decoder_ok = (
            is_finite(decoder_count)
            and int(float(decoder_count)) > 0
            and not decoder_mismatch
            and decoder_source
            and decoder_source != "not_found"
            and "disabled" not in decoder_source
        )
        if not decoder_ok:
            self.add_anomaly(
                "high",
                "red_team_input_manifest.json",
                "decoder_ok",
                f"source={decoder_source}, count={decoder_count}, expected={expected_count}, mismatch={decoder_mismatch}",
                "decoder layer source must be found and match expected config count",
                "decoder_not_clean",
                "red_team_input_manifest.json",
            )

        prompt_budget_ok = "prompt_budget_overflow_warnings.csv" not in names
        if not prompt_budget_ok:
            overflow = self.csv("prompt_budget_overflow_warnings.csv")
            rows = len(overflow) if overflow is not None else "unknown"
            self.add_anomaly(
                "high",
                "prompt_budget_overflow_warnings.csv",
                "prompt_budget_ok",
                rows,
                "prompt budget warning file should be absent or empty",
                "prompt_budget_overflow",
                "prompt_budget_overflow_warnings.csv",
            )

        numeric_path = "analysis_notes/extracted_narrative_columns/numeric_integrity_check.csv"
        numeric_df = self.csv(numeric_path)
        numeric_integrity_ok = False
        if numeric_df is not None and "status" in numeric_df.columns:
            numeric_integrity_ok = not (numeric_df["status"].astype(str).str.lower() == "fail").any()
            if not numeric_integrity_ok:
                self.add_anomaly(
                    "high",
                    numeric_path,
                    "numeric_integrity_ok",
                    "fail",
                    "numeric integrity rows must not contain status=fail",
                    "numeric_integrity_failed",
                    numeric_path,
                )
        else:
            self.add_anomaly(
                "medium",
                numeric_path,
                "numeric_integrity_ok",
                "missing",
                "clean-evidence packages should include numeric integrity audit",
                "not_available_numeric_integrity",
                numeric_path,
            )

        arch_df = self.csv("architecture_module_delta_summary.csv", max_bytes=50_000_000)
        architecture_hooks_ok = arch_df is not None and len(arch_df) > 0
        if not architecture_hooks_ok:
            self.add_anomaly(
                "medium",
                "architecture_module_delta_summary.csv",
                "architecture_hooks_ok",
                "missing_or_empty",
                "architecture/module delta table should be non-empty for mechanistic evidence",
                "not_available_architecture_hooks",
                "architecture_module_delta_summary.csv",
            )

        causal_status_present = "causal_intervention_status.csv" in names
        if causal_status_present:
            status_df = self.csv("causal_intervention_status.csv")
            observed = status_df.to_dict("records")[:3] if status_df is not None else "unreadable"
            self.add_anomaly(
                "high",
                "causal_intervention_status.csv",
                "causal_intervention_status",
                observed,
                "causal_intervention_status.csv should be absent for completed causal blocks",
                "causal_intervention_status_present",
                "causal_intervention_status.csv",
            )

        quarantine_path = "analysis_notes/extracted_narrative_columns/quarantine_index.csv"
        q_df = self.csv(quarantine_path)
        if q_df is not None and "status" in q_df.columns:
            real_q = q_df[~q_df["status"].astype(str).eq("no_quarantine_needed")]
            if len(real_q):
                self.add_anomaly(
                    "info",
                    quarantine_path,
                    "quarantine_rows",
                    len(real_q),
                    "quarantined narrative should be reviewed but does not alter numeric evidence",
                    "quarantine_contains_removed_values",
                    quarantine_path,
                )

        causal_blocks_available = any(
            name in names
            for name in [
                "causal_symmetry_score_summary.csv",
                "causal_bidirectional_symmetry_summary.csv",
                "grade4_axis_component_causal_symmetry_summary.csv",
                "grade4_axis_component_causal_projection_summary.csv",
            ]
        )
        behavior_blocks_available = any(
            name in names
            for name in [
                "behavior_random_p95_gate.csv",
                "behavioral_control_axis_hard_random_summary.csv",
                "behavioral_control_axis_similarity_summary.csv",
            ]
        )
        gate4_detected = any(name.startswith("grade4_axis_") for name in names)
        gate3_detected = any(
            name in names
            for name in [
                "middle_layer_condition_summary.csv",
                "paired_target_vs_control_tests.csv",
                "claim_ladder_final.csv",
            ]
        )
        valid_package = bool(self.manifest) and gate3_detected

        return {
            "valid_package": valid_package,
            "decoder_ok": decoder_ok,
            "decoder_layer_source": decoder_source,
            "decoder_layer_count": decoder_count,
            "expected_decoder_layer_count": expected_count,
            "prompt_budget_ok": prompt_budget_ok,
            "numeric_integrity_ok": numeric_integrity_ok,
            "architecture_hooks_ok": architecture_hooks_ok,
            "causal_blocks_available": causal_blocks_available and not causal_status_present,
            "behavior_blocks_available": behavior_blocks_available,
            "gate3_detected": gate3_detected,
            "gate4_detected": gate4_detected,
        }

    def claim_ladder(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        df = self.csv("claim_ladder_final.csv")
        levels: Dict[str, Dict[str, Any]] = {}
        pass_vector: Dict[str, Any] = {}
        if df is None or not len(df):
            self.add_anomaly(
                "medium",
                "claim_ladder_final.csv",
                "claim_ladder",
                "missing",
                "claim_ladder_final.csv should exist for complete reviewer scoring",
                "not_available_claim_ladder",
                "claim_ladder_final.csv",
            )
            return levels, pass_vector
        for _, row in df.iterrows():
            level_name = str(row.get("level_name", ""))
            key = safe_name(level_name.lower())
            levels[key] = row.to_dict()
            pass_vector[key] = int(to_float(row.get("pass", 0), 0))
        return levels, pass_vector

    def level_by_substring(self, levels: Dict[str, Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        text = text.lower()
        for key, row in levels.items():
            if text in key.lower() or text in str(row.get("level_name", "")).lower():
                return row
        return None

    def extract_primary_metrics(self, flags: Dict[str, Any]) -> Dict[str, Any]:
        levels, pass_vector = self.claim_ladder()
        geometry_level = self.level_by_substring(levels, "geometry")
        specificity_level = self.level_by_substring(levels, "specificity")
        causal_level = self.level_by_substring(levels, "causal")
        behavioral_level = self.level_by_substring(levels, "behavior")
        mechanistic_level = self.level_by_substring(levels, "mechanistic")

        mid = self.csv("middle_layer_condition_summary.csv")
        target_middle_projection = float("nan")
        target_direction_cosine = float("nan")
        if mid is not None and len(mid):
            target = mid[mid.get("condition", pd.Series(dtype=str)).astype(str).eq("target")]
            if len(target):
                target_middle_projection = to_float(target.iloc[0].get("projection_fraction_on_vector_x_loo_mean"))
                target_direction_cosine = to_float(target.iloc[0].get("direction_cosine_with_vector_x_loo_mean"))

        null_df = self.csv("null_vector_baseline_summary.csv")
        observed_minus_random_null = float("nan")
        if null_df is not None and len(null_df):
            observed_minus_random_null = to_float(null_df.iloc[0].get("observed_minus_null_mean"))

        paired = self.csv("paired_target_vs_control_tests.csv")
        best_target_control_gap = float("nan")
        if paired is not None and len(paired) and "metric" in paired.columns:
            sub = paired[paired["metric"].astype(str).eq("mean_projection_fraction_on_vector_x_loo")]
            if len(sub) and "target_minus_control_mean" in sub.columns:
                best_target_control_gap = float(pd.to_numeric(sub["target_minus_control_mean"], errors="coerce").max())

        causal_score = to_float(causal_level.get("score")) if causal_level else float("nan")
        if not np.isfinite(causal_score):
            sym = self.csv("causal_symmetry_score_summary.csv")
            if sym is not None and len(sym) and "symmetry_pass_rate" in sym.columns:
                causal_score = float(pd.to_numeric(sym["symmetry_pass_rate"], errors="coerce").mean())

        behavior_score = to_float(behavioral_level.get("score")) if behavioral_level else float("nan")
        if not np.isfinite(behavior_score):
            behavior_gate = self.csv("behavior_random_p95_gate.csv")
            if behavior_gate is not None and len(behavior_gate) and "win_rate_vs_random_p95" in behavior_gate.columns:
                behavior_score = float(pd.to_numeric(behavior_gate["win_rate_vs_random_p95"], errors="coerce").mean())

        geometry_score = to_float(geometry_level.get("score")) if geometry_level else float("nan")
        specificity_score = to_float(specificity_level.get("score")) if specificity_level else float("nan")
        mechanistic_score = to_float(mechanistic_level.get("score")) if mechanistic_level else float("nan")

        geometry_pass = bool(to_float(geometry_level.get("pass"), 0) > 0) if geometry_level else bool(target_middle_projection > 0.5)
        specificity_pass = bool(to_float(specificity_level.get("pass"), 0) > 0) if specificity_level else bool(best_target_control_gap > 0.0)
        strict_causal_symmetry_pass = bool(to_float(causal_level.get("pass"), 0) > 0) if causal_level else bool(causal_score >= 0.5)
        behavior_random_p95_pass = bool(to_float(behavioral_level.get("pass"), 0) > 0) if behavioral_level else bool(behavior_score >= 0.5)

        failure_codes = []
        for row in levels.values():
            if to_float(row.get("pass", 1), 1) == 0:
                code = str(row.get("failure_code", "")).strip()
                failure_codes.append(code if code and code.lower() != "nan" else "below_threshold")
        if not flags.get("decoder_ok"):
            failure_codes.insert(0, "decoder_not_clean")
        if not flags.get("prompt_budget_ok"):
            failure_codes.insert(0, "prompt_budget_overflow")
        if not flags.get("numeric_integrity_ok"):
            failure_codes.append("not_available_numeric_integrity")

        gate4_component_available = flags.get("gate4_detected", False)
        component_order_orth_pass = None
        component_df = self.csv("grade4_axis_component_causal_rank_summary.csv")
        if component_df is not None and len(component_df):
            cols = {c.lower(): c for c in component_df.columns}
            axis_col = cols.get("axis_name") or cols.get("axis")
            if axis_col:
                oo = component_df[component_df[axis_col].astype(str).eq("x_order_orth")]
                if len(oo):
                    numeric_cols = [c for c in oo.columns if c.lower() in {"rank_score", "mean_gap", "mean_plus_minus_gap", "component_score"}]
                    if numeric_cols:
                        component_order_orth_pass = bool(pd.to_numeric(oo[numeric_cols[0]], errors="coerce").max() > 0)

        return {
            "claim_ladder_pass_vector": pass_vector,
            "geometry_score": geometry_score,
            "specificity_score": specificity_score,
            "causal_symmetry_score": causal_score,
            "behavior_random_p95_score": behavior_score,
            "mechanistic_localization_score": mechanistic_score,
            "geometry_pass": geometry_pass,
            "specificity_pass": specificity_pass,
            "strict_causal_symmetry_pass": strict_causal_symmetry_pass,
            "behavior_random_p95_pass": behavior_random_p95_pass,
            "component_order_orth_pass": component_order_orth_pass,
            "gate4_component_available": gate4_component_available,
            "target_middle_projection": target_middle_projection,
            "target_direction_cosine": target_direction_cosine,
            "observed_minus_random_null": observed_minus_random_null,
            "best_target_control_gap": best_target_control_gap,
            "main_failure_code": ";".join(dict.fromkeys([c for c in failure_codes if c])) or "",
        }

    def geometry_peaks(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        mid = self.csv("middle_layer_condition_summary.csv")
        if mid is not None and len(mid):
            for _, row in mid.iterrows():
                rows.append(
                    {
                        "peak_type": "condition_middle_summary",
                        "condition": row.get("condition"),
                        "layer": np.nan,
                        "question_index": np.nan,
                        "metric": "projection_fraction_on_vector_x_loo_mean",
                        "value": row.get("projection_fraction_on_vector_x_loo_mean"),
                        "direction_cosine": row.get("direction_cosine_with_vector_x_loo_mean"),
                        "source_file": "middle_layer_condition_summary.csv",
                    }
                )
        layer = self.csv("layerwise_geometry_summary.csv", max_bytes=20_000_000)
        if layer is not None and len(layer) and "condition" in layer.columns:
            target = layer[layer["condition"].astype(str).eq("target")].copy()
            for metric in ["mean_projection_fraction_on_vector_x_loo", "mean_direction_cosine_with_vector_x_loo"]:
                if metric in target.columns:
                    top = target.sort_values(metric, ascending=False).head(10)
                    for _, row in top.iterrows():
                        rows.append(
                            {
                                "peak_type": "target_layer_peak",
                                "condition": "target",
                                "layer": row.get("layer"),
                                "question_index": np.nan,
                                "metric": metric,
                                "value": row.get(metric),
                                "direction_cosine": row.get("mean_direction_cosine_with_vector_x_loo"),
                                "source_file": "layerwise_geometry_summary.csv",
                            }
                        )
        q = self.csv("question_level_middle_layer_summary.csv")
        if q is not None and len(q) and "condition" in q.columns:
            target = q[q["condition"].astype(str).eq("target")].copy()
            metric = "mean_projection_fraction_on_vector_x_loo"
            if len(target) and metric in target.columns:
                for peak_type, sub in [
                    ("strongest_question", target.sort_values(metric, ascending=False).head(5)),
                    ("weakest_question", target.sort_values(metric, ascending=True).head(5)),
                ]:
                    for _, row in sub.iterrows():
                        rows.append(
                            {
                                "peak_type": peak_type,
                                "condition": "target",
                                "layer": np.nan,
                                "question_index": row.get("question_index"),
                                "metric": metric,
                                "value": row.get(metric),
                                "direction_cosine": row.get("mean_direction_cosine_with_vector_x_loo"),
                                "source_file": "question_level_middle_layer_summary.csv",
                            }
                        )
        df = pd.DataFrame(rows)
        if len(df):
            target_mid = df[(df["peak_type"] == "condition_middle_summary") & (df["condition"].astype(str) == "target")]
            if len(target_mid):
                self.add_peak("geometry", "target_middle_projection", target_mid.iloc[0]["value"], "middle_layer_condition_summary.csv")
        return df

    def specificity_peaks(self, target_middle_projection: float) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        paired = self.csv("paired_target_vs_control_tests.csv")
        if paired is not None and len(paired):
            sub = paired[paired.get("metric", pd.Series(dtype=str)).astype(str).eq("mean_projection_fraction_on_vector_x_loo")]
            for _, row in sub.sort_values("target_minus_control_mean", ascending=False).iterrows():
                rows.append(
                    {
                        "peak_type": "paired_target_control_gap",
                        "control_condition": row.get("control_condition"),
                        "layer": np.nan,
                        "metric": row.get("metric"),
                        "value": row.get("target_minus_control_mean"),
                        "p_value": row.get("paired_sign_permutation_p"),
                        "fdr_q_value": row.get("fdr_q_value"),
                        "fdr_significant": row.get("fdr_significant"),
                        "source_file": "paired_target_vs_control_tests.csv",
                    }
                )
        spec = self.csv("geometry_specificity_summary.csv")
        if spec is not None and len(spec):
            for _, row in spec.sort_values("specificity_lift", ascending=False).iterrows():
                rows.append(
                    {
                        "peak_type": "geometry_specificity_lift",
                        "control_condition": row.get("control_condition"),
                        "layer": np.nan,
                        "metric": "specificity_lift",
                        "value": row.get("specificity_lift"),
                        "p_value": np.nan,
                        "fdr_q_value": np.nan,
                        "fdr_significant": row.get("pass_specificity"),
                        "source_file": "geometry_specificity_summary.csv",
                    }
                )
                control_mean = to_float(row.get("control_mean_projection"))
                if np.isfinite(target_middle_projection) and np.isfinite(control_mean) and control_mean > 0.8 * target_middle_projection:
                    self.add_anomaly(
                        "medium",
                        "geometry_specificity_summary.csv",
                        "control_leakage",
                        f"{row.get('control_condition')}={control_mean}",
                        "control projection should stay well below target projection",
                        "control_projection_close_to_target",
                        "geometry_specificity_summary.csv",
                    )
        fdr = self.csv("layerwise_fdr_target_vs_control.csv")
        if fdr is not None and len(fdr):
            sig = fdr[pd.to_numeric(fdr.get("fdr_significant", 0), errors="coerce").fillna(0) > 0]
            if len(sig):
                grouped = sig.groupby("control_condition", as_index=False).agg(
                    significant_layers=("layer", "nunique"),
                    max_target_minus_control=("target_minus_control_mean", "max"),
                )
                for _, row in grouped.sort_values("significant_layers", ascending=False).iterrows():
                    rows.append(
                        {
                            "peak_type": "fdr_significant_layer_cluster",
                            "control_condition": row.get("control_condition"),
                            "layer": "multi",
                            "metric": "significant_layers",
                            "value": row.get("significant_layers"),
                            "p_value": np.nan,
                            "fdr_q_value": np.nan,
                            "fdr_significant": 1,
                            "source_file": "layerwise_fdr_target_vs_control.csv",
                        }
                    )
        df = pd.DataFrame(rows)
        if len(df):
            best = pd.to_numeric(df["value"], errors="coerce").idxmax()
            if pd.notna(best):
                row = df.loc[best]
                self.add_peak("specificity", str(row.get("peak_type")), row.get("value"), row.get("source_file", ""))
        return df

    def component_peaks(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        file_specs = [
            ("grade4_axis_component_norm_summary.csv", "component_norm"),
            ("grade4_axis_projection_geometry_summary.csv", "component_projection"),
            ("grade4_axis_component_causal_projection_summary.csv", "component_causal_projection"),
            ("grade4_axis_component_causal_symmetry_summary.csv", "component_causal_symmetry"),
            ("grade4_axis_component_causal_alpha_scaling_summary.csv", "component_alpha_scaling"),
            ("grade4_axis_component_causal_rank_summary.csv", "component_rank"),
        ]
        for file_name, peak_type in file_specs:
            df = self.csv(file_name, max_bytes=50_000_000)
            if df is None or not len(df):
                continue
            axis_col = next((c for c in ["axis_name", "axis", "component", "component_name"] if c in df.columns), "")
            band_col = next((c for c in ["band", "layer_band", "intervention_layer_band", "readout_layer_band"] if c in df.columns), "")
            metric_cols = [
                c
                for c in [
                    "order_orth_energy_fraction_of_full",
                    "mean_projection_fraction_on_axis_loo",
                    "plus_minus_projection_gap",
                    "mean_plus_minus_gap",
                    "alpha_projection_slope",
                    "rank_score",
                    "component_score",
                ]
                if c in df.columns
            ]
            if not metric_cols:
                numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                metric_cols = numeric[:1]
            for metric in metric_cols[:3]:
                top = df.copy()
                top["_metric_value"] = pd.to_numeric(top[metric], errors="coerce")
                top = top.sort_values("_metric_value", ascending=False).head(15)
                for _, row in top.iterrows():
                    rows.append(
                        {
                            "peak_type": peak_type,
                            "axis_name": row.get(axis_col, "") if axis_col else "",
                            "component": row.get(axis_col, "") if axis_col else "",
                            "layer_band": row.get(band_col, "") if band_col else "",
                            "alpha": row.get("alpha", row.get("alpha_abs", np.nan)),
                            "metric": metric,
                            "value": row.get(metric),
                            "source_file": file_name,
                        }
                    )
        df = pd.DataFrame(rows)
        oo = df[df.get("axis_name", pd.Series(dtype=str)).astype(str).eq("x_order_orth")] if len(df) else pd.DataFrame()
        if len(oo):
            top_idx = pd.to_numeric(oo["value"], errors="coerce").idxmax()
            if pd.notna(top_idx):
                row = oo.loc[top_idx]
                self.add_peak("component", "x_order_orth_peak", row.get("value"), row.get("source_file", ""))
        return df

    def causal_peaks(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        bidir = self.csv("causal_bidirectional_symmetry_summary.csv")
        if bidir is not None and len(bidir):
            top = bidir.copy()
            if "plus_minus_projection_gap" in top.columns:
                top["_abs_gap"] = pd.to_numeric(top["plus_minus_projection_gap"], errors="coerce").abs()
                for _, row in top.sort_values("_abs_gap", ascending=False).head(20).iterrows():
                    rows.append(
                        {
                            "peak_type": "bidirectional_gap",
                            "layer_band": row.get("layer_band"),
                            "alpha": row.get("alpha_abs"),
                            "base_condition": row.get("base_condition"),
                            "metric": "plus_minus_projection_gap",
                            "value": row.get("plus_minus_projection_gap"),
                            "source_file": "causal_bidirectional_symmetry_summary.csv",
                        }
                    )
        strict = self.csv("causal_symmetry_score_summary.csv")
        if strict is not None and len(strict):
            for _, row in strict.iterrows():
                rows.append(
                    {
                        "peak_type": "strict_symmetry",
                        "layer_band": row.get("layer_band"),
                        "alpha": row.get("alpha"),
                        "base_condition": "",
                        "metric": "symmetry_pass_rate",
                        "value": row.get("symmetry_pass_rate"),
                        "source_file": "causal_symmetry_score_summary.csv",
                    }
                )
        alpha = self.csv("causal_alpha_scaling_summary.csv")
        if alpha is not None and len(alpha):
            for _, row in alpha.iterrows():
                rows.append(
                    {
                        "peak_type": "alpha_dose_slope",
                        "layer_band": row.get("layer_band"),
                        "alpha": "slope",
                        "base_condition": row.get("base_condition"),
                        "metric": "alpha_projection_slope",
                        "value": row.get("alpha_projection_slope"),
                        "source_file": "causal_alpha_scaling_summary.csv",
                    }
                )
        grade4_sym = self.csv("grade4_axis_component_causal_symmetry_summary.csv")
        if grade4_sym is not None and len(grade4_sym):
            value_col = next((c for c in ["plus_minus_projection_gap", "mean_plus_minus_gap", "symmetry_pass_rate"] if c in grade4_sym.columns), "")
            if value_col:
                for _, row in grade4_sym.sort_values(value_col, ascending=False).head(30).iterrows():
                    rows.append(
                        {
                            "peak_type": "grade4_component_causal",
                            "layer_band": row.get("layer_band", row.get("intervention_layer_band", "")),
                            "alpha": row.get("alpha", row.get("alpha_abs", "")),
                            "base_condition": row.get("base_condition", ""),
                            "metric": value_col,
                            "value": row.get(value_col),
                            "source_file": "grade4_axis_component_causal_symmetry_summary.csv",
                        }
                    )
        return pd.DataFrame(rows)

    def behavior_peaks(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        gate = self.csv("behavior_random_p95_gate.csv")
        if gate is not None and len(gate):
            for _, row in gate.iterrows():
                rows.append(
                    {
                        "peak_type": "random_p95_gate",
                        "layer_band": row.get("layer_band"),
                        "alpha": row.get("alpha"),
                        "metric": "plus_specific_lift",
                        "value": row.get("plus_specific_lift"),
                        "quality_or_win_rate": row.get("win_rate_vs_random_p95"),
                        "source_file": "behavior_random_p95_gate.csv",
                    }
                )
                if is_finite(row.get("plus_specific_lift")) and to_float(row.get("plus_specific_lift")) <= 0:
                    self.add_anomaly(
                        "medium",
                        "behavior_random_p95_gate.csv",
                        "random_p95_not_beaten",
                        row.get("plus_specific_lift"),
                        "behavioral plus effect should exceed random p95 for behavioral support",
                        "below_random_p95",
                        "behavior_random_p95_gate.csv",
                    )
        hard = self.csv("behavioral_control_axis_hard_random_summary.csv")
        if hard is not None and len(hard):
            for _, row in hard.iterrows():
                rows.append(
                    {
                        "peak_type": "hard_random_summary",
                        "layer_band": row.get("layer_band"),
                        "alpha": row.get("alpha_abs"),
                        "metric": "mean_lift_over_random_p95",
                        "value": row.get("mean_lift_over_random_p95"),
                        "quality_or_win_rate": row.get("win_rate_vs_random_p95"),
                        "source_file": "behavioral_control_axis_hard_random_summary.csv",
                    }
                )
        quality = self.csv("quality_adjusted_behavior_summary.csv")
        if quality is not None and len(quality):
            for _, row in quality.iterrows():
                rows.append(
                    {
                        "peak_type": "quality_adjusted_behavior",
                        "layer_band": row.get("layer_band"),
                        "alpha": row.get("alpha"),
                        "metric": "quality_adjusted_effect",
                        "value": row.get("quality_adjusted_effect"),
                        "quality_or_win_rate": row.get("degeneration_rate"),
                        "source_file": "quality_adjusted_behavior_summary.csv",
                    }
                )
                if is_finite(row.get("degeneration_rate")) and to_float(row.get("degeneration_rate")) >= 0.5:
                    self.add_anomaly(
                        "medium",
                        "quality_adjusted_behavior_summary.csv",
                        "quality_degenerate_rate",
                        row.get("degeneration_rate"),
                        "degeneration_rate should stay below 0.5 for visible behavior evidence",
                        "quality_degeneration_high",
                        "quality_adjusted_behavior_summary.csv",
                    )
        coupling = self.csv("internal_visible_coupling_summary.csv")
        if coupling is not None and len(coupling):
            for _, row in coupling.iterrows():
                rows.append(
                    {
                        "peak_type": "internal_visible_coupling",
                        "layer_band": row.get("layer_band"),
                        "alpha": row.get("alpha"),
                        "metric": "pearson_r",
                        "value": row.get("pearson_r"),
                        "quality_or_win_rate": row.get("pass_coupling"),
                        "source_file": "internal_visible_coupling_summary.csv",
                    }
                )
        return pd.DataFrame(rows)

    def architecture_peaks(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        arch = self.csv("architecture_module_delta_summary.csv", max_bytes=50_000_000)
        if arch is not None and len(arch):
            work = arch.copy()
            if "condition" in work.columns:
                work = work[work["condition"].astype(str).eq("target")]
            metric = "mean_abs_delta" if "mean_abs_delta" in work.columns else "l2_distance_to_reference"
            if metric in work.columns:
                top = work.sort_values(metric, ascending=False).head(80)
                for _, row in top.iterrows():
                    rows.append(
                        {
                            "peak_type": "target_module_delta",
                            "condition": row.get("condition", "target"),
                            "layer": row.get("layer"),
                            "module": row.get("module"),
                            "unit_index": np.nan,
                            "metric": metric,
                            "value": row.get(metric),
                            "target_control_overlap": np.nan,
                            "sign_agreement": np.nan,
                            "source_file": "architecture_module_delta_summary.csv",
                        }
                    )
        overlap = self.csv("architecture_target_vs_control_overlap.csv", max_bytes=50_000_000)
        if overlap is not None and len(overlap):
            if "target_control_top_unit_jaccard" in overlap.columns:
                top = overlap.sort_values("target_control_top_unit_jaccard", ascending=True).head(80)
                for _, row in top.iterrows():
                    rows.append(
                        {
                            "peak_type": "low_target_control_overlap",
                            "condition": row.get("control_condition"),
                            "layer": row.get("layer"),
                            "module": row.get("module"),
                            "unit_index": np.nan,
                            "metric": "target_control_top_unit_jaccard",
                            "value": row.get("target_control_top_unit_jaccard"),
                            "target_control_overlap": row.get("target_control_top_unit_jaccard"),
                            "sign_agreement": row.get("sign_agreement_on_intersection"),
                            "source_file": "architecture_target_vs_control_overlap.csv",
                        }
                    )
        return pd.DataFrame(rows)

    def detect_behavior_threshold_mismatch(self) -> None:
        threshold = self.csv("behavioral_control_axis_threshold_eval.csv")
        hard = self.csv("behavioral_control_axis_hard_random_summary.csv")
        if threshold is None or not len(threshold):
            return
        rows = threshold[threshold.get("criterion", pd.Series(dtype=str)).astype(str).eq("plus_x_beats_random_p95")]
        if not len(rows):
            return
        row = rows.iloc[0]
        metric_name = str(row.get("metric_name", ""))
        metric_value = to_float(row.get("metric_value"))
        if "p95" not in metric_name.lower():
            self.add_anomaly(
                "high",
                "behavioral_control_axis_threshold_eval.csv",
                "p95_metric_name_mismatch",
                metric_name,
                "criterion plus_x_beats_random_p95 must use a p95-derived metric",
                "behavior_p95_metric_mismatch",
                "behavioral_control_axis_threshold_eval.csv",
            )
        if hard is not None and len(hard):
            h = hard[
                (hard.get("base_condition", pd.Series(dtype=str)).astype(str).eq("neutral"))
                & (hard.get("sign_name", pd.Series(dtype=str)).astype(str).eq("plus_x"))
            ].copy()
            if len(h) and "mean_lift_over_random_p95" in h.columns:
                alpha = to_float(row.get("alpha"))
                if "alpha_abs" in h.columns and np.isfinite(alpha):
                    h = h[np.isclose(pd.to_numeric(h["alpha_abs"], errors="coerce"), alpha)]
                if len(h):
                    p95_value = to_float(h.iloc[0].get("mean_lift_over_random_p95"))
                    if np.isfinite(metric_value) and np.isfinite(p95_value) and abs(metric_value - p95_value) > 1e-6:
                        self.add_anomaly(
                            "high",
                            "behavioral_control_axis_threshold_eval.csv",
                            "p95_metric_value_mismatch",
                            f"threshold={metric_value}, hard_random_p95={p95_value}",
                            "threshold p95 row should match hard-random mean_lift_over_random_p95",
                            "behavior_p95_metric_mismatch",
                            "behavioral_control_axis_threshold_eval.csv",
                        )

    def scan_forbidden_labels(self) -> None:
        inventory = self.package.inventory()
        csv_files = [item for item in inventory if item.name.endswith(".csv") and item.size <= 10_000_000]
        for item in csv_files:
            if any(part in item.name for part in ["trajectory_metrics_raw", "response_audit", "top_changed_units", "hidden_top_changed"]):
                continue
            df = self.csv(item.name)
            if df is None or not len(df):
                continue
            artifact_type = ""
            if "artifact_type" in df.columns and len(df):
                artifact_type = str(df["artifact_type"].iloc[0])
            if artifact_type == "raw_measurement":
                continue
            for col in df.columns:
                if df[col].dtype != object:
                    continue
                values = set(str(v) for v in df[col].dropna().astype(str).unique())
                found = values & FORBIDDEN_RESULT_LABEL_VALUES
                if found:
                    self.add_anomaly(
                        "high",
                        item.name,
                        "forbidden_label_present",
                        ",".join(sorted(found)),
                        "main evidence CSV should not contain verdict/discovery labels",
                        "forbidden_label_present",
                        item.name,
                    )

    def pick_best_fields(self, component_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
        best_axis = ""
        best_component = ""
        best_layer_band = "not_established"
        best_alpha = float("nan")
        if component_df is not None and len(component_df):
            work = component_df.copy()
            work["_value"] = pd.to_numeric(work.get("value", np.nan), errors="coerce")
            if work["_value"].notna().any():
                row = work.sort_values("_value", ascending=False).iloc[0]
                best_axis = str(row.get("axis_name", ""))
                best_component = str(row.get("component", ""))
                best_layer_band = str(row.get("layer_band", ""))
                best_alpha = to_float(row.get("alpha"))
        elif metrics.get("strict_causal_symmetry_pass"):
            strict = self.csv("causal_symmetry_score_summary.csv")
            if strict is not None and len(strict) and "symmetry_pass_rate" in strict.columns:
                work = strict.copy()
                work["_pass_rate"] = pd.to_numeric(work["symmetry_pass_rate"], errors="coerce")
                if work["_pass_rate"].notna().any():
                    # Only strict symmetry can establish a "best" causal band.
                    # Large absolute gaps alone can be destructive perturbations.
                    row = work.sort_values(["_pass_rate", "layer_band"], ascending=[False, True]).iloc[0]
                    best_layer_band = str(row.get("layer_band", ""))
                    best_alpha = to_float(row.get("alpha"))
        return {
            "best_layer_band": best_layer_band,
            "best_axis": best_axis,
            "best_component": best_component,
            "best_alpha": best_alpha,
        }

    def recommend_next(self, flags: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        if not flags.get("valid_package") or not flags.get("decoder_ok"):
            return "fix_preflight_or_decoder_compatibility"
        if metrics.get("geometry_pass") and metrics.get("specificity_pass") and not flags.get("gate4_detected"):
            return "run_gate4_axis_decomposition"
        if flags.get("gate4_detected") and metrics.get("component_order_orth_pass") is False:
            return "inspect_content_dominance_and_component_specific_interventions"
        if not metrics.get("strict_causal_symmetry_pass"):
            return "inspect_component_axes_and_layer_specific_causal_symmetry"
        if not metrics.get("behavior_random_p95_pass"):
            return "treat_visible_behavior_as_secondary_and_check_random_p95_controls"
        return "replicate_on_next_model_or_seed"

    def summary_markdown(self, scoreboard: Dict[str, Any], peaks: Dict[str, pd.DataFrame]) -> str:
        lines = [
            f"# Hidden Geometry Result Analysis: {scoreboard['run_label']}",
            "",
            "This is an external read-only analysis. The source package was not modified.",
            "",
            "## Run Validity",
            "",
            f"- Source: `{scoreboard['result_path']}`",
            f"- Model: `{scoreboard.get('model_id', '')}`",
            f"- Gate: `{scoreboard.get('gate', '')}`",
            f"- Decoder OK: `{scoreboard.get('decoder_ok')}` from `red_team_input_manifest.json`",
            f"- Prompt budget OK: `{scoreboard.get('prompt_budget_ok')}` from `prompt_budget_overflow_warnings.csv` presence",
            f"- Numeric integrity OK: `{scoreboard.get('numeric_integrity_ok')}` from `analysis_notes/extracted_narrative_columns/numeric_integrity_check.csv`",
            "",
            "## Primary Metrics",
            "",
            f"- Geometry pass: `{scoreboard.get('geometry_pass')}`; target middle projection `{scoreboard.get('target_middle_projection')}`. Source: `middle_layer_condition_summary.csv`.",
            f"- Specificity pass: `{scoreboard.get('specificity_pass')}`; best target-control gap `{scoreboard.get('best_target_control_gap')}`. Source: `paired_target_vs_control_tests.csv`.",
            f"- Strict causal symmetry pass: `{scoreboard.get('strict_causal_symmetry_pass')}`; score `{scoreboard.get('strict_causal_symmetry_score')}`. Source: `claim_ladder_final.csv` / `causal_symmetry_score_summary.csv`.",
            f"- Behavior random p95 pass: `{scoreboard.get('behavior_random_p95_pass')}`; score `{scoreboard.get('behavior_random_p95_score')}`. Source: `claim_ladder_final.csv` / `behavior_random_p95_gate.csv`.",
            "",
            "## Mechanistic Reading",
            "",
        ]
        if scoreboard.get("geometry_pass") and scoreboard.get("specificity_pass"):
            lines.append("The package supports a hidden-geometry/readout shift against the available controls.")
        else:
            lines.append("The package does not establish a clean hidden-geometry/readout shift under the available controls.")
        if not scoreboard.get("strict_causal_symmetry_pass"):
            lines.append("It weakens or does not establish strict bidirectional causal symmetry for the tested intervention.")
        if not scoreboard.get("behavior_random_p95_pass"):
            lines.append("It does not establish visible behavioral steering against random-p95 controls.")
        lines.extend(
            [
                "",
                "## Boundary",
                "",
                "This analysis does not create discovery/verdict labels in machine CSV outputs. It reports pass/fail fields, failure_code values, source files, and conservative missingness.",
                "",
                "## Recommended Next Experiment",
                "",
                f"`{scoreboard.get('recommended_next_experiment')}`",
                "",
                "## Top Anomalies",
                "",
            ]
        )
        for anomaly in self.anomalies[:12]:
            lines.append(
                f"- `{anomaly['severity']}` `{anomaly['failure_code']}` in `{anomaly['source_file']}`: "
                f"{anomaly['metric']} = {anomaly['observed_value']}"
            )
        if not self.anomalies:
            lines.append("- No analyzer anomalies recorded.")
        lines.extend(["", "## Peak Tables", ""])
        for name, df in peaks.items():
            lines.append(f"- `{name}` rows: `{len(df) if df is not None else 0}`")
        return "\n".join(lines).strip() + "\n"

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.load_manifest()

        inventory_rows = [
            {"source_file": item.name, "size_bytes": item.size}
            for item in sorted(self.package.inventory(), key=lambda x: x.name)
        ]
        write_csv(self.out_dir / "source_file_inventory.csv", inventory_rows, ["source_file", "size_bytes"])

        flags = self.validity_flags()
        metrics = self.extract_primary_metrics(flags)
        self.detect_behavior_threshold_mismatch()
        self.scan_forbidden_labels()

        geometry_df = self.geometry_peaks()
        specificity_df = self.specificity_peaks(metrics.get("target_middle_projection", float("nan")))
        component_df = self.component_peaks()
        causal_df = self.causal_peaks()
        behavior_df = self.behavior_peaks()
        architecture_df = self.architecture_peaks()

        best = self.pick_best_fields(component_df, metrics)

        run_label = self.run_label_arg or str(self.manifest.get("run_label") or safe_name(self.package.source.stem))
        gate = "gate4" if flags.get("gate4_detected") else "gate3" if flags.get("gate3_detected") else "unknown"
        anomaly_codes = [
            str(row.get("failure_code", "")).strip()
            for row in self.anomalies
            if str(row.get("severity", "")).lower() in {"high", "critical"} and str(row.get("failure_code", "")).strip()
        ]
        if anomaly_codes:
            existing_codes = [c for c in str(metrics.get("main_failure_code", "")).split(";") if c]
            metrics["main_failure_code"] = ";".join(dict.fromkeys(existing_codes + anomaly_codes))

        scoreboard = {
            "run_label": run_label,
            "model_id": self.manifest.get("model_id", ""),
            "gate": gate,
            "result_path": str(self.package.source),
            **flags,
            **metrics,
            **best,
        }
        scoreboard["recommended_next_experiment"] = self.recommend_next(flags, metrics)
        scoreboard["strict_causal_symmetry_score"] = metrics.get("causal_symmetry_score")
        scoreboard["behavior_random_p95_score"] = metrics.get("behavior_random_p95_score")
        for col in DEFAULT_SCOREBOARD_COLUMNS:
            scoreboard.setdefault(col, "")

        peak_dir = self.out_dir / "peak_tables"
        peak_dfs = {
            "geometry_peaks.csv": geometry_df,
            "specificity_peaks.csv": specificity_df,
            "component_peaks.csv": component_df,
            "causal_peaks.csv": causal_df,
            "behavior_peaks.csv": behavior_df,
            "architecture_peaks.csv": architecture_df,
            "anomaly_flags.csv": pd.DataFrame(self.anomalies),
        }
        for name, df in peak_dfs.items():
            write_csv(peak_dir / name, df if df is not None else pd.DataFrame())

        anomaly_columns = [
            "severity",
            "artifact",
            "metric",
            "observed_value",
            "expected_rule",
            "failure_code",
            "source_file",
        ]
        write_csv(peak_dir / "anomaly_flags.csv", self.anomalies, anomaly_columns)
        write_csv(self.out_dir / "scoreboard_row.csv", [scoreboard], DEFAULT_SCOREBOARD_COLUMNS)

        summary = {
            "scoreboard": {col: scoreboard.get(col) for col in DEFAULT_SCOREBOARD_COLUMNS},
            "validity_flags": flags,
            "primary_metrics": metrics,
            "top_peaks": self.top_peaks,
            "anomalies": self.anomalies,
            "source_files": inventory_rows,
        }
        write_json(self.out_dir / "analysis_summary.json", summary)
        (self.out_dir / "analysis_summary.md").write_text(
            self.summary_markdown(scoreboard, peak_dfs),
            encoding="utf-8",
        )

        # Self-check analyzer machine CSV outputs for forbidden labels.
        for path in self.out_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            for col in df.columns:
                if df[col].dtype == object:
                    found = set(str(v) for v in df[col].dropna().astype(str).unique()) & FORBIDDEN_RESULT_LABEL_VALUES
                    if found:
                        raise RuntimeError(f"Analyzer machine output contains forbidden labels in {path}: {sorted(found)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hidden-geometry result zip/folder without modifying it.")
    parser.add_argument("--results", required=True, help="Input result package: .zip file or unpacked result directory.")
    parser.add_argument("--out", required=True, help="External output directory for analyzer artifacts.")
    parser.add_argument("--run-label", default="", help="Optional run label override for scoreboard_row.csv.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    package = ResultPackage(Path(args.results))
    try:
        analyzer = Analyzer(package, Path(args.out), args.run_label)
        analyzer.run()
    finally:
        package.close()
    print(f"Analysis written to: {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
