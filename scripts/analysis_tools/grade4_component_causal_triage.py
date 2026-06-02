#!/usr/bin/env python3
"""
Compact triage for Grade 4 component-causal runs.

This script does not run the model and does not replace the full metric lab.
It reads either a raw Grade 4 run directory or a metric-lab/analyzer directory
and produces a short verdict focused on the causal question:

    Does x_order_orth causally modulate the target-like generation trajectory
    more cleanly than x_content?
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


RAW_FILES = {
    "symmetry": "grade4_axis_component_causal_symmetry_summary.csv",
    "alpha": "grade4_axis_component_causal_alpha_scaling_summary.csv",
    "rank": "grade4_axis_component_causal_rank_summary.csv",
    "projection": "grade4_axis_component_causal_projection_summary.csv",
    "response": "grade4_axis_component_causal_response_audit.csv",
}

ANALYZER_FILES = {
    "causal_matrix": "grade4_component_causal_matrix.csv",
    "alpha_matrix": "grade4_component_alpha_matrix.csv",
    "rank_matrix": "grade4_component_rank_matrix.csv",
    "final_evidence": "FINAL_DERIVED_METRIC_EVIDENCE.csv",
    "anomalies": "anomaly_flags.csv",
}

PRIMARY_AXIS = "x_order_orth"
CONTROL_AXIS = "x_content"

CRITERIA = {
    "order_gap_stable_positive_rate_min": 0.70,
    "order_beats_content_win_rate_min": 0.60,
    "alpha_scaling_positive_slope_rate_min": 0.60,
    "strong_support_required_pass_count": 4,
    "partial_support_required_pass_count": 3,
    "notes": [
        "These are rubric thresholds, not proof thresholds.",
        "The script does not change raw metrics or invent missing evidence.",
        "A positive verdict requires metrics read from the run artifacts.",
        "The script can also return weak/inconclusive, content-stronger, or missing-data verdicts.",
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_file(root: Path, filename: str) -> Optional[Path]:
    direct = root / filename
    if direct.exists():
        return direct
    hits = sorted(root.rglob(filename))
    return hits[0] if hits else None


def extract_if_zip(input_path: Path, work_dir: Path, force: bool = False) -> Tuple[Path, Optional[Path]]:
    if input_path.is_dir():
        return input_path, None
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Input is neither a directory nor a .zip file: {input_path}")
    extract_dir = work_dir / "extracted"
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    if not extract_dir.exists():
        ensure_dir(extract_dir)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
    return extract_dir, extract_dir


def to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def finite_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def finite_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else float("nan")


def safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def summarize_symmetry(symmetry: pd.DataFrame) -> pd.DataFrame:
    if symmetry.empty:
        return pd.DataFrame()
    required = {"axis_name", "plus_minus_projection_gap"}
    if not required.issubset(symmetry.columns):
        return pd.DataFrame()

    df = to_numeric(symmetry, ["alpha_abs", "plus_minus_projection_gap", "bidirectional_symmetry_supported", "n_questions"])
    group_cols = ["axis_name"]
    rows: List[Dict[str, Any]] = []
    for axis_name, g in df.groupby(group_cols, dropna=False):
        gap = pd.to_numeric(g["plus_minus_projection_gap"], errors="coerce")
        positive = gap > 0
        rows.append(
            {
                "axis_name": axis_name,
                "rows": int(len(g)),
                "mean_gap": finite_mean(gap),
                "median_gap": finite_median(gap),
                "min_gap": float(gap.min()) if gap.notna().any() else float("nan"),
                "max_gap": float(gap.max()) if gap.notna().any() else float("nan"),
                "positive_gap_rate": float(positive.mean()) if len(positive) else float("nan"),
                "mean_n_questions": finite_mean(g.get("n_questions", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_gap"], ascending=False)


def compare_axes(symmetry: pd.DataFrame) -> pd.DataFrame:
    if symmetry.empty or "axis_name" not in symmetry.columns:
        return pd.DataFrame()
    needed = [
        "base_condition",
        "intervention_layer_band",
        "readout_layer_band",
        "alpha_abs",
        "axis_name",
        "plus_minus_projection_gap",
    ]
    if not set(needed).issubset(symmetry.columns):
        return pd.DataFrame()
    df = to_numeric(symmetry[needed].copy(), ["alpha_abs", "plus_minus_projection_gap"])
    pivot = df.pivot_table(
        index=["base_condition", "intervention_layer_band", "readout_layer_band", "alpha_abs"],
        columns="axis_name",
        values="plus_minus_projection_gap",
        aggfunc="mean",
    ).reset_index()
    if PRIMARY_AXIS not in pivot.columns or CONTROL_AXIS not in pivot.columns:
        return pd.DataFrame()
    pivot["order_minus_content_gap"] = pivot[PRIMARY_AXIS] - pivot[CONTROL_AXIS]
    pivot["order_beats_content"] = pivot["order_minus_content_gap"] > 0
    return pivot.sort_values(["alpha_abs", "base_condition", "intervention_layer_band", "readout_layer_band"])


def summarize_alpha(alpha: pd.DataFrame) -> pd.DataFrame:
    if alpha.empty or "axis_name" not in alpha.columns:
        return pd.DataFrame()
    slope_col = "signed_alpha_projection_slope"
    if slope_col not in alpha.columns:
        return pd.DataFrame()
    df = to_numeric(alpha.copy(), [slope_col, "projection_range", "n_alpha_points"])
    rows: List[Dict[str, Any]] = []
    for axis_name, g in df.groupby("axis_name", dropna=False):
        slope = pd.to_numeric(g[slope_col], errors="coerce")
        rows.append(
            {
                "axis_name": axis_name,
                "rows": int(len(g)),
                "mean_signed_alpha_projection_slope": finite_mean(slope),
                "median_signed_alpha_projection_slope": finite_median(slope),
                "positive_slope_rate": float((slope > 0).mean()) if len(slope) else float("nan"),
                "mean_projection_range": finite_mean(g.get("projection_range", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_signed_alpha_projection_slope"], ascending=False)


def summarize_rank(rank: pd.DataFrame) -> pd.DataFrame:
    if rank.empty or "axis_name" not in rank.columns or "rank_by_gap" not in rank.columns:
        return pd.DataFrame()
    df = to_numeric(rank.copy(), ["rank_by_gap", "plus_minus_projection_gap", "alpha_abs"])
    rows: List[Dict[str, Any]] = []
    for axis_name, g in df.groupby("axis_name", dropna=False):
        rows.append(
            {
                "axis_name": axis_name,
                "rows": int(len(g)),
                "mean_rank_by_gap": finite_mean(g["rank_by_gap"]),
                "best_rank_by_gap": float(pd.to_numeric(g["rank_by_gap"], errors="coerce").min()),
                "mean_rank_gap": finite_mean(g.get("plus_minus_projection_gap", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_rank_by_gap"], ascending=True)


def anomaly_summary(anomalies: pd.DataFrame) -> Dict[str, Any]:
    if anomalies.empty:
        return {"rows": 0, "high_or_critical": 0, "top_failure_codes": []}
    sev = anomalies.get("severity", pd.Series(dtype=str)).astype(str).str.lower()
    high = anomalies[sev.isin(["high", "critical"])]
    codes = []
    if "failure_code" in anomalies.columns:
        codes = (
            anomalies["failure_code"]
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"index": "failure_code", "failure_code": "count"})
            .to_dict(orient="records")
        )
    return {"rows": int(len(anomalies)), "high_or_critical": int(len(high)), "top_failure_codes": codes}


def row_for_axis(df: pd.DataFrame, axis: str) -> Dict[str, Any]:
    if df.empty or "axis_name" not in df.columns:
        return {}
    sub = df[df["axis_name"].astype(str).eq(axis)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def build_verdict(
    symmetry_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> Dict[str, Any]:
    order_sym = row_for_axis(symmetry_summary, PRIMARY_AXIS)
    content_sym = row_for_axis(symmetry_summary, CONTROL_AXIS)
    order_alpha = row_for_axis(alpha_summary, PRIMARY_AXIS)
    order_rank = row_for_axis(rank_summary, PRIMARY_AXIS)

    order_positive_rate = safe_float(order_sym.get("positive_gap_rate"))
    order_mean_gap = safe_float(order_sym.get("mean_gap"))
    order_slope_rate = safe_float(order_alpha.get("positive_slope_rate"))
    pairwise_win_rate = None
    pairwise_mean_delta = None
    if not pairwise.empty and "order_beats_content" in pairwise.columns:
        pairwise_win_rate = float(pairwise["order_beats_content"].mean())
        pairwise_mean_delta = finite_mean(pairwise["order_minus_content_gap"])

    anom = anomaly_summary(anomalies)

    passes = {
        "order_gap_positive": bool(order_mean_gap is not None and order_mean_gap > 0),
        "order_gap_stable": bool(order_positive_rate is not None and order_positive_rate >= CRITERIA["order_gap_stable_positive_rate_min"]),
        "order_beats_content": bool(pairwise_win_rate is not None and pairwise_win_rate >= CRITERIA["order_beats_content_win_rate_min"] and (pairwise_mean_delta or 0) > 0),
        "alpha_scaling_positive": bool(order_slope_rate is not None and order_slope_rate >= CRITERIA["alpha_scaling_positive_slope_rate_min"]),
        "no_high_anomaly": bool(anom["high_or_critical"] == 0),
    }
    pass_count = sum(1 for v in passes.values() if v)

    if not order_sym:
        verdict = "not_available_missing_x_order_orth"
        claim = "Cannot evaluate component-causal claim because x_order_orth rows are missing."
    elif pass_count >= CRITERIA["strong_support_required_pass_count"]:
        verdict = "strong_component_causal_support"
        claim = (
            "The run supports upgrading from descriptive latent-state shift to a component-causal claim: "
            "x_order_orth appears causally involved in the shifted generation trajectory."
        )
    elif pass_count >= CRITERIA["partial_support_required_pass_count"]:
        verdict = "partial_component_causal_support"
        claim = (
            "The run gives partial component-causal support. x_order_orth moves the trajectory, "
            "but at least one key criterion needs inspection before making a stronger claim."
        )
    elif content_sym and safe_float(content_sym.get("mean_gap")) is not None and order_mean_gap is not None and safe_float(content_sym.get("mean_gap")) > order_mean_gap:
        verdict = "content_component_stronger_than_order"
        claim = (
            "The causal signal is stronger for x_content than x_order_orth. This weakens the order/response-mode "
            "interpretation and points toward a content-family explanation."
        )
    else:
        verdict = "weak_or_inconclusive_component_causal_support"
        claim = (
            "The component-causal evidence is weak or inconclusive. Treat the prior result as descriptive "
            "until the causal run is strengthened."
        )

    return {
        "verdict": verdict,
        "claim": claim,
        "passes": passes,
        "criteria": CRITERIA,
        "pass_count": pass_count,
        "primary_axis": PRIMARY_AXIS,
        "control_axis": CONTROL_AXIS,
        "x_order_orth": order_sym,
        "x_content": content_sym,
        "x_order_orth_alpha": order_alpha,
        "x_order_orth_rank": order_rank,
        "pairwise_order_vs_content_win_rate": pairwise_win_rate,
        "pairwise_order_minus_content_mean_gap": pairwise_mean_delta,
        "anomalies": anom,
    }


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_not available_"
    return df.head(max_rows).to_markdown(index=False)


def write_report(
    output_dir: Path,
    verdict: Dict[str, Any],
    symmetry_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    found_paths: Dict[str, Optional[Path]],
) -> None:
    lines = [
        "# Grade 4 Component Causal Triage",
        "",
        f"Verdict: `{verdict['verdict']}`",
        "",
        verdict["claim"],
        "",
        "## Pass Checks",
        "",
    ]
    for key, value in verdict["passes"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Rubric Thresholds",
            "",
            f"- order gap stable positive rate >= `{CRITERIA['order_gap_stable_positive_rate_min']}`",
            f"- order beats content paired win rate >= `{CRITERIA['order_beats_content_win_rate_min']}`",
            f"- positive alpha-slope rate >= `{CRITERIA['alpha_scaling_positive_slope_rate_min']}`",
            f"- strong support pass count >= `{CRITERIA['strong_support_required_pass_count']}`",
            f"- partial support pass count >= `{CRITERIA['partial_support_required_pass_count']}`",
            "",
            "These thresholds are a transparent triage rubric, not a formal proof rule.",
        ]
    )
    lines.extend(
        [
            "",
            "## Key Numbers",
            "",
            f"- pairwise order-vs-content win rate: `{verdict.get('pairwise_order_vs_content_win_rate')}`",
            f"- pairwise order-minus-content mean gap: `{verdict.get('pairwise_order_minus_content_mean_gap')}`",
            f"- high/critical anomaly rows: `{verdict['anomalies']['high_or_critical']}`",
            "",
            "## Axis Symmetry Summary",
            "",
            markdown_table(symmetry_summary),
            "",
            "## Order vs Content Paired Rows",
            "",
            markdown_table(pairwise),
            "",
            "## Alpha Scaling Summary",
            "",
            markdown_table(alpha_summary),
            "",
            "## Rank Summary",
            "",
            markdown_table(rank_summary),
            "",
            "## Source Files",
            "",
        ]
    )
    for name, path in sorted(found_paths.items()):
        lines.append(f"- `{name}`: `{path if path else 'missing'}`")
    (output_dir / "grade4_causal_triage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compact readout for Grade 4 component-causal metrics.")
    parser.add_argument("--input", required=True, help="Raw Grade4 run dir, analyzer output dir, or zip.")
    parser.add_argument("--output-dir", required=True, help="Where to write triage report files.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract zip input into output _work dir.")
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    work_dir = output_dir / "_work"
    ensure_dir(work_dir)

    root, temp_root = extract_if_zip(input_path, work_dir, force=args.force_extract)

    found_paths: Dict[str, Optional[Path]] = {}
    for key, filename in {**RAW_FILES, **ANALYZER_FILES}.items():
        found_paths[key] = find_file(root, filename)

    symmetry = read_csv_if_exists(found_paths["symmetry"])
    alpha = read_csv_if_exists(found_paths["alpha"])
    rank = read_csv_if_exists(found_paths["rank"])
    anomalies = read_csv_if_exists(found_paths["anomalies"])

    symmetry_summary = summarize_symmetry(symmetry)
    pairwise = compare_axes(symmetry)
    alpha_summary = summarize_alpha(alpha)
    rank_summary = summarize_rank(rank)
    verdict = build_verdict(symmetry_summary, pairwise, alpha_summary, rank_summary, anomalies)

    symmetry_summary.to_csv(output_dir / "grade4_causal_triage_axis_summary.csv", index=False)
    pairwise.to_csv(output_dir / "grade4_causal_triage_order_vs_content.csv", index=False)
    alpha_summary.to_csv(output_dir / "grade4_causal_triage_alpha_summary.csv", index=False)
    rank_summary.to_csv(output_dir / "grade4_causal_triage_rank_summary.csv", index=False)
    (output_dir / "grade4_causal_triage_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "grade4_causal_triage_criteria.json").write_text(
        json.dumps(CRITERIA, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, verdict, symmetry_summary, pairwise, alpha_summary, rank_summary, found_paths)

    print(json.dumps({"output_dir": str(output_dir), "verdict": verdict["verdict"], "claim": verdict["claim"]}, ensure_ascii=True, indent=2))

    # Keep variable referenced for clarity if future temp extraction mode changes.
    _ = temp_root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
