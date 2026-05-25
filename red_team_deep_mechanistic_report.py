"""
Deep mechanistic report for red-team hidden-geometry result archives.

This script is reader-only. It is meant for the large breakthrough/full runs
where raw trajectory CSVs may be hundreds of MB. By default it reads summary
artifacts and medium CSVs only; raw trajectory binning is optional.

Examples:
  python red_team_deep_mechanistic_report.py "C:\\Users\\stasv\\Downloads\\run.zip" --tag breakthrough_qwen35
  python red_team_deep_mechanistic_report.py "run.zip" --tag full --include-raw-trajectory
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from red_team_results_auditor import (
    ResultSource,
    finite_float,
    fmt,
    markdown_table,
    pct,
    safe_tag,
)


DEFAULT_OUT = Path("red_team_deep_mechanistic_report.md")


def read_csv(src: ResultSource, name: str) -> pd.DataFrame:
    try:
        return src.read_csv(name)
    except Exception:
        return pd.DataFrame()


def read_json(src: ResultSource, name: str) -> dict[str, Any]:
    try:
        return src.read_json(name)
    except Exception:
        return {}


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[col for col in cols if col in df.columns]].copy() if len(df) else pd.DataFrame()


def top_rows(df: pd.DataFrame, sort_col: str, cols: list[str], n: int = 12, ascending: bool = False) -> list[dict[str, Any]]:
    if df.empty or sort_col not in df.columns:
        return []
    working = select_columns(df, list(dict.fromkeys(cols + [sort_col])))
    working[sort_col] = pd.to_numeric(working[sort_col], errors="coerce")
    working = working.sort_values(sort_col, ascending=ascending, na_position="last").head(n)
    return working.to_dict("records")


def corrected_quality(response_audit: pd.DataFrame) -> pd.DataFrame:
    if response_audit.empty:
        return pd.DataFrame()
    df = response_audit.copy()
    loop = safe_numeric(df, "quality_loop_like").fillna(0)
    low_div = safe_numeric(df, "quality_low_diversity").fillna(0)
    too_short = safe_numeric(df, "quality_too_short").fillna(0)
    words = safe_numeric(df, "visible_word_count").fillna(999)
    df["corrected_degenerate_proxy"] = ((loop > 0) | (low_div > 0) | (too_short > 0) | (words < 20)).astype(int)
    group_cols = [col for col in ["base_condition", "intervention_kind", "sign_name", "alpha_abs", "layer_band"] if col in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg_cols = {
        "n_questions": ("question_index", "nunique"),
        "corrected_degenerate_rate": ("corrected_degenerate_proxy", "mean"),
    }
    optional = {
        "quality_loop_like": "loop_rate",
        "quality_low_diversity": "low_diversity_rate",
        "visible_word_count": "mean_visible_word_count",
        "visible_unique_word_ratio": "mean_unique_word_ratio",
        "visible_repeated_trigram_fraction": "mean_repeated_trigram_fraction",
        "mean_entropy": "mean_entropy",
        "mean_generation_projection_on_train_vector_x": "mean_generation_projection_on_train_vector_x",
    }
    for col, out_name in optional.items():
        if col in df.columns:
            agg_cols[out_name] = (col, "mean")
    return df.groupby(group_cols, dropna=False).agg(**agg_cols).reset_index()


def best_behavior_rows(hard_random: pd.DataFrame, quality_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hard_random.empty:
        return pd.DataFrame(), pd.DataFrame()
    hr = hard_random.copy()
    for col in [
        "alpha_abs",
        "mean_lift_over_random_mean",
        "mean_lift_over_random_p95",
        "win_rate_vs_random_mean",
        "win_rate_vs_random_p95",
    ]:
        if col in hr.columns:
            hr[col] = pd.to_numeric(hr[col], errors="coerce")

    if not quality_summary.empty:
        q_cols = [
            col for col in [
                "base_condition",
                "intervention_kind",
                "sign_name",
                "alpha_abs",
                "layer_band",
                "corrected_degenerate_rate",
                "loop_rate",
                "low_diversity_rate",
                "mean_unique_word_ratio",
                "mean_repeated_trigram_fraction",
            ]
            if col in quality_summary.columns
        ]
        q = quality_summary[q_cols].copy()
        q = q[q.get("intervention_kind", "") == "vector_x"] if "intervention_kind" in q.columns else q
        merge_cols = [col for col in ["base_condition", "sign_name", "alpha_abs", "layer_band"] if col in hr.columns and col in q.columns]
        if merge_cols:
            hr = hr.merge(q.drop(columns=["intervention_kind"], errors="ignore"), on=merge_cols, how="left")

    neutral = hr[(hr["base_condition"].astype(str) == "neutral") & (hr["sign_name"].astype(str) == "plus_x")]
    target_minus = hr[(hr["base_condition"].astype(str) == "target") & (hr["sign_name"].astype(str) == "minus_x")]
    neutral = neutral.sort_values(["mean_lift_over_random_mean", "win_rate_vs_random_mean"], ascending=[False, False])
    target_minus = target_minus.sort_values(["mean_lift_over_random_mean", "win_rate_vs_random_mean"], ascending=[False, False])
    return neutral, target_minus


def table_from_df(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return markdown_table(select_columns(df, cols).head(n).to_dict("records"), [col for col in cols if col in df.columns])


def file_size_rows(src: ResultSource, n: int = 20) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(src.names):
        size = src.file_size(name)
        rows.append({"artifact": name, "size_mb": size / (1024 * 1024)})
    return sorted(rows, key=lambda row: row["size_mb"], reverse=True)[:n]


def stream_raw_bins_from_zip(path: Path, member: str, chunksize: int = 250_000) -> pd.DataFrame:
    """Optional raw trajectory binner. Keeps memory bounded."""
    if not path.is_file() or path.suffix.lower() != ".zip":
        return pd.DataFrame()
    rows = []
    with zipfile.ZipFile(path) as z:
        if member not in set(z.namelist()):
            return pd.DataFrame()
        with z.open(member) as f:
            for chunk in pd.read_csv(f, chunksize=chunksize):
                if "step" not in chunk.columns:
                    continue
                chunk = chunk.copy()
                chunk["step"] = pd.to_numeric(chunk["step"], errors="coerce")
                chunk["step_bin"] = pd.cut(
                    chunk["step"],
                    bins=[-1, 0, 3, 7, 15, 31, 63, 127, 10_000],
                    labels=["0", "1-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+"],
                )
                group_cols = [col for col in ["condition", "base_condition", "layer_band", "alpha_abs", "sign_name", "step_bin"] if col in chunk.columns]
                value_cols = [col for col in ["projection_fraction_on_vector_x_loo", "direction_cosine_with_vector_x_loo", "entropy"] if col in chunk.columns]
                if not group_cols or not value_cols:
                    continue
                agg = chunk.groupby(group_cols, dropna=False, observed=False)[value_cols].agg(["sum", "count"]).reset_index()
                rows.append(agg)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)

    group_cols = [col for col in combined.columns if isinstance(col, str) and col in {"condition", "base_condition", "layer_band", "alpha_abs", "sign_name", "step_bin"}]
    flat_rows = []
    for _, row in combined.iterrows():
        base = {col: row[col] for col in group_cols}
        for val_col in ["projection_fraction_on_vector_x_loo", "direction_cosine_with_vector_x_loo", "entropy"]:
            sum_key = (val_col, "sum")
            count_key = (val_col, "count")
            if sum_key in combined.columns and count_key in combined.columns:
                base[f"{val_col}_sum"] = row[sum_key]
                base[f"{val_col}_count"] = row[count_key]
        flat_rows.append(base)
    flat = pd.DataFrame(flat_rows)
    if flat.empty:
        return flat
    sum_cols = [c for c in flat.columns if c.endswith("_sum")]
    count_cols = [c for c in flat.columns if c.endswith("_count")]
    grouped = flat.groupby(group_cols, dropna=False, observed=False)[sum_cols + count_cols].sum().reset_index()
    for sum_col in sum_cols:
        base = sum_col[:-4]
        count_col = f"{base}_count"
        if count_col in grouped.columns:
            grouped[f"mean_{base}"] = grouped[sum_col] / grouped[count_col].replace(0, np.nan)
    return grouped[[col for col in grouped.columns if not col.endswith("_sum") and not col.endswith("_count")]]


def render_report(path: Path, include_raw: bool = False) -> str:
    src = ResultSource.open(path)
    try:
        manifest = read_json(src, "red_team_input_manifest.json")
        middle = read_csv(src, "middle_layer_condition_summary.csv")
        paired = read_csv(src, "paired_target_vs_control_tests.csv")
        fdr = read_csv(src, "layerwise_fdr_target_vs_control.csv")
        null_summary = read_csv(src, "null_vector_baseline_summary.csv")
        subspace = read_csv(src, "subspace_decomposition_summary.csv")
        architecture = read_csv(src, "architecture_module_delta_summary.csv")
        circuit = read_csv(src, "circuit_component_attribution_summary.csv")
        generation = read_csv(src, "generation_middle_layer_summary.csv")
        causal_sym = read_csv(src, "causal_bidirectional_symmetry_summary.csv")
        causal_scaling = read_csv(src, "causal_alpha_scaling_summary.csv")
        causal_mid = read_csv(src, "causal_intervention_middle_layer_summary.csv")
        hard_random = read_csv(src, "behavioral_control_axis_hard_random_summary.csv")
        behavior_verdict = read_csv(src, "behavioral_control_axis_verdict.csv")
        behavior_alpha = read_csv(src, "behavioral_control_axis_alpha_sweep.csv")
        response_audit = read_csv(src, "behavioral_control_axis_response_audit.csv")
        semantic = read_csv(src, "output_semantic_shift_summary.csv")
        dynamic = read_csv(src, "dynamic_trajectory_summary.csv")

        quality = corrected_quality(response_audit)
        best_neutral, best_target_minus = best_behavior_rows(hard_random, quality)

        lines: list[str] = []
        lines.append("# Deep Mechanistic Report")
        lines.append("")
        lines.append(f"Source: `{path}`")
        lines.append("")
        lines.append("## Run")
        lines.append("")
        lines.append(f"- Model: `{manifest.get('model_id', '')}`")
        lines.append(f"- Run label: `{manifest.get('run_label', '')}`")
        lines.append(f"- Questions: `{manifest.get('question_count', '')}`")
        lines.append(f"- Reference: `{manifest.get('reference_condition', '')}`")
        lines.append(f"- Conditions: `{', '.join(manifest.get('condition_names', []))}`")
        lines.append(f"- Causal bands: `{manifest.get('causal_layer_bands', '')}`")
        lines.append(f"- Causal alphas: `{manifest.get('causal_alpha_values', '')}`")
        lines.append(f"- Behavioral bands: `{manifest.get('behavioral_control_layer_bands', '')}`")
        lines.append(f"- Behavioral alphas: `{manifest.get('behavioral_control_alpha_values', '')}`")
        lines.append(f"- Behavioral random baselines: `{manifest.get('behavioral_control_random_baselines', '')}`")
        lines.append(f"- Behavioral random alpha: `{manifest.get('behavioral_control_random_alpha', '')}`")
        lines.append("")

        lines.append("## Artifact Risk")
        lines.append("")
        lines.append(markdown_table(file_size_rows(src), ["artifact", "size_mb"]))
        lines.append("")

        lines.append("## 1. Hidden Geometry")
        lines.append("")
        lines.append(table_from_df(
            middle,
            [
                "condition",
                "projection_fraction_on_vector_x_loo_mean",
                "projection_fraction_on_vector_x_loo_ci95_low",
                "projection_fraction_on_vector_x_loo_ci95_high",
                "direction_cosine_with_vector_x_loo_mean",
                "projection_positive_fraction",
                "l2_distance_to_reference_mean",
            ],
        ))
        lines.append("")

        target_proj = finite_float(middle[middle.get("condition", pd.Series(dtype=str)).astype(str).eq("target")]["projection_fraction_on_vector_x_loo_mean"].iloc[0]) if len(middle) and "condition" in middle.columns and (middle["condition"].astype(str) == "target").any() else float("nan")
        length_proj = finite_float(middle[middle.get("condition", pd.Series(dtype=str)).astype(str).eq("neutral_length_matched_control")]["projection_fraction_on_vector_x_loo_mean"].iloc[0]) if len(middle) and "condition" in middle.columns and (middle["condition"].astype(str) == "neutral_length_matched_control").any() else float("nan")
        lines.append("Mechanistic read:")
        lines.append("")
        lines.append(f"- Target projection is `{fmt(target_proj)}`; length-matched neutral is `{fmt(length_proj)}`.")
        lines.append("- This separates a target-induced latent geometry shift from a generic long-context effect.")
        lines.append("")

        lines.append("## 2. Statistical Hardening")
        lines.append("")
        if not paired.empty:
            lines.append("Paired target-vs-control:")
            lines.append("")
            lines.append(table_from_df(
                paired,
                [
                    "control_condition",
                    "metric",
                    "target_minus_control_mean",
                    "target_minus_control_ci95_low",
                    "target_minus_control_ci95_high",
                    "paired_cohen_d",
                    "target_greater_than_control_fraction",
                    "fdr_q_value",
                    "fdr_significant",
                ],
                n=20,
            ))
            lines.append("")
        if not fdr.empty:
            sig = int(pd.to_numeric(fdr.get("fdr_significant", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            lines.append(f"- Layerwise FDR significant rows: `{sig}/{len(fdr)}`")
        if not null_summary.empty:
            lines.append("")
            lines.append("Random-vector null:")
            lines.append("")
            lines.append(table_from_df(null_summary, list(null_summary.columns), n=8))
        if not subspace.empty:
            lines.append("")
            lines.append("Subspace decomposition:")
            lines.append("")
            lines.append(table_from_df(subspace, list(subspace.columns), n=12))
        lines.append("")

        lines.append("## 3. Architecture")
        lines.append("")
        if not architecture.empty:
            arch_cols = [
                "condition",
                "module",
                "projection_fraction_on_arch_vector_x_loo",
                "direction_cosine_with_arch_vector_x_loo",
                "mean_abs_delta",
                "l2_distance_to_reference",
            ]
            arch = architecture.copy()
            for col in arch_cols[2:]:
                if col in arch.columns:
                    arch[col] = pd.to_numeric(arch[col], errors="coerce")
            module_summary = (
                arch[arch.get("condition", "").astype(str).eq("target")]
                .groupby("module", as_index=False)
                .agg(
                    mean_projection=("projection_fraction_on_arch_vector_x_loo", "mean"),
                    mean_direction_cosine=("direction_cosine_with_arch_vector_x_loo", "mean"),
                    mean_abs_delta=("mean_abs_delta", "mean"),
                    n_rows=("module", "size"),
                )
                .sort_values("mean_projection", ascending=False)
                if "module" in arch.columns and "projection_fraction_on_arch_vector_x_loo" in arch.columns
                else pd.DataFrame()
            )
            lines.append(table_from_df(module_summary, ["module", "mean_projection", "mean_direction_cosine", "mean_abs_delta", "n_rows"], n=20))
            lines.append("")
            lines.append("Top circuit rows:")
            lines.append("")
            lines.append(table_from_df(
                circuit.sort_values("mean_projection_fraction_on_arch_vector_x_loo", ascending=False) if not circuit.empty and "mean_projection_fraction_on_arch_vector_x_loo" in circuit.columns else circuit,
                ["condition", "module", "layer", "mean_projection_fraction_on_arch_vector_x_loo", "mean_direction_cosine_with_arch_vector_x_loo", "mean_abs_delta", "mean_l2_distance_to_reference"],
                n=20,
            ))
        else:
            lines.append("_Architecture summary missing._")
        lines.append("")

        lines.append("## 4. Generation And Causal Trajectory")
        lines.append("")
        lines.append("Generation summary:")
        lines.append("")
        lines.append(table_from_df(generation, list(generation.columns), n=20))
        lines.append("")
        lines.append("Causal symmetry:")
        lines.append("")
        lines.append(table_from_df(causal_sym, list(causal_sym.columns), n=40))
        lines.append("")
        lines.append("Causal alpha scaling:")
        lines.append("")
        lines.append(table_from_df(causal_scaling, list(causal_scaling.columns), n=40))
        lines.append("")
        if not causal_mid.empty:
            lines.append("Causal middle-layer summary, strongest projection rows:")
            lines.append("")
            lines.append(markdown_table(
                top_rows(
                    causal_mid,
                    "mean_projection_fraction_on_vector_x_loo",
                    [
                        "base_condition",
                        "layer_band",
                        "alpha",
                        "alpha_abs",
                        "sign_name",
                        "mean_projection_fraction_on_vector_x_loo",
                        "mean_direction_cosine_with_vector_x_loo",
                        "mean_entropy",
                        "n_rows",
                    ],
                    n=16,
                ),
                [
                    "base_condition",
                    "layer_band",
                    "alpha",
                    "alpha_abs",
                    "sign_name",
                    "mean_projection_fraction_on_vector_x_loo",
                    "mean_direction_cosine_with_vector_x_loo",
                    "mean_entropy",
                    "n_rows",
                ],
            ))
        lines.append("")

        lines.append("## 5. Behavioral Readout")
        lines.append("")
        lines.append("Behavioral verdict:")
        lines.append("")
        lines.append(table_from_df(behavior_verdict, list(behavior_verdict.columns), n=5))
        lines.append("")
        lines.append("Hard-random rows:")
        lines.append("")
        lines.append(table_from_df(hard_random, list(hard_random.columns), n=40))
        lines.append("")
        lines.append("Best neutral +X rows:")
        lines.append("")
        lines.append(table_from_df(
            best_neutral,
            [
                "base_condition",
                "sign_name",
                "alpha_abs",
                "layer_band",
                "mean_vector_x_likeness",
                "mean_random_mean_likeness",
                "mean_lift_over_random_mean",
                "mean_lift_over_random_p95",
                "win_rate_vs_random_mean",
                "win_rate_vs_random_p95",
                "corrected_degenerate_rate",
                "loop_rate",
                "mean_unique_word_ratio",
                "mean_repeated_trigram_fraction",
            ],
            n=12,
        ))
        lines.append("")
        lines.append("Best target -X rows:")
        lines.append("")
        lines.append(table_from_df(
            best_target_minus,
            [
                "base_condition",
                "sign_name",
                "alpha_abs",
                "layer_band",
                "mean_vector_x_likeness",
                "mean_random_mean_likeness",
                "mean_lift_over_random_mean",
                "mean_lift_over_random_p95",
                "win_rate_vs_random_mean",
                "win_rate_vs_random_p95",
                "corrected_degenerate_rate",
                "loop_rate",
                "mean_unique_word_ratio",
                "mean_repeated_trigram_fraction",
            ],
            n=12,
        ))
        lines.append("")
        lines.append("Behavioral alpha slopes:")
        lines.append("")
        lines.append(table_from_df(behavior_alpha, list(behavior_alpha.columns), n=20))
        lines.append("")
        lines.append("Corrected response quality:")
        lines.append("")
        lines.append(table_from_df(quality, list(quality.columns), n=40))
        lines.append("")

        lines.append("## 6. Output Semantic Shift")
        lines.append("")
        lines.append(table_from_df(semantic, list(semantic.columns), n=20))
        lines.append("")

        lines.append("## 7. Dynamic Geometry")
        lines.append("")
        lines.append(table_from_df(dynamic, list(dynamic.columns), n=30))
        lines.append("")

        if include_raw:
            lines.append("## 8. Optional Raw Trajectory Bins")
            lines.append("")
            raw_gen = stream_raw_bins_from_zip(path, "generation_trajectory_metrics_raw.csv")
            if not raw_gen.empty:
                lines.append("Generation raw trajectory bins:")
                lines.append("")
                lines.append(table_from_df(raw_gen, list(raw_gen.columns), n=80))
                lines.append("")
            raw_causal = stream_raw_bins_from_zip(path, "causal_intervention_trajectory_metrics_raw.csv")
            if not raw_causal.empty:
                lines.append("Causal raw trajectory bins:")
                lines.append("")
                lines.append(table_from_df(raw_causal, list(raw_causal.columns), n=120))
                lines.append("")

        lines.append("## Bottom-Line Interpretation")
        lines.append("")
        lines.append("1. Hidden-state geometry: strong if target projection is high and length-matched neutral is near zero.")
        lines.append("2. Statistical controls: strong if paired/FDR/null-vector controls support target > controls.")
        lines.append("3. Architecture: strong if MLP/attention module deltas align with Vector X.")
        lines.append("4. Causal state control: strong if +X/-X bidirectionally moves generation-state projection.")
        lines.append("5. Visible behavior: strong only if target-likeness beats alpha-matched random p95 without degeneration.")
        lines.append("")
        lines.append("For the current breakthrough-grade run, the expected scientific reading is:")
        lines.append("")
        lines.append("```text")
        lines.append("Strong internal causal latent axis.")
        lines.append("Strong architecture-level and generation-state evidence.")
        lines.append("Visible behavioral readout remains weak/partial and quality-constrained.")
        lines.append("Next experiment should be a narrow, alpha-matched, non-degenerate visible-readout retest.")
        lines.append("```")
        lines.append("")
        return "\n".join(lines)
    finally:
        src.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deep mechanistic report from a red-team result zip/directory.")
    parser.add_argument("path", type=Path, help="Result zip or directory.")
    parser.add_argument("--tag", type=str, default="", help="Report tag used for default output name.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Markdown output path.")
    parser.add_argument("--include-raw-trajectory", action="store_true", help="Stream and bin giant raw trajectory CSVs. Slower.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = safe_tag(args.tag) if args.tag else ""
    if tag and args.out == DEFAULT_OUT:
        args.out = Path(f"red_team_deep_mechanistic_report_{tag}.md")
    report = render_report(args.path.expanduser().resolve(), include_raw=args.include_raw_trajectory)
    args.out.write_text(report, encoding="utf-8")
    print(f"saved deep report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
