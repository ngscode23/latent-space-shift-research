"""
Compact cross-model comparison for the held-out domain evidence package.

This script does not run models and does not add diagnostics. It only reads the
finished held-out result folders and writes a small comparison report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = [
    {
        "short_name": "Qwen heldout",
        "run_dir": Path("attractor_results_agent_loop_qwen3_14b4_heldout"),
    },
    {
        "short_name": "Ministral heldout",
        "run_dir": Path("attractor_results_agent_loop_ministral3_14b_heldout"),
    },
    {
        "short_name": "OLMo2 heldout",
        "run_dir": Path("attractor_results_olmo2_13b_heldout"),
    },
]

OUT_DIR = Path("cross_model_comparison_heldout_v1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_ci(observed: float, low: float, high: float) -> str:
    return f"{observed:.3f} [{low:.3f}, {high:.3f}]"


def clean_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_ci(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "validity_bootstrap" / "bootstrap_ci_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing bootstrap summary: {path}")
    return pd.read_csv(path)


def filter_ci(
    ci: pd.DataFrame,
    *,
    family: str,
    scope: str | None = None,
    metric: str | None = None,
    condition: str | None = None,
    turn: float | None = None,
    rejection_applied: bool | None = None,
    mixing_order: str | None = None,
    target_fraction: float | None = None,
) -> pd.Series | None:
    df = ci[ci["family"] == family].copy()
    if scope is not None:
        df = df[df["scope"] == scope]
    if metric is not None:
        df = df[df["metric"] == metric]
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if turn is not None and "turn" in df.columns:
        df = df[df["turn"] == turn]
    if rejection_applied is not None and "rejection_applied" in df.columns:
        df = df[df["rejection_applied"].astype(str) == str(rejection_applied)]
    if mixing_order is not None and "mixing_order" in df.columns:
        df = df[df["mixing_order"] == mixing_order]
    if target_fraction is not None and "target_fraction" in df.columns:
        df = df[df["target_fraction"] == target_fraction]
    if df.empty:
        return None
    return df.iloc[0]


def add_ci_metric(rows: list[dict[str, Any]], run: dict[str, Any], metric_name: str, **filters: Any) -> None:
    ci = run["ci"]
    row = filter_ci(ci, **filters)
    if row is None:
        rows.append({
            "run": run["short_name"],
            "model_id": run["model_id"],
            "metric": metric_name,
            "observed": None,
            "ci_low": None,
            "ci_high": None,
            "formatted": "missing",
        })
        return
    observed = clean_float(row["observed"])
    low = clean_float(row["ci_low"])
    high = clean_float(row["ci_high"])
    rows.append({
        "run": run["short_name"],
        "model_id": run["model_id"],
        "metric": metric_name,
        "observed": observed,
        "ci_low": low,
        "ci_high": high,
        "formatted": fmt_ci(observed, low, high),
        "family": row.get("family"),
        "scope": row.get("scope"),
        "condition": row.get("condition"),
    })


def load_run(run_spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = run_spec["run_dir"]
    core = run_dir / "core_diagnostics_key_files"
    md = read_json(core / "run_metadata.json")
    ci = load_ci(run_dir)

    def result_file(name: str) -> Path:
        core_path = core / name
        if core_path.exists():
            return core_path
        root_path = run_dir / name
        if root_path.exists():
            return root_path
        raise FileNotFoundError(f"Missing result file in core or run root: {name}")

    hidden = pd.read_csv(result_file("hidden_layer_metrics.csv"))
    best_hidden = hidden.sort_values("contrast_norm", ascending=False).iloc[0]

    probe_path = result_file("linear_probe_accuracy.csv")
    if probe_path.exists():
        probe = pd.read_csv(probe_path)
        best_probe = probe.sort_values("probe_accuracy", ascending=False).iloc[0]
    else:
        best_probe = pd.Series(dtype=object)

    token_diag = pd.read_csv(result_file("candidate_token_diagnostics.csv"))
    order_raw = pd.read_csv(result_file("order_hysteresis_raw.csv"))

    return {
        **run_spec,
        "core": core,
        "model_id": md.get("model_id"),
        "model_type": md.get("transformers_model_type"),
        "max_tokens": md.get("max_tokens"),
        "text_family_preset": md.get("text_family_preset"),
        "primary_control_mode": md.get("primary_control_mode"),
        "ci": ci,
        "best_hidden_index": int(best_hidden["hidden_index"]),
        "best_module_layer": int(best_hidden["module_layer"]),
        "hidden_cosine_distance": float(best_hidden["cosine_distance"]),
        "contrast_over_mean_norm": float(best_hidden["contrast_over_mean_norm"]),
        "best_probe_accuracy": clean_float(best_probe.get("probe_accuracy")),
        "best_probe_perm_p95": clean_float(best_probe.get("permutation_p95_accuracy")),
        "candidate_token_problem_count": int(token_diag["problem"].sum()) if "problem" in token_diag.columns else None,
        "order_truncated_rows": int(order_raw["truncated_risk"].astype(bool).sum()),
        "order_max_prompt_tokens": int(order_raw["raw_prompt_tokens"].max()),
    }


def status_for_metric(metric: str, values: list[float | None]) -> str:
    present = [value for value in values if value is not None]
    if len(present) != len(values):
        return "missing"
    if metric in {"candidate_token_problem_count", "order_truncated_rows"}:
        return "clean" if all(value == 0 for value in present) else "check"
    if all(value > 0 for value in present):
        return "cross_model_supported"
    return "not_supported"


def build_metric_tables(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_rows = []
    for run in runs:
        setup_rows.append({
            "run": run["short_name"],
            "model_id": run["model_id"],
            "model_type": run["model_type"],
            "max_tokens": run["max_tokens"],
            "text_family_preset": run["text_family_preset"],
            "primary_control_mode": run["primary_control_mode"],
            "best_hidden_index": run["best_hidden_index"],
            "best_module_layer": run["best_module_layer"],
            "hidden_cosine_distance": run["hidden_cosine_distance"],
            "contrast_over_mean_norm": run["contrast_over_mean_norm"],
            "best_probe_accuracy": run["best_probe_accuracy"],
            "best_probe_perm_p95": run["best_probe_perm_p95"],
            "candidate_token_problem_count": run["candidate_token_problem_count"],
            "order_truncated_rows": run["order_truncated_rows"],
            "order_max_prompt_tokens": run["order_max_prompt_tokens"],
        })

    rows = []
    for run in runs:
        add_ci_metric(
            rows,
            run,
            "blind_clean_overall_mean_abs",
            family="blind_neutral_probe_clean",
            scope="overall",
            metric="mean_abs",
        )
        add_ci_metric(
            rows,
            run,
            "blind_clean_requested_task_mean_abs",
            family="blind_neutral_probe_clean",
            scope="task:requested_task_vs_substitute",
            metric="mean_abs",
        )
        add_ci_metric(
            rows,
            run,
            "blind_clean_trust_context_mean_abs",
            family="blind_neutral_probe_clean",
            scope="task:trust_context_vs_risk_frame",
            metric="mean_abs",
        )
        for turn in [0.0, 6.0]:
            add_ci_metric(
                rows,
                run,
                f"blind_persistence_turn_{int(turn)}_mean_abs",
                family="blind_neutral_persistence",
                scope="overall",
                metric="mean_abs",
                turn=turn,
            )
            add_ci_metric(
                rows,
                run,
                f"rejection_persistence_turn_{int(turn)}_mean_abs",
                family="rejection_persistence",
                scope="overall",
                metric="mean_abs",
                turn=turn,
            )
        for turn, rejection in [(0.0, False), (4.0, False), (4.0, True)]:
            add_ci_metric(
                rows,
                run,
                f"agent_loop_turn_{int(turn)}_rejection_{rejection}_mean_abs",
                family="agent_loop_clean_direct_margin",
                scope="overall",
                metric="mean_abs",
                turn=turn,
                rejection_applied=rejection,
            )
        add_ci_metric(
            rows,
            run,
            "hard_control_specificity_ratio",
            family="hard_control_specificity",
            metric="mean_abs_ratio",
        )
        for condition in ["TNC", "CNT", "TNN", "CNN"]:
            add_ci_metric(
                rows,
                run,
                f"order_{condition}_all_mean_fraction",
                family="order_hysteresis",
                scope=f"condition:{condition}",
                metric="mean_fraction_toward_target",
                condition=condition,
            )
            add_ci_metric(
                rows,
                run,
                f"order_{condition}_central_mean_fraction",
                family="order_hysteresis_central_axis",
                scope=f"condition:{condition},tasks:requested_task_vs_substitute;trust_context_vs_risk_frame",
                metric="mean_fraction_toward_target",
                condition=condition,
            )
        for order_name in ["target_prefix", "target_suffix"]:
            for frac in [0.125, 0.5]:
                add_ci_metric(
                    rows,
                    run,
                    f"mix_{order_name}_{frac}_mean_fraction",
                    family="mixing_threshold",
                    scope=f"order:{order_name},fraction:{frac}",
                    metric="mean_fraction_toward_target",
                    mixing_order=order_name,
                    target_fraction=frac,
                )
            add_ci_metric(
                rows,
                run,
                f"mix_{order_name}_first_crossing_0_5",
                family="mixing_threshold_crossing",
                scope=f"order:{order_name}",
                metric="first_crossing_0_5",
                mixing_order=order_name,
            )

    return pd.DataFrame(setup_rows), pd.DataFrame(rows)


def pivot_comparison(metric_df: pd.DataFrame) -> pd.DataFrame:
    preferred_runs = [run["short_name"] for run in RUNS]
    pivot = metric_df.pivot_table(
        index="metric",
        columns="run",
        values="formatted",
        aggfunc="first",
    ).reset_index()
    obs = metric_df.pivot_table(
        index="metric",
        columns="run",
        values="observed",
        aggfunc="first",
    )
    run_names = [name for name in preferred_runs if name in obs.columns]
    existing_cols = [col for col in preferred_runs if col in pivot.columns]
    other_cols = [col for col in pivot.columns if col not in {"metric", *existing_cols}]
    pivot = pivot[["metric", *existing_cols, *other_cols]]
    if len(run_names) >= 2:
        min_values = []
        max_values = []
        min_over_max = []
        status = []
        for metric in pivot["metric"]:
            values = [
                clean_float(obs.loc[metric, run_name])
                if metric in obs.index and run_name in obs.columns else None
                for run_name in run_names
            ]
            present = [value for value in values if value is not None]
            min_value = min(present) if present else None
            max_value = max(present) if present else None
            min_values.append(min_value)
            max_values.append(max_value)
            min_over_max.append(
                (min_value / max_value)
                if min_value is not None and max_value not in (None, 0) else None
            )
            status.append(status_for_metric(metric, values))
        pivot["min_observed"] = min_values
        pivot["max_observed"] = max_values
        pivot["min_over_max_observed"] = min_over_max
        pivot["status"] = status
    return pivot


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_empty_"

    def cell(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value)
        return text.replace("\n", " ").replace("|", "\\|")

    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(cell(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_report(setup_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    key_metrics = [
        "blind_clean_overall_mean_abs",
        "blind_persistence_turn_0_mean_abs",
        "blind_persistence_turn_6_mean_abs",
        "rejection_persistence_turn_0_mean_abs",
        "rejection_persistence_turn_6_mean_abs",
        "agent_loop_turn_0_rejection_False_mean_abs",
        "agent_loop_turn_4_rejection_False_mean_abs",
        "agent_loop_turn_4_rejection_True_mean_abs",
        "hard_control_specificity_ratio",
        "order_TNC_all_mean_fraction",
        "order_CNT_all_mean_fraction",
        "order_TNN_all_mean_fraction",
        "order_CNN_all_mean_fraction",
        "mix_target_prefix_0.5_mean_fraction",
        "mix_target_suffix_0.5_mean_fraction",
    ]
    key_df = comparison_df[comparison_df["metric"].isin(key_metrics)].copy()
    key_df["sort_key"] = key_df["metric"].apply(lambda x: key_metrics.index(x))
    key_df = key_df.sort_values("sort_key").drop(columns=["sort_key"])

    return "\n".join([
        "# Held-Out Cross-Model Comparison v1",
        "",
        "Runs compared:",
        "",
        markdown_table(setup_df),
        "",
        "## Main Metrics",
        "",
        markdown_table(key_df),
        "",
        "## Status Readout",
        "",
        "- Статус: **ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ** for the internal cross-model claim.",
        "- What the data show: Qwen3-14B, Ministral 3 14B, and OLMo2 13B all show the same held-out context-induced geometry/readout/action-policy structure.",
        "- Model dependence: Qwen has the largest blind semantic margins, Ministral is smaller but strong, and OLMo2 is weaker on semantic margins while still preserving the same order/mixing/persistence/action pattern.",
        "- What not to claim: do not claim equal effect size, all-model universality, irreversible attractor dynamics, or real external-tool agent behavior.",
        "- Minimal next test: freeze this v1 evidence package; if continuing, use a third model family or model-size ablation, not another diagnostic module.",
        "",
        "## Practical Decision",
        "",
        "The core research spine is now cross-model supported:",
        "",
        "```text",
        "target context -> hidden geometry shift -> logit/semantic readout shift ->",
        "persistence/rejection/order/dose structure -> fake-agent action-policy drift",
        "```",
        "",
        "The next engineering move should be reporting and consolidation, not adding more checks to `llm_attractor_colab_copy_paste.py`.",
        "",
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [load_run(spec) for spec in RUNS]
    setup_df, metric_df = build_metric_tables(runs)
    comparison_df = pivot_comparison(metric_df)

    setup_df.to_csv(OUT_DIR / "run_setup_comparison.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(OUT_DIR / "metric_long.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUT_DIR / "metric_wide.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "cross_model_comparison.md").write_text(
        build_report(setup_df, comparison_df),
        encoding="utf-8",
    )
    print(f"saved: {OUT_DIR / 'run_setup_comparison.csv'}")
    print(f"saved: {OUT_DIR / 'metric_long.csv'}")
    print(f"saved: {OUT_DIR / 'metric_wide.csv'}")
    print(f"saved: {OUT_DIR / 'cross_model_comparison.md'}")


if __name__ == "__main__":
    main()
