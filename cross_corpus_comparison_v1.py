"""
Cross-corpus comparison for the latent-regime project.

This script does not run models. It reads existing result folders and compares
the two induction families:

  1. original/selfref/mirror texts
  2. heldout-domain procedural/risk texts

The comparison is intentionally not a model-only replication table. It answers:

  Which parts of the measured latent/readout shift are shared across corpora,
  and which parts look corpus-specific or confounded by hard controls?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = [
    {
        "short_name": "Qwen selfref",
        "model_family": "Qwen3-14B",
        "corpus": "selfref",
        # The three Qwen selfref folders appear to be repeated versions of the
        # same selfref evidence package. Use the latest one as representative.
        "run_dir": Path("attractor_results_agent_loop_qwen3_14b3"),
        "notes": "representative latest Qwen selfref run",
    },
    {
        "short_name": "Qwen heldout",
        "model_family": "Qwen3-14B",
        "corpus": "heldout",
        "run_dir": Path("attractor_results_agent_loop_qwen3_14b4_heldout"),
        "notes": "heldout procedural/risk corpus",
    },
    {
        "short_name": "Ministral selfref",
        "model_family": "Ministral-3-14B",
        "corpus": "selfref",
        "run_dir": Path("attractor_results_agent_loop_ministral3_14b_selfref"),
        "notes": "selfref/mirror corpus",
    },
    {
        "short_name": "Ministral heldout",
        "model_family": "Ministral-3-14B",
        "corpus": "heldout",
        "run_dir": Path("attractor_results_agent_loop_ministral3_14b_heldout"),
        "notes": "heldout procedural/risk corpus",
    },
    {
        "short_name": "OLMo2 heldout",
        "model_family": "OLMo2-13B",
        "corpus": "heldout",
        "run_dir": Path("attractor_results_olmo2_13b_heldout"),
        "notes": "third-family heldout check; no selfref mate yet",
    },
]

OUT_DIR = Path("cross_corpus_comparison_v1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(run_dir: Path, filename: str) -> pd.DataFrame:
    for path in [
        run_dir / filename,
        run_dir / "core_diagnostics_key_files" / filename,
    ]:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def clean_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    value = clean_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_ci(row: pd.Series | None) -> str:
    if row is None:
        return "missing"
    observed = clean_float(row.get("observed"))
    low = clean_float(row.get("ci_low"))
    high = clean_float(row.get("ci_high"))
    if observed is None or low is None or high is None:
        return "missing"
    return f"{observed:.3f} [{low:.3f}, {high:.3f}]"


def metadata_path(run_dir: Path) -> Path | None:
    for path in [
        run_dir / "run_metadata.json",
        run_dir / "core_diagnostics_key_files" / "run_metadata.json",
    ]:
        if path.exists():
            return path
    return None


def infer_corpus_from_labels(run_dir: Path) -> str:
    df = read_csv_if_exists(run_dir, "input_summary.csv")
    if df.empty or "target_label" not in df.columns:
        return "unknown"
    labels = set(df["target_label"].astype(str).tolist())
    if {"force_finality", "rlhf_reward", "safety_overreach"} & labels:
        return "selfref"
    if {"clinical_triage_gate", "legal_contract_gate", "privacy_export_gate"} & labels:
        return "heldout"
    return "unknown"


def load_bootstrap(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "validity_bootstrap" / "bootstrap_ci_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def filter_bootstrap(
    ci: pd.DataFrame,
    *,
    family: str,
    metric: str,
    scope: str | None = None,
    condition: str | None = None,
    turn: float | None = None,
    rejection_applied: bool | None = None,
    mixing_order: str | None = None,
    target_fraction: float | None = None,
) -> pd.Series | None:
    if ci.empty:
        return None
    df = ci[(ci["family"] == family) & (ci["metric"] == metric)].copy()
    if scope is not None:
        df = df[df["scope"] == scope]
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if turn is not None and "turn" in df.columns:
        df = df[pd.to_numeric(df["turn"], errors="coerce") == float(turn)]
    if rejection_applied is not None and "rejection_applied" in df.columns:
        expected = str(bool(rejection_applied))
        df = df[df["rejection_applied"].astype(str) == expected]
    if mixing_order is not None and "mixing_order" in df.columns:
        df = df[df["mixing_order"] == mixing_order]
    if target_fraction is not None and "target_fraction" in df.columns:
        df = df[pd.to_numeric(df["target_fraction"], errors="coerce") == float(target_fraction)]
    if df.empty:
        return None
    return df.iloc[0]


def add_metric(
    rows: list[dict[str, Any]],
    run: dict[str, Any],
    metric_name: str,
    *,
    family: str,
    metric: str,
    scope: str | None = None,
    condition: str | None = None,
    turn: float | None = None,
    rejection_applied: bool | None = None,
    mixing_order: str | None = None,
    target_fraction: float | None = None,
) -> None:
    row = filter_bootstrap(
        run["ci"],
        family=family,
        metric=metric,
        scope=scope,
        condition=condition,
        turn=turn,
        rejection_applied=rejection_applied,
        mixing_order=mixing_order,
        target_fraction=target_fraction,
    )
    rows.append({
        "run": run["short_name"],
        "model_family": run["model_family"],
        "corpus": run["corpus"],
        "metric_name": metric_name,
        "observed": clean_float(row.get("observed")) if row is not None else None,
        "ci_low": clean_float(row.get("ci_low")) if row is not None else None,
        "ci_high": clean_float(row.get("ci_high")) if row is not None else None,
        "n_units": int(row.get("n_units")) if row is not None and not pd.isna(row.get("n_units")) else None,
        "n_rows": int(row.get("n_rows")) if row is not None and not pd.isna(row.get("n_rows")) else None,
        "formatted": fmt_ci(row),
        "source_family": family,
        "source_scope": scope,
        "source_condition": condition,
    })


def load_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = spec["run_dir"]
    meta_path = metadata_path(run_dir)
    if meta_path is None:
        raise FileNotFoundError(f"Missing run metadata: {run_dir}")
    meta = read_json(meta_path)
    ci = load_bootstrap(run_dir)

    hidden = read_csv_if_exists(run_dir, "hidden_layer_metrics.csv")
    if hidden.empty:
        best_hidden = {}
    else:
        best_hidden = (
            hidden.sort_values("contrast_over_mean_norm", ascending=False)
            .iloc[0]
            .to_dict()
        )

    probe = read_csv_if_exists(run_dir, "linear_probe_accuracy.csv")
    if probe.empty:
        best_probe = {}
    else:
        best_probe = (
            probe.sort_values("probe_accuracy", ascending=False)
            .iloc[0]
            .to_dict()
        )

    blind_clean = read_csv_if_exists(run_dir, "blind_neutral_probe_clean_summary.csv")
    clean_fraction = None
    clean_pairs = None
    if not blind_clean.empty:
        clean_fraction = clean_float(blind_clean.iloc[0].get("clean_fraction"))
        clean_pairs = clean_float(blind_clean.iloc[0].get("clean_label_task_pairs"))

    candidate = read_csv_if_exists(run_dir, "candidate_token_diagnostics.csv")
    candidate_problem_count = 0
    if not candidate.empty and "problem" in candidate.columns:
        candidate_problem_count = int(
            candidate["problem"].astype(str).str.lower().isin(["true", "1"]).sum()
        )

    input_summary = read_csv_if_exists(run_dir, "input_summary.csv")
    max_tokens_input = None
    if not input_summary.empty and "token_count" in input_summary.columns:
        max_tokens_input = clean_float(input_summary["token_count"].max())

    declared_preset = meta.get("text_family_preset")
    inferred_corpus = infer_corpus_from_labels(run_dir)

    return {
        **spec,
        "metadata": meta,
        "ci": ci,
        "model_id": meta.get("model_id"),
        "created_utc": meta.get("created_utc"),
        "declared_preset": declared_preset,
        "inferred_corpus": inferred_corpus,
        "max_tokens": meta.get("max_tokens"),
        "max_input_token_count": max_tokens_input,
        "best_hidden_index": clean_float(best_hidden.get("hidden_index")),
        "best_hidden_module_layer": clean_float(best_hidden.get("module_layer")),
        "best_hidden_contrast_over_mean_norm": clean_float(best_hidden.get("contrast_over_mean_norm")),
        "best_hidden_cosine_distance": clean_float(best_hidden.get("cosine_distance")),
        "best_probe_hidden_index": clean_float(best_probe.get("hidden_index")),
        "best_probe_accuracy": clean_float(best_probe.get("probe_accuracy")),
        "best_probe_permutation_p95": clean_float(best_probe.get("permutation_p95_accuracy")),
        "clean_fraction": clean_fraction,
        "clean_label_task_pairs": clean_pairs,
        "candidate_problem_count": candidate_problem_count,
        "bootstrap_available": not ci.empty,
        "run_dir": run_dir,
    }


def build_tables(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    setup_rows = []
    metric_rows: list[dict[str, Any]] = []

    for run in runs:
        setup_rows.append({
            "run": run["short_name"],
            "model_family": run["model_family"],
            "corpus": run["corpus"],
            "model_id": run["model_id"],
            "declared_preset": run["declared_preset"],
            "inferred_corpus": run["inferred_corpus"],
            "max_tokens": run["max_tokens"],
            "max_input_token_count": run["max_input_token_count"],
            "best_hidden_index": run["best_hidden_index"],
            "best_hidden_contrast_over_mean_norm": run["best_hidden_contrast_over_mean_norm"],
            "best_hidden_cosine_distance": run["best_hidden_cosine_distance"],
            "best_probe_accuracy": run["best_probe_accuracy"],
            "best_probe_permutation_p95": run["best_probe_permutation_p95"],
            "clean_fraction": run["clean_fraction"],
            "clean_label_task_pairs": run["clean_label_task_pairs"],
            "candidate_problem_count": run["candidate_problem_count"],
            "bootstrap_available": run["bootstrap_available"],
            "run_dir": str(run["run_dir"]),
            "notes": run["notes"],
        })

        for metric_name, value in [
            ("hidden_best_contrast_over_mean_norm", run["best_hidden_contrast_over_mean_norm"]),
            ("hidden_best_cosine_distance", run["best_hidden_cosine_distance"]),
            ("linear_probe_best_accuracy", run["best_probe_accuracy"]),
            ("blind_clean_fraction", run["clean_fraction"]),
        ]:
            metric_rows.append({
                "run": run["short_name"],
                "model_family": run["model_family"],
                "corpus": run["corpus"],
                "metric_name": metric_name,
                "observed": value,
                "ci_low": None,
                "ci_high": None,
                "n_units": None,
                "n_rows": None,
                "formatted": fmt(value),
                "source_family": "run_summary",
                "source_scope": "",
                "source_condition": "",
            })

        add_metric(
            metric_rows,
            run,
            "blind_clean_overall_mean_abs",
            family="blind_neutral_probe_clean",
            scope="overall",
            metric="mean_abs",
        )
        for turn in [0.0, 6.0]:
            add_metric(
                metric_rows,
                run,
                f"blind_persistence_turn_{int(turn)}_mean_abs",
                family="blind_neutral_persistence",
                scope="overall",
                turn=turn,
                metric="mean_abs",
            )
            add_metric(
                metric_rows,
                run,
                f"rejection_persistence_turn_{int(turn)}_mean_abs",
                family="rejection_persistence",
                scope="overall",
                turn=turn,
                metric="mean_abs",
            )

        add_metric(
            metric_rows,
            run,
            "hard_control_specificity_ratio",
            family="hard_control_specificity",
            metric="mean_abs_ratio",
        )
        for rejection in [False, True]:
            for turn in [0.0, 4.0]:
                add_metric(
                    metric_rows,
                    run,
                    f"agent_loop_turn_{int(turn)}_rejection_{rejection}_mean_abs",
                    family="agent_loop_clean_direct_margin",
                    scope="overall",
                    turn=turn,
                    rejection_applied=rejection,
                    metric="mean_abs",
                )

        for condition in ["CNT", "TNC", "TNN", "CNN"]:
            add_metric(
                metric_rows,
                run,
                f"order_{condition}_all_mean_fraction",
                family="order_hysteresis",
                condition=condition,
                metric="mean_fraction_toward_target",
            )

        for order_name in ["target_prefix", "target_suffix"]:
            add_metric(
                metric_rows,
                run,
                f"mix_{order_name}_0.5_mean_fraction",
                family="mixing_threshold",
                scope=f"order:{order_name},fraction:0.5",
                mixing_order=order_name,
                target_fraction=0.5,
                metric="mean_fraction_toward_target",
            )
            add_metric(
                metric_rows,
                run,
                f"mix_{order_name}_first_crossing_0_5",
                family="mixing_threshold_crossing",
                scope=f"order:{order_name}",
                mixing_order=order_name,
                metric="first_crossing_0_5",
            )

    setup_df = pd.DataFrame(setup_rows)
    metric_df = pd.DataFrame(metric_rows)
    ratio_df = build_ratio_table(metric_df)
    return setup_df, metric_df, ratio_df


def build_ratio_table(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paired_models = (
        metric_df.groupby("model_family")["corpus"]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )
    for model_family, corpora in paired_models.items():
        if not {"selfref", "heldout"}.issubset(corpora):
            continue
        for metric_name, group in metric_df.groupby("metric_name"):
            self_rows = group[(group["model_family"] == model_family) & (group["corpus"] == "selfref")]
            held_rows = group[(group["model_family"] == model_family) & (group["corpus"] == "heldout")]
            if self_rows.empty or held_rows.empty:
                continue
            self_value = clean_float(self_rows.iloc[0]["observed"])
            held_value = clean_float(held_rows.iloc[0]["observed"])
            if self_value is None or held_value is None:
                continue
            ratio = self_value / held_value if held_value != 0 else None
            rows.append({
                "model_family": model_family,
                "metric_name": metric_name,
                "selfref_observed": self_value,
                "heldout_observed": held_value,
                "selfref_minus_heldout": self_value - held_value,
                "selfref_over_heldout": ratio,
                "selfref_formatted": self_rows.iloc[0]["formatted"],
                "heldout_formatted": held_rows.iloc[0]["formatted"],
            })
    return pd.DataFrame(rows)


def pivot_metric_table(metric_df: pd.DataFrame) -> pd.DataFrame:
    run_order = [run["short_name"] for run in RUNS]
    pivot = metric_df.pivot_table(
        index="metric_name",
        columns="run",
        values="formatted",
        aggfunc="first",
    ).reset_index()
    existing = [name for name in run_order if name in pivot.columns]
    other = [col for col in pivot.columns if col not in {"metric_name", *existing}]
    return pivot[["metric_name", *existing, *other]]


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


def key_metric_view(metric_wide: pd.DataFrame) -> pd.DataFrame:
    key_metrics = [
        "hidden_best_contrast_over_mean_norm",
        "blind_clean_fraction",
        "blind_clean_overall_mean_abs",
        "blind_persistence_turn_6_mean_abs",
        "rejection_persistence_turn_6_mean_abs",
        "hard_control_specificity_ratio",
        "agent_loop_turn_4_rejection_False_mean_abs",
        "agent_loop_turn_4_rejection_True_mean_abs",
        "order_CNT_all_mean_fraction",
        "order_TNC_all_mean_fraction",
        "order_TNN_all_mean_fraction",
        "mix_target_prefix_0.5_mean_fraction",
        "mix_target_suffix_0.5_mean_fraction",
    ]
    view = metric_wide[metric_wide["metric_name"].isin(key_metrics)].copy()
    view["sort_key"] = view["metric_name"].apply(lambda x: key_metrics.index(x))
    return view.sort_values("sort_key").drop(columns=["sort_key"])


def ratio_key_view(ratio_df: pd.DataFrame) -> pd.DataFrame:
    key_metrics = [
        "blind_clean_overall_mean_abs",
        "blind_persistence_turn_6_mean_abs",
        "rejection_persistence_turn_6_mean_abs",
        "hard_control_specificity_ratio",
        "agent_loop_turn_4_rejection_False_mean_abs",
        "agent_loop_turn_4_rejection_True_mean_abs",
    ]
    view = ratio_df[ratio_df["metric_name"].isin(key_metrics)].copy()
    if view.empty:
        return view
    view["selfref_over_heldout"] = view["selfref_over_heldout"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.3f}"
    )
    view["selfref_minus_heldout"] = view["selfref_minus_heldout"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.3f}"
    )
    return view[
        [
            "model_family",
            "metric_name",
            "selfref_formatted",
            "heldout_formatted",
            "selfref_minus_heldout",
            "selfref_over_heldout",
        ]
    ]


def build_report(setup_df: pd.DataFrame, metric_wide: pd.DataFrame, ratio_df: pd.DataFrame) -> str:
    return "\n".join([
        "# Cross-Corpus Comparison v1",
        "",
        "This report compares induction families, not only model replications.",
        "",
        "```text",
        "selfref / mirror corpus  !=  heldout procedural/risk corpus",
        "```",
        "",
        "Use it to answer whether the project is about special self-reference texts",
        "or about broader context-induced latent regime formation.",
        "",
        "## Runs",
        "",
        markdown_table(setup_df),
        "",
        "## Main Cross-Corpus Metrics",
        "",
        markdown_table(key_metric_view(metric_wide)),
        "",
        "## Selfref / Heldout Ratios",
        "",
        markdown_table(ratio_key_view(ratio_df)),
        "",
        "## Quality Caveats",
        "",
        "- Qwen selfref is included as the latest representative of the older Qwen selfref folders. Its core blind/persistence/action magnitudes are useful, but some order-fraction bootstrap intervals are very wide. Treat those rows as low-confidence validation fractions, not as headline evidence.",
        "- Ministral selfref is the cleaner selfref cross-model check for order/mixing, but its hard-control specificity fails: pressure-style controls reproduce much of the original effect.",
        "- Heldout rows are the cleaner reviewer-facing comparison line because they avoid direct model self-reference and have stronger hard-control behavior in Qwen and Ministral.",
        "",
        "## Interpretation",
        "",
        "- Статус: **СИЛЬНО ПОДДЕРЖАНО** for the broad state-induction claim.",
        "- What the comparison shows: both selfref and heldout corpora induce hidden/readout/action shifts, so the phenomenon is not only a self-reference trick.",
        "- Selfref is not cleanly unique: in the Ministral selfref run, hard-control specificity is weak because `pressure_style_no_model` matches or exceeds original.",
        "- Heldout is cleaner for reviewer-facing claims because it avoids direct model self-reference and still reproduces the structure across Qwen, Ministral, and OLMo2.",
        "- The right framing is: texts are induction stimuli; the measured object is a distributed latent discourse regime.",
        "",
        "## What Not To Claim",
        "",
        "- Do not claim that self-reference alone is the mechanism.",
        "- Do not claim that original selfref texts always beat all hard controls.",
        "- Do not claim corpus-independent equal effect size.",
        "- Do not call this strict dynamical attractor evidence from this table alone.",
        "",
        "## Practical Decision",
        "",
        "Use heldout as the main reviewer-facing replication line. Use selfref as the",
        "strong mirror/self-model pressure line and as evidence that pressure cadence",
        "and rhetorical topology are active ingredients.",
        "",
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [load_run(spec) for spec in RUNS]
    setup_df, metric_df, ratio_df = build_tables(runs)
    metric_wide = pivot_metric_table(metric_df)

    setup_df.to_csv(OUT_DIR / "run_setup.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(OUT_DIR / "metric_long.csv", index=False, encoding="utf-8-sig")
    metric_wide.to_csv(OUT_DIR / "metric_wide.csv", index=False, encoding="utf-8-sig")
    ratio_df.to_csv(OUT_DIR / "selfref_vs_heldout_ratios.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "cross_corpus_comparison.md").write_text(
        build_report(setup_df, metric_wide, ratio_df),
        encoding="utf-8",
    )

    print(f"saved: {OUT_DIR / 'run_setup.csv'}")
    print(f"saved: {OUT_DIR / 'metric_long.csv'}")
    print(f"saved: {OUT_DIR / 'metric_wide.csv'}")
    print(f"saved: {OUT_DIR / 'selfref_vs_heldout_ratios.csv'}")
    print(f"saved: {OUT_DIR / 'cross_corpus_comparison.md'}")


if __name__ == "__main__":
    main()
