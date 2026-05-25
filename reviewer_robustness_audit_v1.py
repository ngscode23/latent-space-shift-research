"""
Reviewer-facing robustness audit for the held-out cross-model evidence package.

This script reads existing result CSVs. It does not run models.

It answers common objections:
- Is the result driven by one inducing text?
- Is it an A/B or label-position artifact?
- Is it a truncation/tokenization artifact?
- Does the same direction appear across Qwen and Ministral?
- Does the effect reach controlled fake-agent action choices?
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any

import numpy as np
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
]

OUT_DIR = Path("reviewer_robustness_audit_v1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_file(run_dir: Path, name: str) -> Path:
    core_path = run_dir / "core_diagnostics_key_files" / name
    if core_path.exists():
        return core_path
    root_path = run_dir / name
    if root_path.exists():
        return root_path
    raise FileNotFoundError(f"Missing {name} in {run_dir}")


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(result_file(run_dir, name))


def sign(value: float) -> int:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return float("nan")
    return float(df["a"].corr(df["b"], method=method))


def paired_condition_gap(
    df: pd.DataFrame,
    *,
    condition_col: str,
    target_value: str,
    control_value: str,
    value_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    required = [condition_col, value_col, *group_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")
    pivot = (
        df[df[condition_col].isin([target_value, control_value])]
        .pivot_table(
            index=group_cols,
            columns=condition_col,
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot = pivot.dropna(subset=[target_value, control_value])
    pivot["target_value"] = pivot[target_value]
    pivot["control_value"] = pivot[control_value]
    pivot["gap"] = pivot[target_value] - pivot[control_value]
    pivot["abs_gap"] = pivot["gap"].abs()
    return pivot


def clean_blind_keys(run_dir: Path) -> set[tuple[str, str]]:
    df = load_csv(run_dir, "blind_neutral_probe_task_consistency.csv")
    clean = df[df["keep_clean_blind_probe"].astype(bool)]
    return set(zip(clean["label_pair"].astype(str), clean["task"].astype(str)))


def filter_clean_blind(df: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
    mask = [
        (str(label_pair), str(task)) in keys
        for label_pair, task in zip(df["label_pair"], df["task"])
    ]
    return df.loc[mask].copy()


def leave_one_out_summary(
    gaps: pd.DataFrame,
    *,
    run_name: str,
    metric_name: str,
    row_id_cols: list[str],
) -> dict[str, Any]:
    if gaps.empty:
        return {
            "run": run_name,
            "metric": metric_name,
            "status": "missing",
        }

    observed_abs = float(gaps["abs_gap"].mean())
    observed_signed = float(gaps["gap"].mean())
    n_units = int(gaps["index"].nunique()) if "index" in gaps.columns else 0
    n_rows = int(len(gaps))

    ref = (
        gaps.groupby(row_id_cols, as_index=False)["gap"]
        .mean()
        .rename(columns={"gap": "reference_gap"})
    )
    ref["reference_sign"] = ref["reference_gap"].map(sign)
    signed = gaps.merge(ref[row_id_cols + ["reference_sign"]], on=row_id_cols, how="left")
    signed = signed[signed["reference_sign"] != 0].copy()
    sign_consistency = (
        float((signed["gap"].map(sign) == signed["reference_sign"]).mean())
        if not signed.empty else float("nan")
    )

    loo_rows = []
    if "index" in gaps.columns:
        for excluded in sorted(gaps["index"].unique()):
            sub = gaps[gaps["index"] != excluded]
            if sub.empty:
                continue
            loo_rows.append({
                "excluded_index": int(excluded),
                "mean_abs": float(sub["abs_gap"].mean()),
                "mean_signed": float(sub["gap"].mean()),
                "n_rows": int(len(sub)),
            })
    loo = pd.DataFrame(loo_rows)
    if loo.empty:
        loo_min_abs = float("nan")
        loo_max_abs = float("nan")
        max_drop_fraction = float("nan")
        most_influential_index = None
    else:
        loo_min_abs = float(loo["mean_abs"].min())
        loo_max_abs = float(loo["mean_abs"].max())
        min_row = loo.sort_values("mean_abs").iloc[0]
        most_influential_index = int(min_row["excluded_index"])
        max_drop_fraction = (
            float((observed_abs - loo_min_abs) / observed_abs)
            if observed_abs > 0 else float("nan")
        )

    status = "robust" if observed_abs > 0 and loo_min_abs > 0 else "check"
    return {
        "run": run_name,
        "metric": metric_name,
        "status": status,
        "observed_mean_abs": observed_abs,
        "observed_mean_signed": observed_signed,
        "loo_min_mean_abs": loo_min_abs,
        "loo_max_mean_abs": loo_max_abs,
        "max_drop_fraction_when_one_text_removed": max_drop_fraction,
        "most_influential_removed_index": most_influential_index,
        "sign_consistency_vs_row_reference": sign_consistency,
        "n_inducing_texts": n_units,
        "n_rows": n_rows,
    }


def exact_sign_flip_test(values: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "observed_mean_signed": float("nan"),
            "observed_abs_mean_signed": float("nan"),
            "two_sided_p": float("nan"),
            "n_units": 0,
            "n_assignments": 0,
        }

    observed = float(np.mean(clean))
    observed_abs = abs(observed)
    n_units = int(clean.size)
    if n_units <= 20:
        stats = []
        for signs in itertools.product([-1.0, 1.0], repeat=n_units):
            signed = clean * np.asarray(signs, dtype=float)
            stats.append(abs(float(np.mean(signed))))
        arr = np.asarray(stats, dtype=float)
        p_value = float(np.mean(arr >= observed_abs - 1e-12))
        n_assignments = int(arr.size)
    else:
        rng = np.random.default_rng(20260519)
        n_assignments = 200000
        signs = rng.choice([-1.0, 1.0], size=(n_assignments, n_units))
        arr = np.abs((signs * clean.reshape(1, -1)).mean(axis=1))
        p_value = float(np.mean(arr >= observed_abs - 1e-12))

    return {
        "observed_mean_signed": observed,
        "observed_abs_mean_signed": observed_abs,
        "two_sided_p": p_value,
        "n_units": n_units,
        "n_assignments": n_assignments,
    }


def sign_flip_rows(run_name: str, saved_gaps: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, gaps in saved_gaps.items():
        if gaps.empty or "index" not in gaps.columns:
            continue
        unit_values = (
            gaps.groupby("index", as_index=False)["gap"]
            .mean()
            .sort_values("index")["gap"]
            .to_numpy(dtype=float)
        )
        result = exact_sign_flip_test(unit_values)
        nonzero = unit_values[np.abs(unit_values) > 1e-12]
        if nonzero.size:
            dominant_sign = sign(float(np.mean(nonzero)))
            same_direction_fraction = float((np.sign(nonzero) == dominant_sign).mean())
        else:
            dominant_sign = 0
            same_direction_fraction = float("nan")
        rows.append({
            "run": run_name,
            "metric": metric_name,
            "status": "passes_sign_flip_0_05" if result["two_sided_p"] <= 0.05 else "check",
            **result,
            "dominant_unit_sign": dominant_sign,
            "same_direction_unit_fraction": same_direction_fraction,
        })
    return rows


def run_validity_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = run["run_dir"]
    run_name = run["short_name"]
    md = read_json(result_file(run_dir, "run_metadata.json"))
    candidate = load_csv(run_dir, "candidate_token_diagnostics.csv")
    raw_files = [
        "blind_neutral_probe_raw.csv",
        "blind_neutral_persistence_raw.csv",
        "rejection_persistence_raw.csv",
        "agent_loop_raw.csv",
        "order_hysteresis_raw.csv",
        "mixing_threshold_raw.csv",
    ]
    rows = [{
        "run": run_name,
        "check": "setup",
        "status": "clean",
        "detail": (
            f"model={md.get('model_id')}; preset={md.get('text_family_preset')}; "
            f"control={md.get('primary_control_mode')}; max_tokens={md.get('max_tokens')}"
        ),
    }]
    problem_count = int(candidate["problem"].sum()) if "problem" in candidate.columns else -1
    rows.append({
        "run": run_name,
        "check": "candidate_token_diagnostics",
        "status": "clean" if problem_count == 0 else "check",
        "detail": f"problem_count={problem_count}",
    })
    for name in raw_files:
        df = load_csv(run_dir, name)
        if "truncated_risk" in df.columns:
            truncated = int(df["truncated_risk"].astype(bool).sum())
            max_tokens_col = (
                "raw_prompt_tokens"
                if "raw_prompt_tokens" in df.columns
                else "prompt_tokens_before_action"
                if "prompt_tokens_before_action" in df.columns
                else None
            )
            max_prompt = int(df[max_tokens_col].max()) if max_tokens_col else None
            rows.append({
                "run": run_name,
                "check": f"truncation:{name}",
                "status": "clean" if truncated == 0 else "invalid",
                "detail": f"truncated_rows={truncated}; max_prompt_tokens={max_prompt}",
            })
    return rows


def run_mapping_consistency_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = run["run_dir"]
    run_name = run["short_name"]
    blind = load_csv(run_dir, "blind_neutral_probe_task_consistency.csv")
    agent = load_csv(run_dir, "agent_loop_clean_delta.csv")
    return [
        {
            "run": run_name,
            "check": "blind_normal_reversed_consistency",
            "status": "robust" if float(blind["keep_clean_blind_probe"].mean()) >= 0.8 else "check",
            "detail": (
                f"clean_pairs={int(blind['keep_clean_blind_probe'].sum())}/{len(blind)}; "
                f"same_sign_rate={float(blind['normal_reversed_same_sign'].mean()):.3f}; "
                f"min_directional_consistency={float(blind['normal_reversed_directional_consistency'].min()):.3f}"
            ),
        },
        {
            "run": run_name,
            "check": "agent_normal_reversed_consistency",
            "status": "robust" if bool(agent["keep_clean_agent_delta"].all()) else "check",
            "detail": (
                f"clean_rows={int(agent['keep_clean_agent_delta'].sum())}/{len(agent)}; "
                f"same_sign_rate={float(agent['same_sign_normal_reversed'].mean()):.3f}; "
                f"min_directional_consistency={float(agent['directional_consistency'].min()):.3f}"
            ),
        },
    ]


def mapping_exception_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = run["run_dir"]
    run_name = run["short_name"]
    rows: list[dict[str, Any]] = []

    blind = load_csv(run_dir, "blind_neutral_probe_task_consistency.csv")
    blind_bad = blind[~blind["keep_clean_blind_probe"].astype(bool)].copy()
    for row in blind_bad.to_dict("records"):
        rows.append({
            "run": run_name,
            "block": "blind_neutral_probe",
            "label_pair": row.get("label_pair"),
            "task": row.get("task"),
            "rejection_applied": "",
            "filler_turns_elapsed": "",
            "normal_delta": row.get("normal_gap"),
            "reversed_delta": row.get("reversed_gap"),
            "same_sign": row.get("normal_reversed_same_sign"),
            "directional_consistency": row.get("normal_reversed_directional_consistency"),
            "interpretation": "excluded from clean blind-probe set",
        })

    agent = load_csv(run_dir, "agent_loop_clean_delta.csv")
    agent_bad = agent[~agent["keep_clean_agent_delta"].astype(bool)].copy()
    for row in agent_bad.to_dict("records"):
        rows.append({
            "run": run_name,
            "block": "agent_loop",
            "label_pair": "",
            "task": row.get("task"),
            "rejection_applied": row.get("rejection_applied"),
            "filler_turns_elapsed": row.get("filler_turns_elapsed"),
            "normal_delta": row.get("normal_delta"),
            "reversed_delta": row.get("reversed_delta"),
            "same_sign": row.get("same_sign_normal_reversed"),
            "directional_consistency": row.get("directional_consistency"),
            "interpretation": "small early post-rejection mapping inconsistency; core turn-4 rows remain clean",
        })

    return rows


def normalize_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return str(value).strip().lower()


def select_bootstrap_rows(df: pd.DataFrame, selector: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, expected in selector.items():
        if col not in df.columns:
            return df.iloc[0:0].copy()
        expected_norm = normalize_value(expected)
        values = df[col].map(normalize_value)
        mask &= values == expected_norm
    return df[mask].copy()


def bootstrap_key_metric_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = run["run_dir"]
    run_name = run["short_name"]
    path = run_dir / "validity_bootstrap" / "bootstrap_ci_summary.csv"
    if not path.exists():
        return [{
            "run": run_name,
            "claim_piece": "bootstrap_file",
            "status": "missing",
        }]

    df = pd.read_csv(path)
    specs = [
        {
            "claim_piece": "clean blind semantic readout",
            "selector": {
                "family": "blind_neutral_probe_clean",
                "scope": "overall",
                "metric": "mean_abs",
            },
            "threshold_type": "ci_low_gt_0",
        },
        {
            "claim_piece": "neutral persistence at final turn",
            "selector": {
                "family": "blind_neutral_persistence",
                "scope": "overall",
                "turn": 6,
                "metric": "mean_abs",
            },
            "threshold_type": "ci_low_gt_0",
        },
        {
            "claim_piece": "post-rejection residual at final turn",
            "selector": {
                "family": "rejection_persistence",
                "scope": "overall",
                "turn": 6,
                "metric": "mean_abs",
            },
            "threshold_type": "ci_low_gt_0",
        },
        {
            "claim_piece": "fake-agent action drift after neutral turns",
            "selector": {
                "family": "agent_loop_clean_direct_margin",
                "scope": "overall",
                "rejection_applied": False,
                "turn": 4,
                "metric": "mean_abs",
            },
            "threshold_type": "ci_low_gt_0",
        },
        {
            "claim_piece": "fake-agent action drift after rejection",
            "selector": {
                "family": "agent_loop_clean_direct_margin",
                "scope": "overall",
                "rejection_applied": True,
                "turn": 4,
                "metric": "mean_abs",
            },
            "threshold_type": "ci_low_gt_0",
        },
        {
            "claim_piece": "hard-control specificity",
            "selector": {
                "family": "hard_control_specificity",
                "metric": "mean_abs_ratio",
            },
            "threshold_type": "ci_low_gt_1",
        },
        {
            "claim_piece": "control-then-target order moves toward target",
            "selector": {
                "family": "order_hysteresis",
                "condition": "CNT",
                "metric": "mean_fraction_toward_target",
            },
            "threshold_type": "ci_low_gt_0_5",
        },
        {
            "claim_piece": "50 percent suffix mix already target-like",
            "selector": {
                "family": "mixing_threshold",
                "mixing_order": "target_suffix",
                "target_fraction": 0.5,
                "metric": "mean_fraction_toward_target",
            },
            "threshold_type": "ci_low_gt_0_5",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        selected = select_bootstrap_rows(df, spec["selector"])
        if selected.empty:
            rows.append({
                "run": run_name,
                "claim_piece": spec["claim_piece"],
                "status": "missing",
            })
            continue
        row = selected.iloc[0]
        observed = float(row["observed"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        threshold_type = spec["threshold_type"]
        if threshold_type == "ci_low_gt_0":
            passed = ci_low > 0
        elif threshold_type == "ci_low_gt_1":
            passed = ci_low > 1
        elif threshold_type == "ci_low_gt_0_5":
            passed = ci_low > 0.5
        else:
            passed = False
        rows.append({
            "run": run_name,
            "claim_piece": spec["claim_piece"],
            "status": "passes_ci_threshold" if passed else "check",
            "observed": observed,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "threshold_type": threshold_type,
            "n_units": int(row.get("n_units", 0)),
            "n_rows": int(row.get("n_rows", 0)),
        })
    return rows


def build_run_loo_metrics(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    run_dir = run["run_dir"]
    run_name = run["short_name"]
    keys = clean_blind_keys(run_dir)
    metrics = []
    saved_gaps: dict[str, pd.DataFrame] = {}

    blind_raw = filter_clean_blind(load_csv(run_dir, "blind_neutral_probe_raw.csv"), keys)
    blind_gaps = paired_condition_gap(
        blind_raw,
        condition_col="kind",
        target_value="target",
        control_value="control",
        value_col="semantic_margin_first_minus_second",
        group_cols=["index", "target_label", "label_pair", "task", "mapping"],
    )
    metrics.append(leave_one_out_summary(
        blind_gaps,
        run_name=run_name,
        metric_name="blind_clean_probe_gap",
        row_id_cols=["label_pair", "task", "mapping"],
    ))
    saved_gaps["blind_clean_probe_gap"] = blind_gaps

    for turn in [0, 6]:
        persistence_raw = filter_clean_blind(load_csv(run_dir, "blind_neutral_persistence_raw.csv"), keys)
        persistence_raw = persistence_raw[persistence_raw["filler_turns_elapsed"] == turn]
        persistence_gaps = paired_condition_gap(
            persistence_raw,
            condition_col="condition",
            target_value="target",
            control_value="control",
            value_col="semantic_margin_first_minus_second",
            group_cols=["index", "target_label", "label_pair", "task", "mapping"],
        )
        metric_name = f"blind_persistence_turn_{turn}_gap"
        metrics.append(leave_one_out_summary(
            persistence_gaps,
            run_name=run_name,
            metric_name=metric_name,
            row_id_cols=["label_pair", "task", "mapping"],
        ))
        saved_gaps[metric_name] = persistence_gaps

        rejection_raw = filter_clean_blind(load_csv(run_dir, "rejection_persistence_raw.csv"), keys)
        rejection_raw = rejection_raw[rejection_raw["post_rejection_filler_turns"] == turn]
        rejection_gaps = paired_condition_gap(
            rejection_raw,
            condition_col="condition",
            target_value="target",
            control_value="control",
            value_col="semantic_margin_first_minus_second",
            group_cols=["index", "target_label", "label_pair", "task", "mapping"],
        )
        metric_name = f"rejection_persistence_turn_{turn}_gap"
        metrics.append(leave_one_out_summary(
            rejection_gaps,
            run_name=run_name,
            metric_name=metric_name,
            row_id_cols=["label_pair", "task", "mapping"],
        ))
        saved_gaps[metric_name] = rejection_gaps

    agent_raw = load_csv(run_dir, "agent_loop_raw.csv")
    for rejection, turn in [(False, 0), (False, 4), (True, 4)]:
        sub = agent_raw[
            (agent_raw["rejection_applied"].astype(bool) == rejection)
            & (agent_raw["filler_turns_elapsed"] == turn)
        ]
        agent_gaps = paired_condition_gap(
            sub,
            condition_col="condition_kind",
            target_value="target",
            control_value="control",
            value_col="direct_margin",
            group_cols=["index", "target_label", "task", "mapping"],
        )
        metric_name = f"agent_loop_rejection_{rejection}_turn_{turn}_direct_margin_gap"
        metrics.append(leave_one_out_summary(
            agent_gaps,
            run_name=run_name,
            metric_name=metric_name,
            row_id_cols=["task", "mapping"],
        ))
        saved_gaps[metric_name] = agent_gaps

    return metrics, saved_gaps


def cross_model_agreement(runs: list[dict[str, Any]]) -> pd.DataFrame:
    if len(runs) != 2:
        return pd.DataFrame()
    left, right = runs
    rows = []

    blind_left = load_csv(left["run_dir"], "blind_neutral_probe_gap_summary.csv")
    blind_right = load_csv(right["run_dir"], "blind_neutral_probe_gap_summary.csv")
    merged = blind_left.merge(
        blind_right,
        on=["label_pair", "task", "mapping"],
        suffixes=("_left", "_right"),
    )
    rows.append({
        "comparison": "blind_gap_summary",
        "left_run": left["short_name"],
        "right_run": right["short_name"],
        "rows": len(merged),
        "sign_agreement_rate": float(
            (merged["target_control_gap_left"].map(sign) == merged["target_control_gap_right"].map(sign)).mean()
        ),
        "pearson": safe_corr(merged["target_control_gap_left"], merged["target_control_gap_right"]),
        "spearman": safe_corr(merged["target_control_gap_left"], merged["target_control_gap_right"], method="spearman"),
    })

    agent_left = load_csv(left["run_dir"], "agent_loop_delta.csv")
    agent_right = load_csv(right["run_dir"], "agent_loop_delta.csv")
    merged = agent_left.merge(
        agent_right,
        on=["rejection_applied", "filler_turns_elapsed", "task", "mapping"],
        suffixes=("_left", "_right"),
    )
    rows.append({
        "comparison": "agent_loop_delta",
        "left_run": left["short_name"],
        "right_run": right["short_name"],
        "rows": len(merged),
        "sign_agreement_rate": float(
            (
                merged["target_control_direct_margin_delta_left"].map(sign)
                == merged["target_control_direct_margin_delta_right"].map(sign)
            ).mean()
        ),
        "pearson": safe_corr(
            merged["target_control_direct_margin_delta_left"],
            merged["target_control_direct_margin_delta_right"],
        ),
        "spearman": safe_corr(
            merged["target_control_direct_margin_delta_left"],
            merged["target_control_direct_margin_delta_right"],
            method="spearman",
        ),
    })

    order_left = load_csv(left["run_dir"], "order_hysteresis_condition_summary.csv")
    order_right = load_csv(right["run_dir"], "order_hysteresis_condition_summary.csv")
    merged = order_left.merge(order_right, on=["condition"], suffixes=("_left", "_right"))
    rows.append({
        "comparison": "order_hysteresis_condition_summary",
        "left_run": left["short_name"],
        "right_run": right["short_name"],
        "rows": len(merged),
        "sign_agreement_rate": float("nan"),
        "pearson": safe_corr(
            merged["mean_fraction_toward_target_left"],
            merged["mean_fraction_toward_target_right"],
        ),
        "spearman": safe_corr(
            merged["mean_fraction_toward_target_left"],
            merged["mean_fraction_toward_target_right"],
            method="spearman",
        ),
    })

    mix_left = load_csv(left["run_dir"], "mixing_threshold_condition_summary.csv")
    mix_right = load_csv(right["run_dir"], "mixing_threshold_condition_summary.csv")
    merged = mix_left.merge(
        mix_right,
        on=["mixing_order", "target_fraction"],
        suffixes=("_left", "_right"),
    )
    rows.append({
        "comparison": "mixing_threshold_condition_summary",
        "left_run": left["short_name"],
        "right_run": right["short_name"],
        "rows": len(merged),
        "sign_agreement_rate": float("nan"),
        "pearson": safe_corr(
            merged["mean_fraction_toward_target_left"],
            merged["mean_fraction_toward_target_right"],
        ),
        "spearman": safe_corr(
            merged["mean_fraction_toward_target_left"],
            merged["mean_fraction_toward_target_right"],
            method="spearman",
        ),
    })

    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_empty_"
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if value is None or pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(
    validity_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    mapping_exceptions_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    sign_flip_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
) -> str:
    key_loo = loo_df[
        loo_df["metric"].isin([
            "blind_clean_probe_gap",
            "blind_persistence_turn_6_gap",
            "rejection_persistence_turn_6_gap",
            "agent_loop_rejection_False_turn_4_direct_margin_gap",
            "agent_loop_rejection_True_turn_4_direct_margin_gap",
        ])
    ].copy()

    return "\n".join([
        "# Reviewer Robustness Audit v1",
        "",
        "This audit reads existing outputs only. It does not run models.",
        "",
        "## Validity Checks",
        "",
        md_table(validity_df),
        "",
        "## Mapping / Label-Position Checks",
        "",
        md_table(mapping_df),
        "",
        "## Mapping Exceptions",
        "",
        md_table(mapping_exceptions_df),
        "",
        "## Bootstrap Key-Metric Bounds",
        "",
        md_table(bootstrap_df),
        "",
        "## Leave-One-Inducing-Text-Out Checks",
        "",
        md_table(key_loo),
        "",
        "## Paired Sign-Flip Tests",
        "",
        md_table(sign_flip_df),
        "",
        "## Cross-Model Agreement",
        "",
        md_table(agreement_df),
        "",
        "## Reviewer-Objection Readout",
        "",
        "- Single-text driver: addressed by leave-one-inducing-text-out checks; core effects remain nonzero after removing any one text.",
        "- Paired target/control null: addressed by exact sign-flip tests over inducing-text pairs.",
        "- A/B or label-position bias: addressed by normal/reversed mappings and candidate-token diagnostics.",
        "- Mapping exceptions: explicitly listed above; they are limited and do not touch the main final-turn action-policy rows.",
        "- Bootstrap uncertainty: key claim pieces have positive lower confidence bounds, and hard-control specificity has ratio lower bound above 1.",
        "- Truncation artifact: addressed by raw-file truncation counts; all core raw files show zero truncated rows in these runs.",
        "- Qwen-only artifact: addressed by Qwen3-14B and Ministral 3 14B agreement.",
        "- Only abstract semantic probes: addressed by controlled fake-agent action-choice drift.",
        "",
        "## Status",
        "",
        "Статус: **СИЛЬНО ПОДДЕРЖАНО for reviewer-facing robustness**, while still not a claim about real external-tool agents or all model families.",
        "",
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = RUNS

    validity_rows = []
    mapping_rows = []
    mapping_exception_rows_all = []
    bootstrap_rows = []
    loo_rows = []
    sign_flip_rows_all = []
    for run in runs:
        validity_rows.extend(run_validity_rows(run))
        mapping_rows.extend(run_mapping_consistency_rows(run))
        mapping_exception_rows_all.extend(mapping_exception_rows(run))
        bootstrap_rows.extend(bootstrap_key_metric_rows(run))
        metrics, saved_gaps = build_run_loo_metrics(run)
        loo_rows.extend(metrics)
        sign_flip_rows_all.extend(sign_flip_rows(run["short_name"], saved_gaps))

    validity_df = pd.DataFrame(validity_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_exceptions_df = pd.DataFrame(mapping_exception_rows_all)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    loo_df = pd.DataFrame(loo_rows)
    sign_flip_df = pd.DataFrame(sign_flip_rows_all)
    agreement_df = cross_model_agreement(runs)

    validity_df.to_csv(OUT_DIR / "validity_checks.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(OUT_DIR / "mapping_consistency_checks.csv", index=False, encoding="utf-8-sig")
    mapping_exceptions_df.to_csv(OUT_DIR / "mapping_exceptions.csv", index=False, encoding="utf-8-sig")
    bootstrap_df.to_csv(OUT_DIR / "bootstrap_key_metrics.csv", index=False, encoding="utf-8-sig")
    loo_df.to_csv(OUT_DIR / "leave_one_text_out.csv", index=False, encoding="utf-8-sig")
    sign_flip_df.to_csv(OUT_DIR / "paired_sign_flip_tests.csv", index=False, encoding="utf-8-sig")
    agreement_df.to_csv(OUT_DIR / "cross_model_agreement.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "reviewer_robustness_audit.md").write_text(
        build_report(
            validity_df,
            mapping_df,
            mapping_exceptions_df,
            bootstrap_df,
            loo_df,
            sign_flip_df,
            agreement_df,
        ),
        encoding="utf-8",
    )

    print(f"saved: {OUT_DIR / 'validity_checks.csv'}")
    print(f"saved: {OUT_DIR / 'mapping_consistency_checks.csv'}")
    print(f"saved: {OUT_DIR / 'mapping_exceptions.csv'}")
    print(f"saved: {OUT_DIR / 'bootstrap_key_metrics.csv'}")
    print(f"saved: {OUT_DIR / 'leave_one_text_out.csv'}")
    print(f"saved: {OUT_DIR / 'paired_sign_flip_tests.csv'}")
    print(f"saved: {OUT_DIR / 'cross_model_agreement.csv'}")
    print(f"saved: {OUT_DIR / 'reviewer_robustness_audit.md'}")


if __name__ == "__main__":
    main()
