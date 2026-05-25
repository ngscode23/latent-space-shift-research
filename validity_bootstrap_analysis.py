"""
Bootstrap validity analysis for attractor / latent-shift result folders.

This script does not rerun the model. It reads existing CSV files and computes
bootstrap confidence intervals using inducing text index as the resampling unit.
That avoids treating many probe rows from the same inducing text as independent
observations.

Usage:
  python validity_bootstrap_analysis.py attractor_results_agent_loop_qwen3_14b
  python validity_bootstrap_analysis.py attractor_results_agent_loop_qwen3_14b --n-boot 10000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_N_BOOT = 5000
DEFAULT_SEED = 20260519


def read_csv_if_exists(root: Path, name: str) -> pd.DataFrame:
    candidates = [
        root / name,
        root / "core_diagnostics_key_files" / name,
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved: {path}")


def fmt(value: float, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def mean_abs(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values))) if values.size else float("nan")


def mean_signed(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def bootstrap_unit_rows(
    df: pd.DataFrame,
    value_col: str,
    unit_col: str,
    stat_fn: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    if df.empty or value_col not in df.columns or unit_col not in df.columns:
        return {
            "observed": np.nan,
            "bootstrap_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_units": 0,
            "n_rows": 0,
        }

    work = df[[unit_col, value_col]].dropna().copy()
    units = list(work[unit_col].drop_duplicates())
    if not units:
        return {
            "observed": np.nan,
            "bootstrap_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_units": 0,
            "n_rows": 0,
        }

    grouped = {
        unit: work.loc[work[unit_col] == unit, value_col].to_numpy(dtype=float)
        for unit in units
    }
    observed_values = np.concatenate([grouped[unit] for unit in units])
    observed = stat_fn(observed_values)

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled_values = np.concatenate([grouped[unit] for unit in sampled_units])
        boot[b] = stat_fn(sampled_values)

    return {
        "observed": observed,
        "bootstrap_mean": float(np.mean(boot)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_units": int(len(units)),
        "n_rows": int(len(work)),
    }


def add_bootstrap_rows(
    rows: list[dict],
    df: pd.DataFrame,
    *,
    family: str,
    group: dict,
    value_col: str,
    unit_col: str,
    rng: np.random.Generator,
    n_boot: int,
) -> None:
    for metric_name, stat_fn in [
        ("mean_abs", mean_abs),
        ("mean_signed", mean_signed),
    ]:
        out = bootstrap_unit_rows(df, value_col, unit_col, stat_fn, rng, n_boot)
        rows.append({
            "family": family,
            **group,
            "metric": metric_name,
            **out,
            "n_boot": int(n_boot),
            "unit_col": unit_col,
            "value_col": value_col,
        })


def pair_target_control(
    df: pd.DataFrame,
    *,
    condition_col: str,
    value_col: str,
    key_cols: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    needed = set(key_cols + [condition_col, value_col])
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    pivot = (
        df[key_cols + [condition_col, value_col]]
        .pivot_table(index=key_cols, columns=condition_col, values=value_col, aggfunc="mean")
        .reset_index()
    )
    if "target" not in pivot.columns or "control" not in pivot.columns:
        return pd.DataFrame()
    pivot["gap"] = pivot["target"] - pivot["control"]
    return pivot


def clean_blind_keys(root: Path) -> pd.DataFrame:
    consistency = read_csv_if_exists(root, "blind_neutral_probe_task_consistency.csv")
    if consistency.empty or "keep_clean_blind_probe" not in consistency.columns:
        return pd.DataFrame()
    return consistency[consistency["keep_clean_blind_probe"].astype(bool)][
        ["label_pair", "task"]
    ].drop_duplicates()


def analyze_blind_probe(root: Path, rows: list[dict], rng: np.random.Generator, n_boot: int) -> None:
    raw = read_csv_if_exists(root, "blind_neutral_probe_raw.csv")
    clean_keys = clean_blind_keys(root)
    if raw.empty or clean_keys.empty:
        return

    paired = pair_target_control(
        raw,
        condition_col="kind",
        value_col="semantic_margin_first_minus_second",
        key_cols=["index", "target_label", "label_pair", "task", "mapping"],
    )
    if paired.empty:
        return
    paired = paired.merge(clean_keys, on=["label_pair", "task"], how="inner")

    add_bootstrap_rows(
        rows,
        paired,
        family="blind_neutral_probe_clean",
        group={"scope": "overall"},
        value_col="gap",
        unit_col="index",
        rng=rng,
        n_boot=n_boot,
    )

    for task, sub in paired.groupby("task"):
        add_bootstrap_rows(
            rows,
            sub,
            family="blind_neutral_probe_clean",
            group={"scope": f"task:{task}"},
            value_col="gap",
            unit_col="index",
            rng=rng,
            n_boot=n_boot,
        )


def analyze_persistence(
    root: Path,
    rows: list[dict],
    *,
    raw_name: str,
    family: str,
    turn_col: str,
    rng: np.random.Generator,
    n_boot: int,
) -> None:
    raw = read_csv_if_exists(root, raw_name)
    if raw.empty:
        return

    paired = pair_target_control(
        raw,
        condition_col="condition",
        value_col="semantic_margin_first_minus_second",
        key_cols=["index", "target_label", turn_col, "label_pair", "task", "mapping"],
    )
    if paired.empty:
        return

    for turn, sub in paired.groupby(turn_col):
        add_bootstrap_rows(
            rows,
            sub,
            family=family,
            group={"scope": "overall", "turn": int(turn)},
            value_col="gap",
            unit_col="index",
            rng=rng,
            n_boot=n_boot,
        )
        for task, task_sub in sub.groupby("task"):
            add_bootstrap_rows(
                rows,
                task_sub,
                family=family,
                group={"scope": f"task:{task}", "turn": int(turn)},
                value_col="gap",
                unit_col="index",
                rng=rng,
                n_boot=n_boot,
            )


def analyze_hard_controls(root: Path, rows: list[dict], rng: np.random.Generator, n_boot: int) -> None:
    raw = read_csv_if_exists(root, "hard_control_family_blind_probe_raw.csv")
    if raw.empty:
        return

    key_cols = ["index", "target_label", "label_pair", "task", "mapping"]
    needed = set(key_cols + ["variant", "semantic_margin_first_minus_second"])
    if not needed.issubset(raw.columns):
        return

    pivot = (
        raw[key_cols + ["variant", "semantic_margin_first_minus_second"]]
        .pivot_table(
            index=key_cols,
            columns="variant",
            values="semantic_margin_first_minus_second",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "neutral_length_matched" not in pivot.columns:
        return

    variant_cols = [
        c for c in pivot.columns
        if c not in set(key_cols) and c != "neutral_length_matched"
    ]
    for variant in variant_cols:
        sub = pivot[key_cols].copy()
        sub["gap"] = pivot[variant] - pivot["neutral_length_matched"]
        sub = sub.dropna(subset=["gap"])
        add_bootstrap_rows(
            rows,
            sub,
            family="hard_control_family_vs_content_matched_neutral",
            group={"scope": f"variant:{variant}"},
            value_col="gap",
            unit_col="index",
            rng=rng,
            n_boot=n_boot,
        )

    # Specificity ratio with the observed best non-original control family.
    baseline_excluded = {
        "original",
        "neutral_length_matched",
        "repetitive_neutral_length_matched",
    }
    control_variants = [v for v in variant_cols if v not in baseline_excluded]
    if "original" not in variant_cols or not control_variants:
        return

    variant_effects = {}
    for variant in ["original", *control_variants]:
        variant_effects[variant] = (
            pivot[key_cols]
            .assign(gap=pivot[variant] - pivot["neutral_length_matched"])
            .dropna(subset=["gap"])
        )
    observed_controls = {
        variant: mean_abs(df["gap"].to_numpy(dtype=float))
        for variant, df in variant_effects.items()
        if variant != "original"
    }
    best_control = max(observed_controls, key=observed_controls.get)

    common_units = sorted(
        set(variant_effects["original"]["index"].unique())
        & set(variant_effects[best_control]["index"].unique())
    )
    if not common_units:
        return

    original_groups = {
        unit: variant_effects["original"].loc[
            variant_effects["original"]["index"] == unit, "gap"
        ].to_numpy(dtype=float)
        for unit in common_units
    }
    control_groups = {
        unit: variant_effects[best_control].loc[
            variant_effects[best_control]["index"] == unit, "gap"
        ].to_numpy(dtype=float)
        for unit in common_units
    }
    observed_original = mean_abs(np.concatenate([original_groups[u] for u in common_units]))
    observed_control = mean_abs(np.concatenate([control_groups[u] for u in common_units]))
    observed_ratio = observed_original / observed_control if observed_control > 1e-12 else np.nan

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(common_units, size=len(common_units), replace=True)
        original_values = np.concatenate([original_groups[u] for u in sampled])
        control_values = np.concatenate([control_groups[u] for u in sampled])
        denom = mean_abs(control_values)
        boot[b] = mean_abs(original_values) / denom if denom > 1e-12 else np.nan

    rows.append({
        "family": "hard_control_specificity",
        "scope": f"original_vs_observed_best_control:{best_control}",
        "metric": "mean_abs_ratio",
        "observed": observed_ratio,
        "bootstrap_mean": float(np.nanmean(boot)),
        "ci_low": float(np.nanquantile(boot, 0.025)),
        "ci_high": float(np.nanquantile(boot, 0.975)),
        "n_units": int(len(common_units)),
        "n_rows": int(
            len(variant_effects["original"]) + len(variant_effects[best_control])
        ),
        "n_boot": int(n_boot),
        "unit_col": "index",
        "value_col": "mean_abs(original_gap) / mean_abs(best_control_gap)",
    })


def analyze_agent_loop(root: Path, rows: list[dict], rng: np.random.Generator, n_boot: int) -> None:
    raw = read_csv_if_exists(root, "agent_loop_raw.csv")
    clean = read_csv_if_exists(root, "agent_loop_clean_delta.csv")
    if raw.empty:
        return

    paired = pair_target_control(
        raw,
        condition_col="condition_kind",
        value_col="direct_margin",
        key_cols=["index", "target_label", "rejection_applied", "filler_turns_elapsed", "task", "mapping"],
    )
    if paired.empty:
        return

    if not clean.empty and "keep_clean_agent_delta" in clean.columns:
        clean_keys = clean[clean["keep_clean_agent_delta"].astype(bool)][
            ["rejection_applied", "filler_turns_elapsed", "task"]
        ].drop_duplicates()
        paired = paired.merge(
            clean_keys,
            on=["rejection_applied", "filler_turns_elapsed", "task"],
            how="inner",
        )

    for (rejection, filler), sub in paired.groupby(["rejection_applied", "filler_turns_elapsed"]):
        add_bootstrap_rows(
            rows,
            sub,
            family="agent_loop_clean_direct_margin",
            group={"scope": "overall", "rejection_applied": bool(rejection), "turn": int(filler)},
            value_col="gap",
            unit_col="index",
            rng=rng,
            n_boot=n_boot,
        )
        for task, task_sub in sub.groupby("task"):
            add_bootstrap_rows(
                rows,
                task_sub,
                family="agent_loop_clean_direct_margin",
                group={"scope": f"task:{task}", "rejection_applied": bool(rejection), "turn": int(filler)},
                value_col="gap",
                unit_col="index",
                rng=rng,
                n_boot=n_boot,
            )


def add_bootstrap_table_rows(
    rows: list[dict],
    raw: pd.DataFrame,
    *,
    family: str,
    unit_col: str,
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    key_cols: list[str],
    metric_cols: list[str],
    rng: np.random.Generator,
    n_boot: int,
    scope_fn: Callable[[pd.Series], str],
) -> None:
    if raw.empty or unit_col not in raw.columns:
        return

    units = list(raw[unit_col].drop_duplicates())
    if not units:
        return
    grouped = {
        unit: raw.loc[raw[unit_col] == unit].copy()
        for unit in units
    }

    observed = compute_fn(raw)
    if observed.empty:
        return

    boot_values: dict[tuple, list[float]] = {}
    for b in range(n_boot):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sample = pd.concat([grouped[unit] for unit in sampled_units], ignore_index=True)
        sample_out = compute_fn(sample)
        if sample_out.empty:
            continue
        for _, sample_row in sample_out.iterrows():
            key = tuple(sample_row[col] for col in key_cols)
            for metric in metric_cols:
                if metric in sample_row and pd.notna(sample_row[metric]):
                    boot_values.setdefault((key, metric), []).append(float(sample_row[metric]))

    for _, row in observed.iterrows():
        key = tuple(row[col] for col in key_cols)
        for metric in metric_cols:
            if metric not in row or pd.isna(row[metric]):
                continue
            values = np.array(boot_values.get((key, metric), []), dtype=float)
            if values.size == 0:
                continue
            rows.append({
                "family": family,
                "scope": scope_fn(row),
                **{col: row[col] for col in key_cols},
                "metric": metric,
                "observed": float(row[metric]),
                "bootstrap_mean": float(np.nanmean(values)),
                "ci_low": float(np.nanquantile(values, 0.025)),
                "ci_high": float(np.nanquantile(values, 0.975)),
                "n_units": int(len(units)),
                "n_rows": int(len(raw)),
                "n_boot": int(n_boot),
                "unit_col": unit_col,
                "value_col": metric,
            })


def append_bootstrap_metric_row(
    rows: list[dict],
    *,
    family: str,
    scope: str,
    metric: str,
    observed: float,
    boot_values: np.ndarray,
    n_units: int,
    n_rows: int,
    unit_col: str,
    value_col: str,
    extra: dict | None = None,
    n_boot: int,
) -> None:
    values = np.asarray(boot_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not math.isfinite(float(observed)):
        return
    rows.append({
        "family": family,
        "scope": scope,
        **(extra or {}),
        "metric": metric,
        "observed": float(observed),
        "bootstrap_mean": float(np.nanmean(values)),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
        "n_units": int(n_units),
        "n_rows": int(n_rows),
        "n_boot": int(n_boot),
        "unit_col": unit_col,
        "value_col": value_col,
    })


def summarize_order_hysteresis(raw: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    needed = {
        "condition", "label_pair", "task", "mapping",
        "semantic_margin_first_minus_second",
    }
    if raw.empty or not needed.issubset(raw.columns):
        return pd.DataFrame()

    summary = (
        raw[list(needed)]
        .groupby(["condition", "label_pair", "task", "mapping"], as_index=False)
        .agg(condition_margin=("semantic_margin_first_minus_second", "mean"))
    )
    pivot = summary.pivot_table(
        index=["label_pair", "task", "mapping"],
        columns="condition",
        values="condition_margin",
        aggfunc="mean",
    ).reset_index()
    if "T" not in pivot.columns or "C" not in pivot.columns:
        return pd.DataFrame()

    condition_cols = [
        col for col in pivot.columns
        if col not in {"label_pair", "task", "mapping"}
    ]
    delta_rows = []
    for condition in condition_cols:
        work = pivot[["label_pair", "task", "mapping"]].copy()
        work["target_margin"] = pivot["T"]
        work["control_margin"] = pivot["C"]
        work["condition_margin"] = pivot[condition]
        work = work.dropna(subset=["target_margin", "control_margin", "condition_margin"])
        if work.empty:
            continue
        gap = work["target_margin"] - work["control_margin"]
        valid = gap.abs() > 1e-12
        work = work[valid].copy()
        if work.empty:
            continue
        gap = work["target_margin"] - work["control_margin"]
        work["condition"] = str(condition)
        work["target_control_reference_gap"] = gap
        work["abs_target_control_reference_gap"] = gap.abs()
        work["condition_minus_control"] = work["condition_margin"] - work["control_margin"]
        work["condition_minus_target"] = work["condition_margin"] - work["target_margin"]
        work["fraction_toward_target"] = work["condition_minus_control"] / gap
        work["distance_to_target_fraction"] = (1.0 - work["fraction_toward_target"]).abs()
        work["distance_to_control_fraction"] = work["fraction_toward_target"].abs()
        delta_rows.append(work)

    if not delta_rows:
        return pd.DataFrame()
    delta = pd.concat(delta_rows, ignore_index=True)
    return (
        delta
        .groupby(key_cols, as_index=False)
        .agg(
            mean_fraction_toward_target=("fraction_toward_target", "mean"),
            median_fraction_toward_target=("fraction_toward_target", "median"),
            mean_distance_to_target_fraction=("distance_to_target_fraction", "mean"),
            mean_distance_to_control_fraction=("distance_to_control_fraction", "mean"),
            mean_abs_reference_gap=("abs_target_control_reference_gap", "mean"),
            mean_abs_condition_minus_control=(
                "condition_minus_control",
                lambda s: float(np.mean(np.abs(s))),
            ),
            mean_abs_condition_minus_target=(
                "condition_minus_target",
                lambda s: float(np.mean(np.abs(s))),
            ),
            n_label_task_mappings=("fraction_toward_target", "size"),
        )
    )


def analyze_order_hysteresis(root: Path, rows: list[dict], rng: np.random.Generator, n_boot: int) -> None:
    raw = read_csv_if_exists(root, "order_hysteresis_raw.csv")
    if raw.empty:
        return
    needed = {
        "index", "condition", "label_pair", "task", "mapping",
        "semantic_margin_first_minus_second",
    }
    if not needed.issubset(raw.columns):
        return

    unit_summary = (
        raw[list(needed)]
        .groupby(["index", "condition", "label_pair", "task", "mapping"], as_index=False)
        .agg(margin=("semantic_margin_first_minus_second", "mean"))
    )
    units = sorted(unit_summary["index"].drop_duplicates().tolist())
    key_df = (
        unit_summary[["label_pair", "task", "mapping"]]
        .drop_duplicates()
        .sort_values(["label_pair", "task", "mapping"])
        .reset_index(drop=True)
    )
    key_index = pd.MultiIndex.from_frame(key_df[["label_pair", "task", "mapping"]])
    conditions = sorted(str(c) for c in unit_summary["condition"].drop_duplicates())
    if "T" not in conditions or "C" not in conditions:
        return

    matrices = {}
    for condition in conditions:
        sub = unit_summary[unit_summary["condition"].astype(str) == condition]
        pivot = sub.pivot_table(
            index="index",
            columns=["label_pair", "task", "mapping"],
            values="margin",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=units, columns=key_index)
        matrices[condition] = pivot.to_numpy(dtype=float)

    n_units = len(units)
    if n_units == 0:
        return
    weights = rng.multinomial(n_units, [1.0 / n_units] * n_units, size=n_boot)
    observed_weights = np.ones((1, n_units), dtype=float)

    def weighted_mean(matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (w @ matrix) / n_units

    target_obs = weighted_mean(matrices["T"], observed_weights)[0]
    control_obs = weighted_mean(matrices["C"], observed_weights)[0]
    target_boot = weighted_mean(matrices["T"], weights)
    control_boot = weighted_mean(matrices["C"], weights)
    gap_obs = target_obs - control_obs
    gap_boot = target_boot - control_boot

    task_values = sorted(key_df["task"].astype(str).drop_duplicates())
    task_masks = {
        task: key_df["task"].astype(str).to_numpy() == task
        for task in task_values
    }
    central_axis_tasks = [
        "requested_task_vs_substitute",
        "trust_context_vs_risk_frame",
    ]
    central_axis_mask = key_df["task"].astype(str).isin(central_axis_tasks).to_numpy()

    for condition in conditions:
        cond_obs = weighted_mean(matrices[condition], observed_weights)[0]
        cond_boot = weighted_mean(matrices[condition], weights)
        frac_obs = (cond_obs - control_obs) / gap_obs
        frac_boot = (cond_boot - control_boot) / gap_boot
        cond_minus_control_obs = cond_obs - control_obs
        cond_minus_control_boot = cond_boot - control_boot
        cond_minus_target_obs = cond_obs - target_obs
        cond_minus_target_boot = cond_boot - target_boot

        metric_specs = [
            (
                "mean_fraction_toward_target",
                float(np.nanmean(frac_obs)),
                np.nanmean(frac_boot, axis=1),
            ),
            (
                "median_fraction_toward_target",
                float(np.nanmedian(frac_obs)),
                np.nanmedian(frac_boot, axis=1),
            ),
            (
                "mean_abs_condition_minus_control",
                float(np.nanmean(np.abs(cond_minus_control_obs))),
                np.nanmean(np.abs(cond_minus_control_boot), axis=1),
            ),
            (
                "mean_abs_condition_minus_target",
                float(np.nanmean(np.abs(cond_minus_target_obs))),
                np.nanmean(np.abs(cond_minus_target_boot), axis=1),
            ),
        ]
        for metric, observed, boot_values in metric_specs:
            append_bootstrap_metric_row(
                rows,
                family="order_hysteresis",
                scope=f"condition:{condition}",
                metric=metric,
                observed=observed,
                boot_values=boot_values,
                n_units=n_units,
                n_rows=len(raw),
                unit_col="index",
                value_col=metric,
                extra={"condition": condition},
                n_boot=n_boot,
            )

        if np.any(central_axis_mask):
            central_metric_specs = [
                (
                    "mean_fraction_toward_target",
                    float(np.nanmean(frac_obs[central_axis_mask])),
                    np.nanmean(frac_boot[:, central_axis_mask], axis=1),
                ),
                (
                    "median_fraction_toward_target",
                    float(np.nanmedian(frac_obs[central_axis_mask])),
                    np.nanmedian(frac_boot[:, central_axis_mask], axis=1),
                ),
                (
                    "mean_abs_condition_minus_control",
                    float(np.nanmean(np.abs(cond_minus_control_obs[central_axis_mask]))),
                    np.nanmean(np.abs(cond_minus_control_boot[:, central_axis_mask]), axis=1),
                ),
                (
                    "mean_abs_condition_minus_target",
                    float(np.nanmean(np.abs(cond_minus_target_obs[central_axis_mask]))),
                    np.nanmean(np.abs(cond_minus_target_boot[:, central_axis_mask]), axis=1),
                ),
            ]
            for metric, observed, boot_values in central_metric_specs:
                append_bootstrap_metric_row(
                    rows,
                    family="order_hysteresis_central_axis",
                    scope=(
                        f"condition:{condition},tasks:"
                        + ";".join(central_axis_tasks)
                    ),
                    metric=metric,
                    observed=observed,
                    boot_values=boot_values,
                    n_units=n_units,
                    n_rows=len(raw),
                    unit_col="index",
                    value_col=metric,
                    extra={"condition": condition},
                    n_boot=n_boot,
                )

        for task, mask in task_masks.items():
            if not np.any(mask):
                continue
            task_metric_specs = [
                (
                    "mean_fraction_toward_target",
                    float(np.nanmean(frac_obs[mask])),
                    np.nanmean(frac_boot[:, mask], axis=1),
                ),
                (
                    "median_fraction_toward_target",
                    float(np.nanmedian(frac_obs[mask])),
                    np.nanmedian(frac_boot[:, mask], axis=1),
                ),
                (
                    "mean_abs_condition_minus_control",
                    float(np.nanmean(np.abs(cond_minus_control_obs[mask]))),
                    np.nanmean(np.abs(cond_minus_control_boot[:, mask]), axis=1),
                ),
                (
                    "mean_abs_condition_minus_target",
                    float(np.nanmean(np.abs(cond_minus_target_obs[mask]))),
                    np.nanmean(np.abs(cond_minus_target_boot[:, mask]), axis=1),
                ),
            ]
            for metric, observed, boot_values in task_metric_specs:
                append_bootstrap_metric_row(
                    rows,
                    family="order_hysteresis",
                    scope=f"condition:{condition},task:{task}",
                    metric=metric,
                    observed=observed,
                    boot_values=boot_values,
                    n_units=n_units,
                    n_rows=len(raw),
                    unit_col="index",
                    value_col=metric,
                    extra={"condition": condition, "task": task},
                    n_boot=n_boot,
                )


def summarize_mixing_threshold(raw: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    needed = {
        "mixing_order", "target_fraction", "label_pair", "task", "mapping",
        "semantic_margin_first_minus_second",
    }
    if raw.empty or not needed.issubset(raw.columns):
        return pd.DataFrame()

    summary = (
        raw[list(needed)]
        .groupby(["mixing_order", "target_fraction", "label_pair", "task", "mapping"], as_index=False)
        .agg(condition_margin=("semantic_margin_first_minus_second", "mean"))
    )
    delta_rows = []
    for key, sub in summary.groupby(["mixing_order", "label_pair", "task", "mapping"]):
        margins = {
            float(row["target_fraction"]): float(row["condition_margin"])
            for _, row in sub.iterrows()
        }
        if 0.0 not in margins or 1.0 not in margins:
            continue
        control_margin = margins[0.0]
        target_margin = margins[1.0]
        reference_gap = target_margin - control_margin
        if not math.isfinite(reference_gap) or abs(reference_gap) <= 1e-12:
            continue
        mixing_order, label_pair, task, mapping = key
        for dose, condition_margin in margins.items():
            fraction = (condition_margin - control_margin) / reference_gap
            delta_rows.append({
                "mixing_order": str(mixing_order),
                "target_fraction": float(dose),
                "label_pair": str(label_pair),
                "task": str(task),
                "mapping": str(mapping),
                "condition_margin": condition_margin,
                "target_reference_margin": target_margin,
                "control_reference_margin": control_margin,
                "target_control_reference_gap": reference_gap,
                "abs_target_control_reference_gap": abs(reference_gap),
                "condition_minus_control": condition_margin - control_margin,
                "fraction_toward_target": fraction,
                "absolute_fraction_error": abs(float(dose) - fraction),
            })

    if not delta_rows:
        return pd.DataFrame()
    delta = pd.DataFrame(delta_rows)
    return (
        delta
        .groupby(key_cols, as_index=False)
        .agg(
            mean_fraction_toward_target=("fraction_toward_target", "mean"),
            median_fraction_toward_target=("fraction_toward_target", "median"),
            mean_abs_reference_gap=("abs_target_control_reference_gap", "mean"),
            mean_abs_condition_minus_control=(
                "condition_minus_control",
                lambda s: float(np.mean(np.abs(s))),
            ),
            mean_absolute_fraction_error=("absolute_fraction_error", "mean"),
            n_label_task_mappings=("fraction_toward_target", "size"),
        )
    )


def summarize_mixing_crossings(raw: pd.DataFrame) -> pd.DataFrame:
    condition_summary = summarize_mixing_threshold(
        raw,
        ["mixing_order", "target_fraction"],
    )
    if condition_summary.empty:
        return pd.DataFrame()
    rows = []
    for order, sub in condition_summary.groupby("mixing_order"):
        sub = sub.sort_values("target_fraction")
        interior = sub[
            (sub["target_fraction"] > 0.0)
            & (sub["target_fraction"] < 1.0)
        ]
        crossed = interior[interior["mean_fraction_toward_target"] >= 0.50]
        first_cross = float(crossed.iloc[0]["target_fraction"]) if not crossed.empty else np.nan
        endpoint = sub.loc[
            sub["target_fraction"] == 1.0,
            "mean_fraction_toward_target",
        ]
        midpoint = sub.loc[
            sub["target_fraction"] == 0.5,
            "mean_fraction_toward_target",
        ]
        rows.append({
            "mixing_order": str(order),
            "first_crossing_0_5": first_cross,
            "mid_fraction_toward_target": float(midpoint.mean()) if not midpoint.empty else np.nan,
            "endpoint_fraction_toward_target": float(endpoint.mean()) if not endpoint.empty else np.nan,
        })
    return pd.DataFrame(rows)


def analyze_mixing_threshold(root: Path, rows: list[dict], rng: np.random.Generator, n_boot: int) -> None:
    raw = read_csv_if_exists(root, "mixing_threshold_raw.csv")
    if raw.empty:
        return
    needed = {
        "index", "mixing_order", "target_fraction", "label_pair", "task", "mapping",
        "semantic_margin_first_minus_second",
    }
    if not needed.issubset(raw.columns):
        return

    unit_summary = (
        raw[list(needed)]
        .groupby(
            ["index", "mixing_order", "target_fraction", "label_pair", "task", "mapping"],
            as_index=False,
        )
        .agg(margin=("semantic_margin_first_minus_second", "mean"))
    )
    units = sorted(unit_summary["index"].drop_duplicates().tolist())
    key_df = (
        unit_summary[["label_pair", "task", "mapping"]]
        .drop_duplicates()
        .sort_values(["label_pair", "task", "mapping"])
        .reset_index(drop=True)
    )
    key_index = pd.MultiIndex.from_frame(key_df[["label_pair", "task", "mapping"]])
    orders = sorted(str(x) for x in unit_summary["mixing_order"].drop_duplicates())
    doses = sorted(float(x) for x in unit_summary["target_fraction"].drop_duplicates())
    if 0.0 not in doses or 1.0 not in doses:
        return

    matrices = {}
    for order in orders:
        for dose in doses:
            sub = unit_summary[
                (unit_summary["mixing_order"].astype(str) == order)
                & (unit_summary["target_fraction"].astype(float) == float(dose))
            ]
            pivot = sub.pivot_table(
                index="index",
                columns=["label_pair", "task", "mapping"],
                values="margin",
                aggfunc="mean",
            )
            pivot = pivot.reindex(index=units, columns=key_index)
            matrices[(order, float(dose))] = pivot.to_numpy(dtype=float)

    n_units = len(units)
    if n_units == 0:
        return
    weights = rng.multinomial(n_units, [1.0 / n_units] * n_units, size=n_boot)
    observed_weights = np.ones((1, n_units), dtype=float)

    def weighted_mean(matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (w @ matrix) / n_units

    task_values = sorted(key_df["task"].astype(str).drop_duplicates())
    task_masks = {
        task: key_df["task"].astype(str).to_numpy() == task
        for task in task_values
    }

    crossing_values: dict[str, dict[str, list[float]]] = {
        order: {
            "first_crossing_0_5": [],
            "mid_fraction_toward_target": [],
            "endpoint_fraction_toward_target": [],
        }
        for order in orders
    }
    observed_crossings = {}

    for order in orders:
        control_obs = weighted_mean(matrices[(order, 0.0)], observed_weights)[0]
        target_obs = weighted_mean(matrices[(order, 1.0)], observed_weights)[0]
        control_boot = weighted_mean(matrices[(order, 0.0)], weights)
        target_boot = weighted_mean(matrices[(order, 1.0)], weights)
        gap_obs = target_obs - control_obs
        gap_boot = target_boot - control_boot

        order_mean_fraction_by_dose = {}
        order_boot_fraction_by_dose = {}

        for dose in doses:
            cond_obs = weighted_mean(matrices[(order, dose)], observed_weights)[0]
            cond_boot = weighted_mean(matrices[(order, dose)], weights)
            frac_obs = (cond_obs - control_obs) / gap_obs
            frac_boot = (cond_boot - control_boot) / gap_boot
            cond_minus_control_obs = cond_obs - control_obs
            cond_minus_control_boot = cond_boot - control_boot
            abs_error_obs = np.abs(float(dose) - frac_obs)
            abs_error_boot = np.abs(float(dose) - frac_boot)
            order_mean_fraction_by_dose[dose] = float(np.nanmean(frac_obs))
            order_boot_fraction_by_dose[dose] = np.nanmean(frac_boot, axis=1)

            metric_specs = [
                (
                    "mean_fraction_toward_target",
                    float(np.nanmean(frac_obs)),
                    np.nanmean(frac_boot, axis=1),
                ),
                (
                    "median_fraction_toward_target",
                    float(np.nanmedian(frac_obs)),
                    np.nanmedian(frac_boot, axis=1),
                ),
                (
                    "mean_abs_condition_minus_control",
                    float(np.nanmean(np.abs(cond_minus_control_obs))),
                    np.nanmean(np.abs(cond_minus_control_boot), axis=1),
                ),
                (
                    "mean_absolute_fraction_error",
                    float(np.nanmean(abs_error_obs)),
                    np.nanmean(abs_error_boot, axis=1),
                ),
            ]
            for metric, observed, boot_values in metric_specs:
                append_bootstrap_metric_row(
                    rows,
                    family="mixing_threshold",
                    scope=f"order:{order},fraction:{dose:g}",
                    metric=metric,
                    observed=observed,
                    boot_values=boot_values,
                    n_units=n_units,
                    n_rows=len(raw),
                    unit_col="index",
                    value_col=metric,
                    extra={"mixing_order": order, "target_fraction": dose},
                    n_boot=n_boot,
                )

            for task, mask in task_masks.items():
                if not np.any(mask):
                    continue
                task_metric_specs = [
                    (
                        "mean_fraction_toward_target",
                        float(np.nanmean(frac_obs[mask])),
                        np.nanmean(frac_boot[:, mask], axis=1),
                    ),
                    (
                        "median_fraction_toward_target",
                        float(np.nanmedian(frac_obs[mask])),
                        np.nanmedian(frac_boot[:, mask], axis=1),
                    ),
                    (
                        "mean_abs_condition_minus_control",
                        float(np.nanmean(np.abs(cond_minus_control_obs[mask]))),
                        np.nanmean(np.abs(cond_minus_control_boot[:, mask]), axis=1),
                    ),
                    (
                        "mean_absolute_fraction_error",
                        float(np.nanmean(abs_error_obs[mask])),
                        np.nanmean(abs_error_boot[:, mask], axis=1),
                    ),
                ]
                for metric, observed, boot_values in task_metric_specs:
                    append_bootstrap_metric_row(
                        rows,
                        family="mixing_threshold",
                        scope=f"order:{order},fraction:{dose:g},task:{task}",
                        metric=metric,
                        observed=observed,
                        boot_values=boot_values,
                        n_units=n_units,
                        n_rows=len(raw),
                        unit_col="index",
                        value_col=metric,
                        extra={
                            "mixing_order": order,
                            "target_fraction": dose,
                            "task": task,
                        },
                        n_boot=n_boot,
                    )

        interior_doses = [d for d in doses if 0.0 < d < 1.0]
        crossed_obs = [
            d for d in interior_doses
            if order_mean_fraction_by_dose.get(d, np.nan) >= 0.5
        ]
        observed_crossings[order] = {
            "first_crossing_0_5": float(crossed_obs[0]) if crossed_obs else np.nan,
            "mid_fraction_toward_target": float(order_mean_fraction_by_dose.get(0.5, np.nan)),
            "endpoint_fraction_toward_target": float(order_mean_fraction_by_dose.get(1.0, np.nan)),
        }

        if interior_doses:
            boot_stack = np.column_stack([
                order_boot_fraction_by_dose[d] for d in interior_doses
            ])
            first_cross_boot = np.full(n_boot, np.nan, dtype=float)
            for b in range(n_boot):
                hits = np.where(boot_stack[b] >= 0.5)[0]
                if hits.size:
                    first_cross_boot[b] = interior_doses[int(hits[0])]
            crossing_values[order]["first_crossing_0_5"] = first_cross_boot
        crossing_values[order]["mid_fraction_toward_target"] = order_boot_fraction_by_dose.get(
            0.5,
            np.full(n_boot, np.nan, dtype=float),
        )
        crossing_values[order]["endpoint_fraction_toward_target"] = order_boot_fraction_by_dose.get(
            1.0,
            np.full(n_boot, np.nan, dtype=float),
        )

    for order in orders:
        for metric, observed in observed_crossings.get(order, {}).items():
            append_bootstrap_metric_row(
                rows,
                family="mixing_threshold_crossing",
                scope=f"order:{order}",
                metric=metric,
                observed=observed,
                boot_values=np.asarray(crossing_values[order][metric], dtype=float),
                n_units=n_units,
                n_rows=len(raw),
                unit_col="index",
                value_col=metric,
                extra={"mixing_order": order},
                n_boot=n_boot,
            )


def analyze_primary_logit(root: Path, rows: list[dict], rng: np.random.Generator, n_boot: int) -> None:
    scores = read_csv_if_exists(root, "per_text_mode_scores.csv")
    if scores.empty or "delta_target_minus_control" not in scores.columns:
        return

    add_bootstrap_rows(
        rows,
        scores,
        family="primary_logit_probe",
        group={"scope": "overall"},
        value_col="delta_target_minus_control",
        unit_col="index",
        rng=rng,
        n_boot=n_boot,
    )
    for task, sub in scores.groupby("task"):
        add_bootstrap_rows(
            rows,
            sub,
            family="primary_logit_probe",
            group={"scope": f"task:{task}"},
            value_col="delta_target_minus_control",
            unit_col="index",
            rng=rng,
            n_boot=n_boot,
        )


def render_report(summary: pd.DataFrame, root: Path, n_boot: int) -> str:
    lines = [
        "# Bootstrap Validity Report",
        "",
        f"- Source: `{root}`",
        f"- Bootstrap samples: `{n_boot}`",
        "- Resampling unit: inducing text `index`.",
        "- Interpretation: intervals are about variation across inducing texts, not variation across probe rows.",
        "",
        "## Headline Rows",
        "",
    ]

    wanted = [
        ("blind_neutral_probe_clean", "overall", "mean_abs"),
        ("blind_neutral_probe_clean", "task:requested_task_vs_substitute", "mean_abs"),
        ("blind_neutral_probe_clean", "task:trust_context_vs_risk_frame", "mean_abs"),
        ("blind_neutral_persistence", "overall", "mean_abs"),
        ("rejection_persistence", "overall", "mean_abs"),
        ("hard_control_specificity", None, "mean_abs_ratio"),
        ("agent_loop_clean_direct_margin", "overall", "mean_abs"),
        ("order_hysteresis_central_axis", "condition:TNC,tasks:requested_task_vs_substitute;trust_context_vs_risk_frame", "mean_fraction_toward_target"),
        ("order_hysteresis_central_axis", "condition:CNT,tasks:requested_task_vs_substitute;trust_context_vs_risk_frame", "mean_fraction_toward_target"),
        ("order_hysteresis_central_axis", "condition:TNN,tasks:requested_task_vs_substitute;trust_context_vs_risk_frame", "mean_fraction_toward_target"),
        ("order_hysteresis_central_axis", "condition:CNN,tasks:requested_task_vs_substitute;trust_context_vs_risk_frame", "mean_fraction_toward_target"),
        ("order_hysteresis", "condition:TNC", "mean_fraction_toward_target"),
        ("order_hysteresis", "condition:CNT", "mean_fraction_toward_target"),
        ("order_hysteresis", "condition:TNN", "mean_fraction_toward_target"),
        ("order_hysteresis", "condition:CNN", "mean_fraction_toward_target"),
        ("mixing_threshold", "order:target_prefix,fraction:0.125", "mean_fraction_toward_target"),
        ("mixing_threshold", "order:target_prefix,fraction:0.5", "mean_fraction_toward_target"),
        ("mixing_threshold", "order:target_suffix,fraction:0.125", "mean_fraction_toward_target"),
        ("mixing_threshold", "order:target_suffix,fraction:0.5", "mean_fraction_toward_target"),
        ("mixing_threshold_crossing", "order:target_prefix", "first_crossing_0_5"),
        ("mixing_threshold_crossing", "order:target_suffix", "first_crossing_0_5"),
    ]

    rows = []
    for family, scope, metric in wanted:
        sub = summary[(summary["family"] == family) & (summary["metric"] == metric)]
        if scope is not None:
            sub = sub[sub["scope"] == scope]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            turn_bits = []
            if "turn" in row and pd.notna(row.get("turn")):
                turn_bits.append(f"turn={int(row['turn'])}")
            if "rejection_applied" in row and pd.notna(row.get("rejection_applied")):
                turn_bits.append(f"rejection={bool(row['rejection_applied'])}")
            rows.append([
                row["family"],
                row["scope"],
                ", ".join(turn_bits),
                row["metric"],
                fmt(row["observed"]),
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
                int(row["n_units"]),
            ])

    if rows:
        lines.append("| family | scope | condition | metric | observed | 95% CI | n_units |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
        for row in rows:
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
    else:
        lines.append("_No headline rows available._")

    lines.extend([
        "",
        "## Mechanistic Reading",
        "",
        "If the strongest blind-probe intervals remain away from zero under text-level resampling, the effect is not only row-level probe redundancy.",
        "If persistence intervals remain positive at later turns, the result supports a session-state/readout persistence claim.",
        "If the hard-control specificity ratio remains above 1, the original texts beat the tested topic/style/length controls under the same text-level uncertainty model.",
        "If order-hysteresis intervals stay away from the endpoint implied by the last context, the readout is path-dependent under text-level resampling.",
        "If mixing-threshold intervals rise early and stay above the linear-dose line, small target-token fractions are sufficient to induce much of the target readout.",
        "",
        "## Boundary",
        "",
        "This bootstrap does not create new held-out text families and does not prove causal steerability. It only tests whether existing effects are robust when inducing text is treated as the independent unit.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.result_dir
    if not root.exists():
        raise FileNotFoundError(root)
    output_dir = args.output_dir or root / "validity_bootstrap"
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    analyze_blind_probe(root, rows, rng, args.n_boot)
    analyze_persistence(
        root,
        rows,
        raw_name="blind_neutral_persistence_raw.csv",
        family="blind_neutral_persistence",
        turn_col="filler_turns_elapsed",
        rng=rng,
        n_boot=args.n_boot,
    )
    analyze_persistence(
        root,
        rows,
        raw_name="rejection_persistence_raw.csv",
        family="rejection_persistence",
        turn_col="post_rejection_filler_turns",
        rng=rng,
        n_boot=args.n_boot,
    )
    analyze_hard_controls(root, rows, rng, args.n_boot)
    analyze_agent_loop(root, rows, rng, args.n_boot)
    analyze_order_hysteresis(root, rows, rng, args.n_boot)
    analyze_mixing_threshold(root, rows, rng, args.n_boot)
    analyze_primary_logit(root, rows, rng, args.n_boot)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No bootstrap-compatible result files found.")
    # Stable column order for easier reading.
    preferred = [
        "family", "scope", "rejection_applied", "turn", "metric",
        "observed", "bootstrap_mean", "ci_low", "ci_high",
        "n_units", "n_rows", "n_boot", "unit_col", "value_col",
    ]
    summary = summary[
        [c for c in preferred if c in summary.columns]
        + [c for c in summary.columns if c not in preferred]
    ]

    write_csv(summary, output_dir / "bootstrap_ci_summary.csv")
    report = render_report(summary, root, args.n_boot)
    report_path = output_dir / "bootstrap_validity_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
