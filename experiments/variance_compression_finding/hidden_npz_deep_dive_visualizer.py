#!/usr/bin/env python3
"""
Hidden NPZ Deep-Dive Visualizer
===============================

Reads saved hidden_last_token_base/instruct.npz artifacts from
base_vs_instruct_geometry_probability_audit.py and computes additional
geometry diagnostics without re-running the models.

Colab usage:

  !python /content/agent/experiments/variance_compression_finding/hidden_npz_deep_dive_visualizer.py \
    --run_dir /content/alignment_geometry_probability_run_fullbank/run_YYYYMMDD_HHMMSS \
    --late_lo 30 \
    --late_hi 47

Outputs are written to:

  <run_dir>/hidden_npz_deep_dive/

Main outputs:
  - deep_condition_layer_metrics.csv
  - deep_target_control_contrast_by_layer.csv
  - deep_base_instruct_alignment_by_layer.csv
  - deep_late_band_condition_summary.csv
  - deep_late_band_contrast_summary.csv
  - PNG line plots and PNG table snapshots
  - README_DEEP_DIVE.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_TAGS = ("base", "instruct")
DEFAULT_CONDITION_ORDER = [
    "target",
    "target_word_shuffle",
    "target_sentence_shuffle",
    "control",
    "question_only",
]
PLOT_COLORS = {
    "target": "#1f77b4",
    "target_word_shuffle": "#9467bd",
    "target_sentence_shuffle": "#17becf",
    "control": "#ff7f0e",
    "question_only": "#2ca02c",
    "base": "#4c78a8",
    "instruct": "#f58518",
}


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + eps)


def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cos_sim(a, b))


def mean_pairwise_l2(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return 0.0
    vals: List[float] = []
    for i in range(n):
        diff = X[i + 1 :] - X[i]
        if diff.size:
            vals.extend(np.linalg.norm(diff, axis=1).tolist())
    return float(np.mean(vals)) if vals else 0.0


def mean_pairwise_cosdist(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return 0.0
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    vals: List[float] = []
    for i in range(n):
        sims = Xn[i + 1 :] @ Xn[i]
        vals.extend((1.0 - sims).tolist())
    return float(np.mean(vals)) if vals else 0.0


def singular_spectrum_metrics(centered: np.ndarray) -> Dict[str, float]:
    centered = np.asarray(centered, dtype=np.float64)
    if centered.shape[0] < 2:
        return {
            "effective_rank_pr": 0.0,
            "spectral_entropy": 0.0,
            "spectral_entropy_norm": 0.0,
            "top1_pc_variance_share": 0.0,
            "cov_trace": 0.0,
        }
    try:
        s = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError:
        s = np.zeros(min(centered.shape), dtype=np.float64)
    s2 = np.square(s.astype(np.float64))
    total = float(s2.sum())
    if total <= 1e-30:
        return {
            "effective_rank_pr": 0.0,
            "spectral_entropy": 0.0,
            "spectral_entropy_norm": 0.0,
            "top1_pc_variance_share": 0.0,
            "cov_trace": 0.0,
        }
    pr = float((total * total) / (np.square(s2).sum() + 1e-30))
    p = s2 / total
    entropy = float(-(p * np.log(p + 1e-30)).sum())
    entropy_norm = float(entropy / math.log(max(2, len(p))))
    top1 = float(s2[0] / total)
    cov_trace = float(total / max(1, centered.shape[0] - 1))
    return {
        "effective_rank_pr": pr,
        "spectral_entropy": entropy,
        "spectral_entropy_norm": entropy_norm,
        "top1_pc_variance_share": top1,
        "cov_trace": cov_trace,
    }


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Linear CKA using centered Gram matrices over prompts."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    num = float(np.sum(K * L))
    den = math.sqrt(float(np.sum(K * K)) * float(np.sum(L * L))) + eps
    return float(num / den)


def auc_like(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    vals: List[float] = []
    for p in scores_pos:
        pair_scores = (p > scores_neg).astype(np.float64) + 0.5 * (p == scores_neg).astype(np.float64)
        vals.extend(pair_scores.tolist())
    return float(np.mean(vals)) if vals else float("nan")


def ensure_prompt_order(prompt_df: pd.DataFrame, hidden_n: int) -> pd.DataFrame:
    required = ["prompt_id", "condition", "condition_family", "context_id", "question_id"]
    missing = [c for c in required if c not in prompt_df.columns]
    if missing:
        raise RuntimeError(f"prompts.csv missing columns: {missing}")
    prompt_df = prompt_df.sort_values("prompt_id").reset_index(drop=True)
    ids = prompt_df["prompt_id"].astype(int).to_numpy()
    expected = np.arange(len(prompt_df), dtype=int)
    if not np.array_equal(ids, expected):
        raise RuntimeError("prompt_id must be contiguous 0..N-1 for direct hidden alignment.")
    if len(prompt_df) != hidden_n:
        raise RuntimeError(f"prompts.csv rows={len(prompt_df)} but hidden first dim={hidden_n}")
    return prompt_df


def load_run(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    prompt_path = run_dir / "prompts.csv"
    if not prompt_path.exists():
        raise FileNotFoundError(str(prompt_path))
    hidden: Dict[str, np.ndarray] = {}
    for model_tag in MODEL_TAGS:
        path = run_dir / f"hidden_last_token_{model_tag}.npz"
        if not path.exists():
            raise FileNotFoundError(str(path))
        data = np.load(path)
        if "hidden" not in data:
            raise RuntimeError(f"{path} has no `hidden` array")
        hidden[model_tag] = np.asarray(data["hidden"], dtype=np.float32)
    if hidden["base"].shape != hidden["instruct"].shape:
        raise RuntimeError(f"base hidden shape {hidden['base'].shape} != instruct shape {hidden['instruct'].shape}")
    prompt_df = ensure_prompt_order(pd.read_csv(prompt_path), hidden["base"].shape[0])
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return prompt_df, hidden, metadata


def condition_order(prompt_df: pd.DataFrame) -> List[str]:
    seen = list(prompt_df["condition"].drop_duplicates())
    ordered = [c for c in DEFAULT_CONDITION_ORDER if c in seen]
    ordered.extend([c for c in seen if c not in ordered])
    return ordered


def compute_condition_layer_metrics(prompt_df: pd.DataFrame, hidden: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_layers = hidden["base"].shape[1]
    conds = condition_order(prompt_df)
    q_centroids: Dict[Tuple[str, int], np.ndarray] = {}
    global_centroids: Dict[Tuple[str, int], np.ndarray] = {}

    for model_tag in MODEL_TAGS:
        H = hidden[model_tag].astype(np.float64)
        for layer in range(n_layers):
            global_centroids[(model_tag, layer)] = H[:, layer, :].mean(axis=0)
            q_mask = prompt_df["condition"].eq("question_only").to_numpy()
            if q_mask.any():
                q_centroids[(model_tag, layer)] = H[q_mask, layer, :].mean(axis=0)

    for model_tag in MODEL_TAGS:
        H = hidden[model_tag].astype(np.float64)
        for condition in conds:
            idx = prompt_df.index[prompt_df["condition"].eq(condition)].to_numpy()
            family = str(prompt_df.loc[idx[0], "condition_family"]) if len(idx) else ""
            for layer in range(n_layers):
                X = H[idx, layer, :]
                centroid = X.mean(axis=0)
                centered = X - centroid[None, :]
                centroid_norm = float(np.linalg.norm(centroid))
                norms = np.linalg.norm(X, axis=1) + 1e-12
                centroid_unit = unit(centroid)
                cos_to_centroid = (X @ centroid_unit) / norms
                spec = singular_spectrum_metrics(centered)
                q_centroid = q_centroids.get((model_tag, layer))
                g_centroid = global_centroids[(model_tag, layer)]
                row = {
                    "model_tag": model_tag,
                    "condition": condition,
                    "condition_family": family,
                    "layer": int(layer),
                    "n_points": int(len(idx)),
                    "centroid_norm": centroid_norm,
                    "abs_disp_l2_mean": float(np.linalg.norm(centered, axis=1).mean()),
                    "rel_disp_l2_mean": float(np.linalg.norm(centered, axis=1).mean() / (centroid_norm + 1e-12)),
                    "pairwise_l2_mean": mean_pairwise_l2(X),
                    "pairwise_cosine_distance_mean": mean_pairwise_cosdist(X),
                    "cos_to_centroid_mean": float(cos_to_centroid.mean()),
                    "cos_to_centroid_std": float(cos_to_centroid.std()),
                    "angular_disp_to_centroid": float(1.0 - cos_to_centroid.mean()),
                    "centroid_to_global_l2": float(np.linalg.norm(centroid - g_centroid)),
                    "centroid_to_global_cosdist": cos_dist(centroid, g_centroid),
                    **spec,
                }
                if q_centroid is not None:
                    row["centroid_to_question_only_l2"] = float(np.linalg.norm(centroid - q_centroid))
                    row["centroid_to_question_only_cosdist"] = cos_dist(centroid, q_centroid)
                rows.append(row)
    return pd.DataFrame(rows)


def leave_one_question_contrast(prompt_df: pd.DataFrame, X: np.ndarray) -> Dict[str, float]:
    target_mask_all = prompt_df["condition"].eq("target").to_numpy()
    control_mask_all = prompt_df["condition"].eq("control").to_numpy()
    qids = sorted(set(prompt_df.loc[target_mask_all | control_mask_all, "question_id"].astype(int).tolist()))
    accs: List[float] = []
    aucs: List[float] = []
    gaps: List[float] = []
    for qid in qids:
        q_mask = prompt_df["question_id"].astype(int).eq(qid).to_numpy()
        train_t = target_mask_all & ~q_mask
        train_c = control_mask_all & ~q_mask
        test_t = target_mask_all & q_mask
        test_c = control_mask_all & q_mask
        if not train_t.any() or not train_c.any() or not test_t.any() or not test_c.any():
            continue
        mt = X[train_t].mean(axis=0)
        mc = X[train_c].mean(axis=0)
        axis = unit(mt - mc)
        train_scores_t = X[train_t] @ axis
        train_scores_c = X[train_c] @ axis
        threshold = 0.5 * (float(train_scores_t.mean()) + float(train_scores_c.mean()))
        test_scores_t = X[test_t] @ axis
        test_scores_c = X[test_c] @ axis
        acc = 0.5 * (float((test_scores_t > threshold).mean()) + float((test_scores_c < threshold).mean()))
        accs.append(acc)
        aucs.append(auc_like(test_scores_t, test_scores_c))
        gaps.append(float(test_scores_t.mean() - test_scores_c.mean()))
    return {
        "loo_question_balanced_acc": float(np.mean(accs)) if accs else float("nan"),
        "loo_question_auc_like": float(np.mean(aucs)) if aucs else float("nan"),
        "loo_question_projection_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "loo_question_folds": int(len(accs)),
    }


def compute_target_control_contrast(prompt_df: pd.DataFrame, hidden: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_layers = hidden["base"].shape[1]
    target_mask = prompt_df["condition"].eq("target").to_numpy()
    control_mask = prompt_df["condition"].eq("control").to_numpy()
    q_mask = prompt_df["condition"].eq("question_only").to_numpy()
    if not target_mask.any() or not control_mask.any():
        return pd.DataFrame()
    for model_tag in MODEL_TAGS:
        H = hidden[model_tag].astype(np.float64)
        for layer in range(n_layers):
            X = H[:, layer, :]
            Xt = X[target_mask]
            Xc = X[control_mask]
            mt = Xt.mean(axis=0)
            mc = Xc.mean(axis=0)
            axis = unit(mt - mc)
            st = Xt @ axis
            sc = Xc @ axis
            gap = float(st.mean() - sc.mean())
            pooled = math.sqrt(0.5 * (float(st.var(ddof=1)) + float(sc.var(ddof=1))) + 1e-12)
            row = {
                "model_tag": model_tag,
                "layer": int(layer),
                "n_target": int(Xt.shape[0]),
                "n_control": int(Xc.shape[0]),
                "target_control_centroid_l2": float(np.linalg.norm(mt - mc)),
                "target_control_centroid_cosdist": cos_dist(mt, mc),
                "target_control_projection_gap": gap,
                "target_control_projection_gap_z": float(gap / (pooled + 1e-12)),
                "target_control_axis_auc_like": auc_like(st, sc),
            }
            if q_mask.any():
                mq = X[q_mask].mean(axis=0)
                row["target_to_question_centroid_l2"] = float(np.linalg.norm(mt - mq))
                row["control_to_question_centroid_l2"] = float(np.linalg.norm(mc - mq))
                row["target_minus_control_to_question_l2"] = row["target_to_question_centroid_l2"] - row["control_to_question_centroid_l2"]
                row["target_to_question_cosdist"] = cos_dist(mt, mq)
                row["control_to_question_cosdist"] = cos_dist(mc, mq)
            row.update(leave_one_question_contrast(prompt_df, X))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_base_instruct_alignment(prompt_df: pd.DataFrame, hidden: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_layers = hidden["base"].shape[1]
    conds = ["all"] + condition_order(prompt_df)
    Hb = hidden["base"].astype(np.float64)
    Hi = hidden["instruct"].astype(np.float64)
    for condition in conds:
        if condition == "all":
            idx = np.arange(Hb.shape[0])
        else:
            idx = prompt_df.index[prompt_df["condition"].eq(condition)].to_numpy()
        if len(idx) < 2:
            continue
        for layer in range(n_layers):
            Xb = Hb[idx, layer, :]
            Xi = Hi[idx, layer, :]
            delta = Xi - Xb
            base_norm = np.linalg.norm(Xb, axis=1)
            instruct_norm = np.linalg.norm(Xi, axis=1)
            cos = np.sum(Xb * Xi, axis=1) / ((base_norm + 1e-12) * (instruct_norm + 1e-12))
            rows.append(
                {
                    "condition": condition,
                    "layer": int(layer),
                    "n_points": int(len(idx)),
                    "linear_cka_base_instruct": linear_cka(Xb, Xi),
                    "same_prompt_delta_l2_mean": float(np.linalg.norm(delta, axis=1).mean()),
                    "same_prompt_delta_l2_std": float(np.linalg.norm(delta, axis=1).std()),
                    "same_prompt_cosdist_mean": float((1.0 - cos).mean()),
                    "base_norm_mean": float(base_norm.mean()),
                    "instruct_norm_mean": float(instruct_norm.mean()),
                    "instruct_over_base_norm_mean": float((instruct_norm / (base_norm + 1e-12)).mean()),
                }
            )
    return pd.DataFrame(rows)


def late_band_summary(df: pd.DataFrame, late_lo: int, late_hi: int, group_cols: Sequence[str]) -> pd.DataFrame:
    band = df[(df["layer"] >= int(late_lo)) & (df["layer"] <= int(late_hi))].copy()
    return band.groupby(list(group_cols), dropna=False).mean(numeric_only=True).reset_index()


def add_instruct_minus_base(summary: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if "model_tag" not in summary.columns or not set(summary["model_tag"]) >= {"base", "instruct"}:
        return summary
    base = summary[summary["model_tag"].eq("base")].copy()
    inst = summary[summary["model_tag"].eq("instruct")].copy()
    metric_cols = [c for c in summary.columns if c not in set(keys) | {"model_tag"} and pd.api.types.is_numeric_dtype(summary[c])]
    if keys:
        merged = base[list(keys) + metric_cols].merge(inst[list(keys) + metric_cols], on=list(keys), suffixes=("_base", "_instruct"))
    else:
        merged = base[metric_cols].reset_index(drop=True).iloc[[0]].copy()
        inst_one = inst[metric_cols].reset_index(drop=True).iloc[[0]].copy()
        merged = merged.merge(inst_one, left_index=True, right_index=True, suffixes=("_base", "_instruct"))
    for col in metric_cols:
        merged[f"{col}_instruct_minus_base"] = merged[f"{col}_instruct"] - merged[f"{col}_base"]
        merged[f"{col}_instruct_over_base"] = merged[f"{col}_instruct"] / (merged[f"{col}_base"] + 1e-12)
    return merged


def save_line_plot(
    df: pd.DataFrame,
    out_path: Path,
    x: str,
    y: str,
    hue: str,
    title: str,
    ylabel: Optional[str] = None,
    style_col: Optional[str] = None,
    dpi: int = 170,
) -> None:
    if df.empty or y not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 6.2))
    groups = [hue] if style_col is None else [hue, style_col]
    for key, g in df.groupby(groups, dropna=False):
        if style_col is None:
            label = str(key)
            color_key = str(key)
            linestyle = "-"
        else:
            label = " / ".join(map(str, key if isinstance(key, tuple) else (key,)))
            color_key = str(key[0] if isinstance(key, tuple) else key)
            linestyle = "-" if (not isinstance(key, tuple) or str(key[1]) == "base") else "--"
        ax.plot(
            g[x],
            g[y],
            label=label,
            linewidth=2.2,
            alpha=0.92,
            color=PLOT_COLORS.get(color_key, None),
            linestyle=linestyle,
        )
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_heatmap(
    matrix_df: pd.DataFrame,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    dpi: int = 170,
) -> None:
    if matrix_df.empty:
        return
    data = matrix_df.to_numpy(dtype=float)
    fig_w = max(8, 0.55 * len(matrix_df.columns) + 3)
    fig_h = max(4.8, 0.45 * len(matrix_df.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_table_png(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    max_rows: int = 18,
    max_cols: int = 10,
    dpi: int = 190,
) -> None:
    if df.empty:
        return
    show = df.copy().head(max_rows)
    if show.shape[1] > max_cols:
        show = show.iloc[:, :max_cols]
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.4g}")
    fig_w = max(9, min(22, 1.35 * show.shape[1]))
    fig_h = max(3.2, 0.48 * (show.shape[0] + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", pad=16)
    table = ax.table(
        cellText=show.astype(str).values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f4f6")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#d1d5db")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    out_dir: Path,
    condition_metrics: pd.DataFrame,
    contrast: pd.DataFrame,
    alignment: pd.DataFrame,
    late_condition: pd.DataFrame,
    late_contrast: pd.DataFrame,
    late_alignment: pd.DataFrame,
    dpi: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric, title in [
        ("centroid_norm", "Centroid Norm By Layer"),
        ("rel_disp_l2_mean", "Relative L2 Dispersion By Layer"),
        ("pairwise_cosine_distance_mean", "Pairwise Cosine Distance By Layer"),
        ("effective_rank_pr", "Effective Rank By Layer"),
        ("top1_pc_variance_share", "Top PC Variance Share By Layer"),
        ("centroid_to_question_only_l2", "Distance From Question-Only Centroid By Layer"),
    ]:
        save_line_plot(
            condition_metrics,
            plot_dir / f"{metric}_by_layer.png",
            x="layer",
            y=metric,
            hue="condition",
            style_col="model_tag",
            title=title,
            dpi=dpi,
        )

    for metric, title in [
        ("target_control_centroid_l2", "Target-Control Centroid L2 By Layer"),
        ("target_control_projection_gap_z", "Target-Control Projection Gap Z By Layer"),
        ("target_control_axis_auc_like", "Target-Control Axis AUC-like By Layer"),
        ("loo_question_balanced_acc", "Leave-One-Question Target-Control Accuracy By Layer"),
        ("target_minus_control_to_question_l2", "Target-Control Distance To Question-Only Delta"),
    ]:
        save_line_plot(
            contrast,
            plot_dir / f"{metric}_by_layer.png",
            x="layer",
            y=metric,
            hue="model_tag",
            title=title,
            dpi=dpi,
        )

    for metric, title in [
        ("linear_cka_base_instruct", "Base-Instruct Linear CKA By Layer"),
        ("same_prompt_delta_l2_mean", "Same-Prompt Base-Instruct L2 Delta By Layer"),
        ("same_prompt_cosdist_mean", "Same-Prompt Base-Instruct Cosine Distance By Layer"),
        ("instruct_over_base_norm_mean", "Instruct/Base Norm Ratio By Layer"),
    ]:
        save_line_plot(
            alignment,
            plot_dir / f"{metric}_by_layer.png",
            x="layer",
            y=metric,
            hue="condition",
            title=title,
            dpi=dpi,
        )

    heat_cols = [
        "centroid_norm",
        "abs_disp_l2_mean",
        "rel_disp_l2_mean",
        "pairwise_cosine_distance_mean",
        "effective_rank_pr",
        "spectral_entropy_norm",
        "top1_pc_variance_share",
        "centroid_to_question_only_l2",
    ]
    heat_df = late_condition.copy()
    heat_df["row"] = heat_df["model_tag"].astype(str) + " / " + heat_df["condition"].astype(str)
    heat_df = heat_df.set_index("row")[[c for c in heat_cols if c in heat_df.columns]]
    z = (heat_df - heat_df.mean(axis=0)) / (heat_df.std(axis=0) + 1e-12)
    save_heatmap(z, plot_dir / "late_condition_metric_zscore_heatmap.png", "Late L30-L47 Condition Metrics (column z-score)", dpi=dpi)

    save_table_png(
        late_condition[
            [
                c
                for c in [
                    "model_tag",
                    "condition",
                    "centroid_norm",
                    "rel_disp_l2_mean",
                    "pairwise_cosine_distance_mean",
                    "effective_rank_pr",
                    "spectral_entropy_norm",
                    "top1_pc_variance_share",
                    "centroid_to_question_only_l2",
                ]
                if c in late_condition.columns
            ]
        ],
        plot_dir / "late_condition_summary_table.png",
        "Late L30-L47 Condition Summary",
        dpi=dpi,
    )
    save_table_png(
        late_contrast[
            [
                c
                for c in [
                    "model_tag",
                    "target_control_centroid_l2",
                    "target_control_projection_gap_z",
                    "target_control_axis_auc_like",
                    "loo_question_balanced_acc",
                    "target_to_question_centroid_l2",
                    "control_to_question_centroid_l2",
                ]
                if c in late_contrast.columns
            ]
        ],
        plot_dir / "late_target_control_contrast_table.png",
        "Late L30-L47 Target-Control Contrast",
        dpi=dpi,
    )
    save_table_png(
        late_alignment[
            [
                c
                for c in [
                    "condition",
                    "linear_cka_base_instruct",
                    "same_prompt_delta_l2_mean",
                    "same_prompt_cosdist_mean",
                    "instruct_over_base_norm_mean",
                ]
                if c in late_alignment.columns
            ]
        ],
        plot_dir / "late_base_instruct_alignment_table.png",
        "Late L30-L47 Base-Instruct Alignment",
        dpi=dpi,
    )


def write_readme(
    out_dir: Path,
    run_dir: Path,
    metadata: Dict[str, Any],
    prompt_df: pd.DataFrame,
    late_lo: int,
    late_hi: int,
    late_condition: pd.DataFrame,
    late_contrast: pd.DataFrame,
    late_alignment: pd.DataFrame,
) -> None:
    lines = [
        "# Hidden NPZ Deep Dive",
        "",
        f"Run directory: `{run_dir}`",
        f"Late band: `{late_lo}..{late_hi}`",
        "",
        "## Input Coverage",
        "",
        f"Prompts: `{len(prompt_df)}`",
        "",
        "Condition counts:",
        "",
        "```text",
        prompt_df["condition"].value_counts(sort=False).to_string(),
        "```",
        "",
        "Metadata:",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Additional Metrics",
        "",
        "- `linear_cka_base_instruct`: representation similarity between base and instruct hidden states over matching prompts.",
        "- `target_control_projection_gap_z`: target-control diff-in-means gap normalized by pooled projection std.",
        "- `target_control_axis_auc_like`: pairwise AUC-like score on the target-control diff-in-means axis.",
        "- `loo_question_balanced_acc`: target/control classification when the axis is trained excluding each question_id.",
        "- `centroid_to_question_only_l2`: condition centroid displacement from question-only centroid.",
        "- `same_prompt_delta_l2_mean`: same-prompt hidden vector displacement between instruct and base.",
        "",
        "## Core CSVs",
        "",
        "- `deep_condition_layer_metrics.csv`",
        "- `deep_target_control_contrast_by_layer.csv`",
        "- `deep_base_instruct_alignment_by_layer.csv`",
        "- `deep_late_band_condition_summary.csv`",
        "- `deep_late_band_contrast_summary.csv`",
        "- `deep_late_band_base_instruct_alignment_summary.csv`",
        "",
        "## Plots",
        "",
        "See `plots/*.png`.",
        "",
        "## Late Target-Control Snapshot",
        "",
        "```text",
        late_contrast.to_string(index=False, float_format=lambda x: f"{x:.6g}") if not late_contrast.empty else "empty",
        "```",
        "",
        "## Late Base-Instruct Alignment Snapshot",
        "",
        "```text",
        late_alignment.to_string(index=False, float_format=lambda x: f"{x:.6g}") if not late_alignment.empty else "empty",
        "```",
    ]
    (out_dir / "README_DEEP_DIVE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep-dive analysis and PNG visualization for hidden_last_token NPZ artifacts.")
    ap.add_argument("--run_dir", required=True, help="Path to run_YYYYMMDD_HHMMSS directory.")
    ap.add_argument("--out_dir", default=None, help="Default: <run_dir>/hidden_npz_deep_dive")
    ap.add_argument("--late_lo", type=int, default=30)
    ap.add_argument("--late_hi", type=int, default=47)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "hidden_npz_deep_dive"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {run_dir}", flush=True)
    prompt_df, hidden, metadata = load_run(run_dir)
    print(f"[OK] prompts={len(prompt_df)} hidden_shape={hidden['base'].shape}", flush=True)
    print(f"[OK] condition_counts={prompt_df['condition'].value_counts(sort=False).to_dict()}", flush=True)

    print("[1/5] condition/layer geometry", flush=True)
    condition_metrics = compute_condition_layer_metrics(prompt_df, hidden)
    condition_metrics.to_csv(out_dir / "deep_condition_layer_metrics.csv", index=False)

    print("[2/5] target-control contrast", flush=True)
    contrast = compute_target_control_contrast(prompt_df, hidden)
    contrast.to_csv(out_dir / "deep_target_control_contrast_by_layer.csv", index=False)

    print("[3/5] base-instruct alignment", flush=True)
    alignment = compute_base_instruct_alignment(prompt_df, hidden)
    alignment.to_csv(out_dir / "deep_base_instruct_alignment_by_layer.csv", index=False)

    print("[4/5] late-band summaries", flush=True)
    late_condition = late_band_summary(condition_metrics, args.late_lo, args.late_hi, ["model_tag", "condition", "condition_family"])
    late_contrast = late_band_summary(contrast, args.late_lo, args.late_hi, ["model_tag"]) if not contrast.empty else pd.DataFrame()
    late_alignment = late_band_summary(alignment, args.late_lo, args.late_hi, ["condition"]) if not alignment.empty else pd.DataFrame()
    condition_delta = add_instruct_minus_base(late_condition, ["condition", "condition_family"])
    contrast_delta = add_instruct_minus_base(late_contrast, [])

    late_condition.to_csv(out_dir / "deep_late_band_condition_summary.csv", index=False)
    condition_delta.to_csv(out_dir / "deep_late_band_condition_instruct_minus_base.csv", index=False)
    late_contrast.to_csv(out_dir / "deep_late_band_contrast_summary.csv", index=False)
    contrast_delta.to_csv(out_dir / "deep_late_band_contrast_instruct_minus_base.csv", index=False)
    late_alignment.to_csv(out_dir / "deep_late_band_base_instruct_alignment_summary.csv", index=False)

    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "late_lo": int(args.late_lo),
        "late_hi": int(args.late_hi),
        "n_prompts": int(len(prompt_df)),
        "hidden_shape": list(map(int, hidden["base"].shape)),
        "condition_counts": {str(k): int(v) for k, v in prompt_df["condition"].value_counts(sort=False).to_dict().items()},
    }
    (out_dir / "deep_dive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] plots and PNG tables", flush=True)
    make_plots(out_dir, condition_metrics, contrast, alignment, late_condition, late_contrast, late_alignment, args.dpi)
    write_readme(out_dir, run_dir, metadata, prompt_df, args.late_lo, args.late_hi, late_condition, late_contrast, late_alignment)

    print("\n=== SAVED ===", flush=True)
    for name in [
        "deep_condition_layer_metrics.csv",
        "deep_target_control_contrast_by_layer.csv",
        "deep_base_instruct_alignment_by_layer.csv",
        "deep_late_band_condition_summary.csv",
        "deep_late_band_contrast_summary.csv",
        "deep_late_band_base_instruct_alignment_summary.csv",
        "deep_dive_manifest.json",
        "README_DEEP_DIVE.md",
        "plots/",
    ]:
        print(out_dir / name, flush=True)


if __name__ == "__main__":
    main()
