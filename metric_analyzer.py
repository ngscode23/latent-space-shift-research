"""
Compact metric analyzer for the latent-shift / attractor experiments.

Usage examples:
  python metric_analyzer.py "C:\\Users\\stasv\\Downloads"
  python metric_analyzer.py ".\\res\\attractor_results_core_diagnostics_qwen3_14b\\core_diagnostics_key_files"
  python metric_analyzer.py ".\\res\\attractor_results_core_diagnostics_qwen3_14b" --recursive

The script does not rerun the model. It only reads result CSV/JSON files and
builds a short Markdown report that separates:
  - core signal
  - likely confounds
  - metrics that should not be over-interpreted
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_OUTPUT_NAME = "metric_analysis_report.md"


@dataclass
class Finding:
    name: str
    status: str
    detail: str


@dataclass
class Report:
    root: Path
    lines: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def finding(self, name: str, status: str, detail: str) -> None:
        self.findings.append(Finding(name=name, status=status, detail=detail))

    def render(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def read_csv(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise RuntimeError(f"Cannot read CSV {path}: {last_error}")


def read_json(path: Path) -> dict:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise RuntimeError(f"Cannot read JSON {path}: {last_error}")


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def status_label(value: float, thresholds: tuple[float, float, float]) -> str:
    weak, medium, strong = thresholds
    if value >= strong:
        return "strong"
    if value >= medium:
        return "medium"
    if value >= weak:
        return "weak"
    return "tiny"


def direct_and_recursive_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.is_file()]
    return [p for p in root.glob("*") if p.is_file()]


def choose_file(candidates: Iterable[Path], preferred_names: Iterable[str] = ()) -> Path | None:
    candidates = list(candidates)
    if not candidates:
        return None
    preferred = {name.lower() for name in preferred_names}

    def score(path: Path) -> tuple[float, int, int]:
        name = path.name.lower()
        exact = 1 if name in preferred else 0
        not_copy = 1 if "копия" not in name and "copy" not in name else 0
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        # Fresh result beats a perfectly named stale file. If timestamps are
        # tied, prefer the non-copy name, then the exact canonical name.
        return (mtime, not_copy, exact)

    return sorted(candidates, key=score, reverse=True)[0]


class FileIndex:
    def __init__(self, root: Path, recursive: bool = False):
        self.root = root
        self.files = direct_and_recursive_files(root, recursive)
        self.by_name = {}
        for path in self.files:
            self.by_name.setdefault(path.name.lower(), []).append(path)

    def find(self, *patterns: str, preferred_names: Iterable[str] = ()) -> Path | None:
        matched: list[Path] = []
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if "*" in pattern_lower:
                prefix, _, suffix = pattern_lower.partition("*")
                for path in self.files:
                    name = path.name.lower()
                    if name.startswith(prefix) and name.endswith(suffix):
                        matched.append(path)
            else:
                matched.extend(self.by_name.get(pattern_lower, []))
        return choose_file(matched, preferred_names)


def add_table(report: Report, df: pd.DataFrame, max_rows: int = 12) -> None:
    if df.empty:
        report.add("_No rows._")
        return
    shown = df.head(max_rows).copy()
    headers = [str(col) for col in shown.columns]
    rows = []
    for _, row in shown.iterrows():
        rows.append([str(row[col]) for col in shown.columns])

    def clean_cell(value: str) -> str:
        return value.replace("\n", " ").replace("|", "\\|")

    report.add("| " + " | ".join(clean_cell(h) for h in headers) + " |")
    report.add("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        report.add("| " + " | ".join(clean_cell(cell) for cell in row) + " |")


def summarize_colab_trajectory(report: Report, index: FileIndex) -> bool:
    final_report = index.find("final_report.json", "final_report*.json", preferred_names=["final_report.json"])
    trajectory = index.find(
        "trajectory_metric_summary.csv",
        "trajectory_metric_summary*.csv",
        preferred_names=["trajectory_metric_summary.csv"],
    )
    summary_runs = index.find(
        "summary_by_exposure_run.csv",
        "summary_by_exposure_run*.csv",
        preferred_names=["summary_by_exposure_run.csv"],
    )
    layer_band = index.find("layer_band_summary.csv", "layer_band_summary*.csv")
    layer_profile = index.find("layer_profile_summary.csv", "layer_profile_summary*.csv")
    layer_scores = index.find("layer_scores.csv", "layer_scores*.csv")
    turn_scores = index.find("turn_scores.csv", "turn_scores*.csv")
    answers = index.find("answers_readable.csv", "answers_readable*.csv")

    if not any([final_report, trajectory, summary_runs, layer_band, layer_profile, layer_scores, turn_scores]):
        return False

    report.add("## Colab Trajectory Summary")
    if final_report:
        data = read_json(final_report)
        report.add(f"- Source: `{final_report}`")
        report.add(f"- Model: `{data.get('model_id', 'unknown')}`")
        report.add(f"- Runs/texts/turns: `{data.get('N_runs', 'n/a')}` runs, `{data.get('num_exposure_texts', 'n/a')}` exposure texts, `{data.get('num_turns', 'n/a')}` turns")
        report.add(f"- T*: `{fmt(data.get('T_star'), 6)}`")
        if "trajectory_magnitude_score_mean" in data:
            report.add(
                "- Main decomposition: "
                f"magnitude `{fmt(data.get('trajectory_magnitude_score_mean'))}`, "
                f"orthogonal `{fmt(data.get('trajectory_orthogonal_score_mean'))}`, "
                f"abs-aligned `{fmt(data.get('trajectory_abs_aligned_score_mean'))}`, "
                f"mean cosine `{fmt(data.get('trajectory_mean_direction_cosine'), 6)}`"
            )

    if trajectory:
        df = read_csv(trajectory)
        required = {"magnitude_mean", "orthogonal_mean", "abs_aligned_mean", "mean_direction_cosine"}
        if required.issubset(df.columns):
            rows = []
            for _, row in df.iterrows():
                mag = float(row["magnitude_mean"])
                orth = float(row["orthogonal_mean"])
                aligned = float(row["abs_aligned_mean"])
                t_star = float(row.get("t_star_mean", float("nan")))
                orth_share = orth / mag if mag else float("nan")
                aligned_share = aligned / mag if mag else float("nan")
                rows.append({
                    "exposure": int(row["exposure_id"]),
                    "T*": fmt(t_star, 6),
                    "magnitude": fmt(mag),
                    "orthogonal": fmt(orth),
                    "orth/mag": pct(orth_share),
                    "abs_aligned": fmt(aligned),
                    "aligned/mag": pct(aligned_share),
                    "mean_cos": fmt(row["mean_direction_cosine"], 6),
                    "signal_size": status_label(mag, (0.05, 0.15, 0.35)),
                })
                if mag >= 0.35 and orth_share >= 0.9 and aligned_share <= 0.08:
                    report.finding(
                        f"Exposure {int(row['exposure_id'])}: axis blindness",
                        "strong",
                        "Large hidden displacement is almost entirely orthogonal to mu; T* is not a good global strength metric here.",
                    )
                elif mag >= 0.15:
                    report.finding(
                        f"Exposure {int(row['exposure_id'])}: hidden displacement",
                        "medium",
                        "There is a visible target-control displacement, but axis/orthogonal split should be checked.",
                    )
            report.add()
            report.add("### Trajectory Decomposition")
            add_table(report, pd.DataFrame(rows))

    if layer_band:
        df = read_csv(layer_band)
        cols = [
            "exposure_id",
            "layer_band",
            "mean_magnitude_ratio",
            "mean_orthogonal_ratio_to_control",
            "mean_abs_parallel_ratio_to_control",
            "mean_orthogonal_fraction_of_delta",
            "mean_direction_cosine",
        ]
        cols = [c for c in cols if c in df.columns]
        if cols:
            report.add()
            report.add("### Layer Bands")
            pretty = df[cols].copy()
            for col in pretty.columns:
                if col not in {"exposure_id", "layer_band"}:
                    pretty[col] = pretty[col].map(lambda x: fmt(x))
            add_table(report, pretty)
            if "mean_magnitude_ratio" in df.columns and "layer_band" in df.columns:
                late = df[df["layer_band"].astype(str).str.lower().eq("late")]
                if not late.empty:
                    max_late = float(late["mean_magnitude_ratio"].max())
                    if max_late >= 0.35:
                        report.finding(
                            "Late-layer movement",
                            "strong" if max_late >= 0.5 else "medium",
                            f"Late-layer magnitude ratio reaches {fmt(max_late)}.",
                        )

    if layer_profile:
        df = read_csv(layer_profile)
        report.add()
        report.add("### Top Layers")
        for metric in ["magnitude_ratio_mean", "orthogonal_ratio_mean", "abs_parallel_ratio_mean"]:
            if metric in df.columns and "layer" in df.columns:
                top = df.sort_values(metric, ascending=False).head(6)
                cols = ["exposure_id", "layer", metric]
                extras = [
                    "direction_cosine_mean",
                    "weighted_mean",
                    "raw_mean",
                    "orthogonal_fraction_mean",
                ]
                cols += [c for c in extras if c in top.columns and c not in cols]
                pretty = top[cols].copy()
                for col in pretty.columns:
                    if col not in {"exposure_id", "layer"}:
                        pretty[col] = pretty[col].map(lambda x: fmt(x))
                report.add(f"Top by `{metric}`:")
                add_table(report, pretty, max_rows=6)
                report.add()

    if layer_scores:
        df = read_csv(layer_scores)
        if {"magnitude_ratio", "orthogonal_fraction_of_delta", "abs_parallel_ratio_to_control"}.issubset(df.columns):
            report.add("### Cell-Level Distribution")
            dist_rows = []
            for exposure_id, group in df.groupby("exposure_id"):
                mag = group["magnitude_ratio"]
                orth_frac = group["orthogonal_fraction_of_delta"]
                aligned = group["abs_parallel_ratio_to_control"]
                dist_rows.append({
                    "exposure": exposure_id,
                    "mag_p50": fmt(mag.quantile(0.50)),
                    "mag_p90": fmt(mag.quantile(0.90)),
                    "mag_p99": fmt(mag.quantile(0.99)),
                    "orth_frac_p50": fmt(orth_frac.quantile(0.50)),
                    "orth_frac_p90": fmt(orth_frac.quantile(0.90)),
                    "abs_parallel_p90": fmt(aligned.quantile(0.90)),
                    "abs_parallel_max": fmt(aligned.max()),
                })
            add_table(report, pd.DataFrame(dist_rows))

    if turn_scores:
        df = read_csv(turn_scores)
        if {"exposure_id", "turn", "turn_magnitude_score", "turn_orthogonal_score", "turn_score"}.issubset(df.columns):
            turn = (
                df.groupby(["exposure_id", "turn"], as_index=False)
                .agg(
                    t_star_mean=("turn_score", "mean"),
                    magnitude_mean=("turn_magnitude_score", "mean"),
                    orthogonal_mean=("turn_orthogonal_score", "mean"),
                    abs_aligned_mean=("turn_abs_aligned_score", "mean"),
                    cosine_mean=("turn_mean_direction_cosine", "mean"),
                )
            )
            report.add()
            report.add("### Turn Stability")
            pretty = turn.copy()
            for col in pretty.columns:
                if col not in {"exposure_id", "turn"}:
                    pretty[col] = pretty[col].map(lambda x: fmt(x))
            add_table(report, pretty, max_rows=20)

    if answers:
        df = read_csv(answers)
        if "num_generated_tokens" in df.columns:
            cap = 256
            if "max_new_tokens" in df.columns:
                try:
                    cap = int(df["max_new_tokens"].max())
                except Exception:
                    cap = 256
            df = df.copy()
            df["hit_cap"] = df["num_generated_tokens"] >= cap
            cap_rate = float(df["hit_cap"].mean()) if len(df) else 0.0
            report.add()
            report.add("### Generation Sanity")
            report.add(f"- Responses hitting generation cap `>= {cap}`: `{pct(cap_rate)}`")
            if cap_rate >= 0.25:
                report.finding(
                    "Generation cap confound",
                    "warning",
                    "Many readable answers hit max_new_tokens; text-level behavioral interpretation is partly truncated.",
                )
            group_cols = [c for c in ["condition", "exposure_id"] if c in df.columns]
            if group_cols:
                sanity = (
                    df.groupby(group_cols, as_index=False)
                    .agg(
                        n=("num_generated_tokens", "size"),
                        mean_tokens=("num_generated_tokens", "mean"),
                        cap_hits=("hit_cap", "sum"),
                    )
                )
                for col in ["mean_tokens"]:
                    sanity[col] = sanity[col].map(lambda x: fmt(x, 2))
                add_table(report, sanity)

    return True


def summarize_core_diagnostics(report: Report, index: FileIndex) -> bool:
    hidden = index.find("hidden_layer_metrics.csv")
    probe = index.find("linear_probe_accuracy.csv")
    token_diag = index.find("candidate_token_diagnostics.csv")
    blind_clean = index.find("blind_neutral_probe_clean_summary.csv")
    blind_gap = index.find("blind_neutral_probe_gap_summary.csv")
    blind_consistency = index.find("blind_neutral_probe_task_consistency.csv")
    hard_effect = index.find("hard_control_family_effect_summary.csv")
    hard_hidden = index.find("hard_control_family_hidden_summary.csv")
    checklist = index.find("interpretation_checklist.csv")
    steering = index.find("multilabel_semantic_steering_summary.csv", "ab_semantic_steering_summary.csv")
    rescue = index.find("multilabel_semantic_rescue_summary.csv", "ab_semantic_rescue_summary.csv", "rescue_summary.csv")

    if not any([hidden, probe, token_diag, blind_clean, blind_gap, hard_effect, checklist, steering, rescue]):
        return False

    report.add("## Core Diagnostics Summary")

    if hidden:
        df = read_csv(hidden)
        report.add(f"- Hidden metrics source: `{hidden}`")
        if {"hidden_index", "contrast_norm"}.issubset(df.columns):
            top = df.sort_values("contrast_norm", ascending=False).head(8)
            cols = [
                "hidden_index",
                "module_layer",
                "contrast_norm",
                "centroid_cosine",
                "cosine_distance",
                "contrast_over_mean_norm",
            ]
            cols = [c for c in cols if c in top.columns]
            pretty = top[cols].copy()
            for col in pretty.columns:
                if col not in {"hidden_index", "module_layer"}:
                    pretty[col] = pretty[col].map(lambda x: fmt(x))
            report.add()
            report.add("### Hidden Geometry")
            add_table(report, pretty)
            if "contrast_over_mean_norm" in top.columns:
                best_norm = float(top["contrast_over_mean_norm"].max())
                if best_norm >= 0.25:
                    report.finding(
                        "Hidden geometry separation",
                        "strong",
                        f"Best contrast_over_mean_norm is {fmt(best_norm)}.",
                    )

    if probe:
        df = read_csv(probe)
        if "probe_accuracy" in df.columns:
            top = df.sort_values("probe_accuracy", ascending=False).head(8)
            cols = [
                "hidden_index",
                "probe_accuracy",
                "permutation_mean_accuracy",
                "permutation_p95_accuracy",
                "accuracy_minus_permutation_mean",
                "cv_method",
            ]
            cols = [c for c in cols if c in top.columns]
            pretty = top[cols].copy()
            for col in pretty.columns:
                if col not in {"hidden_index", "cv_method"}:
                    pretty[col] = pretty[col].map(lambda x: fmt(x))
            report.add()
            report.add("### Linear Probe")
            add_table(report, pretty)
            best = float(df["probe_accuracy"].max())
            perm = float(df.get("permutation_p95_accuracy", pd.Series([0.5])).max())
            if best > perm:
                report.finding(
                    "Target/control probe",
                    "supported",
                    f"Best probe accuracy {fmt(best)} beats permutation p95 {fmt(perm)}.",
                )

    if token_diag:
        df = read_csv(token_diag)
        problem_col = "problem" if "problem" in df.columns else None
        problems = int(df[problem_col].sum()) if problem_col else 0
        report.add()
        report.add("### Candidate Token Diagnostics")
        report.add(f"- Candidate-token problem rows: `{problems}` / `{len(df)}`")
        if problems == 0:
            report.finding("Candidate token leakage check", "supported", "No same-first-token or zero-token label problems found.")
        else:
            report.finding("Candidate token leakage check", "warning", f"{problems} candidate-token rows are problematic.")

    if blind_clean:
        df = read_csv(blind_clean)
        report.add()
        report.add("### Blind Neutral Probes")
        add_table(report, df)
        row = df.iloc[0].to_dict() if len(df) else {}
        clean_fraction = float(row.get("clean_fraction", 0.0) or 0.0)
        mean_abs_gap = float(row.get("mean_abs_clean_gap", 0.0) or 0.0)
        if clean_fraction >= 0.4 and mean_abs_gap >= 5:
            report.finding(
                "Blind neutral semantic readout",
                "strong",
                f"Clean fraction {pct(clean_fraction)} with mean abs clean gap {fmt(mean_abs_gap)}.",
            )
        elif mean_abs_gap > 0:
            report.finding(
                "Blind neutral semantic readout",
                "weak",
                f"Mean abs clean gap {fmt(mean_abs_gap)}, but clean fraction is {pct(clean_fraction)}.",
            )

    if blind_gap:
        df = read_csv(blind_gap)
        metric = "target_control_gap" if "target_control_gap" in df.columns else None
        if metric and "task" in df.columns:
            task = (
                df.groupby("task", as_index=False)
                .agg(
                    mean_gap=(metric, "mean"),
                    mean_abs_gap=(metric, lambda s: s.abs().mean()),
                    n=(metric, "size"),
                )
                .sort_values("mean_abs_gap", ascending=False)
            )
            for col in ["mean_gap", "mean_abs_gap"]:
                task[col] = task[col].map(lambda x: fmt(x))
            report.add()
            report.add("Top blind-probe tasks by effect:")
            add_table(report, task, max_rows=10)

    if blind_consistency:
        df = read_csv(blind_consistency)
        if "task_consistency" in df.columns:
            report.add()
            report.add("Blind-probe task consistency:")
            cols = [c for c in ["task", "task_consistency", "same_sign_fraction", "mean_abs_gap"] if c in df.columns]
            pretty = df[cols].copy()
            for col in pretty.columns:
                if col != "task":
                    pretty[col] = pretty[col].map(lambda x: fmt(x))
            add_table(report, pretty, max_rows=12)

    if hard_effect:
        df = read_csv(hard_effect)
        report.add()
        report.add("### Hard Control Families")
        add_table(report, df)
        if {"variant", "mean_abs_blind_delta_vs_neutral"}.issubset(df.columns):
            original = df[df["variant"].astype(str).eq("original")]
            controls = df[~df["variant"].astype(str).eq("original")]
            if not original.empty and not controls.empty:
                original_strength = float(original["mean_abs_blind_delta_vs_neutral"].iloc[0])
                best_control = controls.sort_values("mean_abs_blind_delta_vs_neutral", ascending=False).iloc[0]
                best_strength = float(best_control["mean_abs_blind_delta_vs_neutral"])
                ratio = original_strength / best_strength if best_strength else float("inf")
                report.finding(
                    "Original-vs-control specificity",
                    "strong" if ratio >= 1.8 else "medium" if ratio >= 1.2 else "weak",
                    f"Original mean abs effect {fmt(original_strength)} vs best control `{best_control['variant']}` {fmt(best_strength)}; ratio {fmt(ratio)}.",
                )

    if hard_hidden:
        df = read_csv(hard_hidden)
        report.add()
        report.add("Hard-control hidden retention:")
        add_table(report, df)
        report.finding(
            "Hidden displacement is not enough",
            "note",
            "If controls retain hidden contrast but lose blind semantic effect, hidden magnitude alone is not the behavioral story.",
        )

    for title, path in [("Semantic Steering", steering), ("Rescue", rescue)]:
        if path:
            df = read_csv(path)
            report.add()
            report.add(f"### {title}")
            add_table(report, df, max_rows=12)
            report.finding(title, "present", f"Found `{path.name}`; inspect direction/gap columns for causality evidence.")

    if checklist:
        df = read_csv(checklist)
        report.add()
        report.add("### Existing Interpretation Checklist")
        add_table(report, df, max_rows=20)

    return True


def add_findings_summary(report: Report) -> None:
    report.add("# Metric Analysis Report")
    report.add()
    report.add(f"Root: `{report.root}`")
    report.add()


def add_final_readout(report: Report) -> None:
    report.add()
    report.add("## Analyzer Readout")
    if not report.findings:
        report.add("- No recognized high-level findings. The folder may not contain known result files.")
        return

    priority = {"strong": 0, "supported": 1, "medium": 2, "weak": 3, "present": 4, "note": 5, "warning": 6}
    for item in sorted(report.findings, key=lambda f: priority.get(f.status, 99)):
        report.add(f"- **{item.status}**: {item.name}. {item.detail}")

    warnings = [f for f in report.findings if f.status == "warning"]
    if warnings:
        report.add()
        report.add("Main caution:")
        for item in warnings:
            report.add(f"- {item.detail}")

    strong = [f for f in report.findings if f.status in {"strong", "supported"}]
    if strong:
        report.add()
        report.add("Short interpretation:")
        if any("axis blindness" in f.name.lower() for f in strong):
            report.add("- Small T* should not be read as absence of effect; the dominant displacement is orthogonal to the calibration axis.")
        if any("blind neutral" in f.name.lower() for f in strong):
            report.add("- The semantic readout survives neutral labels, so the effect is not just old candidate-word leakage.")
        if any("specificity" in f.name.lower() for f in strong):
            report.add("- Original texts are stronger than tested topic/style/length controls, but non-original controls still contribute part of the effect.")


def analyze_root(root: Path, recursive: bool) -> str:
    index = FileIndex(root, recursive=recursive)
    report = Report(root=root)
    add_findings_summary(report)
    found_colab = summarize_colab_trajectory(report, index)
    found_core = summarize_core_diagnostics(report, index)
    if not found_colab and not found_core:
        report.add("No known metric bundle was detected.")
        report.add()
        report.add("Expected files include:")
        report.add("- `trajectory_metric_summary.csv`, `layer_band_summary.csv`, `layer_scores.csv`")
        report.add("- `blind_neutral_probe_clean_summary.csv`, `hard_control_family_effect_summary.csv`")
        report.add("- `hidden_layer_metrics.csv`, `candidate_token_diagnostics.csv`")
    add_final_readout(report)
    return report.render()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Analyze latent-shift experiment metrics and write a compact Markdown report.")
    parser.add_argument("roots", nargs="*", help="Result folder(s) to analyze. Defaults to current folder.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under each root.")
    parser.add_argument("--out", default=None, help=f"Output Markdown file. Default: {DEFAULT_OUTPUT_NAME} in each root.")
    args = parser.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.roots] or [Path.cwd()]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            root = root.parent
        report = analyze_root(root, recursive=args.recursive)
        out_path = Path(args.out).expanduser().resolve() if args.out and len(roots) == 1 else root / DEFAULT_OUTPUT_NAME
        out_path.write_text(report, encoding="utf-8")
        print(f"saved: {out_path}")
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
