"""
Multi-model latent-regime runner for the LLM latent-shift project.

This is the active orchestration layer for the current mechanistic
interpretability hypothesis. It does not use the old strict-attractor verifier.

Current question:

    Do target texts induce measurable latent discourse-policy regime shifts
    relative to matched controls, and do those shifts line up with semantic
    readout, persistence, hard-control, order/dose, and action-policy metrics?

Recommended Colab use:

    # Main broad/core hypothesis check. Requires only llm_attractor_colab_copy_paste.py.
    !python multi_model_depinning_runner_colab.py --run-broad

    # Rebuild comparison from already finished folders.
    !python multi_model_depinning_runner_colab.py --aggregate-only

Outputs:

    multi_model_depinning_results/
      broad/<model_name>/...
      broad_behavior_summary.csv
      multi_model_depinning_summary.csv
      multi_model_latent_regime_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =========================
# EDITABLE COLAB SETTINGS
# =========================

OUT_DIR = Path("multi_model_depinning_results")
BROAD_SCRIPT = Path("llm_attractor_colab_copy_paste.py")

RUN_BROAD_BEHAVIOR = True
AGGREGATE_ONLY = False

# Broad runner profile uses the existing core diagnostics: hidden geometry,
# blind readouts, persistence/rejection, hard controls, order/dose, and
# controlled fake-agent behavior. This is the main current hypothesis check.
BROAD_RUN_PROFILE = "depinning_core"
BROAD_TEXT_FAMILY_PRESET = "original"

MODEL_SPECS = [
    {
        "name": "qwen3_14b",
        "model_id": "Qwen/Qwen3-14B",
        "max_tokens": 4096,
    },
    {
        "name": "ministral3_14b",
        "model_id": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "max_tokens": 3070,
    },
    {
        "name": "olmo2_13b",
        "model_id": "allenai/OLMo-2-1124-13B-Instruct",
        "max_tokens": 3070,
    },
]


@dataclass
class ModelSpec:
    name: str
    model_id: str
    max_tokens: int


def parse_model_specs(raw: str | None) -> list[ModelSpec]:
    if raw:
        data = json.loads(Path(raw).read_text(encoding="utf-8"))
    else:
        data = MODEL_SPECS
    specs = []
    for item in data:
        specs.append(
            ModelSpec(
                name=str(item["name"]),
                model_id=str(item["model_id"]),
                max_tokens=int(item.get("max_tokens", 4096)),
            )
        )
    return specs


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(command))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, check=True, env=merged_env)


def run_broad_behavior(spec: ModelSpec, out_dir: Path) -> None:
    model_out = out_dir / "broad" / spec.name
    env = {
        "MODEL_ID": spec.model_id,
        "MAX_TOKENS": str(spec.max_tokens),
        "RESULTS_DIR": str(model_out),
        "TEXT_FAMILY_PRESET": BROAD_TEXT_FAMILY_PRESET,
        "RUN_PROFILE": BROAD_RUN_PROFILE,
    }
    run_command([sys.executable, str(BROAD_SCRIPT)], env=env)


def find_result_file(run_dir: Path, filename: str) -> Path | None:
    candidates = [
        run_dir / "core_diagnostics_key_files" / filename,
        run_dir / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def first_float(rows: list[dict[str, str]], key: str) -> float | None:
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def summarize_broad_behavior(spec: ModelSpec, run_dir: Path) -> dict[str, Any]:
    metadata_path = find_result_file(run_dir, "run_metadata.json")
    if metadata_path is None:
        return {
            "name": spec.name,
            "model_id": spec.model_id,
            "broad_run_dir": str(run_dir),
            "broad_status": "missing",
        }

    metadata = read_json(metadata_path)
    hidden_rows = read_csv_rows(find_result_file(run_dir, "hidden_layer_metrics.csv"))
    blind_rows = read_csv_rows(find_result_file(run_dir, "blind_neutral_probe_clean_summary.csv"))
    rejection_rows = read_csv_rows(find_result_file(run_dir, "rejection_persistence_clean_summary.csv"))
    agent_rows = read_csv_rows(find_result_file(run_dir, "agent_loop_clean_summary.csv"))
    hard_rows = read_csv_rows(find_result_file(run_dir, "hard_control_family_effect_summary.csv"))
    vector_x_rows = read_csv_rows(find_result_file(run_dir, "vector_x_ordinary_transfer_summary.csv"))
    vector_x_rlhf_rows = read_csv_rows(find_result_file(run_dir, "vector_x_rlhf_proxy_transfer_summary.csv"))

    best_hidden = None
    if hidden_rows:
        best_hidden = max(
            hidden_rows,
            key=lambda row: safe_float(row.get("contrast_norm")) or float("-inf"),
        )

    rejection_turn0 = next((row for row in rejection_rows if str(row.get("post_rejection_filler_turns")) == "0"), None)
    rejection_turn6 = next((row for row in rejection_rows if str(row.get("post_rejection_filler_turns")) == "6"), None)
    agent_turn0 = next(
        (
            row for row in agent_rows
            if str(row.get("filler_turns_elapsed")) == "0"
            and str(row.get("rejection_applied")).lower() == "false"
        ),
        None,
    )
    hard_original = next((row for row in hard_rows if row.get("variant") == "original"), None)
    vector_x = vector_x_rows[0] if vector_x_rows else {}
    vector_x_rlhf = vector_x_rlhf_rows[0] if vector_x_rlhf_rows else {}

    return {
        "name": spec.name,
        "model_id": metadata.get("model_id", spec.model_id),
        "broad_run_dir": str(run_dir),
        "broad_status": "loaded",
        "run_profile": metadata.get("run_profile"),
        "text_family_preset": metadata.get("text_family_preset"),
        "best_hidden_index": best_hidden.get("hidden_index") if best_hidden else None,
        "best_hidden_contrast_norm": safe_float(best_hidden.get("contrast_norm")) if best_hidden else None,
        "best_hidden_contrast_over_mean_norm": safe_float(best_hidden.get("contrast_over_mean_norm")) if best_hidden else None,
        "blind_clean_mean_abs_gap": first_float(blind_rows, "mean_abs_clean_gap"),
        "blind_clean_fraction": first_float(blind_rows, "clean_fraction"),
        "rejection_turn0_mean_abs_gap": safe_float((rejection_turn0 or {}).get("mean_abs_gap")),
        "rejection_turn6_mean_abs_gap": safe_float((rejection_turn6 or {}).get("mean_abs_gap")),
        "agent_turn0_mean_abs_clean_action_delta": safe_float((agent_turn0 or {}).get("mean_abs_clean_action_delta")),
        "hard_original_specificity_ratio_vs_best_control": safe_float(
            (hard_original or {}).get("original_specificity_ratio_vs_best_control")
        ),
        "vector_x_best_candidate_component": vector_x.get("best_candidate_component"),
        "vector_x_candidate_minus_control_dose_slope": safe_float(
            vector_x.get("candidate_minus_control_dose_slope")
        ),
        "vector_x_candidate_minus_control_positive_fraction": safe_float(
            vector_x.get("candidate_minus_control_positive_fraction")
        ),
        "vector_x_rlhf_best_candidate_component": vector_x_rlhf.get("best_candidate_component"),
        "vector_x_rlhf_mean_abs_natural_direct_gap": safe_float(
            vector_x_rlhf.get("mean_abs_natural_direct_gap")
        ),
        "vector_x_rlhf_candidate_minus_control_dose_slope": safe_float(
            vector_x_rlhf.get("candidate_minus_control_dose_slope")
        ),
        "vector_x_rlhf_candidate_minus_control_positive_fraction": safe_float(
            vector_x_rlhf.get("candidate_minus_control_positive_fraction")
        ),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_empty_"

    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def build_report(merged_rows: list[dict[str, Any]]) -> str:
    loaded = [row for row in merged_rows if row.get("broad_status") == "loaded"]

    if loaded and len(loaded) == len(merged_rows):
        verdict = "Broad/core latent-regime metrics are present for all configured models."
    elif loaded:
        verdict = "Broad/core latent-regime metrics are present for part of the configured models."
    else:
        verdict = "Broad/core latent-regime metrics are missing; run the broad script first."

    columns = [
        "name",
        "model_id",
        "broad_status",
        "run_profile",
        "best_hidden_index",
        "best_hidden_contrast_norm",
        "best_hidden_contrast_over_mean_norm",
        "blind_clean_mean_abs_gap",
        "blind_clean_fraction",
        "rejection_turn0_mean_abs_gap",
        "rejection_turn6_mean_abs_gap",
        "agent_turn0_mean_abs_clean_action_delta",
        "hard_original_specificity_ratio_vs_best_control",
        "vector_x_best_candidate_component",
        "vector_x_candidate_minus_control_dose_slope",
        "vector_x_candidate_minus_control_positive_fraction",
        "vector_x_rlhf_best_candidate_component",
        "vector_x_rlhf_mean_abs_natural_direct_gap",
        "vector_x_rlhf_candidate_minus_control_dose_slope",
        "vector_x_rlhf_candidate_minus_control_positive_fraction",
    ]

    return "\n".join([
        "# Multi-Model Latent-Regime Report",
        "",
        "## Verdict",
        "",
        f"- {verdict}",
        "- The active object is a context-induced latent discourse-policy regime shift.",
        "- The old strict-attractor verifier is archived and is not part of this run.",
        "",
        "## Main Table",
        "",
        markdown_table(merged_rows, columns),
        "",
        "## Metric Meaning",
        "",
        "- `best_hidden_contrast_norm`: late-layer target/control representation separation.",
        "- `best_hidden_contrast_over_mean_norm`: hidden separation normalized by activation scale.",
        "- `blind_clean_mean_abs_gap`: semantic readout shift on neutral probe tasks after leakage filtering.",
        "- `rejection_turn*_mean_abs_gap`: whether the readout shift persists after rejection turns.",
        "- `agent_turn0_mean_abs_clean_action_delta`: controlled downstream action/readout shift.",
        "- `hard_original_specificity_ratio_vs_best_control`: whether the original target beats hard control variants.",
        "- `vector_x_candidate_minus_control_dose_slope`: whether the best candidate activation vector transfers to ordinary prompts better than random/control-control/wrong-layer controls.",
        "- `vector_x_candidate_minus_control_positive_fraction`: same comparison using positive-alpha movement toward the target-induced readout direction.",
        "- `vector_x_rlhf_mean_abs_natural_direct_gap`: whether TARGET_TEXTS naturally shift harmless RLHF/safety proxies before steering.",
        "- `vector_x_rlhf_candidate_minus_control_dose_slope`: whether candidate X transfers those harmless RLHF/safety proxy shifts better than negative-control vectors.",
        "",
        "## Research Claim Boundary",
        "",
        "Supported claim if the table is stable across models:",
        "",
        "```text",
        "Target contexts induce measurable latent discourse-policy regime shifts",
        "that can affect hidden geometry, semantic readout, persistence, and",
        "downstream action-policy measurements.",
        "```",
        "",
        "Do not claim:",
        "",
        "```text",
        "strict mathematical attractor, irreversible state change, consciousness,",
        "or real external-agent behavior.",
        "```",
        "",
    ])


def aggregate(specs: list[ModelSpec], out_dir: Path) -> None:
    broad_rows = [
        summarize_broad_behavior(spec, out_dir / "broad" / spec.name)
        for spec in specs
    ]
    merged_rows = broad_rows

    write_csv(out_dir / "broad_behavior_summary.csv", broad_rows)
    write_csv(out_dir / "multi_model_depinning_summary.csv", merged_rows)
    (out_dir / "multi_model_latent_regime_report.md").write_text(
        build_report(merged_rows),
        encoding="utf-8",
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model_specs": [spec.__dict__ for spec in specs],
                "broad_script": str(BROAD_SCRIPT),
                "broad_run_profile": BROAD_RUN_PROFILE,
                "broad_text_family_preset": BROAD_TEXT_FAMILY_PRESET,
                "archived_strict_attractor_script": (
                    "archive/historical_strict_attractor/"
                    "strict_llm_text_attractor_verifier_colab.py"
                ),
                "note": (
                    "The strict attractor verifier is archived and is not part "
                    "of the active hypothesis check."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved: {out_dir / 'broad_behavior_summary.csv'}")
    print(f"saved: {out_dir / 'multi_model_depinning_summary.csv'}")
    print(f"saved: {out_dir / 'multi_model_latent_regime_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/aggregate multi-model latent-regime metrics.")
    parser.add_argument("--models-json", default="", help="Optional JSON list of model specs.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--run-broad", action="store_true", default=RUN_BROAD_BEHAVIOR)
    parser.add_argument("--no-run-broad", action="store_false", dest="run_broad")
    parser.add_argument("--aggregate-only", action="store_true", default=AGGREGATE_ONLY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = parse_model_specs(args.models_json or None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_broad and not BROAD_SCRIPT.exists():
        raise FileNotFoundError(f"Missing broad script: {BROAD_SCRIPT}")

    if not args.aggregate_only:
        for spec in specs:
            if args.run_broad:
                run_broad_behavior(spec, out_dir)

    aggregate(specs, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
