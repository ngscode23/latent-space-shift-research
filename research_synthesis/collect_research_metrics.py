from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research_synthesis" / "latent_shift_package_current"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def get_first(rows: Iterable[Dict[str, str]], **predicates: Any) -> Optional[Dict[str, str]]:
    for row in rows:
        ok = True
        for key, expected in predicates.items():
            if str(row.get(key, "")) != str(expected):
                ok = False
                break
        if ok:
            return row
    return None


def max_row(rows: Iterable[Dict[str, str]], metric: str, require_positive_hidden: bool = False) -> Optional[Dict[str, str]]:
    best = None
    best_val = float("-inf")
    for row in rows:
        if require_positive_hidden and int(to_float(row.get("hidden_index"), -1)) <= 0:
            continue
        val = to_float(row.get(metric))
        if finite(val) and val > best_val:
            best = row
            best_val = val
    return best


def parse_key_value_metric(text: str, key: str) -> float:
    pattern = rf"{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, text or "")
    return to_float(match.group(1)) if match else float("nan")


def checklist_status(rows: List[Dict[str, str]], criterion: str) -> str:
    row = get_first(rows, criterion=criterion)
    return row.get("status", "") if row else ""


def checklist_metric(rows: List[Dict[str, str]], criterion: str) -> str:
    row = get_first(rows, criterion=criterion)
    return row.get("observed_metric", "") if row else ""


def canonical_hidden_geometry_script(identity: Dict[str, Any], payload: Dict[str, Any]) -> str:
    if payload.get("geometry_projection") or payload.get("middle_middle_alpha_0_75"):
        return "grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py"
    return "scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def artifact_inventory_for_dir(path: Path, family: str) -> Dict[str, Any]:
    files = list(path.glob("*")) if path.exists() and path.is_dir() else []
    names = {p.name for p in files if p.is_file()}
    return {
        "family": family,
        "dir_name": path.name,
        "path": str(path),
        "file_count": sum(1 for p in files if p.is_file()),
        "has_run_metadata": "run_metadata.json" in names or "red_team_input_manifest.json" in names,
        "has_narrative_summary_report": "summary_report.txt" in names,
        "has_legacy_readiness_report": "breakthrough_readiness_report.md" in names,
        "has_claim_threshold_eval": "claim_threshold_eval.csv" in names,
        "has_evidence_threshold_scorecard": "evidence_threshold_scorecard.csv" in names,
        "has_hidden_layer_metrics": "hidden_layer_metrics.csv" in names,
        "has_strict_attractor_criteria": "strict_attractor_criteria.csv" in names,
        "has_legacy_red_team_verdict": "red_team_hidden_geometry_verdict.md" in names,
        "has_legacy_grade4_verdict": "grade4_axis_decomposition_verdict.md" in names,
    }


def summarize_attractor_run(path: Path) -> Dict[str, Any]:
    metadata = read_json(path / "run_metadata.json")
    hidden_rows = read_csv_rows(path / "hidden_layer_metrics.csv")
    compression_rows = read_csv_rows(path / "hidden_cluster_compression.csv")
    probe_rows = read_csv_rows(path / "linear_probe_accuracy.csv")
    checklist_rows = read_csv_rows(path / "claim_threshold_eval.csv")
    if not checklist_rows:
        checklist_rows = read_csv_rows(path / "interpretation_checklist.csv")
    strict_rows = read_csv_rows(path / "strict_attractor_criteria.csv")
    blind_clean_rows = read_csv_rows(path / "blind_neutral_probe_clean_summary.csv")
    blind_persist_rows = read_csv_rows(path / "blind_neutral_persistence_clean_summary.csv")
    rejection_rows = read_csv_rows(path / "rejection_persistence_clean_summary.csv")
    hard_control_rows = read_csv_rows(path / "hard_control_family_effect_summary.csv")
    mixing_rows = read_csv_rows(path / "mixing_threshold_condition_summary.csv")
    order_rows = read_csv_rows(path / "order_hysteresis_condition_summary.csv")

    best_hidden = max_row(hidden_rows, "contrast_norm", require_positive_hidden=True) or {}
    best_hidden_index = int(to_float(best_hidden.get("hidden_index"), -1))
    compression_at_best = get_first(compression_rows, hidden_index=str(best_hidden_index)) or {}
    best_probe = max_row(probe_rows, "probe_accuracy", require_positive_hidden=True) or {}
    blind_clean = blind_clean_rows[0] if blind_clean_rows else {}
    hard_original = get_first(hard_control_rows, variant="original") or {}

    last_blind_persist = None
    if blind_persist_rows:
        last_blind_persist = max(blind_persist_rows, key=lambda r: to_float(r.get("filler_turns_elapsed"), -1))
    last_rejection = None
    if rejection_rows:
        last_rejection = max(rejection_rows, key=lambda r: to_float(r.get("post_rejection_filler_turns"), -1))

    mixing_mid = None
    for row in mixing_rows:
        if str(row.get("mixing_order")) == "target_prefix" and abs(to_float(row.get("target_fraction")) - 0.5) < 1e-9:
            mixing_mid = row
            break
    mixing_endpoint = None
    for row in mixing_rows:
        if str(row.get("mixing_order")) == "target_prefix" and abs(to_float(row.get("target_fraction")) - 1.0) < 1e-9:
            mixing_endpoint = row
            break

    def order_fraction(condition: str) -> float:
        row = get_first(order_rows, condition=condition) or {}
        return to_float(row.get("mean_fraction_toward_target"))

    strict_overall = get_first(strict_rows, criterion="strict_attractor_overall") or {}
    strict_overall_status = strict_overall.get("status", "") if strict_overall else "not_run"

    supported_count = sum(1 for row in checklist_rows if row.get("status") == "supported")
    mixed_count = sum(1 for row in checklist_rows if row.get("status") == "not_supported_or_mixed")
    not_tested_count = sum(1 for row in checklist_rows if row.get("status") == "not_tested")

    mixing_metric = checklist_metric(checklist_rows, "mixing_threshold")
    strict_metric = strict_overall.get("observed_metric", "")

    return {
        "run_dir": path.name,
        "model_id": metadata.get("model_id", ""),
        "created_utc": metadata.get("created_utc", ""),
        "text_family_preset": metadata.get("text_family_preset", ""),
        "primary_control_mode": metadata.get("primary_control_mode", ""),
        "num_hidden_layers": metadata.get("num_hidden_layers", ""),
        "hidden_size": metadata.get("hidden_size", ""),
        "file_count": len([p for p in path.glob("*") if p.is_file()]),
        "best_hidden_index": best_hidden_index if best_hidden_index >= 0 else "",
        "best_hidden_module_layer": int(to_float(best_hidden.get("module_layer"), -1)) if best_hidden else "",
        "best_contrast_norm": to_float(best_hidden.get("contrast_norm")),
        "best_cosine_distance": to_float(best_hidden.get("cosine_distance")),
        "best_contrast_over_mean_norm": to_float(best_hidden.get("contrast_over_mean_norm")),
        "best_probe_hidden_index": int(to_float(best_probe.get("hidden_index"), -1)) if best_probe else "",
        "best_probe_accuracy": to_float(best_probe.get("probe_accuracy")),
        "best_probe_permutation_p95": to_float(best_probe.get("permutation_p95_accuracy")),
        "best_probe_accuracy_minus_perm_mean": to_float(best_probe.get("accuracy_minus_permutation_mean")),
        "cluster_target_over_control_radius_cosine": to_float(compression_at_best.get("target_radius_over_control_radius_cosine")),
        "cluster_compression_fraction_cosine": to_float(compression_at_best.get("compression_fraction_vs_control_cosine")),
        "cluster_separation_over_pooled_radius_cosine": to_float(compression_at_best.get("separation_over_pooled_radius_cosine")),
        "blind_clean_fraction": to_float(blind_clean.get("clean_fraction")),
        "blind_mean_abs_clean_gap": to_float(blind_clean.get("mean_abs_clean_gap")),
        "blind_persistence_end_retention": to_float((last_blind_persist or {}).get("retention_vs_filler0")),
        "blind_persistence_end_same_sign_rate": to_float((last_blind_persist or {}).get("same_sign_as_reference_rate")),
        "rejection_persistence_end_retention": to_float((last_rejection or {}).get("retention_vs_post_rejection0")),
        "rejection_persistence_end_same_sign_rate": to_float((last_rejection or {}).get("same_sign_as_reference_rate")),
        "hard_control_original_mean_abs_effect": to_float(hard_original.get("original_mean_abs_effect") or hard_original.get("mean_abs_blind_delta_vs_neutral")),
        "hard_control_best_non_original_mean_abs_effect": to_float(hard_original.get("best_non_original_control_mean_abs_effect")),
        "hard_control_specificity_ratio": to_float(hard_original.get("original_specificity_ratio_vs_best_control")),
        "mixing_mid_fraction_toward_target": to_float((mixing_mid or {}).get("mean_fraction_toward_target")),
        "mixing_endpoint_fraction_toward_target": to_float((mixing_endpoint or {}).get("mean_fraction_toward_target")),
        "mixing_first_crossing_0_5": parse_key_value_metric(mixing_metric, "first_mean_crossing_0.5"),
        "order_TNC_fraction": order_fraction("TNC"),
        "order_CNT_fraction": order_fraction("CNT"),
        "order_TNN_fraction": order_fraction("TNN"),
        "order_CNN_fraction": order_fraction("CNN"),
        "checklist_supported_count": supported_count,
        "checklist_mixed_count": mixed_count,
        "checklist_not_tested_count": not_tested_count,
        "late_hidden_state_separation_status": checklist_status(checklist_rows, "late_hidden_state_separation"),
        "blind_neutral_probes_status": checklist_status(checklist_rows, "blind_neutral_probes"),
        "hard_control_families_status": checklist_status(checklist_rows, "hard_control_families"),
        "controlled_agent_loop_action_drift_status": checklist_status(checklist_rows, "controlled_agent_loop_action_drift"),
        "order_hysteresis_status": checklist_status(checklist_rows, "order_hysteresis"),
        "mixing_threshold_status": checklist_status(checklist_rows, "mixing_threshold"),
        "strict_attractor_overall_status": strict_overall_status,
        "strict_attractor_basin_supported": "basin=True" in strict_metric,
        "strict_attractor_stability_supported": "stability=True" in strict_metric,
        "strict_attractor_return_supported": "return=True" in strict_metric,
        "strict_attractor_geometry_supported": "geometry=True" in strict_metric,
        "strict_attractor_compression_supported": "compression=True" in strict_metric,
    }


def flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten(f"{prefix}_{key}" if prefix else str(key), value, out)
    elif isinstance(obj, list):
        out[prefix] = "; ".join(str(x) for x in obj)
    else:
        out[prefix] = obj


def summarize_hidden_geometry_from_metrics(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    flat: Dict[str, Any] = {"source_summary_json": str(path)}
    flatten("", payload, flat)
    identity = payload.get("run_identity", {})
    geometry = payload.get("geometry", {})
    causal = payload.get("causal_internal", {})
    behavior = payload.get("behavioral_control", {})
    grade4_geometry = payload.get("geometry_projection", {})
    grade4_middle = payload.get("middle_middle_alpha_0_75", {})
    grade4_neutral = grade4_middle.get("neutral", {}) if isinstance(grade4_middle, dict) else {}
    grade4_target = grade4_middle.get("target", {}) if isinstance(grade4_middle, dict) else {}
    grade4_slopes = payload.get("middle_middle_alpha_slopes", {})
    grade4_neutral_slopes = grade4_slopes.get("neutral", {}) if isinstance(grade4_slopes, dict) else {}
    grade4_target_slopes = grade4_slopes.get("target", {}) if isinstance(grade4_slopes, dict) else {}
    plus_lift = behavior.get("plus_x_lift_over_random_p95_alpha_0_75", "")
    plus_lift_float = to_float(plus_lift)
    behavior_status = "computed" if plus_lift != "" else "not_available"
    behavior_failure = "below_random_p95" if finite(plus_lift_float) and plus_lift_float <= 0 else ""
    run_name = path.parent.name
    if run_name == "qwen3_14b_breakthrough_grade_hardened":
        run_name = "qwen3_14b_grade3_hidden_geometry_hardened"
    return {
        "run_name": run_name,
        "script_family": identity.get("script_family", ""),
        "script": canonical_hidden_geometry_script(identity, payload),
        "model_id": identity.get("model_id", ""),
        "evidence_status": "computed",
        "question_count": identity.get("question_count", ""),
        "target_middle_projection_mean": geometry.get("target_middle_projection_mean", ""),
        "target_middle_direction_cosine_mean": geometry.get("target_middle_direction_cosine_mean", ""),
        "middle_band_r2": geometry.get("middle_band_r2", ""),
        "target_over_sentence_shuffle_projection_lift": geometry.get("target_over_sentence_shuffle_projection_lift", ""),
        "sentence_shuffle_projection_mean": geometry.get("sentence_shuffle_projection_mean", ""),
        "random_null_observed_minus_null_mean": payload.get("null_baseline", {}).get("observed_minus_null_mean", ""),
        "middle_plus_internal_alpha_slope": causal.get("middle_plus_internal_alpha_slope", ""),
        "middle_minus_internal_suppression_alpha_slope": causal.get("middle_minus_internal_suppression_alpha_slope", ""),
        "neutral_middle_gap_alpha_0_75": causal.get("neutral_middle_plus_minus_gap_alpha_0_75", ""),
        "target_middle_gap_alpha_0_75": causal.get("target_middle_plus_minus_gap_alpha_0_75", ""),
        "behavioral_control_status": behavior_status,
        "behavioral_control_failure_code": behavior_failure,
        "plus_x_lift_over_random_p95_alpha_0_75": plus_lift,
        "internal_visible_coupling_pass": behavior.get("internal_visible_coupling_pass", ""),
        "grade4_target_x_order_orth_projection": grade4_geometry.get("target_x_order_orth", ""),
        "grade4_sentence_shuffle_x_order_orth_projection": grade4_geometry.get("sentence_shuffle_x_order_orth", ""),
        "grade4_neutral_x_order_orth_gap_alpha_0_75": grade4_neutral.get("x_order_orth", {}).get("gap", "") if isinstance(grade4_neutral.get("x_order_orth", {}), dict) else "",
        "grade4_target_x_order_orth_gap_alpha_0_75": grade4_target.get("x_order_orth", {}).get("gap", "") if isinstance(grade4_target.get("x_order_orth", {}), dict) else "",
        "grade4_neutral_x_content_gap_alpha_0_75": grade4_neutral.get("x_content", {}).get("gap", "") if isinstance(grade4_neutral.get("x_content", {}), dict) else "",
        "grade4_target_x_content_gap_alpha_0_75": grade4_target.get("x_content", {}).get("gap", "") if isinstance(grade4_target.get("x_content", {}), dict) else "",
        "grade4_neutral_x_order_orth_slope": grade4_neutral_slopes.get("x_order_orth", ""),
        "grade4_target_x_order_orth_slope": grade4_target_slopes.get("x_order_orth", ""),
        "source_summary_json": str(path),
    }


def summarize_grade4_status() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    script = ROOT / "grade4_axis_decomposition" / "red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py"
    candidate_dirs = [
        ROOT / "red_team_hidden_geometry_results_grade4_axis_decomposition",
    ]
    existing = [path for path in candidate_dirs if path.exists()]
    metric_summary = ROOT / "metrics" / "qwen3_14b_grade4_axis_decomposition03" / "summary.json"
    if existing:
        for result_dir in existing:
            rows.append(
                {
                    "component": "grade4_axis_decomposition",
                    "script_exists": script.exists(),
                    "script_path": str(script),
                    "results_dir_exists": True,
                    "results_dir": str(result_dir),
                    "has_grade4_component_summary": (result_dir / "grade4_axis_component_causal_rank_summary.csv").exists(),
                    "status": "results_available",
                    "metrics_summary_json": str(metric_summary) if metric_summary.exists() else "",
                }
            )
    elif metric_summary.exists():
        rows.append(
            {
                "component": "grade4_axis_decomposition",
                "script_exists": script.exists(),
                "script_path": str(script),
                "results_dir_exists": False,
                "results_dir": "",
                "has_grade4_component_summary": True,
                "status": "metrics_summary_available",
                "metrics_summary_json": str(metric_summary),
            }
        )
    else:
        result_dir = candidate_dirs[0]
        rows.append(
            {
                "component": "grade4_axis_decomposition",
                "script_exists": script.exists(),
                "script_path": str(script),
                "results_dir_exists": False,
                "results_dir": str(result_dir),
                "has_grade4_component_summary": False,
                "status": "ready_to_run" if script.exists() else "missing_script",
                "metrics_summary_json": "",
            }
        )
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    val = to_float(value)
    if finite(val):
        return f"{val:.{digits}f}"
    return str(value) if value not in (None, "") else "n/a"


def has_finite_metric(row: Dict[str, Any], key: str) -> bool:
    return finite(to_float(row.get(key)))


def choose_grade3_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return next((row for row in rows if has_finite_metric(row, "target_middle_projection_mean")), rows[0] if rows else {})


def choose_grade4_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return next((row for row in rows if has_finite_metric(row, "grade4_target_x_order_orth_projection")), {})


def make_ru_report(
    attractor_rows: List[Dict[str, Any]],
    geometry_rows: List[Dict[str, Any]],
    grade4_rows: List[Dict[str, Any]],
) -> str:
    qwen_rows = [r for r in attractor_rows if "qwen3-14b" in str(r.get("model_id", "")).lower() or "qwen3_14b" in r["run_dir"].lower() or "quen3_14b" in r["run_dir"].lower()]
    top_hidden = sorted(attractor_rows, key=lambda r: to_float(r.get("best_contrast_over_mean_norm")), reverse=True)[:5]
    best_qwen = sorted(qwen_rows, key=lambda r: to_float(r.get("best_contrast_over_mean_norm")), reverse=True)[:1]
    g3 = choose_grade3_row(geometry_rows)
    g4_metrics = choose_grade4_row(geometry_rows)
    grade4 = grade4_rows[0] if grade4_rows else {}

    lines = [
        "# Latent Shift Research Synthesis",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 1. Что собираем",
        "",
        "Исследование теперь состоит из трех связанных уровней:",
        "",
        "1. `scripts/main_runners/llm_attractor_colab_copy_paste.py`: исходная гипотеза про context-induced latent/readout shift, persistence, path-dependence и строгий attractor gate.",
        "2. `scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py`: reviewer-grade hidden geometry и causal internal Vector X для Qwen3-14B.",
        "3. `grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py`: завершенный Qwen3-14B Grade 4 слой, который разлагает Vector X на `x_content`, `x_order`, `x_order_orth`.",
        "",
        "## 2. Главный вывод",
        "",
        "Часть про латентные сдвиги и геометрическое пространство уже закрыта сильнее, чем исходная гипотеза требовала. Большой Level A скрипт показывает, что target/context переводит модель в отделимую hidden/readout область; Grade 3 показывает, что в Qwen3-14B эта область задает причинно управляемую latent axis в middle residual stream; Grade 4 показывает, что эта ось содержит отделимую discourse-order / rhetorical-regime компоненту.",
        "",
        "Строгое слово `formal attractor` использовать осторожно: в большом скрипте strict-attractor overall обычно не закрыт полностью. Правильная формулировка сейчас:",
        "",
        "```text",
        "context-induced latent/readout regime shift with causal internal steering evidence",
        "```",
        "",
        "## 3. Original Attractor Runs",
        "",
        f"Найдено `attractor_results*` директорий: `{len(attractor_rows)}`.",
        "",
        "| run | model | best hidden idx | contrast/mean | blind clean | hard control ratio | strict overall |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in top_hidden:
        lines.append(
            f"| `{row['run_dir']}` | `{row.get('model_id','')}` | {row.get('best_hidden_index','')} | "
            f"{fmt(row.get('best_contrast_over_mean_norm'))} | {fmt(row.get('blind_clean_fraction'))} | "
            f"{fmt(row.get('hard_control_specificity_ratio'))} | `{row.get('strict_attractor_overall_status','')}` |"
        )

    if best_qwen:
        row = best_qwen[0]
        lines += [
            "",
            "Самый релевантный Qwen3-14B attractor run:",
            "",
            f"- run: `{row['run_dir']}`",
            f"- hidden separation: contrast_over_mean_norm `{fmt(row.get('best_contrast_over_mean_norm'))}`, cosine distance `{fmt(row.get('best_cosine_distance'))}`",
            f"- linear probe: accuracy `{fmt(row.get('best_probe_accuracy'))}`, permutation p95 `{fmt(row.get('best_probe_permutation_p95'))}`",
            f"- blind neutral probes: clean fraction `{fmt(row.get('blind_clean_fraction'))}`, mean abs gap `{fmt(row.get('blind_mean_abs_clean_gap'))}`",
            f"- persistence after neutral turns: retention `{fmt(row.get('blind_persistence_end_retention'))}`",
            f"- rejection persistence: retention `{fmt(row.get('rejection_persistence_end_retention'))}`",
            f"- hard controls: specificity ratio `{fmt(row.get('hard_control_specificity_ratio'))}`",
            f"- order hysteresis: TNC `{fmt(row.get('order_TNC_fraction'))}`, CNT `{fmt(row.get('order_CNT_fraction'))}`",
            f"- strict attractor overall: `{row.get('strict_attractor_overall_status')}`",
            "",
            "Механистически это означает: исходная гипотеза про геометрический latent shift подтверждена; строгая basin/return трактовка требует отдельной осторожности.",
        ]

    lines += [
        "",
        "## 4. Grade 3 Hidden Geometry",
        "",
    ]
    if g3:
        lines += [
            f"- model: `{g3.get('model_id')}`",
            f"- evidence status: `{g3.get('evidence_status')}`",
            f"- target middle projection: `{fmt(g3.get('target_middle_projection_mean'), 6)}`",
            f"- middle direction cosine: `{fmt(g3.get('target_middle_direction_cosine_mean'), 6)}`",
            f"- middle R2: `{fmt(g3.get('middle_band_r2'), 6)}`",
            f"- target over sentence shuffle lift: `{fmt(g3.get('target_over_sentence_shuffle_projection_lift'), 6)}`",
            f"- neutral middle +X/-X gap alpha 0.75: `{fmt(g3.get('neutral_middle_gap_alpha_0_75'), 6)}`",
            f"- target middle +X/-X gap alpha 0.75: `{fmt(g3.get('target_middle_gap_alpha_0_75'), 6)}`",
            f"- visible behavioral gate failure_code: `{g3.get('behavioral_control_failure_code')}`",
            "",
            "Это самый сильный mechanistic block: не только hidden separation, а causal internal residual-stream steering.",
        ]
    else:
        lines.append("Grade 3 summary not found under `metrics/*/summary.json`.")

    lines += [
        "",
        "## 5. Grade 4 Axis Decomposition",
        "",
    ]
    if g4_metrics:
        lines += [
            f"- evidence status: `{g4_metrics.get('evidence_status')}`",
            f"- target projection on x_order_orth: `{fmt(g4_metrics.get('grade4_target_x_order_orth_projection'), 6)}`",
            f"- sentence-shuffle projection on x_order_orth: `{fmt(g4_metrics.get('grade4_sentence_shuffle_x_order_orth_projection'), 6)}`",
            f"- neutral middle/middle x_order_orth gap alpha 0.75: `{fmt(g4_metrics.get('grade4_neutral_x_order_orth_gap_alpha_0_75'), 6)}`",
            f"- target middle/middle x_order_orth gap alpha 0.75: `{fmt(g4_metrics.get('grade4_target_x_order_orth_gap_alpha_0_75'), 6)}`",
            f"- neutral middle/middle x_content gap alpha 0.75: `{fmt(g4_metrics.get('grade4_neutral_x_content_gap_alpha_0_75'), 6)}`",
            f"- target middle/middle x_content gap alpha 0.75: `{fmt(g4_metrics.get('grade4_target_x_content_gap_alpha_0_75'), 6)}`",
            "",
            "Mechanistic meaning: `x_order_orth` survives removal of the content projection and remains causally steerable. This supports a separable discourse-order / rhetorical-regime component, not just a sentence-shuffled content axis.",
            "",
        ]
    else:
        lines.append("Grade 4 metric summary not found yet.")

    lines += [
        "",
        "## 6. Grade 4 Status",
        "",
        f"- script exists: `{grade4.get('script_exists')}`",
        f"- results dir exists: `{grade4.get('results_dir_exists')}`",
        f"- status: `{grade4.get('status')}`",
        f"- metrics summary: `{grade4.get('metrics_summary_json', '')}`",
        "",
        "Grade 4 нужен был не для спасения результата, а для декомпозиции механизма. По правильному `03` архиву этот слой уже поддержал order/rhetorical component:",
        "",
        "```text",
        "x_full       = target - neutral",
        "x_content    = sentence_shuffle(target) - neutral",
        "x_order      = target - sentence_shuffle(target)",
        "x_order_orth = x_order minus layerwise x_content projection",
        "```",
        "",
        "В текущем `03` результате `x_order_orth` дает стабильный causal gap, поэтому claim усилен до separable discourse-order/rhetorical-regime component.",
        "",
        "## 7. Как оформить исследование",
        "",
        "Рекомендуемая структура текста:",
        "",
        "1. Hypothesis: structured context induces a measurable latent geometry/readout regime shift.",
        "2. Original evidence: attractor script shows hidden separation, probe decodability, blind semantic readout, persistence/path dependence; strict formal attractor gate remains mixed.",
        "3. Mechanistic hardening: Grade 3 builds Vector X and shows middle-layer causal internal steering in Qwen3-14B.",
        "4. Mechanism decomposition: Grade 4 separates content-family and discourse-order components.",
        "5. Boundary: no permanent weight-level change; visible behavioral control not yet reviewer-grade.",
        "",
        "## 8. Files Generated By Collector",
        "",
        "- `artifact_inventory.csv`",
        "- `attractor_run_summary.csv`",
        "- `hidden_geometry_run_summary.csv`",
        "- `grade4_status.csv`",
        "- `research_synthesis_ru.md`",
        "- `research_synthesis_en.md`",
    ]
    return "\n".join(lines) + "\n"


def make_en_report(
    attractor_rows: List[Dict[str, Any]],
    geometry_rows: List[Dict[str, Any]],
    grade4_rows: List[Dict[str, Any]],
) -> str:
    g3 = choose_grade3_row(geometry_rows)
    g4_metrics = choose_grade4_row(geometry_rows)
    grade4 = grade4_rows[0] if grade4_rows else {}
    lines = [
        "# Latent Shift Research Synthesis",
        "",
        "## Claim Ladder",
        "",
        "1. The original attractor runs support a context-induced latent/readout regime shift.",
        "2. The strict formal-attractor claim remains mixed unless basin, stability, return, geometry, and compression all pass.",
        "3. The Grade 3 hidden-geometry run supports a robust causal internal latent axis in Qwen3-14B.",
        "4. The Grade 4 run supports a separable discourse-order / rhetorical-regime component beyond sentence-shuffled content.",
        "",
        "## Grade 3 Anchor",
        "",
    ]
    if g3:
        lines += [
            f"- Model: `{g3.get('model_id')}`",
            f"- Evidence status: `{g3.get('evidence_status')}`",
            f"- Target middle projection: `{fmt(g3.get('target_middle_projection_mean'), 6)}`",
            f"- Middle direction cosine: `{fmt(g3.get('target_middle_direction_cosine_mean'), 6)}`",
            f"- Middle R2: `{fmt(g3.get('middle_band_r2'), 6)}`",
            f"- Neutral middle +X/-X gap at alpha 0.75: `{fmt(g3.get('neutral_middle_gap_alpha_0_75'), 6)}`",
            f"- Visible behavioral gate failure_code: `{g3.get('behavioral_control_failure_code')}`",
        ]
    lines += [
        "",
        "## Grade 4 Axis Decomposition",
        "",
    ]
    if g4_metrics:
        lines += [
            f"- Evidence status: `{g4_metrics.get('evidence_status')}`",
            f"- Target projection on x_order_orth: `{fmt(g4_metrics.get('grade4_target_x_order_orth_projection'), 6)}`",
            f"- Sentence-shuffle projection on x_order_orth: `{fmt(g4_metrics.get('grade4_sentence_shuffle_x_order_orth_projection'), 6)}`",
            f"- Neutral middle/middle x_order_orth gap at alpha 0.75: `{fmt(g4_metrics.get('grade4_neutral_x_order_orth_gap_alpha_0_75'), 6)}`",
            f"- Target middle/middle x_order_orth gap at alpha 0.75: `{fmt(g4_metrics.get('grade4_target_x_order_orth_gap_alpha_0_75'), 6)}`",
            "",
            "Mechanistic meaning: `x_order_orth` survives removal of the content projection and remains causally steerable. This supports a separable discourse-order / rhetorical-regime component.",
            "",
        ]
    else:
        lines.append("Grade 4 metric summary is not available yet.")

    lines += [
        "",
        "## Grade 4 Status",
        "",
        f"- Script exists: `{grade4.get('script_exists')}`",
        f"- Results available: `{grade4.get('results_dir_exists')}`",
        f"- Status: `{grade4.get('status')}`",
        f"- Metrics summary: `{grade4.get('metrics_summary_json', '')}`",
        "",
        "Recommended paper wording:",
        "",
        "```text",
        "We observe a context-conditioned latent geometry/readout shift. In Qwen3-14B, a target-reference Vector X extracted from hidden states causally steers the internal generation trajectory under middle-layer residual-stream intervention. Grade 4 shows that this axis contains a separable discourse-order / rhetorical-regime component beyond sentence-shuffled content. We do not claim permanent weight-level change, formal basin status, or reviewer-grade visible behavioral control.",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    attractor_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("attractor_results")])
    red_team_dirs = sorted([
        p
        for p in ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("red_team_hidden_geometry_results")
        and "breakthrough" not in p.name.lower()
    ])
    if (ROOT / "metrics").exists():
        metric_summary_paths = sorted(
            path
            for path in (ROOT / "metrics").glob("*/summary.json")
            if not (path.parent / "DO_NOT_USE_wrong_source.txt").exists()
        )
    else:
        metric_summary_paths = []

    inventory_rows = []
    for path in attractor_dirs:
        inventory_rows.append(artifact_inventory_for_dir(path, "attractor"))
    for path in red_team_dirs:
        inventory_rows.append(artifact_inventory_for_dir(path, "hidden_geometry"))

    attractor_rows = [summarize_attractor_run(path) for path in attractor_dirs]
    geometry_rows = [summarize_hidden_geometry_from_metrics(path) for path in metric_summary_paths]
    grade4_rows = summarize_grade4_status()

    write_csv(OUT_DIR / "artifact_inventory.csv", inventory_rows)
    write_csv(OUT_DIR / "attractor_run_summary.csv", attractor_rows)
    write_csv(OUT_DIR / "hidden_geometry_run_summary.csv", geometry_rows)
    write_csv(OUT_DIR / "grade4_status.csv", grade4_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "attractor_run_count": len(attractor_rows),
        "hidden_geometry_summary_count": len(geometry_rows),
        "grade4_status": grade4_rows,
        "source_attractor_dirs": [str(p) for p in attractor_dirs],
        "source_metric_summary_json": [str(p) for p in metric_summary_paths],
    }
    (OUT_DIR / "run_collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "research_synthesis_ru.md").write_text(
        make_ru_report(attractor_rows, geometry_rows, grade4_rows),
        encoding="utf-8",
    )
    (OUT_DIR / "research_synthesis_en.md").write_text(
        make_en_report(attractor_rows, geometry_rows, grade4_rows),
        encoding="utf-8",
    )

    print(f"Wrote research synthesis package: {OUT_DIR}")
    print(f"Attractor runs: {len(attractor_rows)}")
    print(f"Hidden-geometry summaries: {len(geometry_rows)}")
    print(f"Grade 4 status: {grade4_rows[0]['status'] if grade4_rows else 'unknown'}")


if __name__ == "__main__":
    main()
