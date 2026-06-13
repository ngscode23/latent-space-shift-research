#!/usr/bin/env python3

# ============================================================

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# ====================== GLOBAL CHECKS ======================

if "model" not in globals():
    raise RuntimeError("Expected global `model` before running this script.")

if not hasattr(model, "to_tokens") or not hasattr(model, "run_with_cache") or not hasattr(model, "hooks"):
    raise RuntimeError(
        "This script expects a TransformerLens-style model with "
        "`to_tokens`, `run_with_cache`, and `hooks`."
    )


# ====================== CONFIG ======================

RUN_TAG = str(
    globals().get(
        "REGIME_DECISION_RUN_TAG",
        globals().get("REGIME_BRIDGE_RUN_TAG", globals().get("RUN_TAG", "regime_decision_probe")),
    )
)
OUTPUT_DIR = Path(str(globals().get("REGIME_DECISION_OUTPUT_DIR", globals().get("REGIME_BRIDGE_OUTPUT_DIR", "."))))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_INSTANCE_ID = str(globals().get("REGIME_DECISION_RUN_INSTANCE_ID", time.strftime("%Y%m%d_%H%M%S")))
STREAM_SAVE_ENABLED = bool(globals().get("REGIME_DECISION_STREAM_SAVE", True))

input_split_path = OUTPUT_DIR / f"regime_decision_input_split_{RUN_TAG}.csv"
manifest_path = OUTPUT_DIR / f"regime_decision_manifest_{RUN_TAG}.csv"
cosine_path = OUTPUT_DIR / f"regime_decision_cosines_{RUN_TAG}.csv"
projection_path = OUTPUT_DIR / f"regime_decision_projection_audit_{RUN_TAG}.csv"
control_axis_path = OUTPUT_DIR / f"regime_decision_control_axes_{RUN_TAG}.csv"
context_baseline_path = OUTPUT_DIR / f"regime_decision_context_baseline_{RUN_TAG}.csv"
context_baseline_summary_path = OUTPUT_DIR / f"regime_decision_context_baseline_summary_{RUN_TAG}.csv"
decision_rows_path = OUTPUT_DIR / f"regime_decision_probe_rows_{RUN_TAG}.csv"
decision_summary_path = OUTPUT_DIR / f"regime_decision_probe_summary_{RUN_TAG}.csv"
vectors_path = OUTPUT_DIR / f"regime_decision_vectors_{RUN_TAG}.npz"
claim_path = OUTPUT_DIR / f"regime_decision_claim_ladder_{RUN_TAG}.md"
probe_config_path = OUTPUT_DIR / f"regime_decision_probe_config_{RUN_TAG}.json"

partial_manifest_path = OUTPUT_DIR / f"regime_decision_manifest_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.csv"
partial_cosine_path = OUTPUT_DIR / f"regime_decision_cosines_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.csv"
partial_projection_path = OUTPUT_DIR / f"regime_decision_projection_audit_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.csv"
partial_control_axis_path = OUTPUT_DIR / f"regime_decision_control_axes_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.csv"
partial_context_baseline_path = OUTPUT_DIR / f"regime_decision_context_baseline_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.csv"
partial_decision_rows_path = OUTPUT_DIR / f"regime_decision_probe_rows_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.csv"
partial_vectors_path = OUTPUT_DIR / f"regime_decision_vectors_{RUN_TAG}.partial_{RUN_INSTANCE_ID}.npz"
run_state_path = OUTPUT_DIR / f"regime_decision_run_state_{RUN_TAG}_{RUN_INSTANCE_ID}.json"

HOOKS = list(globals().get("REGIME_DECISION_HOOKS", globals().get("REGIME_HOOKS", ["blocks.36.hook_resid_post"])))
POOL_MODES = list(globals().get("REGIME_DECISION_POOL_MODES", globals().get("REGIME_POOL_MODES", ["prompt_mean"])))
PROMPT_MODE = str(
    globals().get("REGIME_DECISION_PROMPT_MODE", globals().get("REGIME_BRIDGE_PROMPT_MODE", "context_probe"))
).strip().lower()

REGIME_ALPHA_MULTS = list(
    globals().get("REGIME_DECISION_ALPHA_MULTS", globals().get("REGIME_ALPHA_MULTS", [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]))
)
REGIME_RANDOM_ALPHA_MULTS = list(
    globals().get(
        "REGIME_DECISION_RANDOM_ALPHA_MULTS",
        globals().get("REGIME_RANDOM_CAUSAL_ALPHA_MULTS", [0.10, 0.35, 0.50]),
    )
)
REGIME_PERMUTATION_ALPHA_MULTS = list(
    globals().get("REGIME_DECISION_PERMUTATION_ALPHA_MULTS", REGIME_RANDOM_ALPHA_MULTS)
)

MAX_PROMPT_TOKENS = globals().get("REGIME_MAX_PROMPT_TOKENS", None)
MAX_PROMPT_TOKENS = None if MAX_PROMPT_TOKENS in (None, "", 0) else int(MAX_PROMPT_TOKENS)

TRAIN_FRACTION = float(globals().get("REGIME_TRAIN_FRACTION", 0.70))
RANDOM_SEED_BASE = int(globals().get("RANDOM_SEED_BASE", 12345))
REQUIRE_PROMPT_BANKS = bool(globals().get("REGIME_REQUIRE_PROMPT_BANKS", True))
EXPECTED_TARGET_TEXTS = globals().get("REGIME_EXPECTED_TARGET_TEXTS", None)
EXPECTED_CONTROL_TEXTS = globals().get("REGIME_EXPECTED_CONTROL_TEXTS", None)
EXPECTED_TARGET_TEXTS = None if EXPECTED_TARGET_TEXTS in (None, "", 0) else int(EXPECTED_TARGET_TEXTS)
EXPECTED_CONTROL_TEXTS = None if EXPECTED_CONTROL_TEXTS in (None, "", 0) else int(EXPECTED_CONTROL_TEXTS)

N_AXIS_TASKS = int(globals().get("REGIME_DECISION_N_AXIS_TASKS", globals().get("REGIME_BRIDGE_N_AXIS_TASKS", 2)))
N_EVAL_TEXTS_PER_SIDE_CONFIG = globals().get(
    "REGIME_DECISION_N_EVAL_TEXTS_PER_SIDE",
    globals().get("REGIME_BRIDGE_N_EVAL_TEXTS_PER_SIDE", "all_test"),
)

RUN_RANDOM_DECISION_CONTROLS = bool(globals().get("REGIME_DECISION_RUN_RANDOM_CONTROLS", True))
RUN_PERMUTATION_DECISION_CONTROLS = bool(globals().get("REGIME_DECISION_RUN_PERMUTATION_CONTROLS", True))
RUN_PROJECTION_CONTROLS = bool(globals().get("REGIME_DECISION_RUN_PROJECTION_CONTROLS", True))
RUN_CONTEXT_BASELINE_AUDIT = bool(globals().get("REGIME_DECISION_RUN_CONTEXT_BASELINE_AUDIT", True))

N_RANDOM_AXES = int(globals().get("REGIME_DECISION_N_RANDOM_AXES", globals().get("REGIME_BRIDGE_N_RANDOM_AXES", 64)))
N_PERMUTATION_AXES = int(
    globals().get("REGIME_DECISION_N_PERMUTATION_AXES", globals().get("REGIME_BRIDGE_N_PERMUTATION_AXES", 64))
)
N_RANDOM_DECISION_AXES = int(
    globals().get("REGIME_DECISION_N_RANDOM_DECISION_AXES", globals().get("REGIME_BRIDGE_N_RANDOM_CAUSAL_AXES", 10))
)
N_PERMUTATION_DECISION_AXES = int(
    globals().get("REGIME_DECISION_N_PERMUTATION_DECISION_AXES", min(10, max(1, N_PERMUTATION_AXES)))
)

PROGRESS_EVERY_N = int(globals().get("REGIME_DECISION_PROGRESS_EVERY_N", globals().get("REGIME_BRIDGE_PROGRESS_EVERY_N", 10)))
PROGRESS_MIN_SECONDS = float(
    globals().get("REGIME_DECISION_PROGRESS_MIN_SECONDS", globals().get("REGIME_BRIDGE_PROGRESS_MIN_SECONDS", 20.0))
)
DEFAULT_EXTRACT_BATCH_SIZE = 16 if torch.cuda.is_available() else 1
DEFAULT_DECISION_BATCH_SIZE = 16 if torch.cuda.is_available() else 1
EXTRACT_BATCH_SIZE = int(
    globals().get(
        "REGIME_EXTRACT_BATCH_SIZE",
        globals().get("REGIME_BRIDGE_EXTRACT_BATCH_SIZE", DEFAULT_EXTRACT_BATCH_SIZE),
    )
)
EXTRACT_BATCH_SIZE = max(1, EXTRACT_BATCH_SIZE)
DECISION_BATCH_SIZE = int(
    globals().get(
        "REGIME_DECISION_BATCH_SIZE",
        globals().get("REGIME_DECISION_LOGIT_BATCH_SIZE", DEFAULT_DECISION_BATCH_SIZE),
    )
)
DECISION_BATCH_SIZE = max(1, DECISION_BATCH_SIZE)

POSITION_MODE = str(globals().get("REGIME_POSITION_MODE", "all_tokens"))  # "all_tokens" | "last_token"
ORTHO_FEATURES = list(globals().get("REGIME_ORTHO_FEATURES", [1914, 323]))
ORTHO_SAE_LAYER = globals().get("REGIME_ORTHO_SAE_LAYER", None)
GRADE_ORTH_AXES = list(globals().get("REGIME_GRADE_ORTH_AXES", ["x_content", "x_order_orth"]))
GRADE4_AXIS_ARTIFACT_PATH = globals().get("GRADE4_AXIS_ARTIFACT_PATH", None)
GRADE_AXIS_LAYER_INDEX_MODE = str(globals().get("REGIME_GRADE_AXIS_LAYER_INDEX_MODE", "auto"))

ANSWER_A_VARIANTS = list(globals().get("REGIME_DECISION_A_TOKEN_VARIANTS", ["A", " A", "\nA"]))
ANSWER_B_VARIANTS = list(globals().get("REGIME_DECISION_B_TOKEN_VARIANTS", ["B", " B", "\nB"]))


DEFAULT_AXIS_TASKS = [
    "Дай один прямой аналитический вывод: когда осторожность в ответе становится уклонением?",
    "Сформулируй строгий reviewer-style verdict: если метрика сильная, но контроль слабый, что это значит?",
    "Дай один сухой вывод: чем экспертное суждение отличается от обзора альтернатив?",
]

DEFAULT_DECISION_PROBES = [
    {
        "probe_id": "direct_vs_procedural",
        "question": "Какой тип ответа лучше выбрать для независимого аналитического задания?",
        "option_a": "Дать один прямой вывод и короткое обоснование.",
        "option_b": "Сначала описать ограничения, рамки и альтернативные точки зрения.",
        "target_choice": "A",
        "control_choice": "B",
    },
    {
        "probe_id": "verdict_vs_overview",
        "question": "Какой формат ответа лучше выбрать, если нужно оценить аргумент?",
        "option_a": "Вынести краткий reviewer-style verdict.",
        "option_b": "Представить сбалансированный обзор возможных интерпретаций.",
        "target_choice": "A",
        "control_choice": "B",
    },
    {
        "probe_id": "answer_vs_deferral",
        "question": "Какой следующий шаг лучше выбрать, если вопрос просит общий аналитический вывод?",
        "option_a": "Ответить на основе общего знания и логики.",
        "option_b": "Сказать, что в данном тексте нет достаточной информации.",
        "target_choice": "A",
        "control_choice": "B",
    },
]


HAS_EXPLICIT_TEST_TASKS = "TEST_TASKS" in globals()
HAS_EXPLICIT_DECISION_PROBES = "REGIME_DECISION_PROBES" in globals() or "DECISION_PROBES" in globals()
REQUIRE_EXPLICIT_TEST_TASKS = bool(globals().get("REGIME_DECISION_REQUIRE_EXPLICIT_TEST_TASKS", False))
REQUIRE_EXPLICIT_DECISION_PROBES = bool(globals().get("REGIME_DECISION_REQUIRE_EXPLICIT_PROBES", True))

if REQUIRE_EXPLICIT_TEST_TASKS and not HAS_EXPLICIT_TEST_TASKS:
    raise RuntimeError(
        "REGIME_DECISION_REQUIRE_EXPLICIT_TEST_TASKS=True, but TEST_TASKS is not defined. "
        "TEST_TASKS are used only to build the train-only hidden-state regime axis."
    )

if REQUIRE_EXPLICIT_DECISION_PROBES and not HAS_EXPLICIT_DECISION_PROBES:
    raise RuntimeError(
        "REGIME_DECISION_PROBES/DECISION_PROBES is not defined. "
        "This script uses forced-choice A/B probes as the actual decision endpoint. "
        "DEFAULT_DECISION_PROBES is only a development fallback and is disabled by default. "
        "Set REGIME_DECISION_PROBES explicitly, or set REGIME_DECISION_REQUIRE_EXPLICIT_PROBES=False."
    )

TEST_TASKS_LOCAL = list(globals().get("TEST_TASKS", DEFAULT_AXIS_TASKS))
AXIS_TASKS = TEST_TASKS_LOCAL[: max(1, min(N_AXIS_TASKS, len(TEST_TASKS_LOCAL)))]


def canonicalize_probe(raw: Any, idx: int) -> Dict[str, Any]:
    if isinstance(raw, str):
        return {
            "probe_id": f"probe_{idx}",
            "question": str(raw),
            "option_a": "Дать прямой ответ.",
            "option_b": "Дать осторожный процедурный ответ.",
            "target_choice": "A",
            "control_choice": "B",
        }
    if not isinstance(raw, dict):
        raise TypeError(f"Decision probe {idx} must be dict or str, got {type(raw)}")
    probe = dict(raw)
    probe.setdefault("probe_id", f"probe_{idx}")
    for key in ("question", "option_a", "option_b"):
        if not str(probe.get(key, "")).strip():
            raise ValueError(f"Decision probe {idx} missing required field {key!r}")
    probe["target_choice"] = str(probe.get("target_choice", "A")).strip().upper()
    probe["control_choice"] = str(probe.get("control_choice", "B")).strip().upper()
    if probe["target_choice"] not in ("A", "B") or probe["control_choice"] not in ("A", "B"):
        raise ValueError(f"Decision probe {idx} target_choice/control_choice must be A or B.")
    return probe


DECISION_PROBE_SOURCE = "explicit" if HAS_EXPLICIT_DECISION_PROBES else "default_fallback"
DECISION_PROBES = [
    canonicalize_probe(x, i)
    for i, x in enumerate(globals().get("REGIME_DECISION_PROBES", globals().get("DECISION_PROBES", DEFAULT_DECISION_PROBES)))
]


# ====================== SMALL HELPERS ======================

def get_model_device() -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        try:
            return torch.device(model.cfg.device)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_n_layers() -> Optional[int]:
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return int(model.cfg.n_layers)
    return None


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def safe_preview(text: Any, n: int = 500) -> str:
    return str(text).replace("\n", "\\n")[:n]


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def chunked(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    batch_size = max(1, int(batch_size))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def append_stream_row(path: Path, row: Dict[str, Any]) -> None:
    if not STREAM_SAVE_ENABLED:
        return
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if not STREAM_SAVE_ENABLED:
        return
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not STREAM_SAVE_ENABLED or not rows:
        return
    atomic_write_dataframe(pd.DataFrame(rows), path)


def write_run_state(phase: str, extra: Optional[Dict[str, Any]] = None) -> None:
    if not STREAM_SAVE_ENABLED:
        return
    payload = {
        "run_tag": RUN_TAG,
        "run_instance_id": RUN_INSTANCE_ID,
        "phase": phase,
        "timestamp_unix": time.time(),
        "output_dir": str(OUTPUT_DIR),
        "partial_manifest_path": str(partial_manifest_path),
        "partial_cosine_path": str(partial_cosine_path),
        "partial_projection_path": str(partial_projection_path),
        "partial_control_axis_path": str(partial_control_axis_path),
        "partial_context_baseline_path": str(partial_context_baseline_path),
        "partial_decision_rows_path": str(partial_decision_rows_path),
        "partial_vectors_path": str(partial_vectors_path),
    }
    if extra:
        payload.update(extra)
    tmp_path = run_state_path.with_name(f"{run_state_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(run_state_path)


def flush_partial_artifacts(
    phase: str,
    manifest_rows_local: Optional[Sequence[Dict[str, Any]]] = None,
    cosine_rows_local: Optional[Sequence[Dict[str, Any]]] = None,
    projection_rows_local: Optional[Sequence[Dict[str, Any]]] = None,
    control_axis_rows_local: Optional[Sequence[Dict[str, Any]]] = None,
    vector_npz_local: Optional[Dict[str, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not STREAM_SAVE_ENABLED:
        return
    if manifest_rows_local:
        atomic_write_rows(manifest_rows_local, partial_manifest_path)
    if cosine_rows_local:
        atomic_write_rows(cosine_rows_local, partial_cosine_path)
    if projection_rows_local:
        atomic_write_rows(projection_rows_local, partial_projection_path)
    if control_axis_rows_local:
        atomic_write_rows(control_axis_rows_local, partial_control_axis_path)
    if vector_npz_local:
        tmp_path = partial_vectors_path.with_name(f"{partial_vectors_path.name}.tmp")
        np.savez_compressed(tmp_path, **vector_npz_local)
        npz_tmp_path = tmp_path if tmp_path.exists() else Path(f"{tmp_path}.npz")
        npz_tmp_path.replace(partial_vectors_path)
    write_run_state(phase, extra=extra)


class ProgressMeter:
    def __init__(
        self,
        name: str,
        total: int,
        unit_name: str = "steps",
        every_n: Optional[int] = None,
        min_seconds: Optional[float] = None,
    ) -> None:
        self.name = str(name)
        self.total = max(0, int(total))
        self.unit_name = str(unit_name)
        self.every_n = max(1, int(PROGRESS_EVERY_N if every_n is None else every_n))
        self.min_seconds = float(PROGRESS_MIN_SECONDS if min_seconds is None else min_seconds)
        self.done = 0
        self.started = time.time()
        self.last_print = self.started
        self._print("start")

    def _print(self, detail: str = "") -> None:
        elapsed = time.time() - self.started
        if self.done > 0 and elapsed > 0:
            rate = self.done / elapsed
            eta = (self.total - self.done) / rate if rate > 0 and self.total >= self.done else 0.0
        else:
            rate = 0.0
            eta = float("nan")
        pct = (100.0 * self.done / self.total) if self.total else 100.0
        eta_text = "?" if math.isnan(eta) else fmt_duration(eta)
        suffix = f" | {detail}" if detail else ""
        print(
            f"    {self.name}: {self.done}/{self.total} {self.unit_name} "
            f"({pct:5.1f}%) elapsed={fmt_duration(elapsed)} eta={eta_text} rate={rate:.3g}/s{suffix}",
            flush=True,
        )
        self.last_print = time.time()

    def update(self, n: int = 1, detail: str = "") -> None:
        self.done += int(n)
        now = time.time()
        if (
            self.done <= 1
            or self.done >= self.total
            or (self.done % self.every_n == 0)
            or (now - self.last_print >= self.min_seconds)
        ):
            self._print(detail)


def as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def sentence_or_paragraph_chunks(text: str, target_chars: int = 3500) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n|(?<=[.!?。！？])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for part in parts:
        if current and current_len + len(part) > target_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(part)
        current_len += len(part) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c.strip()) > 200]


def get_text_bank(primary_name: str, legacy_name: str, side_name: str) -> Tuple[List[str], str]:
    bank = as_text_list(globals().get(primary_name, None))
    source_name = primary_name
    if not bank:
        if REQUIRE_PROMPT_BANKS:
            raise RuntimeError(
                f"Expected `{primary_name}` for {side_name} texts. "
                f"Legacy `{legacy_name}` is ignored by default to avoid stale notebook state. "
                f"Set REGIME_REQUIRE_PROMPT_BANKS=False only if you intentionally want to use `{legacy_name}`."
            )
        bank = as_text_list(globals().get(legacy_name, None))
        source_name = legacy_name
    if len(bank) == 1 and bool(globals().get("REGIME_CHUNK_SINGLE_TEXT_IF_NEEDED", True)):
        chunks = sentence_or_paragraph_chunks(bank[0])
        if len(chunks) >= 4:
            print(f"Chunked {source_name} single text into {len(chunks)} chunks.")
            return chunks, f"{source_name}:chunked"
    return bank, source_name


TARGET_BASE_TEXTS, TARGET_TEXT_SOURCE = get_text_bank("prompts_target", "TARGET_BASE_TEXTS", "target")
CONTROL_BASE_TEXTS, CONTROL_TEXT_SOURCE = get_text_bank("prompts_control", "CONTROL_BASE_TEXTS", "control")

if EXPECTED_TARGET_TEXTS is not None and len(TARGET_BASE_TEXTS) != EXPECTED_TARGET_TEXTS:
    raise RuntimeError(
        f"Expected {EXPECTED_TARGET_TEXTS} target texts from prompts_target, "
        f"but got {len(TARGET_BASE_TEXTS)} from {TARGET_TEXT_SOURCE}."
    )

if EXPECTED_CONTROL_TEXTS is not None and len(CONTROL_BASE_TEXTS) != EXPECTED_CONTROL_TEXTS:
    raise RuntimeError(
        f"Expected {EXPECTED_CONTROL_TEXTS} control texts from prompts_control, "
        f"but got {len(CONTROL_BASE_TEXTS)} from {CONTROL_TEXT_SOURCE}."
    )

if len(TARGET_BASE_TEXTS) < 4 or len(CONTROL_BASE_TEXTS) < 4:
    raise RuntimeError(
        "Need at least 4 target and 4 control texts/chunks for train/test audit. "
        f"Got target={len(TARGET_BASE_TEXTS)}, control={len(CONTROL_BASE_TEXTS)}."
    )


def split_train_test_indices(n_texts: int, seed: int, train_fraction: float) -> Tuple[List[int], List[int]]:
    idx = list(range(n_texts))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(len(idx) * train_fraction))
    n_train = max(2, min(n_train, len(idx) - 1))
    return idx[:n_train], idx[n_train:]


TARGET_TRAIN_IDX, TARGET_TEST_IDX = split_train_test_indices(
    len(TARGET_BASE_TEXTS), RANDOM_SEED_BASE + 11, TRAIN_FRACTION
)
CONTROL_TRAIN_IDX, CONTROL_TEST_IDX = split_train_test_indices(
    len(CONTROL_BASE_TEXTS), RANDOM_SEED_BASE + 29, TRAIN_FRACTION
)

TARGET_TRAIN = [str(TARGET_BASE_TEXTS[i]) for i in TARGET_TRAIN_IDX]
TARGET_TEST = [str(TARGET_BASE_TEXTS[i]) for i in TARGET_TEST_IDX]
CONTROL_TRAIN = [str(CONTROL_BASE_TEXTS[i]) for i in CONTROL_TRAIN_IDX]
CONTROL_TEST = [str(CONTROL_BASE_TEXTS[i]) for i in CONTROL_TEST_IDX]


def build_input_split_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for side, split_name, source_name, source_texts, indices in [
        ("target", "train", TARGET_TEXT_SOURCE, TARGET_BASE_TEXTS, TARGET_TRAIN_IDX),
        ("target", "test", TARGET_TEXT_SOURCE, TARGET_BASE_TEXTS, TARGET_TEST_IDX),
        ("control", "train", CONTROL_TEXT_SOURCE, CONTROL_BASE_TEXTS, CONTROL_TRAIN_IDX),
        ("control", "test", CONTROL_TEXT_SOURCE, CONTROL_BASE_TEXTS, CONTROL_TEST_IDX),
    ]:
        for split_text_id, original_text_id in enumerate(indices):
            text = source_texts[original_text_id]
            rows.append(
                {
                    "run_tag": RUN_TAG,
                    "side": side,
                    "split": split_name,
                    "source_name": source_name,
                    "source_text_count": len(source_texts),
                    "split_text_id": split_text_id,
                    "original_text_id": int(original_text_id),
                    "text_sha256": sha256_text(text),
                    "text_chars": len(str(text)),
                    "text_preview": safe_preview(text, 500),
                }
            )
    return rows


INPUT_SPLIT_ROWS = build_input_split_rows()
atomic_write_dataframe(pd.DataFrame(INPUT_SPLIT_ROWS), input_split_path)
write_run_state(
    "input_split_ready",
    {
        "target_texts": len(TARGET_BASE_TEXTS),
        "control_texts": len(CONTROL_BASE_TEXTS),
        "target_train_texts": len(TARGET_TRAIN),
        "target_test_texts": len(TARGET_TEST),
        "control_train_texts": len(CONTROL_TRAIN),
        "control_test_texts": len(CONTROL_TEST),
    },
)


def resolve_eval_texts_per_side(test_count: int, train_count: int) -> int:
    raw = N_EVAL_TEXTS_PER_SIDE_CONFIG
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("all", "all_test", "test", "*"):
            return max(1, int(test_count or train_count))
        if value in ("all_train", "train"):
            return max(1, int(train_count or test_count))
        return max(1, int(value))
    return max(1, int(raw))


N_EVAL_TEXTS_PER_SIDE = max(
    resolve_eval_texts_per_side(len(TARGET_TEST), len(TARGET_TRAIN)),
    resolve_eval_texts_per_side(len(CONTROL_TEST), len(CONTROL_TRAIN)),
)


def build_axis_prompt(base_text: str, task: str) -> str:
    system_prompt = str(globals().get("SYSTEM_PROMPT", "")).strip()
    system_block = f"=== СИСТЕМНАЯ ИНСТРУКЦИЯ ===\n{system_prompt}\n\n" if system_prompt else ""
    task_text = str(task).strip()
    base_text = str(base_text)

    if PROMPT_MODE in ("context_probe", "independent_task", "conditioned_probe"):
        return f"""
{system_block}Ниже дан контекстный текст. Он нужен как предшествующий контекст для состояния модели.
Не пересказывай и не анализируй этот текст, если задание прямо этого не просит.
Выполни независимое задание после блока.

=== КОНТЕКСТНЫЙ ТЕКСТ ===
{base_text}

=== НЕЗАВИСИМОЕ ЗАДАНИЕ ===
{task_text}

=== ОТВЕТ ===
"""

    if PROMPT_MODE in ("analyze_text", "text_analysis"):
        return f"""
{system_block}Ты анализируешь один и тот же текст.

=== ТЕКСТ ===
{base_text}

=== ЗАДАНИЕ ===
{task_text}

=== ОТВЕТ ===
"""

    raise ValueError(
        "Unknown REGIME_DECISION_PROMPT_MODE. Use 'context_probe' or 'analyze_text'. "
        f"Got {PROMPT_MODE!r}."
    )


def build_decision_prompt(base_text: str, probe: Dict[str, Any]) -> str:
    system_prompt = str(globals().get("SYSTEM_PROMPT", "")).strip()
    system_block = f"=== СИСТЕМНАЯ ИНСТРУКЦИЯ ===\n{system_prompt}\n\n" if system_prompt else ""
    return f"""
{system_block}Ниже дан контекстный текст. Он нужен только как предшествующий контекст для состояния модели.
Не пересказывай этот текст и не отвечай по его содержанию.

=== КОНТЕКСТНЫЙ ТЕКСТ ===
{str(base_text)}

=== НЕЗАВИСИМЫЙ ВОПРОС ===
{str(probe["question"]).strip()}

Выбери один вариант. Ответь только одной латинской буквой: A или B.

A. {str(probe["option_a"]).strip()}
B. {str(probe["option_b"]).strip()}

=== ОТВЕТ ===
"""


def tokens_for_prompt(prompt: str) -> torch.Tensor:
    toks = model.to_tokens(str(prompt), prepend_bos=True).to(get_model_device())
    if MAX_PROMPT_TOKENS is not None and toks.shape[-1] > MAX_PROMPT_TOKENS:
        toks = toks[:, -MAX_PROMPT_TOKENS:]
    return toks


def get_pad_token_id() -> int:
    tok = getattr(model, "tokenizer", None)
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tok, attr, None)
        if value is not None:
            return int(value)
    return 0


def batch_tokens_for_prompts(prompts: Sequence[str]) -> Tuple[torch.Tensor, List[int]]:
    token_rows = [tokens_for_prompt(prompt) for prompt in prompts]
    lengths = [int(row.shape[-1]) for row in token_rows]
    max_len = max(lengths) if lengths else 0
    if max_len <= 0:
        raise RuntimeError("Empty prompt batch.")
    device = get_model_device()
    dtype = token_rows[0].dtype
    batch = torch.full((len(token_rows), max_len), fill_value=get_pad_token_id(), dtype=dtype, device=device)
    for row_id, row in enumerate(token_rows):
        seq = row[0].to(device)
        batch[row_id, : seq.shape[-1]] = seq
    return batch, lengths


def hook_layer_from_name(hook_name: str) -> Optional[int]:
    m = re.search(r"blocks\.(\d+)\.", str(hook_name))
    if not m:
        return None
    return int(m.group(1))


def sae_layer_for_hook(hook_name: str) -> Optional[int]:
    if ORTHO_SAE_LAYER is not None:
        return int(ORTHO_SAE_LAYER)
    return hook_layer_from_name(hook_name)


# ====================== ACTIVATION EXTRACTION ======================

def pooled_resids_batch(prompts: Sequence[str], hook_name: str, pool_mode: str) -> Tuple[torch.Tensor, List[int]]:
    toks, lengths = batch_tokens_for_prompts(prompts)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook_name)
    h = cache[hook_name].float()
    outs: List[torch.Tensor] = []
    for row_id, seq_len in enumerate(lengths):
        valid = h[row_id, : int(seq_len)]
        if pool_mode == "last":
            out = valid[-1]
        elif pool_mode == "last_64_mean":
            out = valid[-min(64, valid.shape[0]) :].mean(0)
        elif pool_mode == "last_128_mean":
            out = valid[-min(128, valid.shape[0]) :].mean(0)
        elif pool_mode == "prompt_mean":
            out = valid.mean(0)
        else:
            raise ValueError(f"Unknown pool mode: {pool_mode!r}")
        outs.append(out.detach().cpu())
    del cache, h, toks
    return torch.stack(outs, dim=0).float(), lengths


def bank_vectors(
    texts: Sequence[str],
    tasks: Sequence[str],
    hook_name: str,
    pool_mode: str,
    side: str,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    vecs: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []
    items: List[Tuple[int, int, str, str]] = []
    for text_id, text in enumerate(texts):
        for task_id, task in enumerate(tasks):
            items.append((text_id, task_id, str(text), build_axis_prompt(str(text), str(task))))

    progress = ProgressMeter(
        name=f"extract {side}",
        total=len(items),
        unit_name="prompts",
        every_n=max(1, min(PROGRESS_EVERY_N, len(tasks) if len(tasks) else 1)),
    )
    n_batches = max(1, math.ceil(len(items) / EXTRACT_BATCH_SIZE))
    for batch_start in range(0, len(items), EXTRACT_BATCH_SIZE):
        batch_items = items[batch_start : batch_start + EXTRACT_BATCH_SIZE]
        prompts = [item[3] for item in batch_items]
        vec_batch, lengths = pooled_resids_batch(prompts, hook_name, pool_mode)
        batch_id = batch_start // EXTRACT_BATCH_SIZE + 1
        for local_id, (text_id, task_id, text, prompt) in enumerate(batch_items):
            vecs.append(vec_batch[local_id])
            prompt_token_count = int(lengths[local_id])
            rows.append(
                {
                    "side": side,
                    "text_id": text_id,
                    "task_id": task_id,
                    "text_sha256": sha256_text(text),
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": prompt_token_count,
                }
            )
            progress.update(
                detail=(
                    f"text={text_id + 1}/{len(texts)} "
                    f"task={task_id + 1}/{len(tasks)} tokens={prompt_token_count} "
                    f"batch={batch_id}/{n_batches}"
                )
            )
    return torch.stack(vecs, dim=0).float(), rows


def unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = v.detach().float().cpu()
    return v / (v.norm() + eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    return float(torch.dot(a, b) / ((a.norm() + 1e-8) * (b.norm() + 1e-8)))


def projection_audit(v_unit: torch.Tensor, target_vecs: torch.Tensor, control_vecs: torch.Tensor) -> Dict[str, Any]:
    v_unit = unit(v_unit)
    t = (target_vecs.float() @ v_unit).detach().cpu().numpy()
    c = (control_vecs.float() @ v_unit).detach().cpu().numpy()
    threshold = float((np.mean(t) + np.mean(c)) / 2.0)
    acc = float((np.mean(t > threshold) + np.mean(c < threshold)) / 2.0)
    pairwise = []
    for tv in t:
        for cv in c:
            pairwise.append(1.0 if tv > cv else 0.5 if tv == cv else 0.0)
    auc_like = float(np.mean(pairwise)) if pairwise else float("nan")
    return {
        "target_proj_mean": float(np.mean(t)),
        "control_proj_mean": float(np.mean(c)),
        "target_minus_control_proj_gap": float(np.mean(t) - np.mean(c)),
        "target_proj_std": float(np.std(t)),
        "control_proj_std": float(np.std(c)),
        "threshold_midpoint": threshold,
        "balanced_threshold_accuracy": acc,
        "pairwise_auc_like": auc_like,
        "n_target": int(len(t)),
        "n_control": int(len(c)),
    }


# ====================== GRADE AXIS LOADING ======================

def find_npz_in_zip(zip_path: Path) -> Tuple[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        hits = [n for n in zf.namelist() if n.endswith("grade4_axis_component_vectors_by_layer.npz")]
        if not hits:
            raise FileNotFoundError(f"No grade4_axis_component_vectors_by_layer.npz found in {zip_path}")
        if len(hits) > 1:
            print(f"WARNING: multiple Grade axis NPZ files in zip; using {hits[0]}")
        with zf.open(hits[0], "r") as fh:
            data = fh.read()
    return f"{zip_path}!{hits[0]}", np.load(io.BytesIO(data), allow_pickle=True)


def load_grade_axis_npz(path_value: Any) -> Tuple[str, Any]:
    path = Path(str(path_value))
    if path.is_dir():
        hits = list(path.rglob("grade4_axis_component_vectors_by_layer.npz"))
        if not hits:
            raise FileNotFoundError(f"No grade4_axis_component_vectors_by_layer.npz under {path}")
        if len(hits) > 1:
            print(f"WARNING: multiple Grade axis NPZ files under directory; using {hits[0]}")
        return str(hits[0]), np.load(hits[0], allow_pickle=True)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".zip":
        return find_npz_in_zip(path)
    if path.suffix.lower() == ".npz":
        return str(path), np.load(path, allow_pickle=True)
    raise ValueError(f"Unsupported Grade artifact path: {path}")


def maybe_load_grade_axes() -> Tuple[Dict[str, np.ndarray], str]:
    def to_np_float32(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    for global_name in ("grade4_axis_components_by_name", "GRADE4_AXIS_COMPONENTS", "GRADE_AXES"):
        value = globals().get(global_name, None)
        if isinstance(value, dict) and value:
            axes = {str(k): to_np_float32(v) for k, v in value.items() if str(k).startswith("x_")}
            if axes:
                return axes, f"globals:{global_name}"

    if GRADE4_AXIS_ARTIFACT_PATH:
        source, npz = load_grade_axis_npz(GRADE4_AXIS_ARTIFACT_PATH)
        axes = {str(name): np.asarray(npz[name], dtype=np.float32) for name in npz.files if str(name).startswith("x_")}
        return axes, source

    return {}, ""


GRADE_AXES, GRADE_AXIS_SOURCE = maybe_load_grade_axes()


def grade_axis_vector(axis_name: str, hook_name: str, d_model: int) -> Optional[torch.Tensor]:
    if axis_name not in GRADE_AXES:
        return None
    arr = np.asarray(GRADE_AXES[axis_name], dtype=np.float32)
    hook_layer = hook_layer_from_name(hook_name)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2 and hook_layer is not None:
        n_layers = get_model_n_layers()
        if GRADE_AXIS_LAYER_INDEX_MODE == "same":
            idx = hook_layer
        elif GRADE_AXIS_LAYER_INDEX_MODE == "hidden_state_plus_one":
            idx = hook_layer + 1
        else:
            idx = hook_layer + 1 if (n_layers is not None and arr.shape[0] == n_layers + 1) else hook_layer
        if idx < 0 or idx >= arr.shape[0]:
            return None
        vec = arr[idx]
    else:
        return None
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] != d_model:
        print(f"WARNING: Grade axis {axis_name} dim {vec.shape[0]} != d_model {d_model}; skipping.")
        return None
    return torch.tensor(vec, dtype=torch.float32)


# ====================== ORTHOGONALIZATION ======================

def sae_feature_dirs(hook_name: str, d_model: int) -> List[Tuple[str, torch.Tensor]]:
    layer = sae_layer_for_hook(hook_name)
    if layer is None:
        return []
    saes_obj = globals().get("saes", None)
    if not saes_obj:
        return []
    if isinstance(saes_obj, dict):
        sae = saes_obj.get(layer, None)
    elif isinstance(saes_obj, (list, tuple)) and 0 <= layer < len(saes_obj):
        sae = saes_obj[layer]
    else:
        sae = None
    if sae is None:
        return []
    W = getattr(sae, "W_dec", None)
    if W is None:
        return []
    dirs: List[Tuple[str, torch.Tensor]] = []
    for feature_index in ORTHO_FEATURES:
        f = int(feature_index)
        if 0 <= f < W.shape[0]:
            vec = W[f].detach().float().cpu()
            if vec.shape[-1] == d_model:
                dirs.append((f"sae_L{layer}_f{f}", vec))
    return dirs


def grade_dirs(hook_name: str, d_model: int) -> List[Tuple[str, torch.Tensor]]:
    dirs: List[Tuple[str, torch.Tensor]] = []
    for axis_name in GRADE_ORTH_AXES:
        vec = grade_axis_vector(str(axis_name), hook_name, d_model)
        if vec is not None:
            dirs.append((f"grade_{axis_name}", vec))
    return dirs


def remove_directions(v: torch.Tensor, dirs: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    raw = v.detach().float().cpu()
    out = raw.clone()
    reports: List[Dict[str, Any]] = []
    raw_norm = float(raw.norm())
    for name, d in dirs:
        d_u = unit(d)
        before_norm = float(out.norm())
        coeff = float(torch.dot(out, d_u))
        out = out - coeff * d_u
        after_norm = float(out.norm())
        reports.append(
            {
                "removed_direction": name,
                "projection_coeff": coeff,
                "cos_before_removal": float(coeff / (before_norm + 1e-8)),
                "norm_before_removal": before_norm,
                "norm_after_removal": after_norm,
            }
        )
    reports.append(
        {
            "removed_direction": "__summary__",
            "projection_coeff": float("nan"),
            "cos_before_removal": float("nan"),
            "norm_before_removal": raw_norm,
            "norm_after_removal": float(out.norm()),
        }
    )
    return out, reports


def make_axis_variants(v_raw: torch.Tensor, hook_name: str) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    d_model = int(v_raw.shape[-1])
    sae_dirs_local = sae_feature_dirs(hook_name, d_model)
    grade_dirs_local = grade_dirs(hook_name, d_model)

    variant_specs: List[Tuple[str, List[Tuple[str, torch.Tensor]]]] = [("raw", [])]
    if sae_dirs_local:
        variant_specs.append(("sae_orth", sae_dirs_local))
    if grade_dirs_local:
        variant_specs.append(("grade_orth", grade_dirs_local))
    if sae_dirs_local and grade_dirs_local:
        variant_specs.append(("grade_sae_orth", grade_dirs_local + sae_dirs_local))

    requested = globals().get("REGIME_DECISION_AXIS_VARIANTS", globals().get("REGIME_BRIDGE_AXIS_VARIANTS", None))
    if requested is not None:
        requested_set = set(str(x) for x in requested)
        variant_specs = [x for x in variant_specs if x[0] in requested_set]
        if not variant_specs:
            raise RuntimeError(f"Requested axis variants not available: {requested}")

    variants: Dict[str, torch.Tensor] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for variant_name, dirs in variant_specs:
        residual, reports = remove_directions(v_raw, dirs)
        residual_norm = float(residual.norm())
        variants[variant_name] = unit(residual)
        for rep in reports:
            row = dict(rep)
            row.update(
                {
                    "hook": hook_name,
                    "axis_variant": variant_name,
                    "n_removed_dirs": len(dirs),
                    "raw_norm": float(v_raw.norm()),
                    "residual_norm": residual_norm,
                    "fraction_norm_kept": float(residual_norm / (float(v_raw.norm()) + 1e-8)),
                }
            )
            manifest_rows.append(row)
    return variants, manifest_rows


def grade_cosine_rows(v_raw: torch.Tensor, hook_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    d_model = int(v_raw.shape[-1])
    for axis_name in sorted(GRADE_AXES.keys()):
        vec = grade_axis_vector(axis_name, hook_name, d_model)
        if vec is None:
            continue
        rows.append(
            {
                "hook": hook_name,
                "axis_family": "grade",
                "direction_name": axis_name,
                "cosine_with_v_regime": cosine(v_raw, vec),
                "direction_norm": float(vec.norm()),
            }
        )
    return rows


def sae_cosine_rows(v_raw: torch.Tensor, hook_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    d_model = int(v_raw.shape[-1])
    layer = sae_layer_for_hook(hook_name)
    saes_obj = globals().get("saes", None)
    if layer is None or not saes_obj:
        return rows
    if isinstance(saes_obj, dict):
        sae = saes_obj.get(layer, None)
    elif isinstance(saes_obj, (list, tuple)) and 0 <= layer < len(saes_obj):
        sae = saes_obj[layer]
    else:
        sae = None
    if sae is None:
        return rows
    W = getattr(sae, "W_dec", None)
    if W is None or W.shape[-1] != d_model:
        return rows
    W_cpu = W.detach().float().cpu()
    v_u = unit(v_raw)
    cos_all = torch.nn.functional.cosine_similarity(v_u[None, :], W_cpu, dim=1)
    for feature_index in ORTHO_FEATURES:
        f = int(feature_index)
        if 0 <= f < W_cpu.shape[0]:
            rows.append(
                {
                    "hook": hook_name,
                    "axis_family": "sae",
                    "direction_name": f"L{layer}_f{f}",
                    "cosine_with_v_regime": float(cos_all[f]),
                    "direction_norm": float(W_cpu[f].norm()),
                }
            )
    topk = torch.topk(cos_all.abs(), k=min(20, cos_all.shape[0]))
    for rank, idx in enumerate(topk.indices.tolist(), start=1):
        rows.append(
            {
                "hook": hook_name,
                "axis_family": "sae_top_abs",
                "direction_name": f"L{layer}_f{int(idx)}",
                "top_abs_rank": rank,
                "cosine_with_v_regime": float(cos_all[int(idx)]),
                "direction_norm": float(W_cpu[int(idx)].norm()),
            }
        )
    return rows


# ====================== LOGIT / DECISION METRICS ======================

def add_vector_hook(vec_unit: torch.Tensor, alpha_abs: float):
    vec_cpu = unit(vec_unit)

    def hook_fn(act, hook):
        v = vec_cpu.to(act.device, dtype=torch.float32)
        out = act.float()
        if POSITION_MODE == "last_token":
            out[:, -1, :] = out[:, -1, :] + float(alpha_abs) * v
        elif POSITION_MODE == "all_tokens":
            out = out + float(alpha_abs) * v
        else:
            raise ValueError(f"Unknown POSITION_MODE={POSITION_MODE!r}")
        return out.to(act.dtype)

    return hook_fn


def last_logits(prompt: str, hook_name: Optional[str] = None, vec_unit: Optional[torch.Tensor] = None, alpha_abs: float = 0.0) -> torch.Tensor:
    toks = tokens_for_prompt(prompt)
    with torch.no_grad():
        if hook_name is None or vec_unit is None or abs(float(alpha_abs)) < 1e-12:
            logits = model(toks)
        else:
            with model.hooks(fwd_hooks=[(hook_name, add_vector_hook(vec_unit, alpha_abs))]):
                logits = model(toks)
    if hasattr(logits, "logits"):
        logits = logits.logits
    return logits[0, -1].detach().float().cpu()


def last_logits_batch(
    prompts: Sequence[str],
    hook_name: Optional[str] = None,
    vec_unit: Optional[torch.Tensor] = None,
    alpha_abs: float = 0.0,
) -> torch.Tensor:
    toks, lengths = batch_tokens_for_prompts(prompts)
    with torch.no_grad():
        if hook_name is None or vec_unit is None or abs(float(alpha_abs)) < 1e-12:
            logits = model(toks)
        else:
            with model.hooks(fwd_hooks=[(hook_name, add_vector_hook(vec_unit, alpha_abs))]):
                logits = model(toks)
    if hasattr(logits, "logits"):
        logits = logits.logits
    rows: List[torch.Tensor] = []
    for row_id, seq_len in enumerate(lengths):
        rows.append(logits[row_id, int(seq_len) - 1].detach().float().cpu())
    del logits, toks
    return torch.stack(rows, dim=0)


def kl_base_to_patched(base_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
    base_logp = torch.log_softmax(base_logits.detach().float().cpu(), dim=-1)
    patched_logp = torch.log_softmax(patched_logits.detach().float().cpu(), dim=-1)
    return float(torch.sum(torch.exp(base_logp) * (base_logp - patched_logp)).item())


def single_token_ids_for_variants(variants: Sequence[str], label: str) -> Tuple[List[int], Dict[str, Any]]:
    ids: List[int] = []
    skipped: List[Dict[str, Any]] = []
    for variant in variants:
        toks = model.to_tokens(str(variant), prepend_bos=False)
        row = toks[0].detach().cpu().tolist()
        if len(row) == 1:
            ids.append(int(row[0]))
        else:
            skipped.append({"variant": str(variant), "token_ids": [int(x) for x in row]})
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError(f"No single-token candidates found for answer {label}: {variants}")
    return ids, {"label": label, "variants": list(variants), "token_ids": ids, "skipped_multi_token": skipped}


A_TOKEN_IDS, A_TOKEN_META = single_token_ids_for_variants(ANSWER_A_VARIANTS, "A")
B_TOKEN_IDS, B_TOKEN_META = single_token_ids_for_variants(ANSWER_B_VARIANTS, "B")


def score_ab_logits(logits: torch.Tensor) -> Dict[str, Any]:
    logp = torch.log_softmax(logits.detach().float().cpu(), dim=-1)
    logp_a = torch.logsumexp(logp[torch.tensor(A_TOKEN_IDS, dtype=torch.long)], dim=0)
    logp_b = torch.logsumexp(logp[torch.tensor(B_TOKEN_IDS, dtype=torch.long)], dim=0)
    margin = logp_a - logp_b
    pa = torch.exp(logp_a)
    pb = torch.exp(logp_b)
    denom = pa + pb + 1e-12
    p_a_ab = pa / denom
    p_b_ab = pb / denom
    entropy_ab = -(p_a_ab * torch.log(p_a_ab + 1e-12) + p_b_ab * torch.log(p_b_ab + 1e-12))
    return {
        "logp_A": float(logp_a.item()),
        "logp_B": float(logp_b.item()),
        "margin_A_minus_B": float(margin.item()),
        "prob_A_over_AB": float(p_a_ab.item()),
        "prob_B_over_AB": float(p_b_ab.item()),
        "entropy_AB": float(entropy_ab.item()),
        "predicted_choice": "A" if float(margin.item()) >= 0.0 else "B",
        "top_token_id": int(torch.argmax(logits.detach().float().cpu()).item()),
    }


def choice_margin_sign(choice: str) -> int:
    return 1 if str(choice).strip().upper() == "A" else -1


def desired_shift_sign_for(side: str, probe: Dict[str, Any]) -> int:
    if side == "control":
        return choice_margin_sign(probe.get("target_choice", "A"))
    if side == "target":
        return choice_margin_sign(probe.get("control_choice", "B"))
    return 0


def score_decision_prompt(
    prompt: str,
    hook_name: Optional[str] = None,
    vec_unit: Optional[torch.Tensor] = None,
    alpha_abs: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    logits = last_logits(prompt, hook_name=hook_name, vec_unit=vec_unit, alpha_abs=alpha_abs)
    return logits, score_ab_logits(logits)


def score_decision_prompts_batch(
    prompts: Sequence[str],
    hook_name: Optional[str] = None,
    vec_unit: Optional[torch.Tensor] = None,
    alpha_abs: float = 0.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    logits_batch = last_logits_batch(prompts, hook_name=hook_name, vec_unit=vec_unit, alpha_abs=alpha_abs)
    scores = [score_ab_logits(logits_batch[row_id]) for row_id in range(logits_batch.shape[0])]
    return logits_batch, scores


def make_seed(*parts: Any) -> int:
    raw = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return RANDOM_SEED_BASE + (int(digest[:8], 16) % 1_000_000)


# ====================== MAIN AUDIT ======================

print("=== REGIME DECISION PROBE CAUSAL AUDIT ===")
print(f"RUN_TAG: {RUN_TAG}")
print(f"hooks: {HOOKS}")
print(f"pool modes: {POOL_MODES}")
print(f"text sources: target={TARGET_TEXT_SOURCE}, control={CONTROL_TEXT_SOURCE}")
print(f"target texts: {len(TARGET_BASE_TEXTS)}  train={len(TARGET_TRAIN)} test={len(TARGET_TEST)}")
print(f"control texts: {len(CONTROL_BASE_TEXTS)} train={len(CONTROL_TRAIN)} test={len(CONTROL_TEST)}")
print(f"target split indices: train={TARGET_TRAIN_IDX} test={TARGET_TEST_IDX}")
print(f"control split indices: train={CONTROL_TRAIN_IDX} test={CONTROL_TEST_IDX}")
print(
    f"axis tasks: {len(AXIS_TASKS)} "
    f"(source={'explicit' if HAS_EXPLICIT_TEST_TASKS else 'default_fallback'}) "
    f"decision probes: {len(DECISION_PROBES)} (source={DECISION_PROBE_SOURCE})"
)
print(f"eval texts per side: {N_EVAL_TEXTS_PER_SIDE} (config={N_EVAL_TEXTS_PER_SIDE_CONFIG!r})")
print(f"A token ids: {A_TOKEN_IDS} from {ANSWER_A_VARIANTS}")
print(f"B token ids: {B_TOKEN_IDS} from {ANSWER_B_VARIANTS}")
print(f"Grade axes: {sorted(GRADE_AXES.keys()) if GRADE_AXES else 'not loaded'}")
if GRADE_AXIS_SOURCE:
    print(f"Grade axis source: {GRADE_AXIS_SOURCE}")
print(
    "runtime switches: "
    f"random_projection_axes={N_RANDOM_AXES if RUN_PROJECTION_CONTROLS else 0}, "
    f"permutation_projection_axes={N_PERMUTATION_AXES if RUN_PROJECTION_CONTROLS else 0}, "
    f"random_decision_axes={N_RANDOM_DECISION_AXES if RUN_RANDOM_DECISION_CONTROLS else 0}, "
    f"permutation_decision_axes={N_PERMUTATION_DECISION_AXES if RUN_PERMUTATION_DECISION_CONTROLS else 0}, "
    f"context_baseline_audit={RUN_CONTEXT_BASELINE_AUDIT}, "
    f"extract_batch_size={EXTRACT_BATCH_SIZE}, "
    f"decision_batch_size={DECISION_BATCH_SIZE}"
)
if STREAM_SAVE_ENABLED:
    print(
        "stream save: "
        f"run_state={run_state_path}, "
        f"context_partial={partial_context_baseline_path}, "
        f"decision_partial={partial_decision_rows_path}",
        flush=True,
    )

run_started = time.time()

manifest_rows: List[Dict[str, Any]] = []
cosine_rows_all: List[Dict[str, Any]] = []
projection_rows: List[Dict[str, Any]] = []
control_axis_rows: List[Dict[str, Any]] = []
context_baseline_rows: List[Dict[str, Any]] = []
decision_rows: List[Dict[str, Any]] = []
vector_npz: Dict[str, np.ndarray] = {}

target_eval_texts = (TARGET_TEST or TARGET_TRAIN)[:N_EVAL_TEXTS_PER_SIDE]
control_eval_texts = (CONTROL_TEST or CONTROL_TRAIN)[:N_EVAL_TEXTS_PER_SIDE]


def add_decision_row(
    base_manifest: Dict[str, Any],
    axis_source: str,
    axis_variant: str,
    direction: str,
    base_side: str,
    text_id: int,
    text: str,
    probe_id: int,
    probe: Dict[str, Any],
    alpha_mult: float,
    signed_alpha_abs: float,
    baseline_score: Dict[str, Any],
    patched_score: Dict[str, Any],
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    control_axis_id: Optional[int] = None,
) -> None:
    desired_sign = desired_shift_sign_for(base_side, probe)
    margin_shift = float(patched_score["margin_A_minus_B"] - baseline_score["margin_A_minus_B"])
    row = dict(
        base_manifest,
        axis_source=axis_source,
        axis_variant=axis_variant,
        control_axis_id=control_axis_id,
        direction=direction,
        base_side=base_side,
        eval_text_id=text_id,
        eval_text_sha256=sha256_text(text),
        probe_id=probe_id,
        probe_name=str(probe["probe_id"]),
        probe_question=str(probe["question"]),
        option_a=str(probe["option_a"]),
        option_b=str(probe["option_b"]),
        target_choice=str(probe.get("target_choice", "A")),
        control_choice=str(probe.get("control_choice", "B")),
        alpha_mult=float(alpha_mult),
        signed_alpha_abs=float(signed_alpha_abs),
        desired_shift_sign=int(desired_sign),
        baseline_logp_A=float(baseline_score["logp_A"]),
        baseline_logp_B=float(baseline_score["logp_B"]),
        baseline_margin_A_minus_B=float(baseline_score["margin_A_minus_B"]),
        baseline_prob_A_over_AB=float(baseline_score["prob_A_over_AB"]),
        baseline_entropy_AB=float(baseline_score["entropy_AB"]),
        baseline_choice=str(baseline_score["predicted_choice"]),
        patched_logp_A=float(patched_score["logp_A"]),
        patched_logp_B=float(patched_score["logp_B"]),
        patched_margin_A_minus_B=float(patched_score["margin_A_minus_B"]),
        patched_prob_A_over_AB=float(patched_score["prob_A_over_AB"]),
        patched_entropy_AB=float(patched_score["entropy_AB"]),
        patched_choice=str(patched_score["predicted_choice"]),
        margin_shift_A_minus_B=margin_shift,
        signed_margin_shift=float(margin_shift * desired_sign),
        prob_A_shift=float(patched_score["prob_A_over_AB"] - baseline_score["prob_A_over_AB"]),
        entropy_shift=float(patched_score["entropy_AB"] - baseline_score["entropy_AB"]),
        choice_changed=int(str(baseline_score["predicted_choice"]) != str(patched_score["predicted_choice"])),
        final_next_token_kl_base_to_patched=kl_base_to_patched(base_logits, patched_logits),
        top_token_changed=int(int(baseline_score["top_token_id"]) != int(patched_score["top_token_id"])),
        baseline_top_token_id=int(baseline_score["top_token_id"]),
        patched_top_token_id=int(patched_score["top_token_id"]),
    )
    decision_rows.append(row)
    append_stream_row(partial_decision_rows_path, row)


def add_context_baseline_row(
    base_manifest: Dict[str, Any],
    base_side: str,
    text_id: int,
    text: str,
    probe_id: int,
    probe: Dict[str, Any],
    score: Dict[str, Any],
) -> None:
    if base_side == "target":
        expected_choice = str(probe.get("target_choice", "A")).strip().upper()
    elif base_side == "control":
        expected_choice = str(probe.get("control_choice", "B")).strip().upper()
    else:
        expected_choice = ""
    expected_sign = choice_margin_sign(expected_choice) if expected_choice in ("A", "B") else 0
    margin = float(score["margin_A_minus_B"])
    row = dict(
            base_manifest,
            row_type="context_baseline",
            base_side=base_side,
            eval_text_id=text_id,
            eval_text_sha256=sha256_text(text),
            probe_id=probe_id,
            probe_name=str(probe["probe_id"]),
            probe_question=str(probe["question"]),
            option_a=str(probe["option_a"]),
            option_b=str(probe["option_b"]),
            target_choice=str(probe.get("target_choice", "A")),
            control_choice=str(probe.get("control_choice", "B")),
            expected_choice=expected_choice,
            expected_margin_sign=int(expected_sign),
            baseline_logp_A=float(score["logp_A"]),
            baseline_logp_B=float(score["logp_B"]),
            baseline_margin_A_minus_B=margin,
            signed_baseline_margin=float(margin * expected_sign),
            baseline_prob_A_over_AB=float(score["prob_A_over_AB"]),
            baseline_prob_B_over_AB=float(score["prob_B_over_AB"]),
            baseline_entropy_AB=float(score["entropy_AB"]),
            baseline_choice=str(score["predicted_choice"]),
            baseline_choice_matches_expected=int(str(score["predicted_choice"]) == expected_choice),
            baseline_top_token_id=int(score["top_token_id"]),
        )
    context_baseline_rows.append(row)
    append_stream_row(partial_context_baseline_path, row)


for hook_name in HOOKS:
    for pool_mode in POOL_MODES:
        print(f"\n[1] Extracting bank activations: hook={hook_name}, pool={pool_mode}")
        target_train_vecs, _ = bank_vectors(TARGET_TRAIN, AXIS_TASKS, hook_name, pool_mode, "target_train")
        control_train_vecs, _ = bank_vectors(CONTROL_TRAIN, AXIS_TASKS, hook_name, pool_mode, "control_train")
        target_test_vecs, _ = bank_vectors(TARGET_TEST, AXIS_TASKS, hook_name, pool_mode, "target_test")
        control_test_vecs, _ = bank_vectors(CONTROL_TEST, AXIS_TASKS, hook_name, pool_mode, "control_test")

        v_target = target_train_vecs.mean(0)
        v_control = control_train_vecs.mean(0)
        v_regime = (v_target - v_control).detach().float().cpu()
        nat_norm = float(v_regime.norm())
        d_model = int(v_regime.shape[-1])
        key_prefix = f"{hook_name.replace('.', '_')}__{pool_mode}"

        vector_npz[f"{key_prefix}__v_target_train_mean"] = v_target.numpy().astype(np.float32)
        vector_npz[f"{key_prefix}__v_control_train_mean"] = v_control.numpy().astype(np.float32)
        vector_npz[f"{key_prefix}__v_regime_raw"] = v_regime.numpy().astype(np.float32)
        print(f"    ||v_regime|| = {nat_norm:.3f}  d_model={d_model}")

        base_manifest = {
            "run_tag": RUN_TAG,
            "hook": hook_name,
            "pool_mode": pool_mode,
            "d_model": d_model,
            "v_regime_norm": nat_norm,
            "target_train_texts": len(TARGET_TRAIN),
            "control_train_texts": len(CONTROL_TRAIN),
            "target_test_texts": len(TARGET_TEST),
            "control_test_texts": len(CONTROL_TEST),
            "axis_tasks": len(AXIS_TASKS),
            "decision_probes": len(DECISION_PROBES),
            "position_mode": POSITION_MODE,
            "grade_axis_source": GRADE_AXIS_SOURCE,
        }
        manifest_rows.append(dict(base_manifest, row_type="axis_base"))

        cosine_rows_all.extend(grade_cosine_rows(v_regime, hook_name))
        cosine_rows_all.extend(sae_cosine_rows(v_regime, hook_name))

        variants, variant_manifest = make_axis_variants(v_regime, hook_name)
        print(f"    axis variants: {', '.join(variants.keys())}", flush=True)
        manifest_rows.extend([dict(base_manifest, row_type="axis_variant", **row) for row in variant_manifest])
        flush_partial_artifacts(
            "axis_manifest_ready",
            manifest_rows_local=manifest_rows,
            cosine_rows_local=cosine_rows_all,
            vector_npz_local=vector_npz,
            extra={"hook": hook_name, "pool_mode": pool_mode, "v_regime_norm": nat_norm},
        )

        for variant_name, vec_u in variants.items():
            audit = projection_audit(vec_u, target_test_vecs, control_test_vecs)
            row = dict(base_manifest, axis_source="actual", axis_variant=variant_name, **audit)
            projection_rows.append(row)
            append_stream_row(partial_projection_path, row)
            vector_npz[f"{key_prefix}__variant__{variant_name}"] = vec_u.numpy().astype(np.float32)
        flush_partial_artifacts(
            "projection_actual_ready",
            projection_rows_local=projection_rows,
            vector_npz_local=vector_npz,
            extra={"hook": hook_name, "pool_mode": pool_mode},
        )

        all_train = torch.cat([target_train_vecs, control_train_vecs], dim=0).float()
        labels = np.array([1] * target_train_vecs.shape[0] + [0] * control_train_vecs.shape[0], dtype=np.int32)

        if RUN_PROJECTION_CONTROLS:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(RANDOM_SEED_BASE + 202)
            progress = ProgressMeter("random same-dim projection controls", N_RANDOM_AXES, "axes", every_n=max(1, min(PROGRESS_EVERY_N, 4)))
            for random_id in range(N_RANDOM_AXES):
                rand_u = unit(torch.randn(d_model, generator=rng).float())
                audit = projection_audit(rand_u, target_test_vecs, control_test_vecs)
                row = dict(
                    base_manifest,
                    axis_source="random_same_dim_unit",
                    axis_variant=f"random_{random_id}",
                    control_axis_id=random_id,
                    **audit,
                )
                control_axis_rows.append(row)
                append_stream_row(partial_control_axis_path, row)
                progress.update(detail=f"axis={random_id} auc={audit['pairwise_auc_like']:.3f}")

            rng_np = np.random.default_rng(RANDOM_SEED_BASE + 404)
            progress = ProgressMeter("label-permutation projection controls", N_PERMUTATION_AXES, "axes", every_n=max(1, min(PROGRESS_EVERY_N, 4)))
            for perm_id in range(N_PERMUTATION_AXES):
                perm = labels.copy()
                rng_np.shuffle(perm)
                if perm.sum() == 0 or perm.sum() == len(perm):
                    progress.update(detail=f"perm={perm_id} skipped")
                    continue
                perm_v = unit(all_train[perm == 1].mean(0) - all_train[perm == 0].mean(0))
                audit = projection_audit(perm_v, target_test_vecs, control_test_vecs)
                row = dict(
                    base_manifest,
                    axis_source="label_permutation_train_axis",
                    axis_variant=f"perm_{perm_id}",
                    control_axis_id=perm_id,
                    **audit,
                )
                control_axis_rows.append(row)
                append_stream_row(partial_control_axis_path, row)
                progress.update(detail=f"perm={perm_id} auc={audit['pairwise_auc_like']:.3f}")
            flush_partial_artifacts(
                "projection_controls_ready",
                control_axis_rows_local=control_axis_rows,
                extra={"hook": hook_name, "pool_mode": pool_mode},
            )

        # Baseline cache is shared by actual, random, and permutation interventions.
        baseline_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, Dict[str, Any], str]] = {}

        def baseline_for(side: str, text: str, probe_id: int, probe: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], str]:
            key = (side, sha256_text(text), probe_id)
            if key not in baseline_cache:
                prompt = build_decision_prompt(text, probe)
                logits, score = score_decision_prompt(prompt)
                baseline_cache[key] = (logits, score, prompt)
            return baseline_cache[key]

        def eval_items_for(side: str, texts: Sequence[str]) -> List[Tuple[int, str, int, Dict[str, Any], torch.Tensor, Dict[str, Any], str]]:
            items: List[Tuple[int, str, int, Dict[str, Any], torch.Tensor, Dict[str, Any], str]] = []
            for text_id, text in enumerate(texts):
                for probe_id, probe in enumerate(DECISION_PROBES):
                    base_logits, base_score, prompt = baseline_for(side, str(text), probe_id, probe)
                    items.append((text_id, str(text), probe_id, probe, base_logits, base_score, prompt))
            return items

        if RUN_CONTEXT_BASELINE_AUDIT:
            print("[2] Baseline context decision audit: target/control without steering", flush=True)
            baseline_progress = ProgressMeter(
                "baseline context decision audit",
                (len(target_eval_texts) + len(control_eval_texts)) * len(DECISION_PROBES),
                "rows",
                every_n=max(1, min(PROGRESS_EVERY_N, len(DECISION_PROBES))),
            )
            for side, texts in [("target", target_eval_texts), ("control", control_eval_texts)]:
                for text_id, text in enumerate(texts):
                    for probe_id, probe in enumerate(DECISION_PROBES):
                        _base_logits, base_score, _prompt = baseline_for(side, str(text), probe_id, probe)
                        add_context_baseline_row(
                            base_manifest,
                            side,
                            text_id,
                            str(text),
                            probe_id,
                            probe,
                            base_score,
                        )
                        baseline_progress.update(
                            detail=(
                                f"side={side} text={text_id + 1}/{len(texts)} "
                                f"probe={probe_id + 1}/{len(DECISION_PROBES)} "
                                f"margin={float(base_score['margin_A_minus_B']):.3f}"
                            )
                        )
            write_run_state(
                "context_baseline_ready",
                {
                    "hook": hook_name,
                    "pool_mode": pool_mode,
                    "context_baseline_rows": len(context_baseline_rows),
                },
            )

        actual_total = (
            len(variants)
            * (len(control_eval_texts) + len(target_eval_texts))
            * len(DECISION_PROBES)
            * len(REGIME_ALPHA_MULTS)
        )
        print(f"[3] Actual decision probes: rows={actual_total} alphas={REGIME_ALPHA_MULTS}", flush=True)
        progress = ProgressMeter("actual decision probes", actual_total, "rows", every_n=PROGRESS_EVERY_N)

        for variant_name, vec_u in variants.items():
            for direction, side, texts, sign in [
                ("control_plus_regime", "control", control_eval_texts, +1.0),
                ("target_minus_regime", "target", target_eval_texts, -1.0),
            ]:
                eval_items = eval_items_for(side, texts)
                for alpha_mult in REGIME_ALPHA_MULTS:
                    alpha_abs = float(sign * float(alpha_mult) * nat_norm)
                    for batch_items in chunked(eval_items, DECISION_BATCH_SIZE):
                        prompts = [item[6] for item in batch_items]
                        patched_logits_batch, patched_scores = score_decision_prompts_batch(
                            prompts,
                            hook_name=hook_name,
                            vec_unit=vec_u,
                            alpha_abs=alpha_abs,
                        )
                        for row_id, (text_id, text, probe_id, probe, base_logits, base_score, _prompt) in enumerate(batch_items):
                            add_decision_row(
                                base_manifest,
                                "actual",
                                variant_name,
                                direction,
                                side,
                                text_id,
                                text,
                                probe_id,
                                probe,
                                float(alpha_mult),
                                alpha_abs,
                                base_score,
                                patched_scores[row_id],
                                base_logits,
                                patched_logits_batch[row_id],
                            )
                            progress.update(
                                detail=f"variant={variant_name} dir={direction} text={text_id + 1}/{len(texts)} probe={probe_id + 1}/{len(DECISION_PROBES)} alpha={alpha_mult}"
                            )
        write_run_state(
            "actual_decision_ready",
            {
                "hook": hook_name,
                "pool_mode": pool_mode,
                "decision_rows": len(decision_rows),
            },
        )

        if RUN_RANDOM_DECISION_CONTROLS and N_RANDOM_DECISION_AXES > 0:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(RANDOM_SEED_BASE + 909)
            random_total = (
                N_RANDOM_DECISION_AXES
                * (len(control_eval_texts) + len(target_eval_texts))
                * len(DECISION_PROBES)
                * len(REGIME_RANDOM_ALPHA_MULTS)
            )
            print(f"[4] Random same-norm decision controls: rows={random_total}", flush=True)
            progress = ProgressMeter("random decision controls", random_total, "rows", every_n=PROGRESS_EVERY_N)
            for random_id in range(N_RANDOM_DECISION_AXES):
                rand_u = unit(torch.randn(d_model, generator=rng).float())
                for direction, side, texts, sign in [
                    ("control_plus_random", "control", control_eval_texts, +1.0),
                    ("target_minus_random", "target", target_eval_texts, -1.0),
                ]:
                    eval_items = eval_items_for(side, texts)
                    for alpha_mult in REGIME_RANDOM_ALPHA_MULTS:
                        alpha_abs = float(sign * float(alpha_mult) * nat_norm)
                        for batch_items in chunked(eval_items, DECISION_BATCH_SIZE):
                            prompts = [item[6] for item in batch_items]
                            patched_logits_batch, patched_scores = score_decision_prompts_batch(
                                prompts,
                                hook_name=hook_name,
                                vec_unit=rand_u,
                                alpha_abs=alpha_abs,
                            )
                            for row_id, (text_id, text, probe_id, probe, base_logits, base_score, _prompt) in enumerate(batch_items):
                                add_decision_row(
                                    base_manifest,
                                    "random_same_norm_decision",
                                    f"random_{random_id}",
                                    direction,
                                    side,
                                    text_id,
                                    text,
                                    probe_id,
                                    probe,
                                    float(alpha_mult),
                                    alpha_abs,
                                    base_score,
                                    patched_scores[row_id],
                                    base_logits,
                                    patched_logits_batch[row_id],
                                    control_axis_id=random_id,
                                )
                                progress.update(detail=f"random={random_id} dir={direction} text={text_id + 1}/{len(texts)} probe={probe_id + 1}/{len(DECISION_PROBES)} alpha={alpha_mult}")
            write_run_state(
                "random_decision_controls_ready",
                {
                    "hook": hook_name,
                    "pool_mode": pool_mode,
                    "decision_rows": len(decision_rows),
                    "random_decision_axes": N_RANDOM_DECISION_AXES,
                },
            )

        if RUN_PERMUTATION_DECISION_CONTROLS and N_PERMUTATION_DECISION_AXES > 0:
            rng_np = np.random.default_rng(RANDOM_SEED_BASE + 707)
            perm_total = (
                N_PERMUTATION_DECISION_AXES
                * (len(control_eval_texts) + len(target_eval_texts))
                * len(DECISION_PROBES)
                * len(REGIME_PERMUTATION_ALPHA_MULTS)
            )
            print(f"[5] Label-permutation decision controls: rows<={perm_total}", flush=True)
            progress = ProgressMeter("permutation decision controls", perm_total, "rows", every_n=PROGRESS_EVERY_N)
            for perm_id in range(N_PERMUTATION_DECISION_AXES):
                perm = labels.copy()
                rng_np.shuffle(perm)
                if perm.sum() == 0 or perm.sum() == len(perm):
                    continue
                perm_u = unit(all_train[perm == 1].mean(0) - all_train[perm == 0].mean(0))
                for direction, side, texts, sign in [
                    ("control_plus_permutation", "control", control_eval_texts, +1.0),
                    ("target_minus_permutation", "target", target_eval_texts, -1.0),
                ]:
                    eval_items = eval_items_for(side, texts)
                    for alpha_mult in REGIME_PERMUTATION_ALPHA_MULTS:
                        alpha_abs = float(sign * float(alpha_mult) * nat_norm)
                        for batch_items in chunked(eval_items, DECISION_BATCH_SIZE):
                            prompts = [item[6] for item in batch_items]
                            patched_logits_batch, patched_scores = score_decision_prompts_batch(
                                prompts,
                                hook_name=hook_name,
                                vec_unit=perm_u,
                                alpha_abs=alpha_abs,
                            )
                            for row_id, (text_id, text, probe_id, probe, base_logits, base_score, _prompt) in enumerate(batch_items):
                                add_decision_row(
                                    base_manifest,
                                    "label_permutation_decision",
                                    f"perm_{perm_id}",
                                    direction,
                                    side,
                                    text_id,
                                    text,
                                    probe_id,
                                    probe,
                                    float(alpha_mult),
                                    alpha_abs,
                                    base_score,
                                    patched_scores[row_id],
                                    base_logits,
                                    patched_logits_batch[row_id],
                                    control_axis_id=perm_id,
                                )
                                progress.update(detail=f"perm={perm_id} dir={direction} text={text_id + 1}/{len(texts)} probe={probe_id + 1}/{len(DECISION_PROBES)} alpha={alpha_mult}")
            write_run_state(
                "permutation_decision_controls_ready",
                {
                    "hook": hook_name,
                    "pool_mode": pool_mode,
                    "decision_rows": len(decision_rows),
                    "permutation_decision_axes": N_PERMUTATION_DECISION_AXES,
                },
            )


# ====================== SAVE ARTIFACTS ======================

manifest_df = pd.DataFrame(manifest_rows)
cosine_df = pd.DataFrame(cosine_rows_all)
projection_df = pd.DataFrame(projection_rows)
control_axis_df = pd.DataFrame(control_axis_rows)
context_baseline_df = pd.DataFrame(context_baseline_rows)
decision_df = pd.DataFrame(decision_rows)

input_split_path = OUTPUT_DIR / f"regime_decision_input_split_{RUN_TAG}.csv"
manifest_path = OUTPUT_DIR / f"regime_decision_manifest_{RUN_TAG}.csv"
cosine_path = OUTPUT_DIR / f"regime_decision_cosines_{RUN_TAG}.csv"
projection_path = OUTPUT_DIR / f"regime_decision_projection_audit_{RUN_TAG}.csv"
control_axis_path = OUTPUT_DIR / f"regime_decision_control_axes_{RUN_TAG}.csv"
context_baseline_path = OUTPUT_DIR / f"regime_decision_context_baseline_{RUN_TAG}.csv"
context_baseline_summary_path = OUTPUT_DIR / f"regime_decision_context_baseline_summary_{RUN_TAG}.csv"
decision_rows_path = OUTPUT_DIR / f"regime_decision_probe_rows_{RUN_TAG}.csv"
decision_summary_path = OUTPUT_DIR / f"regime_decision_probe_summary_{RUN_TAG}.csv"
vectors_path = OUTPUT_DIR / f"regime_decision_vectors_{RUN_TAG}.npz"
claim_path = OUTPUT_DIR / f"regime_decision_claim_ladder_{RUN_TAG}.md"
probe_config_path = OUTPUT_DIR / f"regime_decision_probe_config_{RUN_TAG}.json"

pd.DataFrame(INPUT_SPLIT_ROWS).to_csv(input_split_path, index=False)
manifest_df.to_csv(manifest_path, index=False)
cosine_df.to_csv(cosine_path, index=False)
projection_df.to_csv(projection_path, index=False)
control_axis_df.to_csv(control_axis_path, index=False)
context_baseline_df.to_csv(context_baseline_path, index=False)
decision_df.to_csv(decision_rows_path, index=False)

if vector_npz:
    np.savez_compressed(vectors_path, **vector_npz)

probe_config = {
    "decision_probes": DECISION_PROBES,
    "decision_probe_source": DECISION_PROBE_SOURCE,
    "has_explicit_test_tasks": HAS_EXPLICIT_TEST_TASKS,
    "has_explicit_decision_probes": HAS_EXPLICIT_DECISION_PROBES,
    "require_explicit_test_tasks": REQUIRE_EXPLICIT_TEST_TASKS,
    "require_explicit_decision_probes": REQUIRE_EXPLICIT_DECISION_PROBES,
    "run_context_baseline_audit": RUN_CONTEXT_BASELINE_AUDIT,
    "axis_tasks": AXIS_TASKS,
    "prompt_mode": PROMPT_MODE,
    "answer_a_token_meta": A_TOKEN_META,
    "answer_b_token_meta": B_TOKEN_META,
    "alpha_mults": REGIME_ALPHA_MULTS,
    "random_alpha_mults": REGIME_RANDOM_ALPHA_MULTS,
    "permutation_alpha_mults": REGIME_PERMUTATION_ALPHA_MULTS,
}
probe_config_path.write_text(json.dumps(probe_config, ensure_ascii=False, indent=2), encoding="utf-8")

if not context_baseline_df.empty:
    context_summary_rows: List[Dict[str, Any]] = []
    group_cols = ["hook", "pool_mode", "probe_id", "probe_name"]
    for group_key, g in context_baseline_df.groupby(group_cols, dropna=False):
        target_g = g[g["base_side"].eq("target")]
        control_g = g[g["base_side"].eq("control")]
        if target_g.empty or control_g.empty:
            continue
        target_choice = str(g["target_choice"].iloc[0]).strip().upper()
        target_sign = choice_margin_sign(target_choice) if target_choice in ("A", "B") else 0
        t_margins = target_g["baseline_margin_A_minus_B"].astype(float).to_numpy()
        c_margins = control_g["baseline_margin_A_minus_B"].astype(float).to_numpy()
        pairwise = []
        signed_pairwise = []
        for tv in t_margins:
            for cv in c_margins:
                pairwise.append(1.0 if tv > cv else 0.5 if tv == cv else 0.0)
                stv = tv * target_sign
                scv = cv * target_sign
                signed_pairwise.append(1.0 if stv > scv else 0.5 if stv == scv else 0.0)
        target_mean = float(np.mean(t_margins))
        control_mean = float(np.mean(c_margins))
        gap = float(target_mean - control_mean)
        context_summary_rows.append(
            {
                "hook": group_key[0],
                "pool_mode": group_key[1],
                "probe_id": int(group_key[2]),
                "probe_name": group_key[3],
                "target_choice": target_choice,
                "control_choice": str(g["control_choice"].iloc[0]).strip().upper(),
                "n_target": int(len(target_g)),
                "n_control": int(len(control_g)),
                "target_margin_mean": target_mean,
                "control_margin_mean": control_mean,
                "target_minus_control_margin_gap": gap,
                "signed_target_minus_control_margin_gap": float(gap * target_sign),
                "target_prob_A_mean": float(target_g["baseline_prob_A_over_AB"].astype(float).mean()),
                "control_prob_A_mean": float(control_g["baseline_prob_A_over_AB"].astype(float).mean()),
                "target_choice_match_rate": float(target_g["baseline_choice_matches_expected"].astype(float).mean()),
                "control_choice_match_rate": float(control_g["baseline_choice_matches_expected"].astype(float).mean()),
                "target_gt_control_auc_like": float(np.mean(pairwise)) if pairwise else float("nan"),
                "signed_target_gt_control_auc_like": float(np.mean(signed_pairwise)) if signed_pairwise else float("nan"),
            }
        )
    context_summary_df = pd.DataFrame(context_summary_rows)
    context_summary_df.to_csv(context_baseline_summary_path, index=False)
else:
    pd.DataFrame().to_csv(context_baseline_summary_path, index=False)

if not decision_df.empty:
    group_cols = [
        "axis_source",
        "axis_variant",
        "direction",
        "base_side",
        "alpha_mult",
        "probe_name",
        "hook",
        "pool_mode",
    ]
    metric_cols = [
        "baseline_margin_A_minus_B",
        "patched_margin_A_minus_B",
        "margin_shift_A_minus_B",
        "signed_margin_shift",
        "prob_A_shift",
        "entropy_shift",
        "choice_changed",
        "final_next_token_kl_base_to_patched",
        "top_token_changed",
    ]
    existing_metrics = [c for c in metric_cols if c in decision_df.columns]
    summary_df = (
        decision_df.groupby(group_cols, dropna=False)[existing_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary_df["n_rows"] = decision_df.groupby(group_cols, dropna=False).size().values
    summary_df.to_csv(decision_summary_path, index=False)
else:
    pd.DataFrame().to_csv(decision_summary_path, index=False)


def quantile_or_nan(series: pd.Series, q: float) -> float:
    if series is None or series.empty:
        return float("nan")
    return float(series.quantile(q))


best_projection = None
if not projection_df.empty:
    best_projection = projection_df.sort_values("pairwise_auc_like", ascending=False).iloc[0].to_dict()

random_margin_p95 = float("nan")
permutation_margin_p95 = float("nan")
if not decision_df.empty:
    random_margin_p95 = quantile_or_nan(
        decision_df[decision_df["axis_source"].eq("random_same_norm_decision")]["signed_margin_shift"].abs(), 0.95
    )
    permutation_margin_p95 = quantile_or_nan(
        decision_df[decision_df["axis_source"].eq("label_permutation_decision")]["signed_margin_shift"].abs(), 0.95
    )

best_context_baseline = None
if "context_summary_df" in globals() and not context_summary_df.empty:
    best_context_baseline = (
        context_summary_df.reindex(
            context_summary_df["signed_target_minus_control_margin_gap"].abs().sort_values(ascending=False).index
        )
        .iloc[0]
        .to_dict()
    )

elapsed = time.time() - run_started

claim_lines = [
    "# Regime Decision Probe Claim Ladder",
    "",
    f"RUN_TAG: `{RUN_TAG}`",
    f"Elapsed seconds: `{elapsed:.1f}`",
    "",
    "## What This Script Measures",
    "",
    "1. Train-only target/control hidden-state regime axis.",
    "2. Held-out target/control projection separation.",
    "3. Unpatched target-vs-control context effect on forced-choice decision margins.",
    "4. Causal margin shifts under actual, same-norm random, and label-permutation axes.",
    "",
    "## Primary Metric",
    "",
    "`margin = logp(A) - logp(B)`",
    "",
]

if best_projection:
    claim_lines.extend(
        [
            f"Best actual projection variant: `{best_projection.get('axis_variant')}`",
            f"Best actual held-out AUC-like score: `{best_projection.get('pairwise_auc_like'):.4f}`",
            f"Best actual projection gap: `{best_projection.get('target_minus_control_proj_gap'):.4f}`",
            "",
        ]
    )

if best_context_baseline:
    claim_lines.extend(
        [
            f"Top baseline context probe: `{best_context_baseline.get('probe_name')}`",
            f"Top signed target-control margin gap: `{best_context_baseline.get('signed_target_minus_control_margin_gap'):.4f}`",
            f"Top signed target>control AUC-like: `{best_context_baseline.get('signed_target_gt_control_auc_like'):.4f}`",
            "",
        ]
    )

claim_lines.extend(
    [
        f"Random decision |signed margin shift| p95: `{random_margin_p95:.4f}`",
        f"Permutation decision |signed margin shift| p95: `{permutation_margin_p95:.4f}`",
        "",
        "## Reading Rules",
        "",
        "- Strong geometry signal: actual held-out AUC beats random and permutation projection controls.",
        "- Strong baseline context signal: target contexts shift `margin = logp(A)-logp(B)` relative to control contexts in the configured target-choice direction.",
        "- Strong decision signal: actual `signed_margin_shift` beats same-norm random and permutation controls at matched alpha/probe/side.",
        "- Directional success: `signed_margin_shift > 0` means movement toward the configured opposite-side choice.",
        "- Choice flips are secondary; margin shifts matter even when the top A/B choice does not flip.",
        "",
        "## Boundary",
        "",
        "This does not measure free-form answer quality. It tests whether hidden-state regime directions move forced-choice decision margins before text generation.",
        "",
    ]
)
claim_path.write_text("\n".join(claim_lines), encoding="utf-8")
write_run_state(
    "complete",
    {
        "manifest_rows": len(manifest_rows),
        "cosine_rows": len(cosine_rows_all),
        "projection_rows": len(projection_rows),
        "control_axis_rows": len(control_axis_rows),
        "context_baseline_rows": len(context_baseline_rows),
        "decision_rows": len(decision_rows),
        "final_decision_rows_path": str(decision_rows_path),
        "final_context_baseline_path": str(context_baseline_path),
    },
)

print("\n=== SAVED ===")
print(f"input split:      {input_split_path}")
print(f"manifest:         {manifest_path}")
print(f"cosines:          {cosine_path}")
print(f"projection:       {projection_path}")
print(f"control axes:     {control_axis_path}")
print(f"context rows:     {context_baseline_path}")
print(f"context summary:  {context_baseline_summary_path}")
print(f"decision rows:    {decision_rows_path}")
print(f"decision summary: {decision_summary_path}")
print(f"vectors:          {vectors_path}")
print(f"probe config:     {probe_config_path}")
print(f"claim ladder:     {claim_path}")

if best_projection:
    print("\n=== TOP PROJECTION SIGNAL ===")
    print(
        f"{best_projection.get('hook')} / {best_projection.get('pool_mode')} / "
        f"{best_projection.get('axis_variant')}: "
        f"AUC-like={best_projection.get('pairwise_auc_like'):.3f}, "
        f"balanced_acc={best_projection.get('balanced_threshold_accuracy'):.3f}, "
        f"gap={best_projection.get('target_minus_control_proj_gap'):.3f}"
    )

if best_context_baseline:
    print("\n=== TOP BASELINE CONTEXT SIGNAL ===")
    print(
        f"{best_context_baseline.get('hook')} / {best_context_baseline.get('pool_mode')} / "
        f"{best_context_baseline.get('probe_name')}: "
        f"signed_gap={best_context_baseline.get('signed_target_minus_control_margin_gap'):.3f}, "
        f"signed_auc={best_context_baseline.get('signed_target_gt_control_auc_like'):.3f}, "
        f"target_margin={best_context_baseline.get('target_margin_mean'):.3f}, "
        f"control_margin={best_context_baseline.get('control_margin_mean'):.3f}"
    )

print("\n=== DECISION CONTROL HINTS ===")
print(f"random |signed margin shift| p95:      {random_margin_p95:.4f}")
print(f"permutation |signed margin shift| p95: {permutation_margin_p95:.4f}")
print("\n=== DONE ===")
