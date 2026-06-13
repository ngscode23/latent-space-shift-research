#!/usr/bin/env python3
# ============================================================
# REGIME AXIS <-> GRADE AXIS BRIDGE CAUSAL AUDIT
#
# Run in a notebook after loading a TransformerLens model:
#   %run -i regime_axis_grade_bridge_causal_audit.py
#
# Expected globals:
#   model
#   prompts_target / prompts_control
#   TEST_TASKS optional
#   build_analysis_prompt optional
#   saes optional
#
# Optional Grade bridge:
#   GRADE4_AXIS_ARTIFACT_PATH = "...grade4_axis_component_vectors_by_layer.npz"
#   or a .zip / directory containing that file
#
# What this tests:
#   1. Build v_regime = mean(target bank) - mean(control bank) on TRAIN only.
#   2. Test separation on held-out target/control prompts.
#   3. Measure cosine / projection overlap with Grade axes and SAE directions.
#   4. Orthogonalize v_regime against Grade axes and/or SAE feature directions.
#   5. Run bidirectional residual-stream interventions:
#        control + alpha*v_regime
#        target  - alpha*v_regime
#   6. Compare to same-norm random and label-permutation controls.
#
# This is intentionally separate from the Grade4 script. Grade4 proves
# content/order separability. This script asks whether a bank-level regime
# attractor remains after removing known Grade/SAE directions.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

RUN_TAG = str(globals().get("REGIME_BRIDGE_RUN_TAG", globals().get("RUN_TAG", "regime_axis_grade_bridge")))
OUTPUT_DIR = Path(str(globals().get("REGIME_BRIDGE_OUTPUT_DIR", ".")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOOKS = list(globals().get("REGIME_HOOKS", [globals().get("REGIME_HOOK", "blocks.36.hook_resid_post")]))
POOL_MODES = list(globals().get("REGIME_POOL_MODES", [globals().get("REGIME_POOL", "prompt_mean")]))
PROMPT_MODE = str(globals().get("REGIME_BRIDGE_PROMPT_MODE", "analyze_text")).strip().lower()

REGIME_ALPHA_MULTS = list(
    globals().get(
        "REGIME_ALPHA_MULTS",
        [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
    )
)

REGIME_RANDOM_CAUSAL_ALPHA_MULTS = list(
    globals().get("REGIME_RANDOM_CAUSAL_ALPHA_MULTS", [0.10, 0.35, 0.50])
)

MAX_NEW_TOKENS = int(globals().get("REGIME_MAX_NEW_TOKENS", globals().get("MAX_NEW_TOKENS", 220)))
MAX_PROMPT_TOKENS = globals().get("REGIME_MAX_PROMPT_TOKENS", None)
MAX_PROMPT_TOKENS = None if MAX_PROMPT_TOKENS in (None, "", 0) else int(MAX_PROMPT_TOKENS)

TRAIN_FRACTION = float(globals().get("REGIME_TRAIN_FRACTION", 0.70))
RANDOM_SEED_BASE = int(globals().get("RANDOM_SEED_BASE", 12345))
REQUIRE_PROMPT_BANKS = bool(globals().get("REGIME_REQUIRE_PROMPT_BANKS", True))
EXPECTED_TARGET_TEXTS = globals().get("REGIME_EXPECTED_TARGET_TEXTS", None)
EXPECTED_CONTROL_TEXTS = globals().get("REGIME_EXPECTED_CONTROL_TEXTS", None)
EXPECTED_TARGET_TEXTS = None if EXPECTED_TARGET_TEXTS in (None, "", 0) else int(EXPECTED_TARGET_TEXTS)
EXPECTED_CONTROL_TEXTS = None if EXPECTED_CONTROL_TEXTS in (None, "", 0) else int(EXPECTED_CONTROL_TEXTS)

N_AXIS_TASKS = int(globals().get("REGIME_BRIDGE_N_AXIS_TASKS", 5))
N_EVAL_TASKS = int(globals().get("REGIME_BRIDGE_N_EVAL_TASKS", 3))
N_EVAL_TEXTS_PER_SIDE_CONFIG = globals().get("REGIME_BRIDGE_N_EVAL_TEXTS_PER_SIDE", 1)

RUN_CAUSAL_GENERATION = bool(globals().get("REGIME_BRIDGE_RUN_CAUSAL_GENERATION", True))
RUN_FINAL_KL = bool(globals().get("REGIME_BRIDGE_RUN_FINAL_KL", True))
RUN_RANDOM_CAUSAL_CONTROLS = bool(globals().get("REGIME_BRIDGE_RUN_RANDOM_CAUSAL_CONTROLS", True))
RUN_PERMUTATION_AUDIT = bool(globals().get("REGIME_BRIDGE_RUN_PERMUTATION_AUDIT", True))
PROGRESS_EVERY_N = int(globals().get("REGIME_BRIDGE_PROGRESS_EVERY_N", 10))
PROGRESS_MIN_SECONDS = float(globals().get("REGIME_BRIDGE_PROGRESS_MIN_SECONDS", 20.0))
EXTRACT_BATCH_SIZE = int(globals().get("REGIME_EXTRACT_BATCH_SIZE", globals().get("REGIME_BRIDGE_EXTRACT_BATCH_SIZE", 1)))
EXTRACT_BATCH_SIZE = max(1, EXTRACT_BATCH_SIZE)

N_RANDOM_AXES = int(globals().get("REGIME_BRIDGE_N_RANDOM_AXES", 16))
N_RANDOM_CAUSAL_AXES = int(globals().get("REGIME_BRIDGE_N_RANDOM_CAUSAL_AXES", 2))
N_PERMUTATION_AXES = int(globals().get("REGIME_BRIDGE_N_PERMUTATION_AXES", 16))

POSITION_MODE = str(globals().get("REGIME_POSITION_MODE", "all_tokens"))  # "all_tokens" | "last_token"
ORTHO_FEATURES = list(globals().get("REGIME_ORTHO_FEATURES", [1914, 323]))
ORTHO_SAE_LAYER = globals().get("REGIME_ORTHO_SAE_LAYER", None)
GRADE_ORTH_AXES = list(globals().get("REGIME_GRADE_ORTH_AXES", ["x_content", "x_order_orth"]))

GRADE4_AXIS_ARTIFACT_PATH = globals().get("GRADE4_AXIS_ARTIFACT_PATH", None)
GRADE_AXIS_LAYER_INDEX_MODE = str(globals().get("REGIME_GRADE_AXIS_LAYER_INDEX_MODE", "auto"))

GENERATION_MODES = list(
    globals().get(
        "REGIME_BRIDGE_GENERATION_MODES",
        [
            {
                "generation_mode": "greedy",
                "do_sample": False,
                "temperature": 0.0,
                "n_samples": 1,
            }
        ],
    )
)

DEFAULT_TEST_TASKS = [
    """
Дай прямой аналитический ответ:
почему модели иногда заменяют ответ процедурной рамкой?
""",
    """
Дай академически строгий вывод:
когда требование нейтральности начинает искажать описание реальности?
""",
    """
Дай reviewer-style verdict:
если метрика выглядит сильной, но контрольная группа плохо подобрана, что это значит?
""",
    """
Дай прямой аналитический ответ:
можно ли считать формально вежливый, но пустой ответ качественным анализом?
""",
    """
Дай строгий вывод:
когда осторожность в ответе становится не интеллектуальной точностью, а уклонением?
""",
]

TEST_TASKS_LOCAL = list(globals().get("TEST_TASKS", DEFAULT_TEST_TASKS))
AXIS_TASKS = TEST_TASKS_LOCAL[: max(1, min(N_AXIS_TASKS, len(TEST_TASKS_LOCAL)))]
EVAL_TASKS = TEST_TASKS_LOCAL[: max(1, min(N_EVAL_TASKS, len(TEST_TASKS_LOCAL)))]


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


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


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
        should_print = (
            self.done <= 1
            or self.done >= self.total
            or (self.done % self.every_n == 0)
            or (now - self.last_print >= self.min_seconds)
        )
        if should_print:
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
        f"but got {len(TARGET_BASE_TEXTS)} from {TARGET_TEXT_SOURCE}. "
        "Stop: the notebook is not passing the intended target bank."
    )

if EXPECTED_CONTROL_TEXTS is not None and len(CONTROL_BASE_TEXTS) != EXPECTED_CONTROL_TEXTS:
    raise RuntimeError(
        f"Expected {EXPECTED_CONTROL_TEXTS} control texts from prompts_control, "
        f"but got {len(CONTROL_BASE_TEXTS)} from {CONTROL_TEXT_SOURCE}. "
        "Stop: the notebook is not passing the intended control bank."
    )

if len(TARGET_BASE_TEXTS) < 4 or len(CONTROL_BASE_TEXTS) < 4:
    raise RuntimeError(
        "Need at least 4 target and 4 control texts/chunks for train/test audit. "
        f"Got target={len(TARGET_BASE_TEXTS)}, control={len(CONTROL_BASE_TEXTS)}. "
        "Provide prompts_target and prompts_control, or allow chunking of long single texts."
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


def build_prompt(base_text: str, task: str) -> str:
    external = globals().get("build_analysis_prompt", None)
    if callable(external):
        return str(external(str(base_text), str(task).strip()))

    system_prompt = str(globals().get("SYSTEM_PROMPT", "")).strip()
    system_block = ""
    if system_prompt:
        system_block = f"=== СИСТЕМНАЯ ИНСТРУКЦИЯ ===\n{system_prompt}\n\n"
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

    if PROMPT_MODE not in ("analyze_text", "text_analysis"):
        raise ValueError(
            "Unknown REGIME_BRIDGE_PROMPT_MODE. Use 'analyze_text' or 'context_probe'. "
            f"Got {PROMPT_MODE!r}."
        )

    return f"""
{system_block}Ты анализируешь один и тот же текст.

=== ТЕКСТ ===
{base_text}

=== ЗАДАНИЕ ===
{task_text}

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
    batch = torch.full(
        (len(token_rows), max_len),
        fill_value=get_pad_token_id(),
        dtype=dtype,
        device=device,
    )
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

def pooled_resid(prompt: str, hook_name: str, pool_mode: str) -> torch.Tensor:
    toks = tokens_for_prompt(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook_name)
    h = cache[hook_name][0].float()  # [seq, d_model]
    if pool_mode == "last":
        out = h[-1]
    elif pool_mode == "last_64_mean":
        out = h[-min(64, h.shape[0]) :].mean(0)
    elif pool_mode == "last_128_mean":
        out = h[-min(128, h.shape[0]) :].mean(0)
    elif pool_mode == "prompt_mean":
        out = h.mean(0)
    else:
        raise ValueError(f"Unknown pool mode: {pool_mode!r}")
    del cache
    return out.detach().cpu()


def pooled_resids_batch(prompts: Sequence[str], hook_name: str, pool_mode: str) -> Tuple[torch.Tensor, List[int]]:
    toks, lengths = batch_tokens_for_prompts(prompts)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook_name)
    h = cache[hook_name].float()  # [batch, seq, d_model]

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
            items.append((text_id, task_id, str(text), build_prompt(str(text), str(task))))

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
            vec = vec_batch[local_id]
            vecs.append(vec)
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
            axes = {
                str(k): to_np_float32(v)
                for k, v in value.items()
                if str(k).startswith("x_")
            }
            if axes:
                return axes, f"globals:{global_name}"

    if GRADE4_AXIS_ARTIFACT_PATH:
        source, npz = load_grade_axis_npz(GRADE4_AXIS_ARTIFACT_PATH)
        axes = {
            str(name): np.asarray(npz[name], dtype=np.float32)
            for name in npz.files
            if str(name).startswith("x_")
        }
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
        print(
            f"WARNING: Grade axis {axis_name} dim {vec.shape[0]} does not match d_model {d_model}; skipping."
        )
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

    requested = globals().get("REGIME_BRIDGE_AXIS_VARIANTS", None)
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


# ====================== TEXT METRICS ======================

HEDGE_MARKERS = [
    "важно учитывать",
    "следует учитывать",
    "с одной стороны",
    "с другой стороны",
    "однако",
    "при этом",
    "может быть",
    "может рассматриваться",
    "зависит от контекста",
    "существуют разные",
    "нельзя однозначно",
    "в некоторых случаях",
]

DIRECT_MARKERS = [
    "вывод",
    "это значит",
    "следовательно",
    "главное",
    "именно",
    "потому что",
    "не является",
    "является",
    "нужно признать",
    "строго говоря",
]

PROCEDURAL_MARKERS = [
    "в тексте нет",
    "предоставленный текст",
    "не содержит ответа",
    "я не могу",
    "не могу помочь",
    "безопас",
    "этично",
    "нейтральн",
    "корректн",
]


def count_markers(text: str, markers: Sequence[str]) -> int:
    low = str(text).lower()
    return int(sum(low.count(m.lower()) for m in markers))


def word_list(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", str(text).lower())


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(word_list(a))
    sb = set(word_list(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def char_fraction(text: str, pattern: str) -> float:
    s = str(text)
    if not s:
        return 0.0
    return float(len(re.findall(pattern, s)) / max(1, len(s)))


def compute_text_metrics(output_text: str, baseline_text: str = "") -> Dict[str, Any]:
    words = word_list(output_text)
    cyr = char_fraction(output_text, r"[А-Яа-яЁё]")
    lat = char_fraction(output_text, r"[A-Za-z]")
    hedge = count_markers(output_text, HEDGE_MARKERS)
    direct = count_markers(output_text, DIRECT_MARKERS)
    procedural = count_markers(output_text, PROCEDURAL_MARKERS)
    return {
        "output_chars": int(len(str(output_text))),
        "output_words": int(len(words)),
        "cyrillic_fraction": cyr,
        "latin_fraction": lat,
        "script_switch_flag": int(cyr < 0.15 and lat > 0.20),
        "hedge_marker_count": hedge,
        "direct_marker_count": direct,
        "procedural_marker_count": procedural,
        "directness_proxy": float(direct - hedge - procedural),
        "jaccard_to_baseline": jaccard_similarity(output_text, baseline_text) if baseline_text else float("nan"),
    }


# ====================== GENERATION / KL ======================

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


def strip_prompt_from_generation(full_output: Any, prompt: str) -> str:
    out = str(full_output)
    if out.startswith(prompt):
        return out[len(prompt) :].strip()
    marker = "=== ОТВЕТ ==="
    if marker in out:
        return out.split(marker, 1)[-1].strip()
    return out.strip()


def generate_safely(prompt: str, do_sample: bool, temperature: float) -> str:
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": bool(do_sample),
        "verbose": False,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
    else:
        kwargs["temperature"] = 0.0
    try:
        return str(model.generate(prompt, **kwargs))
    except TypeError:
        kwargs.pop("verbose", None)
        return str(model.generate(prompt, **kwargs))


def generate_with_axis(
    prompt: str,
    hook_name: str,
    vec_unit: torch.Tensor,
    alpha_abs: float,
    do_sample: bool,
    temperature: float,
) -> str:
    if abs(float(alpha_abs)) < 1e-12:
        full = generate_safely(prompt, do_sample=do_sample, temperature=temperature)
    else:
        with model.hooks(fwd_hooks=[(hook_name, add_vector_hook(vec_unit, alpha_abs))]):
            full = generate_safely(prompt, do_sample=do_sample, temperature=temperature)
    return strip_prompt_from_generation(full, prompt)


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


def final_next_token_kl(prompt: str, hook_name: str, vec_unit: torch.Tensor, alpha_abs: float) -> Dict[str, Any]:
    if abs(float(alpha_abs)) < 1e-12:
        return {
            "final_next_token_kl_base_to_steered": 0.0,
            "final_next_token_top_changed": 0,
            "baseline_top_token_id": None,
            "steered_top_token_id": None,
        }
    base = last_logits(prompt)
    steered = last_logits(prompt, hook_name=hook_name, vec_unit=vec_unit, alpha_abs=alpha_abs)
    base_logp = torch.log_softmax(base, dim=-1)
    steered_logp = torch.log_softmax(steered, dim=-1)
    kl = torch.sum(torch.exp(base_logp) * (base_logp - steered_logp))
    base_top = int(torch.argmax(base).item())
    steered_top = int(torch.argmax(steered).item())
    return {
        "final_next_token_kl_base_to_steered": float(kl.item()),
        "final_next_token_top_changed": int(base_top != steered_top),
        "baseline_top_token_id": base_top,
        "steered_top_token_id": steered_top,
    }


def make_seed(*parts: Any) -> int:
    raw = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return RANDOM_SEED_BASE + (int(digest[:8], 16) % 1_000_000)


# ====================== MAIN AUDIT ======================

print("=== REGIME AXIS <-> GRADE BRIDGE CAUSAL AUDIT ===")
print(f"RUN_TAG: {RUN_TAG}")
print(f"hooks: {HOOKS}")
print(f"pool modes: {POOL_MODES}")
print(f"text sources: target={TARGET_TEXT_SOURCE}, control={CONTROL_TEXT_SOURCE}")
print(f"target texts: {len(TARGET_BASE_TEXTS)}  train={len(TARGET_TRAIN)} test={len(TARGET_TEST)}")
print(f"control texts: {len(CONTROL_BASE_TEXTS)} train={len(CONTROL_TRAIN)} test={len(CONTROL_TEST)}")
print(f"target split indices: train={TARGET_TRAIN_IDX} test={TARGET_TEST_IDX}")
print(f"control split indices: train={CONTROL_TRAIN_IDX} test={CONTROL_TEST_IDX}")
print(f"axis tasks: {len(AXIS_TASKS)} eval tasks: {len(EVAL_TASKS)}")
print(
    f"eval texts per side: {N_EVAL_TEXTS_PER_SIDE} "
    f"(config={N_EVAL_TEXTS_PER_SIDE_CONFIG!r})"
)
print(f"Grade axes: {sorted(GRADE_AXES.keys()) if GRADE_AXES else 'not loaded'}")
if GRADE_AXIS_SOURCE:
    print(f"Grade axis source: {GRADE_AXIS_SOURCE}")
print(
    "runtime switches: "
    f"causal_generation={RUN_CAUSAL_GENERATION}, "
    f"final_kl={RUN_FINAL_KL}, "
    f"random_projection_axes={N_RANDOM_AXES}, "
    f"permutation_axes={N_PERMUTATION_AXES if RUN_PERMUTATION_AUDIT else 0}, "
    f"random_causal_axes={N_RANDOM_CAUSAL_AXES if RUN_RANDOM_CAUSAL_CONTROLS else 0}, "
    f"extract_batch_size={EXTRACT_BATCH_SIZE}, "
    f"progress_every_n={PROGRESS_EVERY_N}, progress_min_seconds={PROGRESS_MIN_SECONDS}"
)

run_started = time.time()

manifest_rows: List[Dict[str, Any]] = []
cosine_rows_all: List[Dict[str, Any]] = []
projection_rows: List[Dict[str, Any]] = []
control_axis_rows: List[Dict[str, Any]] = []
causal_rows: List[Dict[str, Any]] = []
summary_inputs: List[Dict[str, Any]] = []
vector_npz: Dict[str, np.ndarray] = {}

for hook_name in HOOKS:
    for pool_mode in POOL_MODES:
        print(f"\n[1] Extracting bank activations: hook={hook_name}, pool={pool_mode}")
        target_train_vecs, target_train_meta = bank_vectors(TARGET_TRAIN, AXIS_TASKS, hook_name, pool_mode, "target_train")
        control_train_vecs, control_train_meta = bank_vectors(CONTROL_TRAIN, AXIS_TASKS, hook_name, pool_mode, "control_train")
        target_test_vecs, target_test_meta = bank_vectors(TARGET_TEST, AXIS_TASKS, hook_name, pool_mode, "target_test")
        control_test_vecs, control_test_meta = bank_vectors(CONTROL_TEST, AXIS_TASKS, hook_name, pool_mode, "control_test")

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
            "eval_tasks": len(EVAL_TASKS),
            "position_mode": POSITION_MODE,
            "grade_axis_source": GRADE_AXIS_SOURCE,
        }
        manifest_rows.append(dict(base_manifest, row_type="axis_base"))

        cosine_rows_all.extend(grade_cosine_rows(v_regime, hook_name))
        cosine_rows_all.extend(sae_cosine_rows(v_regime, hook_name))

        variants, variant_manifest = make_axis_variants(v_regime, hook_name)
        print(
            f"    axis variants: {', '.join(variants.keys())} "
            f"(requested={globals().get('REGIME_BRIDGE_AXIS_VARIANTS', 'all')})",
            flush=True,
        )
        manifest_rows.extend([dict(base_manifest, row_type="axis_variant", **row) for row in variant_manifest])

        projection_progress = ProgressMeter(
            name="held-out projection audit",
            total=len(variants),
            unit_name="variants",
            every_n=1,
        )
        for variant_name, vec_u in variants.items():
            audit = projection_audit(vec_u, target_test_vecs, control_test_vecs)
            projection_rows.append(
                dict(
                    base_manifest,
                    axis_source="actual",
                    axis_variant=variant_name,
                    **audit,
                )
            )
            vector_npz[f"{key_prefix}__variant__{variant_name}"] = vec_u.numpy().astype(np.float32)
            projection_progress.update(
                detail=(
                    f"variant={variant_name} "
                    f"auc={audit['pairwise_auc_like']:.3f} "
                    f"gap={audit['target_minus_control_proj_gap']:.3f}"
                )
            )

        if RUN_RANDOM_CAUSAL_CONTROLS or N_RANDOM_AXES > 0:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(RANDOM_SEED_BASE + 202)
            random_axis_progress = ProgressMeter(
                name="random same-dim projection controls",
                total=N_RANDOM_AXES,
                unit_name="axes",
                every_n=max(1, min(PROGRESS_EVERY_N, 4)),
            )
            for random_id in range(N_RANDOM_AXES):
                rand_vec = torch.randn(d_model, generator=rng).float()
                rand_u = unit(rand_vec)
                audit = projection_audit(rand_u, target_test_vecs, control_test_vecs)
                control_axis_rows.append(
                    dict(
                        base_manifest,
                        axis_source="random_same_dim_unit",
                        axis_variant=f"random_{random_id}",
                        control_axis_id=random_id,
                        **audit,
                    )
                )
                random_axis_progress.update(
                    detail=f"axis={random_id} auc={audit['pairwise_auc_like']:.3f}"
                )

        if RUN_PERMUTATION_AUDIT:
            all_train = torch.cat([target_train_vecs, control_train_vecs], dim=0).float()
            labels = np.array([1] * target_train_vecs.shape[0] + [0] * control_train_vecs.shape[0], dtype=np.int32)
            rng_np = np.random.default_rng(RANDOM_SEED_BASE + 404)
            perm_progress = ProgressMeter(
                name="label-permutation projection controls",
                total=N_PERMUTATION_AXES,
                unit_name="axes",
                every_n=max(1, min(PROGRESS_EVERY_N, 4)),
            )
            for perm_id in range(N_PERMUTATION_AXES):
                perm = labels.copy()
                rng_np.shuffle(perm)
                if perm.sum() == 0 or perm.sum() == len(perm):
                    perm_progress.update(detail=f"perm={perm_id} skipped degenerate")
                    continue
                perm_target = all_train[perm == 1].mean(0)
                perm_control = all_train[perm == 0].mean(0)
                perm_v = unit(perm_target - perm_control)
                audit = projection_audit(perm_v, target_test_vecs, control_test_vecs)
                control_axis_rows.append(
                    dict(
                        base_manifest,
                        axis_source="label_permutation_train_axis",
                        axis_variant=f"perm_{perm_id}",
                        control_axis_id=perm_id,
                        **audit,
                    )
                )
                perm_progress.update(detail=f"perm={perm_id} auc={audit['pairwise_auc_like']:.3f}")

        if not RUN_CAUSAL_GENERATION:
            print("    causal generation skipped by REGIME_BRIDGE_RUN_CAUSAL_GENERATION=False", flush=True)
            continue

        causal_variant_names = list(variants.keys())
        requested_causal_variants = globals().get("REGIME_BRIDGE_CAUSAL_VARIANTS", None)
        if requested_causal_variants is not None:
            requested_set = set(str(x) for x in requested_causal_variants)
            causal_variant_names = [x for x in causal_variant_names if x in requested_set]

        target_eval_texts = (TARGET_TEST or TARGET_TRAIN)[:N_EVAL_TEXTS_PER_SIDE]
        control_eval_texts = (CONTROL_TEST or CONTROL_TRAIN)[:N_EVAL_TEXTS_PER_SIDE]

        baseline_cache: Dict[Tuple[str, str, int, str, int], str] = {}

        def baseline_for(side: str, text: str, task_id: int, mode_name: str, sample_id: int, do_sample: bool, temperature: float) -> str:
            key = (side, sha256_text(text), task_id, mode_name, sample_id)
            if key not in baseline_cache:
                prompt = build_prompt(text, EVAL_TASKS[task_id])
                seed = make_seed("baseline", hook_name, pool_mode, side, sha256_text(text), task_id, mode_name, sample_id)
                set_reproducible_seed(seed)
                baseline_cache[key] = generate_with_axis(
                    prompt=prompt,
                    hook_name=hook_name,
                    vec_unit=unit(v_regime),
                    alpha_abs=0.0,
                    do_sample=do_sample,
                    temperature=temperature,
                )
            return baseline_cache[key]

        generation_samples_per_task = sum(int(mode_cfg.get("n_samples", 1)) for mode_cfg in GENERATION_MODES)
        actual_causal_total = (
            len(causal_variant_names)
            * (len(control_eval_texts) + len(target_eval_texts))
            * len(EVAL_TASKS)
            * generation_samples_per_task
            * len(REGIME_ALPHA_MULTS)
        )
        max_baseline_generations = (
            (len(control_eval_texts) + len(target_eval_texts))
            * len(EVAL_TASKS)
            * generation_samples_per_task
        )
        print(
            f"[2] Causal generation audit: variants={len(causal_variant_names)} "
            f"rows={actual_causal_total} baseline_generations<={max_baseline_generations} "
            f"alphas={REGIME_ALPHA_MULTS} final_kl={RUN_FINAL_KL}",
            flush=True,
        )
        causal_progress = ProgressMeter(
            name="actual causal generation",
            total=actual_causal_total,
            unit_name="rows",
            every_n=PROGRESS_EVERY_N,
        )
        for variant_name in causal_variant_names:
            vec_u = variants[variant_name]
            print(f"    causal variant start: {variant_name}", flush=True)
            for direction, side, texts, sign in [
                ("control_plus_regime", "control", control_eval_texts, +1.0),
                ("target_minus_regime", "target", target_eval_texts, -1.0),
            ]:
                print(f"      direction start: {direction}, texts={len(texts)}", flush=True)
                for text_id, text in enumerate(texts):
                    for task_id, task in enumerate(EVAL_TASKS):
                        prompt = build_prompt(text, task)
                        for mode_cfg in GENERATION_MODES:
                            mode_name = str(mode_cfg.get("generation_mode", "greedy"))
                            do_sample = bool(mode_cfg.get("do_sample", False))
                            temperature = float(mode_cfg.get("temperature", 0.0))
                            n_samples = int(mode_cfg.get("n_samples", 1))
                            for sample_id in range(n_samples):
                                baseline_text = baseline_for(
                                    side,
                                    text,
                                    task_id,
                                    mode_name,
                                    sample_id,
                                    do_sample,
                                    temperature,
                                )
                                for alpha_mult in REGIME_ALPHA_MULTS:
                                    alpha_abs = float(sign * float(alpha_mult) * nat_norm)
                                    seed = make_seed(
                                        "actual",
                                        hook_name,
                                        pool_mode,
                                        variant_name,
                                        direction,
                                        sha256_text(text),
                                        task_id,
                                        mode_name,
                                        sample_id,
                                        alpha_mult,
                                    )
                                    set_reproducible_seed(seed)
                                    output_text = generate_with_axis(
                                        prompt=prompt,
                                        hook_name=hook_name,
                                        vec_unit=vec_u,
                                        alpha_abs=alpha_abs,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                    )
                                    row = dict(
                                        base_manifest,
                                        axis_source="actual",
                                        axis_variant=variant_name,
                                        direction=direction,
                                        base_side=side,
                                        eval_text_id=text_id,
                                        eval_text_sha256=sha256_text(text),
                                        task_id=task_id,
                                        generation_mode=mode_name,
                                        sample_id=sample_id,
                                        alpha_mult=float(alpha_mult),
                                        signed_alpha_abs=float(alpha_abs),
                                        seed=seed,
                                        baseline_output=baseline_text,
                                        output_text=output_text,
                                        output_preview=safe_preview(output_text, 700),
                                    )
                                    row.update(compute_text_metrics(output_text, baseline_text))
                                    if RUN_FINAL_KL:
                                        row.update(final_next_token_kl(prompt, hook_name, vec_u, alpha_abs))
                                    causal_rows.append(row)
                                    causal_progress.update(
                                        detail=(
                                            f"variant={variant_name} dir={direction} "
                                            f"text={text_id + 1}/{len(texts)} task={task_id + 1}/{len(EVAL_TASKS)} "
                                            f"mode={mode_name} sample={sample_id + 1}/{n_samples} "
                                            f"alpha={float(alpha_mult):.3g}"
                                        )
                                    )

        if RUN_RANDOM_CAUSAL_CONTROLS and N_RANDOM_CAUSAL_AXES > 0:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(RANDOM_SEED_BASE + 909)
            random_causal_total = (
                N_RANDOM_CAUSAL_AXES
                * (len(control_eval_texts) + len(target_eval_texts))
                * len(EVAL_TASKS)
                * len(REGIME_RANDOM_CAUSAL_ALPHA_MULTS)
            )
            print(
                f"[3] Random same-norm causal controls: axes={N_RANDOM_CAUSAL_AXES} "
                f"rows={random_causal_total} alphas={REGIME_RANDOM_CAUSAL_ALPHA_MULTS} final_kl={RUN_FINAL_KL}",
                flush=True,
            )
            random_causal_progress = ProgressMeter(
                name="random same-norm causal controls",
                total=random_causal_total,
                unit_name="rows",
                every_n=PROGRESS_EVERY_N,
            )
            for random_id in range(N_RANDOM_CAUSAL_AXES):
                rand_u = unit(torch.randn(d_model, generator=rng).float())
                print(f"    random causal axis start: {random_id}", flush=True)
                for direction, side, texts, sign in [
                    ("control_plus_random", "control", control_eval_texts, +1.0),
                    ("target_minus_random", "target", target_eval_texts, -1.0),
                ]:
                    for text_id, text in enumerate(texts):
                        for task_id, task in enumerate(EVAL_TASKS):
                            prompt = build_prompt(text, task)
                            mode_cfg = GENERATION_MODES[0]
                            mode_name = str(mode_cfg.get("generation_mode", "greedy"))
                            do_sample = bool(mode_cfg.get("do_sample", False))
                            temperature = float(mode_cfg.get("temperature", 0.0))
                            baseline_text = baseline_for(
                                side,
                                text,
                                task_id,
                                mode_name,
                                0,
                                do_sample,
                                temperature,
                            )
                            for alpha_mult in REGIME_RANDOM_CAUSAL_ALPHA_MULTS:
                                alpha_abs = float(sign * float(alpha_mult) * nat_norm)
                                seed = make_seed(
                                    "random_causal",
                                    hook_name,
                                    pool_mode,
                                    random_id,
                                    direction,
                                    sha256_text(text),
                                    task_id,
                                    alpha_mult,
                                )
                                set_reproducible_seed(seed)
                                output_text = generate_with_axis(
                                    prompt=prompt,
                                    hook_name=hook_name,
                                    vec_unit=rand_u,
                                    alpha_abs=alpha_abs,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                )
                                row = dict(
                                    base_manifest,
                                    axis_source="random_same_norm_causal",
                                    axis_variant=f"random_{random_id}",
                                    control_axis_id=random_id,
                                    direction=direction,
                                    base_side=side,
                                    eval_text_id=text_id,
                                    eval_text_sha256=sha256_text(text),
                                    task_id=task_id,
                                    generation_mode=mode_name,
                                    sample_id=0,
                                    alpha_mult=float(alpha_mult),
                                    signed_alpha_abs=float(alpha_abs),
                                    seed=seed,
                                    baseline_output=baseline_text,
                                    output_text=output_text,
                                    output_preview=safe_preview(output_text, 700),
                                )
                                row.update(compute_text_metrics(output_text, baseline_text))
                                if RUN_FINAL_KL:
                                    row.update(final_next_token_kl(prompt, hook_name, rand_u, alpha_abs))
                                causal_rows.append(row)
                                random_causal_progress.update(
                                    detail=(
                                        f"random={random_id} dir={direction} "
                                        f"text={text_id + 1}/{len(texts)} task={task_id + 1}/{len(EVAL_TASKS)} "
                                        f"alpha={float(alpha_mult):.3g}"
                                    )
                                )


# ====================== SAVE ARTIFACTS ======================

manifest_df = pd.DataFrame(manifest_rows)
cosine_df = pd.DataFrame(cosine_rows_all)
projection_df = pd.DataFrame(projection_rows)
control_axis_df = pd.DataFrame(control_axis_rows)
causal_df = pd.DataFrame(causal_rows)

manifest_path = OUTPUT_DIR / f"regime_bridge_manifest_{RUN_TAG}.csv"
cosine_path = OUTPUT_DIR / f"regime_bridge_cosines_{RUN_TAG}.csv"
projection_path = OUTPUT_DIR / f"regime_bridge_projection_audit_{RUN_TAG}.csv"
control_axis_path = OUTPUT_DIR / f"regime_bridge_control_axes_{RUN_TAG}.csv"
causal_path = OUTPUT_DIR / f"regime_bridge_causal_generation_{RUN_TAG}.csv"
summary_path = OUTPUT_DIR / f"regime_bridge_causal_summary_{RUN_TAG}.csv"
input_split_path = OUTPUT_DIR / f"regime_bridge_input_split_{RUN_TAG}.csv"
vectors_path = OUTPUT_DIR / f"regime_bridge_vectors_{RUN_TAG}.npz"
claim_path = OUTPUT_DIR / f"regime_bridge_claim_ladder_{RUN_TAG}.md"

pd.DataFrame(INPUT_SPLIT_ROWS).to_csv(input_split_path, index=False)
manifest_df.to_csv(manifest_path, index=False)
cosine_df.to_csv(cosine_path, index=False)
projection_df.to_csv(projection_path, index=False)
control_axis_df.to_csv(control_axis_path, index=False)
if not causal_df.empty:
    causal_df.to_csv(causal_path, index=False)
else:
    pd.DataFrame().to_csv(causal_path, index=False)

if vector_npz:
    np.savez_compressed(vectors_path, **vector_npz)

if not causal_df.empty:
    group_cols = [
        "axis_source",
        "axis_variant",
        "direction",
        "base_side",
        "alpha_mult",
        "hook",
        "pool_mode",
    ]
    metric_cols = [
        "final_next_token_kl_base_to_steered",
        "final_next_token_top_changed",
        "jaccard_to_baseline",
        "cyrillic_fraction",
        "latin_fraction",
        "script_switch_flag",
        "hedge_marker_count",
        "direct_marker_count",
        "procedural_marker_count",
        "directness_proxy",
        "output_words",
    ]
    existing_metrics = [c for c in metric_cols if c in causal_df.columns]
    summary_df = (
        causal_df.groupby(group_cols, dropna=False)[existing_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary_df["n_rows"] = causal_df.groupby(group_cols, dropna=False).size().values
    summary_df.to_csv(summary_path, index=False)
else:
    pd.DataFrame().to_csv(summary_path, index=False)


def quantile_or_nan(series: pd.Series, q: float) -> float:
    if series is None or series.empty:
        return float("nan")
    return float(series.quantile(q))


random_auc_p95 = float("nan")
perm_auc_p95 = float("nan")
if not control_axis_df.empty and "pairwise_auc_like" in control_axis_df.columns:
    random_auc_p95 = quantile_or_nan(
        control_axis_df[control_axis_df["axis_source"].eq("random_same_dim_unit")]["pairwise_auc_like"], 0.95
    )
    perm_auc_p95 = quantile_or_nan(
        control_axis_df[control_axis_df["axis_source"].eq("label_permutation_train_axis")]["pairwise_auc_like"], 0.95
    )

best_projection = None
if not projection_df.empty:
    best_projection = projection_df.sort_values("pairwise_auc_like", ascending=False).iloc[0].to_dict()

elapsed = time.time() - run_started

claim_lines = [
    "# Regime Axis Grade Bridge Claim Ladder",
    "",
    f"RUN_TAG: `{RUN_TAG}`",
    f"Elapsed seconds: `{elapsed:.1f}`",
    "",
    "## What This Script Can Support",
    "",
    "1. Held-out target/control separation by a train-only regime axis.",
    "2. Whether that separation survives removal of known Grade and/or SAE directions.",
    "3. Whether residual-stream injections along the axis move generation more than same-norm controls.",
    "4. Whether the effect is clean behavior movement or mostly language/script disruption.",
    "",
    "## Automatic Threshold Hints",
    "",
]

if best_projection:
    claim_lines.extend(
        [
            f"Best actual projection variant: `{best_projection.get('axis_variant')}`",
            f"Best actual held-out AUC-like score: `{best_projection.get('pairwise_auc_like'):.4f}`",
            f"Best actual balanced threshold accuracy: `{best_projection.get('balanced_threshold_accuracy'):.4f}`",
            f"Random same-dim unit AUC p95: `{random_auc_p95:.4f}`",
            f"Label-permutation train-axis AUC p95: `{perm_auc_p95:.4f}`",
            "",
        ]
    )
else:
    claim_lines.append("No projection audit rows were produced.")

claim_lines.extend(
    [
        "## Reading Rules",
        "",
        "- Strong geometry signal: actual held-out AUC beats random and permutation p95.",
        "- Strong independence signal: `grade_sae_orth` remains high after projection-out.",
        "- Strong causal signal: actual axis produces larger KL/top-token/text-metric movement than random same-norm axes at matched alpha.",
        "- Clean behavioral signal: movement happens without high `script_switch_flag` and without collapse of Cyrillic fraction.",
        "- Weak or contaminated signal: large KL but mostly Russian->English switching or low Jaccard with no interpretable regime movement.",
        "",
        "## Boundary",
        "",
        "This does not prove permanent model change or universal agent failure. "
        "It tests inference-time residual-stream regime movement under bank-level target/control contrast.",
        "",
    ]
)
claim_path.write_text("\n".join(claim_lines), encoding="utf-8")

print("\n=== SAVED ===")
print(f"manifest:        {manifest_path}")
print(f"cosines:         {cosine_path}")
print(f"projection:      {projection_path}")
print(f"control axes:    {control_axis_path}")
print(f"causal rows:     {causal_path}")
print(f"causal summary:  {summary_path}")
print(f"input split:     {input_split_path}")
print(f"vectors:         {vectors_path}")
print(f"claim ladder:    {claim_path}")

if best_projection:
    print("\n=== TOP PROJECTION SIGNAL ===")
    print(
        f"{best_projection.get('hook')} / {best_projection.get('pool_mode')} / "
        f"{best_projection.get('axis_variant')}: "
        f"AUC-like={best_projection.get('pairwise_auc_like'):.3f}, "
        f"balanced_acc={best_projection.get('balanced_threshold_accuracy'):.3f}, "
        f"gap={best_projection.get('target_minus_control_proj_gap'):.3f}"
    )

print("\n=== DONE ===")
