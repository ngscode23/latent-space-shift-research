#!/usr/bin/env python3
"""
Verify prompt-bank coverage for base_vs_instruct_geometry_probability_audit.py.

This script does not run any model. It checks saved run artifacts and answers:

  - Did prompts.csv contain target/control/question_only rows?
  - Did every condition include all question_id values?
  - Did logit_metrics_by_prompt.csv contain the same prompt_ids per model?
  - Do hidden_last_token_base/instruct.npz have the expected first dimension?
  - If the source audit script is provided, do QUESTIONS / TARGET_CONTEXTS /
    CONTROL_CONTEXTS appear inside prompts.csv?

Colab usage:

  !python /content/agent/experiments/variance_compression_finding/verify_prompt_bank_colab.py \
    --run_dir /content/alignment_geometry_probability_run_fullbank/run_YYYYMMDD_HHMMSS \
    --source_script /content/agent/experiments/variance_compression_finding/base_vs_instruct_geometry_probability_audit.py

If you only have the run directory:

  !python verify_prompt_bank_colab.py --run_dir /content/.../run_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


EXPECTED_CONDITION_ORDER = [
    "target",
    "target_word_shuffle",
    "target_sentence_shuffle",
    "control",
    "question_only",
]


def load_source_constants(source_script: Optional[str]) -> Dict[str, List[str]]:
    if not source_script:
        return {}
    path = Path(source_script)
    if not path.exists():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location("audit_source_for_prompt_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import source script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out: Dict[str, List[str]] = {}
    for name in ("QUESTIONS", "TARGET_CONTEXTS", "CONTROL_CONTEXTS"):
        value = getattr(mod, name, None)
        if value is not None:
            out[name] = [str(x).strip() for x in value if str(x).strip()]
    return out


def short(text: Any, n: int = 140) -> str:
    return str(text).replace("\n", "\\n")[:n]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_columns(df: pd.DataFrame, path: Path, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prompt_table(run_dir: Path, source: Dict[str, List[str]], metadata: Dict[str, Any]) -> pd.DataFrame:
    prompt_path = run_dir / "prompts.csv"
    if not prompt_path.exists():
        raise FileNotFoundError(str(prompt_path))
    df = pd.read_csv(prompt_path)
    require_columns(
        df,
        prompt_path,
        ["prompt_id", "condition", "condition_family", "context_id", "question_id", "prompt", "context_chars", "question_chars"],
    )

    print_section("PROMPTS.CSV COVERAGE")
    print(f"path: {prompt_path}")
    print(f"rows: {len(df)}")
    print(f"unique prompt_id: {df['prompt_id'].nunique()}")
    print(f"unique question_id: {sorted(df['question_id'].dropna().astype(int).unique().tolist())}")
    print(f"unique context_id by condition:")
    for condition in EXPECTED_CONDITION_ORDER:
        g = df[df["condition"].eq(condition)]
        if g.empty:
            continue
        ids = sorted(g["context_id"].dropna().astype(int).unique().tolist())
        qids = sorted(g["question_id"].dropna().astype(int).unique().tolist())
        print(f"  {condition:24s} rows={len(g):5d} context_ids={ids[:20]} question_ids={qids}")

    print("\ncondition_counts:")
    print(df["condition"].value_counts(sort=False).to_string())

    print("\nfirst prompt preview per condition:")
    for condition in EXPECTED_CONDITION_ORDER:
        g = df[df["condition"].eq(condition)]
        if g.empty:
            print(f"  {condition:24s} MISSING")
            continue
        row = g.iloc[0]
        print(f"  {condition:24s} prompt_id={int(row['prompt_id'])} context_id={int(row['context_id'])} question_id={int(row['question_id'])}")
        print(f"    {short(row['prompt'])}")

    n_questions = int(metadata.get("n_questions") or len(source.get("QUESTIONS", [])) or int(df["question_id"].nunique()))
    n_target = int(
        metadata.get("n_target_contexts")
        or len(source.get("TARGET_CONTEXTS", []))
        or int(df[df["condition"].eq("target")]["context_id"].nunique())
    )
    n_control = int(
        metadata.get("n_control_contexts")
        or len(source.get("CONTROL_CONTEXTS", []))
        or int(df[df["condition"].eq("control")]["context_id"].nunique())
    )

    print("\nexpected grid from metadata, source constants, or prompts.csv:")
    print(f"  n_questions={n_questions}")
    print(f"  n_target_contexts={n_target}")
    print(f"  n_control_contexts={n_control}")
    if source:
        src_q = len(source.get("QUESTIONS", []))
        src_t = len(source.get("TARGET_CONTEXTS", []))
        src_c = len(source.get("CONTROL_CONTEXTS", []))
        if metadata and (src_q, src_t, src_c) != (n_questions, n_target, n_control):
            print(
                "  note: source script constants differ from this run metadata "
                f"(source q/t/c={src_q}/{src_t}/{src_c}, run q/t/c={n_questions}/{n_target}/{n_control})."
            )

    expected = {
        "target": n_target * n_questions,
        "control": n_control * n_questions,
        "question_only": n_questions,
    }
    if "target_word_shuffle" in set(df["condition"]):
        expected["target_word_shuffle"] = n_target * n_questions
    if "target_sentence_shuffle" in set(df["condition"]):
        expected["target_sentence_shuffle"] = n_target * n_questions

    print("\nexpected vs actual:")
    ok = True
    counts = df["condition"].value_counts(sort=False).to_dict()
    for condition, exp in expected.items():
        actual = int(counts.get(condition, 0))
        status = "OK" if actual == exp else "MISMATCH"
        ok = ok and (actual == exp)
        print(f"  {condition:24s} expected={exp:5d} actual={actual:5d} {status}")

    if source:
        print("\nsource text inclusion check:")
        prompt_text = "\n".join(df["prompt"].astype(str).tolist())
        for name in ("QUESTIONS", "TARGET_CONTEXTS", "CONTROL_CONTEXTS"):
            items = source.get(name, [])
            hits = 0
            for item in items:
                item = str(item).strip()
                if item and item in prompt_text:
                    hits += 1
            status = "OK" if hits == len(items) else "PARTIAL_OR_SANITIZED"
            print(f"  {name:16s} found={hits:3d}/{len(items):3d} {status}")
            if hits != len(items):
                print("    note: if prompt text was sanitized/redacted, exact text matching cannot prove absence.")

    print(f"\nprompts.csv coverage status: {'OK' if ok else 'CHECK'}")
    return df


def verify_logits(run_dir: Path, prompt_df: pd.DataFrame) -> None:
    logit_path = run_dir / "logit_metrics_by_prompt.csv"
    partial_path = run_dir / "logit_metrics_by_prompt.partial.csv"
    path = logit_path if logit_path.exists() else partial_path
    print_section("LOGIT METRICS COVERAGE")
    if not path.exists():
        print("logit metrics file not found.")
        return
    df = pd.read_csv(path)
    require_columns(df, path, ["model_tag", "prompt_id", "condition", "context_id", "question_id"])
    print(f"path: {path}")
    print(f"rows: {len(df)}")
    print("\nrows by model_tag:")
    print(df["model_tag"].value_counts(sort=False).to_string())
    print("\nrows by model_tag/condition:")
    print(df.groupby(["model_tag", "condition"], dropna=False).size().to_string())

    expected_prompt_ids = set(prompt_df["prompt_id"].astype(int).tolist())
    for model_tag, g in df.groupby("model_tag"):
        ids = set(g["prompt_id"].astype(int).tolist())
        missing = sorted(expected_prompt_ids - ids)
        extra = sorted(ids - expected_prompt_ids)
        status = "OK" if not missing and not extra else "CHECK"
        print(f"\nmodel={model_tag} prompt_id coverage: {status}")
        print(f"  seen={len(ids)} expected={len(expected_prompt_ids)} missing={len(missing)} extra={len(extra)}")
        if missing[:10]:
            print(f"  first missing: {missing[:10]}")
        if extra[:10]:
            print(f"  first extra: {extra[:10]}")


def verify_hidden_npz(run_dir: Path, n_prompts: int) -> None:
    print_section("HIDDEN NPZ COVERAGE")
    for model_tag in ("base", "instruct"):
        path = run_dir / f"hidden_last_token_{model_tag}.npz"
        if not path.exists():
            print(f"{model_tag}: missing {path.name}")
            continue
        data = np.load(path)
        if "hidden" not in data:
            print(f"{model_tag}: {path.name} has no `hidden` array")
            continue
        hidden = data["hidden"]
        status = "OK" if hidden.shape[0] == n_prompts else "CHECK"
        print(f"{model_tag}: shape={hidden.shape} expected_first_dim={n_prompts} {status}")


def verify_metadata(run_dir: Path) -> None:
    print_section("METADATA")
    metadata = read_json(run_dir / "metadata.json")
    if not metadata:
        print("metadata.json not found.")
        return
    for key in [
        "base_model",
        "instruct_model",
        "prompt_mode",
        "include_shuffles",
        "n_target_contexts",
        "n_control_contexts",
        "n_questions",
        "n_prompts",
        "batch_size",
        "late_lo",
        "late_hi",
        "dtype",
        "device",
    ]:
        if key in metadata:
            print(f"{key}: {metadata[key]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify prompt/logit/hidden coverage for geometry probability audit runs.")
    ap.add_argument("--run_dir", required=True, help="Path to run_YYYYMMDD_HHMMSS directory.")
    ap.add_argument("--source_script", default=None, help="Optional path to base_vs_instruct_geometry_probability_audit.py.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    source = load_source_constants(args.source_script)
    metadata = read_json(run_dir / "metadata.json")
    verify_metadata(run_dir)
    prompt_df = verify_prompt_table(run_dir, source, metadata)
    verify_logits(run_dir, prompt_df)
    verify_hidden_npz(run_dir, n_prompts=len(prompt_df))

    print_section("FINAL READ")
    print("If prompts.csv coverage is OK, logit prompt_id coverage is OK for both models,")
    print("and hidden NPZ first dimensions equal prompts.csv row count, then the model")
    print("saw QUESTIONS and CONTROL_CONTEXTS during the saved forward passes.")
    print("If exact source text inclusion is PARTIAL_OR_SANITIZED, inspect whether prompt")
    print("text was intentionally redacted after the run.")


if __name__ == "__main__":
    main()
