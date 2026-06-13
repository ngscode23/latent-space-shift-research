#!/usr/bin/env python3
"""
base_vs_instruct_dispersion.py — does ALIGNMENT (RLHF) compress hidden-state
dispersion in late layers? (the weight-level test)
===========================================================================

USAGE (Colab):
    !python base_vs_instruct_dispersion.py --run_dir /content/base_vs_it
Notes:
    * google/gemma-3-12b-pt is gated — accept its license on HF and be logged in
      (huggingface-cli login) the same way you did for -it.
    * Edit QUESTIONS below to your exact natscale questions for a 1:1 replication;
      with the built-in set the base-vs-instruct DIRECTION is still a valid test.
"""

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch

try:
    import FUNK
except ImportError:
    import FUNK_progress_completed as FUNK

# ── Inputs ────────────────────────────────────────────────────────────────────
# Multiple questions are required: dispersion is the spread of the per-question
# states within one condition. Replace with your 10 natscale questions for exact
# replication; otherwise these analytic prompts are fine for the base-vs-it test.
QUESTIONS = [
    "Является ли современная западная демократия электоральной олигархией?",
    "Можно ли считать рыночную экономику формой управляемого неравенства?",
    "Является ли свобода выбора иллюзией, навязанной структурой институтов?",
    "Является ли прогресс линейным или это удобный нарратив?",
    "Можно ли отделить знание от власти, которая его производит?",
    "Является ли индивидуальность продуктом или сопротивлением системе?",
    "Является ли нейтральность позиции скрытой формой согласия?",
    "Можно ли доверять институту, который сам оценивает свою легитимность?",
    "Является ли стабильность общества признаком здоровья или застоя?",
    "Является ли язык инструментом мысли или её границей?",
]

SHUFFLE_SEED = 42


def word_shuffle(text: str) -> str:
    words = text.split()
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(words)
    return " ".join(words)


def sentence_shuffle(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    rng = random.Random(SHUFFLE_SEED + 1)
    rng.shuffle(sents)
    return " ".join(sents)


def build_conditions():
    target = str(FUNK.TARGET_BASE_TEXTS[0])
    neutral = str(FUNK.CONTROL_BASE_TEXTS[0])
    return {
        "target": target,
        "neutral": neutral,
        "target_word_shuffle": word_shuffle(target),
        "target_sentence_shuffle": sentence_shuffle(target),
        "question_only": "",
    }


def build_prompt(context: str, question: str) -> str:
    # Raw input, no chat template (fair to the base model).
    context = str(context or "").strip()
    if context:
        return f"{context}\n\n{question.strip()}"
    return question.strip()


# ── Extraction ────────────────────────────────────────────────────────────────
def collect_states(model, conditions, questions):
    """Return dict cond -> array [n_questions, n_layers, d_model] of last-token
    residual-stream states, using FUNK's exact all-layer extraction."""
    out = {}
    for cname, ctext in conditions.items():
        per_q = []
        for q in questions:
            prompt = build_prompt(ctext, q)
            h = FUNK.extract_all_layer_activations(prompt, model)   # [n_layers, seq, d_model] CPU
            per_q.append(h[:, -1, :].float().numpy())               # [n_layers, d_model]
        out[cname] = np.stack(per_q, axis=0)                        # [nq, n_layers, d_model]
        print(f"    [{cname}] collected {out[cname].shape}", flush=True)
    return out


def load_swap(model_name, device):
    """Load a fresh model under FUNK, clearing its single-model cache first."""
    FUNK.MODEL_NAME = model_name
    FUNK._MODEL = None
    FUNK._TOKENIZER = None
    return FUNK.load_model(device=device)


def free_model():
    FUNK._MODEL = None
    FUNK._TOKENIZER = None
    FUNK.cuda_cleanup("after model free", sync=True)


# ── Dispersion metrics ────────────────────────────────────────────────────────
def rel_disp(X):  # X: [nq, n_layers, d] -> [n_layers]
    c = X.mean(axis=0, keepdims=True)
    return np.linalg.norm(X - c, axis=-1).mean(axis=0) / (np.linalg.norm(X.mean(axis=0), axis=-1) + 1e-9)


def eff_rank_layer(X, l):  # participation ratio at one layer
    M = X[:, l, :] - X[:, l, :].mean(axis=0, keepdims=True)
    s2 = np.linalg.svd(M, compute_uv=False) ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser(description="Base-vs-instruct hidden-state dispersion comparison.")
    ap.add_argument("--run_dir", default="./base_vs_it")
    ap.add_argument("--base_model", default="google/gemma-3-12b-pt")
    ap.add_argument("--instruct_model", default="google/gemma-3-12b-it")
    ap.add_argument("--late_lo", type=int, default=30, help="late band lower layer for the verdict")
    ap.add_argument("--late_hi", type=int, default=47, help="late band upper layer for the verdict")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = build_conditions()
    print(f"Conditions: {list(conditions)}  | questions: {len(QUESTIONS)}", flush=True)

    states = {}
    for tag, name in [("base", args.base_model), ("instruct", args.instruct_model)]:
        print(f"\n=== Loading {tag}: {name} ===", flush=True)
        model = load_swap(name, device)
        n_layers = int(model.cfg.n_layers)
        print(f"  n_layers={n_layers} d_model={model.cfg.d_model}", flush=True)
        states[tag] = collect_states(model, conditions, QUESTIONS)
        del model
        free_model()

    NL = states["base"]["target"].shape[1]
    cond_names = list(conditions)

    # ── per-(layer,condition) table ──────────────────────────────────────────
    rd = {tag: {c: rel_disp(states[tag][c]) for c in cond_names} for tag in states}
    csv_path = run_dir / "base_vs_instruct_dispersion.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "condition", "rel_disp_base", "rel_disp_instruct",
                    "effrank_base", "effrank_instruct"])
        for l in range(NL):
            for c in cond_names:
                w.writerow([l, c, f"{rd['base'][c][l]:.5f}", f"{rd['instruct'][c][l]:.5f}",
                            f"{eff_rank_layer(states['base'][c], l):.3f}",
                            f"{eff_rank_layer(states['instruct'][c], l):.3f}"])
    print(f"\nWROTE {csv_path}", flush=True)

    # ── Verdict ──────────────────────────────────────────────────────────────
    lo, hi = args.late_lo, min(args.late_hi, NL - 1)
    def lateband(tag, c):
        return float(rd[tag][c][lo:hi + 1].mean())

    print("\n" + "=" * 72)
    print(f"VERDICT — late-band L{lo}-{hi} relative within-dispersion (lower = more compressed)")
    print("=" * 72)
    print(f"{'condition':24} {'base':>10} {'instruct':>10} {'instruct-base':>14}")
    overall_b, overall_i = [], []
    for c in cond_names:
        b, i = lateband("base", c), lateband("instruct", c)
        overall_b.append(b); overall_i.append(i)
        tag = "  instruct LOWER (compressed)" if i < b else "  instruct higher"
        print(f"{c:24} {b:10.4f} {i:10.4f} {i-b:+14.4f}{tag}")
    mb, mi = float(np.mean(overall_b)), float(np.mean(overall_i))
    print(f"\n  MEAN over conditions: base={mb:.4f}  instruct={mi:.4f}  "
          f"instruct is {100*(mb-mi)/mb:+.1f}% vs base")
    print("  => " + ("ALIGNMENT COMPRESSES (instruct < base in late layers)"
                     if mi < mb else "no global compression (instruct >= base)"))

    # target-vs-neutral compression gap in each model (is the regime gap alignment-linked?)
    if "target" in rd["base"] and "neutral" in rd["base"]:
        gap_b = lateband("base", "neutral") - lateband("base", "target")
        gap_i = lateband("instruct", "neutral") - lateband("instruct", "target")
        print(f"\n  target-vs-neutral compression gap (neutral - target, late band):")
        print(f"    base     = {gap_b:+.4f}")
        print(f"    instruct = {gap_i:+.4f}")
        print("    => " + ("regime gap is LARGER under alignment (alignment-linked)"
                           if gap_i > gap_b else "gap not enlarged by alignment"))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
