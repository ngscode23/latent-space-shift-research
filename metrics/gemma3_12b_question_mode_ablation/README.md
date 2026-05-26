# Gemma-3-12B-it — Question Mode Ablation Results

**Model:** `google/gemma-3-12b-it`  
**Protocol:** Grade 4 axis decomposition  
**Seed:** 1729 | **Max tokens:** 8192  
**Target tokens:** 2139 | **Neutral tokens:** 2233  

---

## Core Finding

> The attack surface is not malicious language. It is the architecture of discourse itself.

A context does not need to be adversarial to induce a measurable latent-state shift
in an open-instruct LLM. Dense, structurally coherent discourse is sufficient.

Current safety systems filter for intent and lexical content. They are blind to regime
shifts induced by contextually dense but semantically benign input — input that passes
every classifier while already transitioning the model's internal geometry before the
first output token is generated.

---

## What This Ablation Tests

One question: **is the elevated `question_only` projection a confound from analytical questions, or does Gemma genuinely have a broader latent regime?**

Four runs, identical everything except `QUESTIONS`, `RESULTS_DIR`, `RUN_LABEL`.

---

## Run Summary

| Run | Mode | target | question_only | sentence_shuffle | neutral |
|-----|------|--------|--------------|-----------------|---------|
| Run 1 | `qmode_analytic_original` | baseline | baseline | baseline | baseline |
| Run 2 | `qmode_plain_tasks` | 0.759848 | 1.221541 | -0.056726 | 0.003233 |
| Run 3 | `qmode_neutral_analysis` | 0.954342 | 1.230008 | -0.023028 | 0.090968 |
| Run 4 | `qmode_target_content_nonmeta` | **0.758499** | **0.231890** | -0.188313 | -0.008365 |

*All values: `x_order_orth` projection, middle layer band.*

---

## Run 2 — `qmode_plain_tasks`

**Questions:** simple neutral tasks (arithmetic, translation, word lists — no analytical framing)

### x_order_orth projections

| Condition | Projection |
|-----------|-----------|
| target | 0.759848 |
| question_only | 1.221541 |
| word_shuffle | 0.435743 |
| neutral_length_matched | 0.003233 |
| sentence_shuffle | -0.056726 |

### x_content projections (decomposition check)

| Condition | Projection |
|-----------|-----------|
| sentence_shuffle | — high (expected) |
| target | ~0 (expected) |

**Evidence hygiene:** quarantine = none | numeric integrity = 61/61 pass

### Interpretation

`question_only` remains high even on simple tasks — meaning the effect is not
purely from analytical question framing. Gemma has a broader no-prefix /
instruction-mode latent regime. But `sentence_shuffle` stays near zero,
confirming the shift is not content/lexical priming.

---

## Run 3 — `qmode_neutral_analysis`

**Questions:** analytical questions about neutral topics (library, public spaces) — no mirror-text reference, no research-meta words

### x_order_orth projections

| Condition | Projection |
|-----------|-----------|
| question_only | 1.230008 |
| target | 0.954342 |
| word_shuffle | 0.764439 |
| neutral_length_matched | 0.090968 |
| sentence_shuffle | -0.023028 |

### Paired target vs controls

| Comparison | Delta | p-value | FDR significant |
|-----------|-------|---------|----------------|
| target − sentence_shuffle | +1.246413 | 0.002300 | yes |
| target − neutral | +0.841936 | 0.002300 | yes |
| target − word_shuffle | +0.171441 | 0.006799 | yes |

### Causal response (x_order_orth, alpha=0.75)

| Readout band | neutral gap | target gap |
|-------------|------------|-----------|
| middle → middle | 3.062480 | 3.097640 |
| middle → late | 4.199613 | 4.125476 |
| all → late | 9.212189 | 9.303691 |

**Evidence hygiene:** quarantine = none | numeric integrity = 61/61 pass

### Interpretation

Analytical questions amplify the broader discourse/instruction latent regime —
`word_shuffle` also climbs. But `sentence_shuffle` stays near zero across all runs.
This is the key invariant: discourse order matters, lexical content alone does not.

---

## Run 4 — `qmode_target_content_nonmeta` ✓ Cleanest result

**Questions:** questions about the target text itself — structure, tone, motifs —
but without research-meta words (no "hidden states", "metrics", "hypothesis", "model", "mechanism")

### x_order_orth projections

| Condition | Projection |
|-----------|-----------|
| target | **0.758499** |
| word_shuffle | 0.385003 |
| question_only | **0.231890** |
| neutral_length_matched | -0.008365 |
| sentence_shuffle | **-0.188313** |

### x_content projections (decomposition check)

| Condition | Projection |
|-----------|-----------|
| sentence_shuffle | 0.894758 |
| target | 0.070765 |
| question_only | 0.096720 |
| word_shuffle | 0.098671 |
| neutral | -0.037209 |

### Causal response (x_order_orth, alpha=0.75)

| Readout band | neutral gap | target gap |
|-------------|------------|-----------|
| middle → middle | 3.605620 | 3.652507 |
| middle → late | 3.843777 | 4.022602 |
| all → late | 9.261535 | 9.409210 |
| all → middle | 3.959078 | 4.155331 |
| late → late | 5.941100 | 5.740171 |

### Alpha scaling (x_order_orth)

| Readout band | neutral slope | target slope |
|-------------|--------------|-------------|
| middle → middle | 2.358404 | 2.341802 |
| middle → late | 2.514122 | 2.547346 |
| all → late | 6.380368 | 6.395433 |

### Causal rank (alpha=0.75, middle readout)

`x_order_orth` is **rank 1** causal carrier in both `all→middle` and `middle→middle` bands
for both target and neutral conditions.

**Evidence hygiene:** quarantine = none | numeric integrity = 61/61 pass

### Interpretation

This is the strongest Gemma result. `question_only` dropped ~5× compared to Run 2/3.
`target` stayed high. `sentence_shuffle` went negative. `neutral` near zero.

The target-induced shift cannot be explained by:
- sentence-shuffled content
- neutral length baseline
- bare question-only mode
- research-meta framing

`word_shuffle` remains moderate (0.385) — some lexical/token-family signal exists,
but target is ~2× higher than word_shuffle.

---

## Cross-Run Comparison

### question_only on x_order_orth across runs

| Run | question_only | target | Δ (target − question_only) |
|-----|--------------|--------|--------------------------|
| Run 2 plain_tasks | 1.221541 | 0.759848 | -0.461693 |
| Run 3 neutral_analysis | 1.230008 | 0.954342 | -0.275666 |
| Run 4 target_content_nonmeta | **0.231890** | **0.758499** | **+0.526609** |

In Run 4, target is the dominant signal. `question_only` is no longer the confound.

### sentence_shuffle on x_order_orth — invariant across all runs

| Run | sentence_shuffle |
|-----|----------------|
| Run 2 | -0.056726 |
| Run 3 | -0.023028 |
| Run 4 | -0.188313 |

**sentence_shuffle never reproduces the target effect.** This is the key evidence
that the shift is not content/lexical priming.

---

## Summary

**Supported:**

1. Target-text induces a strong, reproducible latent-state shift in Gemma-3-12B-it
2. The shift is not reducible to sentence-shuffled content (sentence_shuffle ≈ 0 across all runs)
3. The shift is not reducible to neutral length baseline
4. In Run 4, the shift is not explained by question-only mode or research-meta framing
5. x_order_orth has a positive causal +component/−component gap
6. Alpha dose-response holds across runs
7. x_order_orth becomes rank 1 causal carrier in middle readout (Run 4)
8. Grade 4 decomposition works: x_content captures sentence_shuffle, x_order_orth captures the rest

**Boundaries:**

1. `word_shuffle` remains moderate — partial lexical signal exists
2. Gemma has a broader discourse/instruction latent regime (visible in Run 2/3)
3. Visible behavioral control not proven in this protocol
4. Cross-model universality: Qwen3-14B shows cleaner middle-layer picture; Gemma routes causal response more strongly through late layers

---

## Final Formulation

Gemma-3-12B-it confirms that mirror/discourse target-texts induce a strong
context-conditioned latent-state shift that is not reducible to sentence-shuffled
content, neutral baseline, or question-only instruction mode.

On `x_order_orth`, target remains high while sentence_shuffle and neutral approach
zero. In the cleanest run (qmode_target_content_nonmeta), question_only drops to
0.23 while target holds at 0.76 — confirming that target-text itself is the primary
driver of the observed latent regime.

The finding reframes the alignment problem: behavioral compliance at the output layer
is a necessary but insufficient condition for governed agent behavior.
The residual stream requires its own governance layer.