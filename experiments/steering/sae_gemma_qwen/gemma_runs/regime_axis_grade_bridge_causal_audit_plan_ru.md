# Regime Axis Grade Bridge Causal Audit

## What We Are Testing

This is the next experiment after the Claude-style `regime_diff_steering.py`.
Claude's script asks a first-order causal question:

```text
Does a bank-level target-control residual vector move generation when injected?
```

The new bridge script asks the stronger question:

```text
Is the bank-level regime vector an independent regime attractor, or is it
mostly the already-known Grade/SAE geometry in another form?
```

## Script

```text
experiments/steering/sae_gemma_qwen/gemma_active/regime_axis_grade_bridge_causal_audit.py
```

Expected notebook usage:

```python
TARGET_BASE_TEXTS = [str(x) for x in prompts_target]   # or the 40-text target bank
CONTROL_BASE_TEXTS = [str(x) for x in prompts_control] # real control bank

REGIME_HOOKS = ["blocks.36.hook_resid_post"]
REGIME_POOL_MODES = ["prompt_mean"]
REGIME_ALPHA_MULTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
REGIME_ORTHO_FEATURES = [1914, 323]

# Optional, if the Grade artifact is available:
GRADE4_AXIS_ARTIFACT_PATH = "/content/.../grade4_axis_component_vectors_by_layer.npz"

%run -i experiments/steering/sae_gemma_qwen/gemma_active/regime_axis_grade_bridge_causal_audit.py
```

## Why This Is Stronger Than Claude's Script

Claude's script builds:

```text
v_regime = mean(target_bank) - mean(control_bank)
```

and injects it bidirectionally:

```text
control + alpha*v_regime
target  - alpha*v_regime
```

The bridge script keeps that core but adds reviewer-grade controls:

```text
1. train/test split: axis is built on train texts only;
2. held-out target/control projection audit;
3. cosine with Grade axes: x_content, x_order, x_order_orth, x_full;
4. cosine/top overlap with SAE directions, especially L36 f1914/f323;
5. projection-out variants: raw, SAE-orth, Grade-orth, Grade+SAE-orth;
6. same-dimension random unit axes;
7. label-permuted train axes;
8. bidirectional causal generation with small alpha grid;
9. final next-token KL and top-token-change;
10. visible text metrics: Cyrillic/Latin fraction, script switch, hedging,
    procedural markers, directness proxy, Jaccard to baseline.
```

## What Would Be A Strong Positive Result

Strong result:

```text
actual v_regime held-out AUC > random p95 and permutation p95;
grade_sae_orth variant remains high after removing x_content/x_order_orth
and f1914/f323;
control + alpha*v moves outputs more target-like than random same-norm axes;
target - alpha*v moves outputs away from target-like regime;
effect appears at small alpha without Russian->English script collapse.
```

Scientific reading:

```text
There is a bank-level regime attractor that is not reducible to the known
content/order Grade axes or the currently tracked L36 SAE features.
```

## What Would Be A Negative Or Contaminated Result

Negative result:

```text
raw v_regime separates held-out texts, but the signal disappears after
Grade/SAE projection-out.
```

Reading:

```text
The Claude-style vector is mostly a recombination of already-known geometry.
Useful, but not a new independent axis.
```

Contaminated result:

```text
large KL / top-token movement appears only at high alpha, with Cyrillic
fraction collapse and Russian->English switching.
```

Reading:

```text
This is a hidden-state disruption or language/script threshold effect, not
clean regime control. It resembles the earlier L36 f1914/f323 negative-scale
behavior.
```

## Current Claim Boundary

This experiment should not be merged into the main Grade result as if it were
the same object. Grade4 remains the content/order decomposition result. The
bridge audit is a separate test of a bank-level regime axis and its causal
independence from Grade/SAE directions.

