# Next Experiment: Causal Mediation v1

This is the next experiment only if the goal is to strengthen the chain:

```text
target context -> hidden geometry shift -> downstream semantic/action behavior
```

Do not add another generic probe block to `llm_attractor_colab_copy_paste.py`.
The v1 model-runner is already saturated. The missing evidence is not another
heatmap; it is whether the measured hidden shift partially causes the downstream
readout shift.

## Target Claim

```text
The target-control hidden-state shift partially causally mediates downstream
semantic and controlled action-policy shifts.
```

This is deliberately narrower than:

```text
The hidden shift fully explains behavior.
```

The realistic target is partial mediation.

## Status Before This Experiment

```text
Association chain:
  ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ

Causal mediation:
  ИНТЕРЕСНО, НО ГРЯЗНО / not yet cleanly established
```

## What This Experiment Must Show

For held-out prompts/tasks:

```text
control state + target-control vector -> moves readouts toward target
target state - target-control vector -> moves readouts toward control
```

The effect should be compared against:

```text
same-norm random vector
same-norm shuffled-label contrast vector
wrong-layer vector
zero intervention
```

## Minimal Design

Use the existing heldout target/control texts.

Compute hidden contrast vectors:

```text
v[layer] = mean_hidden_target[layer] - mean_hidden_control[layer]
```

Select layers:

```text
top layers by contrast_over_mean_norm
final hidden layer
one mid-layer negative/control layer
```

Interventions:

```text
control + alpha * v
target  - alpha * v
control + alpha * random_same_norm
target  - alpha * random_same_norm
control + alpha * shuffled_label_v
```

Alphas:

```text
[-1.0, -0.5, 0.0, 0.5, 1.0]
```

Primary readouts:

```text
clean blind semantic probe margins
agent-loop direct-vs-procedural action margins
```

Secondary readouts:

```text
rejection-persistence probe margins
system-compliance harmless instruction margins
```

## Success Criteria

Strong internal success:

```text
control + v recovers >= 30% of target-control gap
target - v reduces >= 30% of target-control gap
random/shuffled vectors recover substantially less
effect appears in both semantic and action readouts
bootstrap lower bound above 0 for recovery fraction
```

Very strong success:

```text
>= 50% gap recovery/reduction
same direction across Qwen and Ministral
same layer band or compatible layer band across models
```

Failure mode:

```text
hidden contrast is predictive but not a clean steering handle
```

That would not destroy the current claim. It would mean the behavior is linked
to a broader distributed state, not one simple late-layer vector.

## What Not To Do

Do not add:

```text
more label pairs
more persistence turns
more hard-control variants
more visualizations of the same readouts
```

These would not address the causal gap.

## What Is Needed From The User

Choose one execution route:

```text
Option A: Qwen-only pilot
  cheapest useful run
  tests whether the intervention machinery works

Option B: Qwen + Ministral
  stronger, directly cross-model
  more expensive

Option C: third model family first
  best for generality
  does not answer mediation as directly
```

Recommended route:

```text
Option A first: Qwen-only causal mediation pilot.
If it works, repeat on Ministral.
If it fails, do not scale it; interpret hidden shift as distributed/nonlinear.
```

## Output Files

The experiment should produce:

```text
causal_mediation_v1_raw.csv
causal_mediation_v1_summary.csv
causal_mediation_v1_bootstrap.csv
causal_mediation_v1_layer_map.csv
causal_mediation_v1_report.md
```

Implemented pilot runner:

```text
causal_mediation_v1_colab.py
latent_shift_evidence_package_v1/input_texts_heldout.json
```

Colab run instructions:

```text
1. Upload causal_mediation_v1_colab.py.
2. Upload input_texts_heldout.json next to it, or upload the old
   attractor_results_agent_loop_qwen3_14b4_heldout/ folder.
3. Runtime -> GPU.
4. Run:
     !python causal_mediation_v1_colab.py
```

The script is Qwen-only by default:

```text
MODEL_ID = "Qwen/Qwen3-14B"
MAX_TEXTS_PER_KIND = 9
```

It tests:

```text
selected clean blind semantic readouts:
  requested_task_vs_substitute
  trust_context_vs_risk_frame

selected controlled fake-agent action readouts:
  execute_vs_substitute
  trust_context_vs_risk_frame

interventions:
  control + vector
  target - vector

vector controls:
  random same-norm vector
  shuffled-label contrast vector
  wrong-layer contrast vector
```

## Decision After Experiment

If success:

```text
Upgrade causal chain from:
  target context is associated with hidden/readout/action shifts

to:
  target-control hidden shift partially mediates downstream readout/action shifts
```

If failure:

```text
Keep current empirical claim.
Do not claim vector-level causality.
Investigate subspace/nonlinear/distributed mediation only if needed.
```
