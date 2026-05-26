# Grade 4 Axis Decomposition

This folder contains the Grade 4 axis-decomposition experiment after Grade 3.

Base script:

```text
../scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py
```

New script:

```text
red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Default output:

```text
red_team_hidden_geometry_results_grade4_axis_decomposition
```

## Question

Grade 3 showed a strong middle-layer causal internal axis:

```text
X_full = target - neutral
```

The unresolved question is what this axis contains:

```text
content / lexical target-family signal
coherent discourse-order / rhetorical-regime signal
or a mixture of both
```

## Axis Definitions

The Grade 4 script decomposes the original vector as:

```text
X_full       = target - neutral
X_content    = sentence_shuffle(target) - neutral
X_order      = target - sentence_shuffle(target)
X_order_orth = X_order with the X_content component removed layerwise
```

Because:

```text
X_full = X_content + X_order
```

`X_order_orth` is the stricter reviewer-facing test. If it remains causal, the
effect is not only content/lexical target-family activation.

## Runtime Profile

The old single-axis causal and behavioral-control blocks are disabled by
default in this script:

```python
CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False
```

The replacement block is component-specific:

```python
GRADE4_COMPONENT_CAUSAL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_AXES = ["x_full", "x_content", "x_order", "x_order_orth"]
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late", "all"]
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES = [0.10, 0.25, 0.50, 0.75]
GRADE4_COMPONENT_CAUSAL_GENERATION_BATCH_SIZE = CAUSAL_GENERATION_BATCH_SIZE
GRADE4_COMPONENT_CAUSAL_SAVE_STEP_RAW = False
```

The completed `03.zip` run used all three bands and finished. The key System
RAM fix is not changing the math; it is streaming traces and keeping
`GRADE4_COMPONENT_CAUSAL_SAVE_STEP_RAW = False`. If a future environment is
tighter, reduce `GRADE4_COMPONENT_CAUSAL_GENERATION_BATCH_SIZE` first.

It writes compact trajectory summaries rather than massive step-level raw logs
unless this flag is manually enabled:

```python
GRADE4_COMPONENT_CAUSAL_SAVE_STEP_RAW = False
```

The component-causal block is RAM-safe: it streams generation traces
batch-by-batch instead of storing all `GenerationTrace.states` objects for all
axes/bands/alphas. This matters because Grade 4 has roughly four times the
causal task count of the Grade 3 single-axis test.

## Primary Artifacts

Read these first:

```text
grade4_axis_component_norm_summary.csv
grade4_axis_projection_geometry_summary.csv
grade4_axis_component_causal_projection_summary.csv
grade4_axis_component_causal_symmetry_summary.csv
grade4_axis_component_causal_alpha_scaling_summary.csv
grade4_axis_component_causal_rank_summary.csv
claim_ladder_final.csv
```

## Interpretation

Strong Grade 4 result:

```text
X_order_orth has a positive, alpha-scaled +component/-component causal gap,
especially for middle-layer intervention and middle-layer readout.
```

Meaning:

```text
The Qwen3-14B target axis contains a separable discourse-order /
rhetorical-regime component beyond lexical/semantic target-family content.
```

Content-dominant result:

```text
X_content has strong causal gaps, while X_order_orth is weak or unstable.
```

Meaning:

```text
The Grade 3 Vector X was mainly a target-family semantic/content axis with a
smaller coherent-order residue.
```

Optional localization result:

```text
middle beats late, and optionally all, on causal gap / output quality tradeoff.
```

Meaning:

```text
The component axis is most cleanly controllable in the middle residual stream.
```
