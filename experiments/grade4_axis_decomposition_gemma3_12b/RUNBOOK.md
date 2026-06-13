# Gemma Grade4 Hidden-Geometry Runbook

This runbook explains the Grade4 source script and how its outputs feed the SAE
and steering pipeline.

## Main Script

```text
experiments/grade4_axis_decomposition_gemma/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

This is the main Gemma hidden-state geometry script. It captures model hidden
states, compares conditions, constructs latent axes from condition deltas,
computes layerwise and generation-time geometry, optionally runs causal
interventions, and writes SAE readout tables.

It is the source producer for the later SAE feature pipeline.

## What The Script Tests

The core question is:

```text
Does a coherent target context only change the visible answer, or does it move
the model into a different measurable internal state before and during
generation?
```

The important control logic:

```text
target
  Coherent target context.

neutral / control
  Reference baseline context.

word_shuffle
  Same or related words with destroyed order.

sentence_shuffle
  Similar content with sentence order / discourse structure disrupted.
```

If the effect were only lexical/content overlap, target and shuffle controls
should look similar in hidden-state geometry. The main Grade4 result is that
they separate differently: sentence-shuffle follows a content-like direction,
while coherent target loads onto an order/residual response-mode direction.

## Constructed Axes

The axes are not universal coordinates of the whole model. They are constructed
directions in residual-stream hidden-state space, derived from target/control
condition deltas.

```text
x_full
  Full target-vs-reference hidden-state displacement.

x_content
  Content-like component captured by shuffled-content controls, especially
  sentence_shuffle.

x_order
  Order/discourse-sensitive component before orthogonalization.

x_order_orth
  Order/residual component after removing the content-like part. This is the
  key axis used to ask whether coherent order / response mode separates from
  mere content overlap.
```

Important boundary:

```text
x_order_orth is not a magic coordinate. It is a constructed residual-stream
axis from condition deltas. It is useful because it separates coherent target
from shuffled-content controls.
```

## Important Configuration Knobs

Exact globals may vary by run, but these are the knobs to check before a run:

```text
DEFAULT_RUN_BASENAME
  Base name for the output folder/package. Use a unique value for every serious
  run so old metrics are not overwritten.

model / tokenizer / prompts_target / prompts_control
  The model and base texts already loaded in the notebook or configured in the
  script.

GENERATION_ENABLED
  Whether generation trajectory and visible-output blocks run.

RESEARCH_GRADE_METRICS_ENABLED
  Enables the reviewer-grade metric layer.

ENABLE_SENTENCE_SHUFFLE_CONTROL / related shuffle settings
  Must be on for the content/order separation claim.

SAE settings
  Must match the Gemma-Scope / SAE configuration if SAE tables are needed.

GRADE4_COMPONENT_CAUSAL_* settings
  Control component-axis causal intervention blocks.
```

`DEFAULT_RUN_BASENAME` is operationally important: downstream scripts expect a
stable ZIP or output folder name. For Colab, use a descriptive basename and zip
the result package after completion.

## Main Outputs

High-value artifacts:

```text
sae_order_feature_contrast.csv
grade4_axis_component_vectors_by_layer.npz
grade4_axis_component_*csv
layerwise_geometry_*csv
generation_trajectory_*csv
causal_intervention_*csv
red_team_hidden_geometry_verdict.md
run_metadata.json
red_team_input_manifest.json
```

### SAE handoff

```text
sae_order_feature_contrast.csv
```

is the primary feature ranking table for:

```text
experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

### Dense axis handoff

```text
grade4_axis_component_vectors_by_layer.npz
```

contains dense component vectors by layer and is the main input for:

```text
experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

### Metric-lab handoff

The whole Grade4 ZIP or output directory can be analyzed by:

```text
scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
```

## Metrics Folder

Packaged outputs and analysis results live in:

```text
experiments/grade4_axis_decomposition_gemma/metrics/
```

Read:

```text
experiments/grade4_axis_decomposition_gemma/metrics/README.md
```

## Claim Boundary

This script supports:

```text
context -> hidden-state shift -> separable content/order coordinates
```

It does not by itself prove permanent weight change or a permanent model state.
The correct claim is a temporary inference-time hidden-state shift measured in
high-dimensional residual-stream space.

The honest causal status is:

```text
descriptive latent shift: strong
content/order separability: strong
causal involvement: supported
stable bidirectional x_order_orth behavioral control: not proven
```

