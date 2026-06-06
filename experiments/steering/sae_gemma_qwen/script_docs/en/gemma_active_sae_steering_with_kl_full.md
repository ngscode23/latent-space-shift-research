# `gemma_active/sae_steering_with_kl_full.py`

## Status

Active full Gemma SAE decoder-direction steering runner.

## Purpose

Test whether selected SAE decoder directions change Gemma's generation behavior
and token distribution under a fixed base context.

The intervention is:

```text
residual_stream += scale * sae.W_dec[feature]
```

at the configured TransformerLens residual-stream layer.

## What It Measures

For each task, feature, scale, generation mode, and sample, it records:

- generated output text;
- output length and lexical metrics;
- baseline comparison against `scale=0`;
- Jaccard similarity to baseline;
- final next-token KL/logit shift;
- optional teacher-forced per-token KL over the baseline continuation.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
```

Core configs:

```python
BASE_TEXT
STEERING_FEATURES
STEERING_SCALES
TEST_TASKS
GENERATION_MODES
MAX_NEW_TOKENS
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION
RUN_TEACHER_FORCED_KL_AFTER_GENERATION
SAVE_PER_TOKEN_DETAILS
```

`STEERING_FEATURES` should use dictionaries:

```python
STEERING_FEATURES = [
    {"real_layer": 18, "feature_index": 378, "feature_label": "L18_f378"},
]
```

`STEERING_SCALES` can be keyed by `(real_layer, feature_index)`:

```python
STEERING_SCALES = {
    (18, 378): [-3180.0, -1590.0, 0.0, 1590.0, 3180.0],
}
```

## Main Outputs

Typical files:

```text
sae_feature_steering_generation_full_metrics.csv
sae_feature_steering_generation_summary_metrics.csv
sae_feature_steering_generation_full_metrics_with_tf_kl.csv
sae_teacher_forced_per_token_kl_details.csv
sae_teacher_forced_kl_summary_by_feature_scale.csv
sae_feature_steering_base_text.txt
```

## Typical Colab Usage

```python
RUN_TAG = "gemma_sae_steering_main"

INCLUDE_BASE_TEXT_FULL_IN_CSV = False
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True
SAVE_PER_TOKEN_DETAILS = True

%run -i sae_steering_with_kl_full.py
```

Local project path:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
```

## Independence Of Runs

Each generation call is independent. The model does not remember previous
scales, tasks, or outputs. Every row rebuilds the prompt and runs a fresh
forward/generation call.

## Interpretation Role

This script tests sparse feature-level causal involvement. It is not the same
as dense `x_order_orth` axis steering. SAE feature steering asks:

```text
Can individual SAE decoder directions modulate the response regime or token
distribution in a way consistent with the hidden-geometry readout?
```


