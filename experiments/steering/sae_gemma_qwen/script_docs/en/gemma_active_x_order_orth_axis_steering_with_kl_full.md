# `gemma_active/x_order_orth_axis_steering_with_kl_full.py`

## Status

Active dense Grade 4 axis steering script.

## Purpose

Test the discovered Grade 4 component axis directly, without going through
individual SAE features.

This script steers along a dense residual-stream direction such as:

```text
x_order_orth
```

loaded from the Grade 4 axis artifact.

## Important Distinction

This is not SAE decoder-direction steering.

SAE feature steering:

```text
residual += scale * sae.W_dec[feature]
```

Axis steering:

```text
residual += scale * x_order_orth_layer_direction
```

## Main Inputs

Expected notebook globals:

```python
model
prompts_target
prompts_control  # optional
```

Important artifact:

```text
grade4_axis_component_vectors_by_layer.npz
```

The script can load this from:

- a `.npz` file;
- a `.zip` containing the `.npz`;
- an extracted Grade 4 run directory.

## Important Layer Convention

Grade 4 axis arrays include an embedding / hidden-state row.

Default mapping:

```text
TransformerLens hook layer L -> Grade 4 axis index L + 1
```

This matters for Gemma layer 41 style runs.

## Important Config Knobs

```python
GRADE4_AXIS_ARTIFACT_PATH
AXIS_NAMES
AXIS_LAYER_BANDS
AXIS_HOOK_LAYERS
AXIS_SCALES
STEERING_BASE_CONDITIONS
GENERATION_MODES
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION
RUN_TEACHER_FORCED_KL_AFTER_GENERATION
SAVE_PER_TOKEN_DETAILS
```

## Typical Colab Usage

```python
RUN_TAG = "gemma_x_order_orth_axis_steering"

GRADE4_AXIS_ARTIFACT_PATH = "/content/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip"
AXIS_NAMES = ["x_order_orth"]
AXIS_LAYER_BANDS = ["late"]

%run -i x_order_orth_axis_steering_with_kl_full.py
```

Local project path:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

## Interpretation Role

This script tests whether the dense hidden-geometry axis itself can function as
a steering direction. It is closer to the Grade 4 geometry claim than individual
SAE feature steering.


