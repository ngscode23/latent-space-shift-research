# `gemma_active/02_scale_calibration.py`

## Status

Active Gemma calibration script.

## Purpose

Choose physically meaningful steering scales for SAE decoder-direction
interventions before running generation.

The full steering script adds:

```text
scale * sae.W_dec[feature]
```

to the residual stream. A small numeric scale such as `1.5` can be almost zero
relative to Gemma residual norms. This calibration script estimates how large a
scale is relative to:

1. the residual-stream norm at the target layer;
2. the feature's native SAE activation magnitude;
3. optional next-token KL response.

## Why It Exists

Without calibration, a steering run can be misleading in two opposite ways:

1. The scale is too small and nothing happens.
2. The scale is too large and the model is damaged rather than steered.

This script gives a principled scale grid: mild, medium, and stress-test
intervention sizes as fractions of residual norm.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
STEERING_FEATURES
```

`STEERING_FEATURES` should use real TransformerLens block indices:

```python
STEERING_FEATURES = [
    (18, 378),
    (18, 373),
    (36, 323),
    (24, 76),
    (41, 207),
]
```

## Important Layer Convention

Use real layer indices, not CSV layer indices.

If a CSV uses one-based layer numbering:

```text
CSV layer 42 -> real_layer 41
CSV layer 37 -> real_layer 36
CSV layer 31 -> real_layer 30
```

## Important Config Knobs

```python
RESID_FRACTIONS
PASTE_FRACTIONS
RUN_KL_CHECK
KL_CHECK_FRACTIONS
MAX_PROMPTS_FOR_RESID_NORM
```

Example:

```python
RESID_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
PASTE_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
RUN_KL_CHECK = True
KL_CHECK_FRACTIONS = [0.02, 0.05, 0.10, 0.20]
```

## Main Outputs

The script prints ready-to-copy scale recommendations and writes a calibration
CSV such as:

```text
sae_scale_calibration.csv
```

The important product is a `STEERING_SCALES` dictionary for the generation
runner:

```python
STEERING_SCALES = {
    (18, 378): [-3180.0, -1590.0, 0.0, 1590.0, 3180.0],
    (41, 207): [-12600.0, -6300.0, 0.0, 6300.0, 12600.0],
}
```

## Typical Colab Usage

```python
STEERING_FEATURES = [
    (18, 378),
    (18, 373),
    (36, 323),
    (24, 76),
    (41, 207),
]

RESID_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
PASTE_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
RUN_KL_CHECK = True
KL_CHECK_FRACTIONS = [0.02, 0.05, 0.10, 0.20]

%run -i 02_scale_calibration.py
```

Local project path:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/02_scale_calibration.py
```

## Interpretation Role

This is not a discovery script and not a final steering experiment. It is a
measurement tool that prevents arbitrary scale choices.

Use it between candidate selection and generation steering.


