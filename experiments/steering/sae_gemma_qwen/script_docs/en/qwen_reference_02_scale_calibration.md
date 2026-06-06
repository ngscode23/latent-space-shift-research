# `qwen_reference/scripts_snapshot/02_scale_calibration.py`

## Status

Qwen reference snapshot.

## Purpose

Calibrate decoder-direction steering scales for Qwen features, analogous to the
Gemma scale calibration script.

## What It Does

- estimates residual-stream norms at selected Qwen layers;
- compares decoder-vector scale to residual norm;
- can check final-token KL response for selected scale fractions;
- proposes scale grids for later Qwen steering runs.

## Main Inputs

Expected inputs depend on the Qwen notebook setup, but typically include:

```python
model
saes
prompts_target
STEERING_FEATURES
```

## Important Boundary

This is not the canonical Gemma calibration script. For Gemma, use:

```text
gemma_active/02_scale_calibration.py
```


