# `gemma_legacy/sae_feature_steering_light.py`

## Status

Legacy quick-probe script.

## Purpose

Run a small SAE feature steering generation test on a few hand-picked Gemma
features.

## What It Does

- steers selected SAE decoder directions at all residual positions;
- uses a small fixed feature list;
- uses a small scale grid such as `[-3.0, -1.5, 0.0, 1.5, 3.0]`;
- samples multiple generations per feature/scale/task;
- writes basic generated outputs.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
```

Main config:

```python
STEERING_FEATURES
STEERING_SCALES
N_SAMPLES
DO_SAMPLE
TEMPERATURE
MAX_NEW_TOKENS
TEST_TASKS
```

## Limitations

This script does not provide the full current metric stack:

- no full baseline comparison;
- no teacher-forced per-token KL;
- no current calibrated scale handling;
- no richer feature metadata.

## Current Recommendation

Use only for quick sanity checks or old-run interpretation. For serious runs,
use:

```text
gemma_active/sae_steering_with_kl_full.py
gemma_active/fast/sae_steering_with_kl_full_fast.py
```


