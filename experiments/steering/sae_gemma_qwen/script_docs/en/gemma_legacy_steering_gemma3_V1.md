# `gemma_legacy/steering_gemma3_V1.py`

## Status

Legacy first-generation Gemma SAE steering runner.

## Purpose

Run early SAE decoder-direction steering tests for Gemma3-12B-IT with a fixed
base text, fixed feature list, and basic output metrics.

## What It Does

- uses `prompts_target[0]` as `BASE_TEXT`;
- steers a small set of late SAE features;
- runs multiple tasks over several scale values;
- records generated outputs and basic text metrics;
- compares outputs to baseline scale `0.0`.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
```

Typical feature set:

```python
(41, 13686)
(41, 208)
(41, 207)
```

## Limitations

This script uses older scale assumptions and does not include the current full
teacher-forced KL metric layer. It is useful because older generated outputs may
refer to it, but it is not the current preferred runner.

## Current Recommendation

Use only to reproduce or interpret old runs. For new runs use:

```text
gemma_active/sae_steering_with_kl_full.py
```


