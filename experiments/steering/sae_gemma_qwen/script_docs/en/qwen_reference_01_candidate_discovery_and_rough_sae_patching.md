# `qwen_reference/scripts_snapshot/01_candidate_discovery_and_rough_sae_patching.py`

## Status

Qwen reference snapshot. The canonical Qwen workspace remains:

```text
model_workspaces/qwen3_5_9b_qwen_scope/
```

## Purpose

Qwen-adapted candidate discovery / rough patching script, analogous to the
Gemma legacy candidate discovery line.

## What It Is For

- inspecting Qwen SAE candidate features;
- comparing Qwen and Gemma candidate discovery logic;
- interpreting old Qwen SAE runs.

## Main Inputs

Typical inputs:

```python
model
saes
prompts_target
CONTRAST_CSV_PATH
```

But Qwen runs may use different loading infrastructure than Gemma-Scope,
depending on the exact script version.

## Important Boundary

Do not treat this as a Gemma script. Qwen-Scope and Gemma-Scope have different
SAE storage/loading conventions.


