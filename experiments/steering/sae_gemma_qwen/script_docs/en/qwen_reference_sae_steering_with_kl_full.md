# `qwen_reference/scripts_snapshot/sae_steering_with_kl_full.py`

## Status

Qwen reference snapshot of the full SAE steering + KL runner.

## Purpose

Run generation and KL metrics for selected Qwen SAE decoder directions, similar
in spirit to the Gemma full runner.

## What It Does

- uses a base text from `prompts_target[0]`;
- applies selected Qwen SAE decoder directions;
- runs generation tasks;
- records output metrics;
- computes final next-token KL;
- optionally computes teacher-forced KL.

## Important Boundary

This is a Qwen workspace copy. It may use Qwen-specific features, scales, or
tokenization assumptions. Do not use it as the main Gemma runner.

For current Gemma work use:

```text
gemma_active/sae_steering_with_kl_full.py
```


