# `qwen_reference/scripts_snapshot/sae_steering_with_kl_ful_v2l.py`

## Status

Qwen-specific self-contained SAE steering runner.

## Purpose

Run Qwen SAE feature steering with its own model/tokenizer/SAE loading logic.

## What It Does

- loads `Qwen/Qwen3.5-9B-Base` via `transformers`;
- loads Qwen-Scope TopK SAE checkpoints;
- supports chat-template controls;
- disables or strips thinking tags when configured;
- applies selected Qwen SAE decoder directions;
- records generation and KL metrics.

## Key Configs

```python
MODEL_NAME
SAE_RELEASE
USE_CHAT_TEMPLATE
DISABLE_THINKING
STRICT_DISABLE_THINKING
STRIP_THINKING_FROM_OUTPUT
STEERING_FEATURES
STEERING_SCALES
TEST_TASKS
GENERATION_MODES
```

## Important Boundary

This script is not a Gemma script. It is useful for Qwen steering and for
comparing Qwen/Gemma behavior, but it should not be mixed into the Gemma
pipeline.


