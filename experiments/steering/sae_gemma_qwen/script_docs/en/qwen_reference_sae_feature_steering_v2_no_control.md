# `qwen_reference/scripts_snapshot/sae_feature_steering_v2_no_control.py`

## Status

Qwen reference copy of the older intermediate steering/diagnostic script.

## Purpose

Historical Qwen-side analogue of the Gemma `sae_feature_steering_v2_no_control`
script.

## What It May Contain

- generation steering;
- next-token KL checks;
- unembed projection;
- positional activation profile;
- optional short-prompt ablation logic.

## Current Recommendation

Use only for historical comparison or if you specifically need one of its old
diagnostic blocks. For new Gemma runs, use the active Gemma full runner. For new
Qwen runs, prefer the canonical Qwen workspace scripts.


