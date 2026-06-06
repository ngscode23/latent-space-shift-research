# `gemma_legacy/sae_feature_steering_v2_no_control.py`

## Status

Legacy intermediate diagnostics script.

## Purpose

Extend early SAE feature steering with additional causal diagnostics before the
full KL runner existed.

## What It Does

It can run:

- generation steering;
- next-token KL checks;
- activation patching from target to control;
- unembed projection for decoder directions;
- positional feature activation profiles;
- short-prompt ablation tests for selected features.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
prompts_control  # optional for target-to-control activation patching
```

Important configs:

```python
STEERING_FEATURES
STEERING_SCALES
RUN_GENERATION
RUN_NEXT_TOKEN_KL
RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL
RUN_UNEMBED_PROJECTION
RUN_POSITIONAL_PROFILE
RUN_SHORT_PROMPT_ABLATION
```

## Main Outputs

Typical historical outputs:

```text
sae_feature_steering_generation_with_causal_metrics.csv
sae_feature_unembed_top_tokens.csv
sae_feature_position_activation_profile.csv
sae_feature_position_activation_profile.png
sae_feature_208_short_prompt_ablation.csv
sae_feature_208_short_prompt_ablation_summary.csv
```

## Limitations

This script is useful but not the clean current runner. It predates the current
full generation baseline / teacher-forced KL workflow and the newer candidate
selection/calibration pipeline.

## Current Recommendation

Use only when you specifically need its old unembed or positional-profile
diagnostics. For current generation+KL steering, use:

```text
gemma_active/sae_steering_with_kl_full.py
```


