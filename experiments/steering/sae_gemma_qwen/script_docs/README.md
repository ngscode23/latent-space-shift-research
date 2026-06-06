# Script Documentation Index

This folder contains one English documentation file for each Python script in
the sorted SAE / Gemma / Qwen steering workspace.

The script files themselves live in:

```text
experiments/steering/sae_gemma_qwen/gemma_active/
experiments/steering/sae_gemma_qwen/gemma_active/fast/
experiments/steering/sae_gemma_qwen/gemma_legacy/
experiments/steering/sae_gemma_qwen/qwen_reference/scripts_snapshot/
```

## Active Gemma

```text
en/gemma_active_01b_full_sae_evidence_candidate_patching_gemma.md
en/gemma_active_02_scale_calibration.md
en/gemma_active_sae_steering_with_kl_full.md
en/gemma_active_fast_sae_steering_with_kl_full_fast.md
en/gemma_active_x_order_orth_axis_steering_with_kl_full.md
en/gemma_active_gemma_revision_audit.md
```

## Legacy Gemma

```text
en/gemma_legacy_01_candidate_discovery_and_rough_sae_patching.md
en/gemma_legacy_sae_feature_steering_light.md
en/gemma_legacy_sae_feature_steering_v2_no_control.md
en/gemma_legacy_steering_gemma3_V1.md
```

## Qwen Reference

```text
en/qwen_reference_01_candidate_discovery_and_rough_sae_patching.md
en/qwen_reference_02_scale_calibration.md
en/qwen_reference_qwen35_9b_sae_mediation_top_k.md
en/qwen_reference_sae_steering_with_kl_full.md
en/qwen_reference_sae_steering_with_kl_ful_v2l.md
en/qwen_reference_sae_feature_steering_light.md
en/qwen_reference_sae_feature_steering_v2_no_control.md
en/qwen_reference_steering_gemma3_V1.md
```

## Current Recommended Pipeline

For current Gemma work, use the active Gemma scripts in this order:

```text
01b_full_sae_evidence_candidate_patching_gemma.py
02_scale_calibration.py
sae_steering_with_kl_full.py
```

Use the fast runner when the full runner is too slow:

```text
fast/sae_steering_with_kl_full_fast.py
```

Use the dense axis runner when testing the Grade 4 component axis directly:

```text
x_order_orth_axis_steering_with_kl_full.py
```


