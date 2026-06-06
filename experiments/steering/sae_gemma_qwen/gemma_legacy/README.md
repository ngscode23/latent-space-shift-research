# Gemma Legacy SAE Steering Scripts

These scripts are preserved for audit trail and historical comparison. They are
not the preferred current pipeline.

Use current active scripts instead:

```text
../gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
../gemma_active/02_scale_calibration.py
../gemma_active/sae_steering_with_kl_full.py
../gemma_active/fast/sae_steering_with_kl_full_fast.py
../gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

## Files

```text
01_candidate_discovery_and_rough_sae_patching.py
```

Original candidate discovery / rough patching script. It reads mainly
`sae_order_feature_contrast.csv`. Superseded by `01b`, which reads the full
Grade 4 SAE table package.

```text
sae_feature_steering_light.py
```

Small early generation steering probe. Useful only as a historical sanity
check.

```text
sae_feature_steering_v2_no_control.py
```

Intermediate script with generation, next-token KL, unembed projection,
positional profile, and optional short-prompt ablation. Superseded by the full
KL runner.

```text
steering_gemma3_V1.py
```

First older Gemma steering runner. Preserved because old outputs may refer to
it.


