# SAE Steering Metrics

This folder contains completed SAE steering run packages.

These are not Grade4 source metrics. They are downstream steering results that
consume selected SAE features and calibrated scales.

## Source Scripts

The current preferred runner is:

```text
experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

The full non-fast runner is:

```text
experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
```

Both scripts apply SAE decoder-direction interventions:

```text
residual_stream += scale * sae.W_dec[feature]
```

and then record generated outputs, text metrics, baseline comparisons, final
next-token KL/logit shifts, and optional teacher-forced per-token KL.

## Current ZIPs

```text
01_gemma_sae_steering_movers_3tasks.zip
gemma_sae_steering_fast_readout_3tasks.zip
```

Interpretation:

```text
01_gemma_sae_steering_movers_3tasks.zip
  SAE decoder-direction steering over the first/mover feature group.

gemma_sae_steering_fast_readout_3tasks.zip
  Fast SAE decoder-direction steering over the readout/probe feature group.
```

Exact feature lists and scales are recorded inside each ZIP in the generated CSV
rows and/or run logs.

## Expected Files Inside A Steering ZIP

```text
sae_feature_steering_generation_full_metrics_*.csv
sae_feature_steering_generation_summary_metrics_*.csv
sae_feature_steering_generation_full_metrics_with_tf_kl_*.csv
sae_teacher_forced_per_token_kl_details_*.csv
sae_teacher_forced_kl_summary_by_feature_scale_*.csv
sae_feature_steering_base_text_*.txt
```

How to read them:

```text
sae_feature_steering_generation_full_metrics_*.csv
  Main table. One row per task / feature / scale / generation mode / sample.
  Contains the actual output text, baseline comparison, final next-token KL,
  logit shift, top-token-change flags, and text metrics.

sae_feature_steering_generation_summary_metrics_*.csv
  Group summary by feature, scale, generation mode, and sometimes task/base.

sae_feature_steering_generation_full_metrics_with_tf_kl_*.csv
  Main table joined with teacher-forced KL summaries.

sae_teacher_forced_per_token_kl_details_*.csv
  Token-localized effect table. Use this when asking where the intervention
  changes the distribution over the baseline continuation.

sae_teacher_forced_kl_summary_by_feature_scale_*.csv
  Compact teacher-forced KL summary by feature and scale.

sae_feature_steering_base_text_*.txt
  Full base text removed from the CSV to keep tables smaller and easier to
  inspect with AI-assisted analysis.
```

## How These Metrics Fit The Pipeline

Upstream:

```text
Grade4 package
-> sae_order_feature_contrast.csv
-> 01b_full_sae_evidence_candidate_patching_gemma.py
-> selected SAE features
-> 02_scale_calibration.py
-> STEERING_SCALES
```

This folder:

```text
STEERING_FEATURES + STEERING_SCALES
-> sae_steering_with_kl_full_fast.py
-> steering ZIPs and CSV metrics
```

Downstream:

```text
Read outputs and KL tables
-> update research_synthesis/
-> decide whether the SAE feature is a behavioral mover, distributional mover,
   weak readout, or likely surface correlate.
```

## Do Not Confuse With Dense Axis Steering

SAE steering uses sparse feature decoder directions.

Dense axis steering uses:

```text
grade4_axis_component_vectors_by_layer.npz
```

and is run by:

```text
experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

