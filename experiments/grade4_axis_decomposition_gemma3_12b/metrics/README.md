# Grade4 Gemma Metrics

This folder contains metric packages and analysis outputs for the Grade4
hidden-state geometry work.

These are source metrics and post-hoc analysis results. They are not SAE
steering runs.

## Main Source Script

The source Grade4 metrics are produced by:

```text
experiments/grade4_axis_decomposition_gemma/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

That script captures hidden states, compares target/control/shuffle conditions,
constructs Grade4 component axes, computes causal/intervention summaries, and
writes SAE readout tables.

## Main Gemma Package

The primary Gemma packages live under:

```text
gemma_full/
```

Current important ZIPs:

```text
grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip
grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale_metric_lab.zip
```

The non-`metric_lab` ZIP is the direct Grade4 source package. The `metric_lab`
ZIP is the package after post-hoc metric analysis.

## High-Value Files Inside A Grade4 ZIP

These files are the main bridge from Grade4 geometry into SAE and steering:

```text
sae_order_feature_contrast.csv
grade4_axis_component_vectors_by_layer.npz
grade4_axis_component_*csv
layerwise_geometry_*csv
generation_trajectory_*csv
red_team_hidden_geometry_verdict.md
```

How to use them:

```text
sae_order_feature_contrast.csv
  Main SAE feature ranking table. This is the primary input for:
  experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py

grade4_axis_component_vectors_by_layer.npz
  Dense component vectors by layer. This is the main input for:
  experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py

grade4_axis_component_*csv
  Component projection, norm, causal, alpha, and rank summaries.

layerwise_geometry_*csv
  Layerwise hidden-state geometry and target/control/shuffle separation.

generation_trajectory_*csv
  Generation-time trajectory metrics.

red_team_hidden_geometry_verdict.md
  Human-readable verdict and run interpretation produced by the Grade4 script.
```

## Post-Hoc Metric Analysis

The folder:

```text
gemma3_12b_it_gate3_analysis/
```

is already a post-hoc analysis output. It is produced by:

```text
scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
```

That analyzer reads an existing Grade4 ZIP or unpacked result directory. It
does not run the model and does not create the original hidden states.

Useful files in an analysis folder:

```text
analysis_summary.md
analysis_summary.json
source_file_inventory.csv
scoreboard_row.csv
peak_tables/anomaly_flags.csv
peak_tables/geometry_peaks.csv
peak_tables/component_peaks.csv
peak_tables/causal_peaks.csv
peak_tables/specificity_peaks.csv
```

## Qwen Folders

Folders such as:

```text
qwen_full/
qwen3_14b_breakthrough_grade_hardened/
qwen3_14b_grade4_axis_decomposition02/
qwen3_14b_grade4_axis_decomposition03/
```

are kept for cross-model context and historical comparison. The current Gemma
mainline starts from `gemma_full/`.

## Reading Order

1. Read `gemma_full/` package inventory or unpack the direct Grade4 ZIP.
2. Inspect `red_team_hidden_geometry_verdict.md`.
3. Inspect `sae_order_feature_contrast.csv` and component CSVs.
4. Use `gemma3_12b_it_gate3_analysis/analysis_summary.md` for post-hoc metric
   interpretation.
5. Hand off the Grade4 ZIP to SAE candidate discovery and steering.

