# latent_attractor_gpu_rapids_analysis.py

Read-only analyzer for hidden-geometry Grade 3 / Grade 4 result packages.

Script:

```text
scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
```

This script does not run a model. It reads an existing result directory or zip
file, audits the artifacts, summarizes numeric CSV/NPZ evidence, builds derived
metric tables, writes plots, and records anomaly rows. It is the analysis layer
after a Grade 3 or Grade 4 run has already produced source artifacts.

## What It Answers

Use it when you have a completed package such as:

```text
red_team_hidden_geometry_results_grade3_*.zip
red_team_hidden_geometry_results_grade4_*.zip
grade4_gemma3_12b_it_sae_l12_36/
```

The analyzer answers:

```text
1. Are the expected artifacts present?
2. Which metric families are available?
3. What is the global and grouped signal by condition/layer/band/component?
4. Does alpha response look monotonic or noisy?
5. Where are the strongest layerwise transitions?
6. Do prompt hidden states form non-X condition clusters?
7. Which Grade 4 component tables and NPZ vectors support the split?
8. Which anomalies/confounds must be read before making a claim?
```

The important distinction:

```text
Grade scripts produce evidence.
This analyzer reads evidence and reorganizes it.
It must not mutate the source package.
```

Current Gemma Grade4 handoff:

```text
Input:
  experiments/grade4_axis_decomposition_gemma/metrics/gemma_full/*.zip

Typical output:
  experiments/grade4_axis_decomposition_gemma/metrics/gemma3_12b_it_gate3_analysis/
```

This analyzer is a post-hoc metric lab. It does not run Gemma/Qwen, does not
capture hidden states, and does not create the original Grade4 source
artifacts. It only reads an existing result package and writes derived summaries,
plots, inventories, and anomaly/confound tables.

## Inputs

Accepted input:

```text
--input path/to/results.zip
--input path/to/unpacked_results_dir
```

The input can contain:

```text
CSV metric tables
NPZ vector/state artifacts
PNG/JPG/SVG plots
JSON manifests
analysis_notes/ quarantine or integrity files
```

Key optional high-value artifacts:

```text
red_team_input_manifest.json
prompt_hidden_states.npz
grade4_axis_component_vectors_by_layer.npz
middle_layer_condition_summary.csv
layerwise_geometry_summary.csv
generation_middle_layer_summary.csv
grade4_axis_projection_geometry_summary.csv
grade4_axis_component_causal_*.csv
sae_*.csv
```

## Quick Start

PowerShell:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "C:\path\to\result_package.zip" `
  --output-dir "metrics\my_run_analysis" `
  --backend auto `
  --strict
```

Directory input:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "C:\path\to\unpacked_result_dir" `
  --output-dir "metrics\my_run_analysis" `
  --backend pandas
```

Colab:

```bash
python /content/latent_attractor_gpu_rapids_analysis.py \
  --input /content/drive/MyDrive/hidden_geometry_runs/grade4_gemma3_12b_it_sae_l12_36.zip \
  --output-dir /content/drive/MyDrive/hidden_geometry_analysis/grade4_gemma3_12b_it_sae_l12_36 \
  --backend auto \
  --strict
```

Small debug run:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "C:\path\to\result_package.zip" `
  --output-dir "metrics\debug_analysis" `
  --limit-files 5 `
  --plots false `
  --strict
```

## Backend

```text
--backend auto    use RAPIDS/cuDF if importable, otherwise pandas
--backend rapids  require cuDF, fail if unavailable
--backend pandas  CPU fallback
```

For RAPIDS on Colab, install a matching CUDA/cuDF stack first. The script has a
pandas fallback, so analysis remains possible without RAPIDS, just slower on
large packages.

Useful cache option when using RAPIDS:

```powershell
--cache-parquet --overwrite-cache
```

This writes selected CSV columns under:

```text
<output-dir>/_work/parquet_cache/
```

## Main Outputs

Core inventory/audit:

```text
analysis_manifest.json
artifact_inventory.csv
artifact_schema_audit.csv
processing_audit.csv
anomaly_flags.csv
index.html
```

Generic metric summaries:

```text
global_metric_summary.csv
grouped_metric_summary.csv
condition_effects.csv
alpha_response_regression.csv
causal_alpha_regression.csv
layerwise_transition_proxy.csv
layerwise_phase_transition.csv
summary_file_numeric_extract.csv
```

NPZ summaries:

```text
npz_inventory.csv
npz_array_summary.csv
npz_layer_norms.csv
npz_adjacent_layer_cosines.csv
grade4_npz_component_geometry.csv
```

Non-X prompt state-space geometry from `prompt_hidden_states.npz`:

```text
state_space_condition_centroids.csv
state_space_condition_distance_matrix.csv
state_space_within_between_variance.csv
state_space_layerwise_pca_summary.csv
state_space_layerwise_pca_coordinates.csv
state_space_non_x_peaks.csv
```

Specialized Grade 3 / Grade 4 matrices:

```text
grade3_geometry_overview.csv
grade3_specificity_control_matrix.csv
grade3_null_baseline_matrix.csv
grade3_causal_symmetry_matrix.csv
grade3_behavior_random_matrix.csv
grade3_architecture_module_matrix.csv
grade3_unit_candidate_matrix.csv
grade3_trajectory_matrix.csv
grade4_component_norm_matrix.csv
grade4_component_projection_matrix.csv
grade4_component_causal_matrix.csv
grade4_component_alpha_matrix.csv
grade4_component_rank_matrix.csv
grade4_axis_cross_correlation.csv
```

Unified machine evidence:

```text
FINAL_DERIVED_METRIC_EVIDENCE.csv
FINAL_LATENT_ATTRACTOR_METRICS.csv
```

Plots are written under:

```text
plots/
plot_manifest.csv
```

## How To Read The Results

Start here:

```text
analysis_manifest.json
anomaly_flags.csv
FINAL_DERIVED_METRIC_EVIDENCE.csv
state_space_non_x_peaks.csv
grade4_component_projection_matrix.csv
grade4_component_causal_matrix.csv
```

Interpretation order:

```text
1. anomaly_flags.csv
   If this has high-severity rows, resolve them before treating metrics as clean.

2. artifact_inventory.csv
   Confirms whether the package is Grade 3, Grade 4, partial, or missing key files.

3. global_metric_summary.csv and grouped_metric_summary.csv
   Broad signal by source file, metric, condition, layer, band, or component.

4. condition_effects.csv
   Condition-minus-baseline gaps. This is where hidden shift strength is easiest
   to scan.

5. alpha_response_regression.csv
   Causal dose-response proxy. Stronger when slope is coherent and R2 is nonzero.

6. state_space_* files
   Non-X geometry: whether conditions separate as state-space regions even
   without projecting onto Vector X.

7. grade4_* matrices
   Content/order decomposition and component causal evidence.
```

## Mechanistic Reading

The analyzer separates three evidence layers:

```text
hidden shift:
  residual-stream geometry, condition centroids, Vector X projections,
  within/between state-space separation

semantic readout:
  generated text, output semantic shift, response trajectory summaries

visible behavior:
  refusal/caution/substitution markers, behavioral-control/random-p95 tables
```

Do not collapse these layers. A run can have strong hidden geometry and weak
visible behavioral control. That is still mechanistically meaningful: it means
the target context organizes internal state without necessarily giving clean
external steering.

## Non-X State-Space Geometry

If the package contains:

```text
prompt_hidden_states.npz
```

the analyzer computes condition centroids, centroid distances,
within/between variance, PCA summaries, selected PCA coordinates, and strongest
condition-pair peaks.

Research meaning:

```text
Vector X is one readout of the shift.
Non-X state-space geometry asks whether the whole prompt endpoint state space
separates by condition even before choosing a projection axis.
```

If `prompt_hidden_states.npz` is missing, the analyzer writes
`not_available_prompt_hidden_states` rows rather than inventing geometry.

## SAE Outputs

If a Grade 4 run produced SAE Lens tables, they are read as normal source CSVs
and appear in generic/global/grouped summaries. The current analyzer does not
treat SAE features as causal by itself. SAE tables remain descriptive
feature-level evidence unless the source package also contains causal or
intervention evidence.

SAE files to inspect manually:

```text
feature_level_interpretability_status.csv
sae_model_compatibility.csv
sae_reconstruction_quality.csv
sae_prompt_feature_delta_summary.csv
sae_grade4_component_feature_summary.csv
sae_generation_feature_summary.csv
```

## Common Commands

Full analysis with plots and HTML:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "result_archives\grade4_run.zip" `
  --output-dir "metrics\grade4_run_analysis" `
  --backend auto `
  --strict
```

Fast CPU/no plots:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "result_archives\grade4_run.zip" `
  --output-dir "metrics\grade4_run_analysis_cpu" `
  --backend pandas `
  --plots false `
  --strict
```

Disable NPZ and state-space analysis:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "result_archives\grade4_run.zip" `
  --output-dir "metrics\csv_only_analysis" `
  --npz-summary false `
  --state-space-summary false
```

## Troubleshooting

`backend=rapids requested but cudf is not importable`

```text
Use --backend pandas, or install a RAPIDS/cuDF build matching your CUDA runtime.
```

`not_available_prompt_hidden_states`

```text
The source package does not include prompt_hidden_states.npz. Vector-X CSVs can
still be analyzed, but non-X state-space geometry is unavailable.
```

`missing_key_grade3_artifact` or `missing_key_grade4_artifact`

```text
The package is partial, from an older script, or a failed run. Treat downstream
claims as incomplete until the missing artifact is explained.
```

High anomaly count:

```text
Read anomaly_flags.csv before the metric summaries. The analyzer deliberately
keeps confounds visible instead of hiding them behind a final verdict.
```

## Verification

Static check:

```powershell
python -m py_compile scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
```

Minimal behavior check:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "path\to\small_or_partial_package" `
  --output-dir "metrics\smoke_analysis" `
  --limit-files 3 `
  --backend pandas `
  --plots false
```

Expected outcome:

```text
process exits 0
analysis_manifest.json exists
artifact_inventory.csv exists
anomaly_flags.csv exists
FINAL_DERIVED_METRIC_EVIDENCE.csv exists or is empty but well-formed
```
