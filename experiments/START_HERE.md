# START HERE: active research map

This is the first file to open when returning to the project.

The current main line is:

```text
Grade4 hidden geometry
-> SAE feature candidate evidence
-> SAE scale calibration
-> SAE decoder-direction steering / KL
-> optional dense x_order_orth axis steering
```

COAST / POS-NEG conceptor work is archived separately and is not the main
pipeline.

## Folder Map

```text
research_synthesis/
  Meaning layer: conclusions, claim boundaries, outreach drafts, paper/readout
  documents, and the current research narrative.

experiments/
  Real experimental code and local run artifacts.

experiments/grade4_axis_decomposition_gemma/
  Gemma Grade4 hidden-state geometry script, component-axis outputs, SAE tables,
  and Grade4 metric packages.

experiments/steering/sae_gemma_qwen/
  SAE feature discovery, Gemma SAE steering, dense x_order_orth axis steering,
  legacy steering scripts, and Qwen reference snapshots.

scripts/analysis_tools/
  Post-hoc analyzers. These read existing ZIP/result folders and build metric
  summaries, plots, and anomaly tables. They do not run the model.

archive/
  Closed or side branches. COAST / POS-NEG conceptor work lives here.

result_archives/
  Raw or packaged result archives that are kept for traceability.
```

Old root-level folders such as `grade4_axis_decomposition/` and `steering/`
are pointer folders only. The real active code is under `experiments/`.

## Golden Path

### 1. Grade4 hidden geometry

Script:

```text
experiments/grade4_axis_decomposition_gemma/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Purpose:

```text
Capture hidden states, build condition-delta axes, compare target/control/shuffle
conditions, produce Grade4 component tables, SAE readout tables, and packaged
metric artifacts.
```

Important outputs inside the Grade4 package:

```text
sae_order_feature_contrast.csv
grade4_axis_component_vectors_by_layer.npz
grade4_axis_component_*csv
layerwise_geometry_*csv
generation_trajectory_*csv
red_team_hidden_geometry_verdict.md
```

Read:

```text
experiments/grade4_axis_decomposition_gemma/RUNBOOK.md
experiments/grade4_axis_decomposition_gemma/metrics/README.md
```

### 2. Metric-lab analysis of a Grade4 ZIP

Script:

```text
scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
```

Purpose:

```text
Read an already-created Grade4 ZIP or directory, audit artifacts, summarize CSV
and NPZ evidence, write derived metric tables and plots, and record anomalies.
```

This script does not run the model and does not create source hidden states.

Typical input:

```text
experiments/grade4_axis_decomposition_gemma/metrics/gemma_full/*.zip
```

Typical output:

```text
experiments/grade4_axis_decomposition_gemma/metrics/gemma3_12b_it_gate3_analysis/
```

### 3. SAE feature candidate evidence

Script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

Purpose:

```text
Read the Grade4 SAE tables from the ZIP, use sae_order_feature_contrast.csv as
the primary ranking table, add supporting SAE evidence, select feature
candidates, optionally run rough zero-ablation and top-context inspection.
```

Core Colab variables:

```python
SAE_TABLE_ZIP_PATH = "/content/<grade4_package>.zip"
PATCH_PROMPTS = prompts_target
```

Outputs include:

```text
selected_sae_order_candidates.csv
ranked_sae_order_candidates_full_evidence.csv
rough_sae_zero_ablation_logit_results.csv
sae_feature_top_activating_contexts.csv
summary.md
```

### 4. SAE scale calibration

Script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/02_scale_calibration.py
```

Purpose:

```text
Convert selected SAE features into sane intervention scales. This matters
because scale=1 for a Gemma SAE decoder direction is often tiny relative to the
residual-stream norm, effectively a fraction-of-a-percent perturbation or pure
noise.
```

Inputs expected in the notebook:

```python
model
saes
prompts_target
STEERING_FEATURES
```

Outputs:

```text
sae_scale_calibration.csv
sae_scale_calibration_kl_check.csv
printed RECOMMENDED_SCALES_BY_FEATURE / STEERING_SCALES
```

### 5. SAE decoder-direction steering

Preferred fast script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

Purpose:

```text
Apply scale * SAE_decoder_direction(feature) to residual-stream hooks, generate
answers, measure output changes, final next-token KL/logit changes, and optional
teacher-forced per-token KL.
```

Typical outputs:

```text
sae_feature_steering_generation_full_metrics_*.csv
sae_feature_steering_generation_summary_metrics_*.csv
sae_feature_steering_generation_full_metrics_with_tf_kl_*.csv
sae_teacher_forced_per_token_kl_details_*.csv
sae_teacher_forced_kl_summary_by_feature_scale_*.csv
sae_feature_steering_base_text_*.txt
```

Local collected ZIPs live in:

```text
experiments/steering/metrics/
```

### 6. Optional dense x_order_orth axis steering

Script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

Purpose:

```text
Steer directly along the dense Grade4 residual-stream axis, usually
x_order_orth, instead of individual SAE feature decoder directions.
```

Important distinction:

```text
SAE steering uses sparse feature decoder directions.
Dense axis steering uses Grade4 component vectors from
grade4_axis_component_vectors_by_layer.npz.
```

## Where To Read Conclusions

Main interpretation hub:

```text
research_synthesis/
```

Important current documents:

```text
research_context_current.md
research_synthesis/document_index.md
research_synthesis/gemma3_grade4_sae_academic_readout/context_induced_latent_state_shift_final_conclusion_ru.md
research_synthesis/geometry_coordinate_evidence_package/grade4_geometry_to_sae_steering_unified_readout_ru.md
```

## Claim Boundary

Current established result:

```text
Coherent target context induces a measurable temporary inference-time
hidden-state shift in Gemma3-12B-IT. The shift is not reducible to content/word
overlap because sentence-shuffle separates into x_content while coherent target
separates into x_order_orth.
```

Current causal boundary:

```text
Descriptive latent shift is strong. Content/order separability is strong.
Causal involvement is supported. Stable bidirectional x_order_orth behavioral
control is not proven.
```

