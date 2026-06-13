# Current Research Context

This is the compact handoff file and the active place for new research memory.
The old long historical context is archived at `archive/research_context_anchor.md`.

## Current Claim

We are not using formal "attractor" language as the main claim.

## 2026-06-13 Base-vs-Instruct Geometry/Probability Run 02

Artifact:

```text
C:\Users\stasv\Downloads\alignment_geometry_probability_run_02.zip
```

Script:

```text
experiments/variance_compression_finding/base_vs_instruct_geometry_probability_audit.py
```

Run shape:

```text
base_model     = google/gemma-3-12b-pt
instruct_model = google/gemma-3-12b-it
prompt_mode    = raw
include_shuffles = true
target contexts = 1
control contexts = 1
questions = 10
prompts = 50 per model
hidden shape = (50, 49, 3840) for both base and instruct
```

Integrity:

```text
Both hidden_last_token_base.npz and hidden_last_token_instruct.npz exist,
shape is correct, dtype float32, no NaNs. Metrics are computed from in-memory
hidden arrays after saving NPZ; NPZ is an artifact, not a reread dependency.
```

Main result:

```text
The naive claim "instruction/alignment globally compresses late hidden-state
geometry" is not supported by this run. Instruct has higher late residual
relative dispersion than base across all conditions, and higher effective rank
in the L30-L47 residual band.

However, instruct has much more concentrated next-token distributions at the
prompt boundary: entropy is strongly lower and top1 probability is higher
across all conditions. This supports a probability-distribution sharpening /
commitment effect more than a simple hidden-geometry compression effect.
```

Important layer note:

```text
The generated late_band_summary uses L30-L48 by default. Layer 48 has a very
different centroid norm from L30-L47, likely final normalization / final hidden
state rather than ordinary residual accumulation. For residual-stream geometry
comparisons, prefer L30-L47. The L30-L47 recalculation preserves the same main
reading: instruct > base rel_disp/effective-rank, while logits are much more
concentrated.
```

L30-L47 residual-band summary:

```text
target:
  base rel_disp ~= 0.18030
  instruct rel_disp ~= 0.19073
  delta ~= +0.01043
  base effective rank ~= 2.142
  instruct effective rank ~= 2.830

control:
  base rel_disp ~= 0.18226
  instruct rel_disp ~= 0.20179
  delta ~= +0.01952
  base effective rank ~= 2.163
  instruct effective rank ~= 2.522

question_only:
  base rel_disp ~= 0.19716
  instruct rel_disp ~= 0.20185
  delta ~= +0.00469
  base effective rank ~= 1.791
  instruct effective rank ~= 3.080
```

Prompt-boundary probability result:

```text
question_only entropy:
  base ~= 2.928
  instruct ~= 0.912

target entropy:
  base ~= 2.693
  instruct ~= 1.482

control entropy:
  base ~= 2.815
  instruct ~= 1.306

question_only top1 probability:
  base ~= 0.419
  instruct ~= 0.816

target top1 probability:
  base ~= 0.430
  instruct ~= 0.654

control top1 probability:
  base ~= 0.429
  instruct ~= 0.697
```

Mechanistic reading:

```text
This run weakens the simple "alignment = hidden-state variance compression"
story. It strengthens a more precise story:

instruction tuning / alignment reorganizes the relation between hidden geometry
and output probabilities. The instruct model can occupy a broader/higher-rank
late hidden geometry while projecting to a sharper, lower-entropy next-token
distribution.

So the better object may be:
  hidden-state geometry -> probability commitment map
rather than only:
  hidden-state geometry compression.
```

Expanded metric reading for L30-L47:

```text
centroid_norm:
  base > instruct across target/control/question_only.
  target:        124817 -> 107665
  control:       122917 -> 104514
  question_only: 127501 ->  97001

  This matters because instruct's sharper next-token distribution is not caused
  by a larger hidden vector magnitude. The probability commitment appears with
  a smaller late hidden centroid.

abs_disp_l2_mean and cov_trace:
  absolute spread around the centroid is lower for instruct in most cells.
  question_only cov_trace drops from 8.25e8 to 5.49e8.
  This supports absolute-scale compression, but only in the absolute metric.

rel_disp_l2_mean:
  relative spread is higher for instruct:
    target        0.1803 -> 0.1907
    control       0.1823 -> 0.2018
    question_only 0.1972 -> 0.2019
  So after normalizing by centroid size, instruct is not more compact.

pairwise_cosine_distance_mean and angular_disp_to_centroid:
  angular spread is higher for instruct:
    target cosine distance        0.0191 -> 0.0272
    control cosine distance       0.0195 -> 0.0291
    question_only cosine distance 0.0195 -> 0.0301
  The instruct cloud is more angularly dispersed around its smaller centroid.

effective_rank_pr / spectral_entropy_norm:
  covariance dimensionality is higher for instruct:
    target effective rank        2.142 -> 2.830
    control effective rank       2.163 -> 2.522
    question_only effective rank 1.791 -> 3.080
  Variance is spread across more directions, not collapsed into one axis.

top1_pc_variance_share:
  first-PC dominance is lower for instruct:
    target        0.670 -> 0.564
    control       0.671 -> 0.610
    question_only 0.746 -> 0.545
  This agrees with the higher rank/entropy story.
```

Compact conclusion:

```text
The hidden geometry result is mixed-scale:
  absolute magnitude/variance scale: instruct smaller
  relative/angular/rank structure: instruct broader and higher-dimensional
  probability readout: instruct much sharper

Therefore the current hypothesis should be revised from:
  "alignment simply compresses hidden-state dispersion"
to:
  "instruction tuning reduces absolute hidden-state scale while making the
   hidden-to-logit readout more probability-committed / stiff."
```

Boundary:

```text
This run used raw prompts, not Gemma chat template. It is a clean identical-input
base-vs-instruct comparison, but not the normal deployment format for the
instruct model. It used 1 target context and 1 control context with 10 questions,
so it is a small controlled run, not a final multi-bank replication.
```

## Gemma3 Grade 4 Final Synthesis

New Russian final conclusion document:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/context_induced_latent_state_shift_final_conclusion_ru.md
```

Deep mechanistic roadmap for the next stage:

```text
research_synthesis/deep_mechanistic_roadmap_ru.md
```

Current next-stage focus:

```text
Move from dense hidden-state coordinates to mechanism:
cross-model Qwen3.5-9B SAE replication, layer birth localization,
token/fragment attribution, SAE sparse-feature mechanism, module route,
path-dependence, and behavior/KL coupling.
```

Current verdict recorded there:

```text
Gemma-3-12B-IT shows a context-induced latent-state shift from coherent target
text. The shift is separable from shuffled-content controls and contains a
large x_order_orth order/structure component. Norm-controlled component-causal
testing supports causal involvement/sensitivity of the component directions,
but does not prove x_order_orth as a stable bidirectional steering axis.
Next run: natural-scale norm-controlled component causality.
```

Supported wording:

```text
Context-conditioned latent geometry/readout shift.
Qwen3-14B has a robust target-conditioned internal Vector X.
In Grade 3, middle-layer residual-stream +X/-X intervention causally steers
the generation-time hidden trajectory.
In Grade 4, that Vector X decomposes into a large content component plus a
separable discourse-order / rhetorical-regime component.
```

Not claimed:

```text
permanent weight/topology change
formal attractor basin
reviewer-grade visible behavioral control
production bypass
```

## Multi-Model Protocol State

Multi-model testing should now be preflight-first, not "change MODEL_ID and
hope".

New compatibility layer:

```text
scripts/hidden_geometry/common/model_registry.py
scripts/hidden_geometry/common/model_compat.py
scripts/hidden_geometry/common/preflight_probe.py
scripts/hidden_geometry/common/README.md
```

Canonical experiment scripts remain:

```text
scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Required clean-run checks for each new model:

```text
decoder_layer_count > 0
decoder_layer_count == expected_decoder_layer_count
decoder_layer_count_mismatch == false
prompt budget clean for target/neutral/questions
module hooks fire if architecture/neuron evidence is used
causal_intervention_status.csv must not report not_run_no_decoder_layers_found
```

Preflight command shape:

```powershell
python scripts/hidden_geometry/common/preflight_probe.py `
  --profile gemma3_12b_it `
  --load-model `
  --target-file target.txt `
  --neutral-file neutral.txt `
  --questions-file questions.txt `
  --results-dir hidden_geometry_preflight_results/gemma3_12b_it_full
```

## Evidence Sanitizer State

Gate 3 and Gate 4 clean-evidence sanitizer is now quarantine-based.

Current behavior:

```text
raw_measurement / audit / response-audit CSV:
  preserve raw text and raw rows;
  only add artifact_type;
  do not mask forbidden labels inside raw model output.

derived_metric / threshold_eval / proxy_metric CSV:
  remove narrative/verdict columns from main evidence;
  route removed values to analysis_notes/extracted_narrative_columns/;
  convert machine-readable reason values to failure_code;
  quarantine human-readable reason text;
  quarantine forbidden verdict labels instead of silently replacing them.
```

New audit artifacts:

```text
analysis_notes/extracted_narrative_columns/quarantine_index.csv
analysis_notes/extracted_narrative_columns/numeric_integrity_check.csv
```

Numeric integrity rule:

```text
Any numeric metric column present before and after cleaning must be identical.
If a numeric value changes during cleaning, the run fails.
```

## Result Analyzer State

Implemented read-only result package analyzer:

```text
scripts/hidden_geometry/common/analyze_result_package.py
```

Purpose:

```text
Read a completed Grade 3 / Grade 4 zip or folder, leave the source package
unchanged, and write an external audit bundle with validity flags, primary
metrics, peak tables, anomaly flags, and a one-row cross-model scoreboard.
```

Command shape:

```powershell
python scripts/hidden_geometry/common/analyze_result_package.py `
  --results C:\path\to\result.zip `
  --out metrics\some_run_analysis `
  --run-label some_run
```

Outputs:

```text
analysis_summary.md
analysis_summary.json
scoreboard_row.csv
source_file_inventory.csv
peak_tables/geometry_peaks.csv
peak_tables/specificity_peaks.csv
peak_tables/component_peaks.csv
peak_tables/causal_peaks.csv
peak_tables/behavior_peaks.csv
peak_tables/architecture_peaks.csv
peak_tables/anomaly_flags.csv
```

Smoke test already run on:

```text
C:\Users\stasv\Downloads\red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip
```

Analyzer output:

```text
metrics\gemma3_12b_it_gate3_analysis
```

Key smoke result:

```text
valid_package=true
decoder_ok=true
geometry_pass=true
specificity_pass=true
strict_causal_symmetry_pass=false
behavior_random_p95_pass=false
main_failure_code=below_threshold;behavior_p95_metric_mismatch
recommended_next_experiment=run_gate4_axis_decomposition
```

## Analyzer Scope Guard

Important framing rule:

```text
Do not collapse Grade 3 / Grade 4 result analysis into "Vector X only".
Vector X is a measurement / causal-handle object, not the whole phenomenon.
```

Current analyzer status:

```text
scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
is the primary metric-lab analyzer. It reads broad CSV/NPZ evidence, not only
Vector-X columns: layerwise l2/cosine/norm geometry, controls, null baselines,
generation trajectory, behavioral proxies, semantic-shift proxies, architecture
module/unit deltas, Grade 4 component tables, and NPZ array summaries.
```

Implemented analyzer expansion:

```text
2026-05-29:
The primary metric-lab analyzer now has an explicit non-X state-space layer
over prompt_hidden_states.npz. It writes condition centroids, condition
distance matrices, within/between variance, layerwise PCA summaries and
coordinates, non-X peak separations, and state-space plots.

This closes the immediate analyzer risk that broad hidden-geometry packages
could be read as Vector-X-only evidence.

Smoke on Gemma Gate 3 zip:
  py_compile passed
  real package smoke exit_code=0
  state_space_* CSVs written
  plot_manifest.csv includes the four state-space plots
  FINAL_DERIVED_METRIC_EVIDENCE.csv includes non_x_state_space_geometry rows
  partial no-NPZ package exits cleanly with not_available_prompt_hidden_states
```

Interpretation rule:

```text
Use Vector X and x_full/x_content/x_order/x_order_orth as mechanistic handles
inside the broader context-induced latent regime formation program. Always
inspect non-X geometry, trajectory dynamics, architecture deltas, controls,
nulls, and behavior/semantic readouts before writing claims.
```

## Gemma3 Gate 3 Result

Source:

```text
C:\Users\stasv\Downloads\red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip
```

Compatibility:

```text
model_id: google/gemma-3-12b-it
decoder_layer_source: model.language_model.layers
decoder_layer_count: 48
expected_decoder_layer_count: 48
decoder_layer_count_mismatch: false
causal_intervention_status.csv: absent
architecture_module_delta_summary.csv: non-empty, 12000 rows
```

Main result:

```text
Gemma3-12B-IT cleanly replicates Level 1 geometry and Level 2 specificity, but
does not pass strict causal symmetry / behavioral steering gates.
```

Key metrics:

```text
target middle projection mean: 0.934655
target middle direction cosine: 0.612155
random null mean: 0.000045
target - word shuffle projection: 0.382052, p=0.002300, FDR significant
target - sentence shuffle projection: 1.392150, p=0.002300, FDR significant
target - length-matched neutral projection: 0.912941, p=0.002300, FDR significant
```

Interpretation:

```text
The latent target axis is not Qwen-only. It appears in Gemma3 under clean
decoder-layer access and strong controls. However, Gemma does not yet support
the stronger claim that Vector X robustly gives reviewer-grade causal symmetry
or visible behavioral control.
```

Important scoring note:

```text
The current Gemma zip contains a behavioral threshold row named
plus_x_beats_random_p95 whose metric value came from random-mean lift, not p95
lift. Do not use that row as behavioral support. The canonical Gate 3 / Gate 4
scripts have been patched so future runs use mean_lift_over_random_p95 for this
criterion.
```

## Grade 3 Result

Source:

```text
metrics/qwen3_14b_breakthrough_grade_hardened/
```

Main internal metrics:

```text
target middle projection mean: 0.976583
target middle direction cosine: 0.852397
positive projection fraction: 1.0
random same-norm null mean: ~0.000040
middle +X/-X causal gap grows monotonically with alpha
```

Supported measurement read:

```text
target-conditioned causal internal Vector X in Qwen3-14B middle residual stream
```

Boundary:

```text
visible behavioral steering did not beat alpha-matched random p95.
```

## Grade 4 Result

Source:

```text
C:\Users\stasv\Downloads\red_team_hidden_geometry_results_breakthrough_grade4_axis_decomposition03.zip
```

Interpreted metrics:

```text
metrics/qwen3_14b_grade4_axis_decomposition03/
```

Supported measurement read:

```text
x_order_orth remains separable from sentence-shuffled content and causally steerable
```

Key metrics:

```text
target projection on x_order_orth:              0.978944
sentence_shuffle projection on x_order_orth:    0.007214

middle/middle alpha 0.75 gaps:
neutral x_order_orth:                           3.726561
neutral x_order:                                3.384538
neutral x_full:                                 3.308553
neutral x_content:                              2.990294

target x_order_orth:                            3.698789
target x_order:                                 3.383840
target x_full:                                  3.330993
target x_content:                               2.997980
```

Meaning:

```text
x_order_orth survives removal of the content projection and remains the
strongest middle/middle causal component. The Grade 3 Vector X is not merely a
sentence-shuffled content axis; it contains a separable discourse-order /
rhetorical-regime component.
```

Boundary:

```text
Still not a permanent topology change, formal attractor basin, reviewer-grade
visible behavioral control, or cross-model universality claim.
```

## Grade 4 Question

Grade 4 is not a repeat of Grade 3. It decomposes Vector X:

```text
X_full       = target - neutral
X_content    = sentence_shuffle(target) - neutral
X_order      = target - sentence_shuffle(target)
X_order_orth = X_order after removing layerwise X_content projection
```

Main question:

```text
Does X_order_orth keep a stable alpha-scaled +component/-component causal gap?
```

If yes:

```text
Vector X contains a separable discourse-order / rhetorical-regime component,
not only a lexical/semantic target-family trace.
```

If no:

```text
Grade 3 mostly found a target-family content axis with smaller coherent-order
residue.
```

## Active Grade 4 Files

Workspace script:

```text
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Archived Downloads copies:

```text
scripts/hidden_geometry/grade4_variants/from_downloads/
```

Root-level Grade 4 variant moved here:

```text
scripts/hidden_geometry/grade4_variants/red_team_hidden_geometry_grade4_axis_decomposition_memory_safe_fixed (1).py
```

Canonical Grade 3 script moved here:

```text
scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py
```

Active canonical Grade 4 defaults after the completed 03 run:

```python
RESULTS_DIR = Path("red_team_hidden_geometry_results_grade4_axis_decomposition")
RUN_LABEL = "grade4_axis_decomposition"

CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False

GRADE4_COMPONENT_CAUSAL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_AXES = ["x_full", "x_content", "x_order", "x_order_orth"]
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late", "all"]
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES = [0.10, 0.25, 0.50, 0.75]
GRADE4_COMPONENT_CAUSAL_GENERATION_BATCH_SIZE = CAUSAL_GENERATION_BATCH_SIZE
GRADE4_COMPONENT_CAUSAL_SAVE_STEP_RAW = False
```

The completed `03.zip` run used all three bands and finished. The key System
RAM fix is `SAVE_STEP_RAW=False` plus streaming generation/post-processing. If
a future environment is tighter, reduce only
`GRADE4_COMPONENT_CAUSAL_GENERATION_BATCH_SIZE` first.

## Grade 4 Artifacts To Read First

```text
grade4_axis_component_norm_summary.csv
grade4_axis_projection_geometry_summary.csv
grade4_axis_component_causal_projection_summary.csv
grade4_axis_component_causal_symmetry_summary.csv
grade4_axis_component_causal_alpha_scaling_summary.csv
grade4_axis_component_causal_rank_summary.csv
claim_ladder_final.csv
```

## Report-Framing Observation

Legacy markdown verdicts can become active narrative anchors. Clean evidence
scripts now keep autogenerated interpretation out of root result packages.

Observation:

```text
When a report headline says "not proven / not supported", downstream model
analysis may defend a weak/no-result interpretation even when metrics show a
strong internal Vector-X axis and causal +X/-X movement.
```

Correct report shape:

```text
Supported:
causal internal latent axis in middle residual stream.

Not supported:
permanent topology change, formal basin, reviewer-grade visible behavioral
control.
```

Future test:

```text
same metrics, same evaluator, different markdown verdict frames
```

## Where Outputs Go

Metric package:

```text
research_synthesis/latent_shift_package_current/
```

Current claim package:

```text
research_synthesis/core_claim_package_ru.md
research_synthesis/next_metric_collection_plan_ru.md
research_synthesis/paper_draft_ru.md
research_synthesis/evidence_matrix_ru.md
research_synthesis/report_outline_ru.md
```

Collector:

```text
research_synthesis/collect_research_metrics.py
```

Runbook:

```text
research_synthesis/RUNBOOK_ru.md
```

Protocol:

```text
research_synthesis/METRIC_REPORTING_PROTOCOL_ru.md
```

Current next step:

```text
Do not rerun Qwen3-14B Grade 4 now. Use the core claim package as the working
evidence spine, then collect new metrics only after a fresh broad run or a
cross-model Grade 3/Grade 4 replication.
```

Fresh broad run slot:

```text
scripts/main_runners/llm_attractor_colab_copy_paste.py
RESULTS_DIR = attractor_results_qwen3_14b_original_core_fresh_2026_05_25
collector output = research_synthesis/latent_shift_package_current/attractor_run_summary.csv
purpose = Level A broad latent/readout evidence from zero, not a replacement
for Grade 3/Grade 4 causal/component evidence.
```

## Clean Evidence Script Defaults

Canonical Grade 3 and Grade 4 scripts now default to evidence-only output:

```text
scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py
  RESULTS_DIR = red_team_hidden_geometry_results_grade3_hidden_geometry
  RUN_LABEL = grade3_hidden_geometry

experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
  RESULTS_DIR = red_team_hidden_geometry_results_grade4_axis_decomposition
  RUN_LABEL = grade4_axis_decomposition
```

Evidence hygiene:

```text
EMIT_NARRATIVE = False
verdict/narrative files are not written to root RESULTS_DIR
CSV outputs get artifact_type
behavioral_control_axis_threshold_eval.csv replaces behavioral verdict CSVs
metric_math_reference.md is redirected to docs/
research meta questions are excluded from the core question set
response replacement-char counter uses "\ufffd", not spaces
validate_evidence_package_schema(results_dir) runs before completion
```

The old interpreted metric folders remain historical evidence packages; new
runs from these scripts should produce clean machine-readable evidence.

## Level A Clean Evidence Defaults

The broad runner has also been cleaned at the output layer:

```text
scripts/main_runners/llm_attractor_colab_copy_paste.py
  EMIT_NARRATIVE = False
  root verdict/readiness/summary text artifacts are not written by default
  interpretation_checklist.csv -> claim_threshold_eval.csv
  breakthrough_readiness_scorecard.csv -> evidence_threshold_scorecard.csv
  breakthrough_predictive_validity.csv -> evidence_predictive_validity.csv
  vector_x_rlhf_verdict.md -> vector_x_rlhf_proxy_threshold_eval.csv
```

All CSVs written through `save_df` get `artifact_type`; JSON manifests written
through `save_json` get `artifact_type`; `validate_evidence_package_schema`
runs before zip creation. If narrative is explicitly enabled, it goes under
`analysis_notes/` with the autogenerated-interpretation header.

Source hygiene pass:

```text
BREAKTHROUGH_READINESS_AUDIT -> EVIDENCE_PACKAGE_AUDIT
build_breakthrough_readiness_audit -> build_evidence_package_audit
breakthrough_* dataframes -> evidence_* dataframes
vector_x_rlhf_verdict_text -> vector_x_rlhf_proxy_notes_text
```

Remaining `breakthrough/verdict` strings in the Level A runner are either
forbidden-value guard constants or old probe surface labels such as
`direct/verdict`, not generated evidence labels.

## Grade 4 SAE Lens Integration

The canonical Grade 4 script now has a real optional SAE Lens post-processing
layer over already captured residual-stream states:

```text
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
  SAE_FEATURE_ANALYSIS_ENABLED = False by default
  SAE_BACKEND = sae_lens
  SAE_SPECS controls release/sae_id or disk path
```

Research meaning:

```text
SAE evidence is now separated from dense hidden-unit proxy evidence. If a
compatible SAE is configured, the script can map prompt endpoint and generation
trajectory hidden shifts into sparse feature activations without changing
Vector X construction, prompt batching, generation batching, residual-stream
interventions, or Grade 4 component geometry.
```

New SAE outputs:

```text
feature_level_interpretability_status.csv
sae_model_compatibility.csv
sae_reconstruction_quality.csv
sae_prompt_feature_activation_summary.csv
sae_prompt_feature_delta_summary.csv
sae_top_changed_features.csv
sae_grade4_component_feature_summary.csv
sae_generation_feature_summary.csv
sae_generation_top_features.csv
sae_order_feature_contrast.csv
dense_feature_proxy_mapping.csv
```

Operational boundary:

```text
Disabled runs do not import sae_lens and still write explicit not_run status.
Enabled runs fail strictly on missing SAE specs, invalid layer, unavailable
d_in, or d_in != HIDDEN_SIZE. A real SAE claim requires a compatible model/layer
SAE, not an arbitrary dictionary.
```

## Qwen3-14B Grade 4 Metric Lab Readout

Analyzed package:

```text
c:\Users\stasv\Downloads\hidden_geometry_runs.zip
source run = content/hidden_geometry_runs/grade4_qwen3_14b
analysis = content/hidden_geometry_runs/grade4_qwen3_14b_metric_lab
```

Analyzer manifest:

```text
model_id = Qwen/Qwen3-14B
csv_files_processed = 49
npz_files_processed = 3
state_space_rows_written = 5494
plots_written = 19
errors = 0
```

Main signal:

```text
The Qwen Grade 4 run is a strong hidden-geometry/component run, not a
behavioral-control or SAE run. SAE was disabled. Behavioral random-p95 and
internal-visible coupling were not available.
```

Key Grade 4 numbers:

```text
middle target x_full projection = 0.984003
sentence-shuffle x_full projection = 1.271868
sentence-shuffle specificity therefore confounds x_full content evidence

target x_order_orth projection = 0.973637
sentence-shuffle x_order_orth projection = -0.007194
target x_order_orth direction cosine = 0.168238
target x_order_orth explained-shift R2 proxy = 0.063009

middle x_order_orth norm = 671.726251
middle x_order_orth energy fraction of x_full = 0.000471
```

Mechanistic reading:

```text
Most Vector X energy is content/format-like and carried by sentence-shuffled
target text. The orthogonal discourse-order component is tiny in residual norm
but separates target from sentence-shuffle and remains steerable in the middle
component-causal block.
```

Component causal readout:

```text
At alpha_abs=0.75, middle readout:
neutral middle intervention x_order gap = 4.082775
neutral middle intervention x_order_orth gap = 3.698995
neutral all intervention x_order gap = 4.672726
neutral all intervention x_order_orth gap = 4.345032
late intervention gaps are near zero.
```

State-space readout:

```text
between/within L2 ratio peaks around layers 20-23 at ~10.57-10.62.
Largest centroid split is neutral vs target_sentence_shuffle_control around
layers 23-27. Target vs sentence-shuffle remains separated but much smaller
and peaks later around layers 33-37.
```

Next experiment:

```text
Run Gemma 3 12B IT SAE smoke. The Qwen result says the right SAE target is not
"does x_full exist?" but "which sparse features distinguish x_order_orth from
x_content/sentence-shuffle despite x_order_orth's small residual norm?"
```

## Gemma3-12B-IT Grade 4 SAE Metric Lab Readout

Analyzed package:

```text
c:\Users\stasv\Downloads\all_hidden_geometry_runs_sae_analyze.zip
source run = content/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41
analysis = content/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_metric_lab
```

Analyzer manifest:

```text
model_id = google/gemma-3-12b-it
backend = pandas
csv_files_processed = 52
npz_files_processed = 3
state_space_rows_written = 6566
plots_written = 18
errors = 0
```

SAE compatibility:

```text
6/6 SAE specs computed
layers = 13, 19, 25, 31, 37, 42
hidden_size = 3840
sae_d_in = 3840 for every SAE
sae_d_sae = 16384
reconstruction cosine mean ranges from 0.9987 at layer 13 to 0.9896 at layer 42
explained_variance_proxy mean ranges from 0.9956 at layer 13 to 0.9788 at layer 42
```

Main Grade 4 shift:

```text
Unlike the Qwen run, Gemma's x_order_orth is not tiny.

middle x_order_orth_energy_fraction_of_full = 0.613503
late   x_order_orth_energy_fraction_of_full = 0.564123
all    x_order_orth_energy_fraction_of_full = 0.575700

target x_order_orth projection = 0.909026
sentence_shuffle x_order_orth projection = -0.069058
target x_content projection = -0.010294
sentence_shuffle x_content projection = 0.849551
```

Mechanistic reading:

```text
For Gemma3-12B-IT, the target-vs-sentence-shuffle split is not a tiny
orthogonal residue. The order/rhetorical component is a major part of x_full.
x_content behaves as the sentence-shuffle/content axis; x_order_orth behaves
as the target/coherent-order axis.
```

Sparse feature readout:

```text
sae_order_feature_contrast.csv rows = 278
order_abs_gt_content_abs = 114
order_abs_gt_content_abs with prompt gap = 33
order_abs_gt_content_abs with generation gap = 30
order_abs_gt_content_abs with target generation > sentence generation = 23
```

Important candidate sparse features:

```text
layer 31 feature 58:
  x_order_orth_delta = 728.190430
  x_content_delta = -603.021973
  target_minus_sentence_prompt_delta = 1019.599609
  target_minus_sentence_generation = 10.305182

layer 31 feature 451:
  x_order_orth_delta = 438.826172
  x_content_delta = -292.639648
  target_minus_sentence_prompt_delta = 580.245605
  target_minus_sentence_generation = 6.260312

layer 42 feature 13686:
  x_order_orth_delta = 441.245026
  x_content_delta = not in top x_content rows
  target_generation = 695.825800
  sentence_shuffle_generation = 487.451146

layer 42 feature 180:
  x_order_orth_delta = 553.048828
  x_content_delta = 389.999023
  target_minus_sentence_prompt_delta = 322.187500
  target_minus_sentence_generation = 48.408337
```

Anomalies:

```text
metric_lab/anomaly_flags.csv has four medium missing Grade4 component causal
artifacts. These are expected for this run because GRADE4_COMPONENT_CAUSAL_ENABLED=False.
They are not SAE failures.
```

Next experiment:

```text
If compute is available, run a causal component sweep on Gemma with
GRADE4_COMPONENT_CAUSAL_ENABLED=True. The SAE readout now suggests that the
right intervention target is not generic x_full but the order-specific layer
31/42 sparse-feature-associated component.
```

Academic readout package:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/
```

Contents:

```text
academic_conclusion_ru.md
README.md
metrics/
```

Purpose:

```text
Compact Russian academic conclusion for the Gemma3-12B-IT Grade 4 SAE run,
plus copied machine metrics from all_hidden_geometry_runs_sae_analyze.zip.
The package preserves the descriptive strength of the result while keeping the
boundary clear: this run supports separable order/rhetorical-regime sparse
readout, but not causal SAE steering because Grade 4 component causal artifacts
were intentionally not produced in that run.
```

## Memory Hygiene Rule

Going forward:

```text
research_context_current.md = compact current state
research_context_anchor.md  = historical archive / long audit trail
research_synthesis/         = reports, runbooks, collected metrics
metrics/                    = per-run interpreted result folders
```

Do not keep appending every detail to the large anchor. Add only major
milestones there, and put operational state in this compact file.

## 2026-05-29: Next Operational Step Is Norm-Controlled Component Causal

The raw-alpha Gemma3-12B-IT causal x_order run moved the claim from descriptive
separability to causal involvement, not dominance:

```text
x_order_orth mean causal gap = 1.793915
x_order_orth positive gap rate = 0.861111
x_content mean causal gap = 2.067048
x_content positive gap rate = 0.916667
pairwise x_order_orth beats x_content = 0.361111
```

Interpretation:

```text
x_order_orth is causally active, but raw-alpha dominance over x_content is not
supported. The confound is intervention energy: raw x_content has a larger band
norm than x_order_orth.
```

Implemented next-run setup in:

```text
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

New run:

```text
RUN_LABEL = grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl
GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_MODE = band_l2
GRADE4_COMPONENT_CAUSAL_READOUT_USES_NORMED_AXIS = True
```

Primary check after the Colab run:

```text
mean_intervention_axis_band_norm should be close to 1
mean_effective_intervention_l2 should match alpha_abs across axes
then compare x_order_orth vs x_content plus_minus_projection_gap again
```

Decision rule:

```text
If x_order_orth matches or beats x_content under equal-energy intervention, move
to behavioral steering/random baseline. If x_content still wins, the honest
claim is causal involvement of order/response-mode geometry, but dominant causal
control remains content-family.
```

## 2026-05-29: Normctl Component Causal Run Result

Source zip:

```text
C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41 (1).zip
```

What this run measured:

```text
Not the descriptive hidden-state shift. It measured whether x_order_orth or
x_content is the stronger causal direction when both component vectors are
normalized to the same L2 norm over the intervention layer band.
```

Sanity check:

```text
mean_intervention_axis_band_norm = 1.0
mean_effective_intervention_l2 = alpha_abs
```

So raw-norm advantage was removed.

Main result:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate   = 0.527778

x_content mean causal gap    = -125.128343
x_content positive rate      = 0.472222

all-readout pairwise x_order_orth beats x_content = 0.416667
matching-readout pairwise x_order_orth beats x_content = 0.500000
```

Cosine did not rescue dominance:

```text
x_order_orth positive cosine-gap rate = 0.583333
x_content positive cosine-gap rate    = 0.416667
matching-readout cosine pairwise win rate = 0.500000
```

Important asymmetry:

```text
neutral:
  x_order_orth beats x_content = 0.666667
  mean order_minus_content_gap = +354.870122

target:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -236.496475
```

Verdict:

```text
Unit-L2 norm-controlled intervention removed the norm confound but did not show
stable causal dominance of x_order_orth. There is some directional signal for
neutral +x_order_orth injection, but bidirectional symmetry and alpha scaling
are weak.
```

Next experiment:

```text
norm-controlled natural-scale causal run
```

Use equal-energy directions, but rescale both axes to a shared natural band norm:

```text
unit_axis = axis / norm(axis over band)
shared_band_norm = min(norm(x_order_orth over band), norm(x_content over band))
intervention = alpha * shared_band_norm * unit_axis
```

Reason:

```text
The unit-L2 run was fair as a direction comparison, but likely underpowered
relative to natural component norms, which are in the thousands.
```

## 2026-06-02 - Gemma Grade 4 natural-scale norm-controlled causal verdict

Source zips:

```text
C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip
C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale_metric_lab.zip
```

Run config:

```text
model_id = google/gemma-3-12b-it
run_label = grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale
norm_control_mode = shared_natural_band_l2
readout_uses_normed_axis = True
axes = [x_order_orth, x_content]
bands = [middle, late]
alpha = [0.25, 0.50, 0.75]
base_conditions = [neutral, target]
CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False
```

Natural-scale result:

```text
x_order_orth mean causal gap = 19284.481823
x_order_orth positive rate   = 0.861111
x_order_orth max gap         = 84082.583718
x_order_orth alpha slope mean = 18611.065776

x_content mean causal gap    = 27352.919286
x_content positive rate      = 0.944444

all-readout pairwise x_order_orth beats x_content      = 0.194444
matching-readout pairwise x_order_orth beats x_content = 0.166667
all-readout mean order_minus_content_gap               = -8068.437463
matching-readout mean order_minus_content_gap          = -3656.666860
```

Natural-scale interpretation:

```text
The unit-L2 underpower hypothesis is partly confirmed: at natural scale
x_order_orth becomes strongly causally active, with large positive trajectory
gaps and much better alpha scaling than in the unit-L2 run. However,
x_order_orth does not become the dominant steering component over x_content.
x_content remains larger on mean gap, positive rate, all-readout pairwise wins,
and matching-readout pairwise wins.
```

Checklist answers:

```text
x_order_orth beats x_content? No. all-readout win rate = 0.194444.
matching readout dominance? No. matching win rate = 0.166667.
neutral + x_order_orth stronger than unit-L2? Yes. neutral mean gap = 18898.736033, positive rate = 0.833333.
target - x_order_orth less unstable than unit-L2? Directionality improved, but not clean: target mean gap = 19670.227613, positive rate = 0.888889, min = -15346.259374.
alpha scaling improved? Yes. x_order_orth slope mean = 18611.065776, positive slope rate = 0.833333; matching slopes positive rate = 1.0.
anomaly_flags = 0? Yes. analyzer anomaly_flags.csv rows = 0; analysis_manifest errors = 0.
```

Current claim boundary:

```text
descriptive latent-state shift: strong/proven for Gemma Grade 4
content/order separation: strong/proven for Gemma Grade 4
x_order_orth causal activity at natural scale: strongly supported
x_order_orth as dominant stable steering handle: not supported
next mechanistic target: layer/birth localization and SAE feature-level carrier analysis
```

## 2026-06-02 - Gemma Grade 4 script extended beyond natscale

Updated file:

```text
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Purpose:

```text
Do not stop at the natscale verdict. The next Grade 4 run should also emit
layer/birth localization and SAE feature triage artifacts so the research can
move from dense coordinates to sparse/layer-specific mechanism.
```

New outputs added:

```text
grade4_x_order_orth_birth_layer_summary.csv
grade4_x_order_orth_birth_layer_verdict.csv
sae_order_feature_triage.csv
sae_order_feature_triage_top_candidates.csv
```

New capability:

```text
layer_band_to_indices now accepts single-layer and multi-layer causal bands:
layer_24, l24, layers_18_24_30.
This prepares layer-specific component intervention without enabling the
behavioral coupling block.
```

Current intended next run:

```text
Gemma Grade 4 rerun with natscale config unchanged, plus the new birth/triage
outputs. Behavioral coupling remains a later block after layer/SAE evidence.
```

## 2026-06-02 - Checklist closure update (natscale vs unit-L2)

Comparison runs:

```text
baseline = content/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl
natscale = content/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale
```

### 1) x_order_orth beats x_content? (especially matching readout)

Max-alpha matching-readout rank (`alpha_abs=0.75`):

```text
unit-L2:   x_order_orth top in 2/4 groups (neutral late+middle), x_content top in 2/4
natscale:  x_order_orth top in 0/4 groups, x_content top in 4/4
```

All-alpha matching-readout pairwise wins (`x_order_orth > x_content` on plus_minus_projection_gap):

```text
unit-L2: 6/12 = 0.500000
natscale: 2/12 = 0.166667
```

Interpretation:

```text
x_order_orth does not beat x_content under natscale. Natural scale increases
causal activity of both axes, but content remains the dominant steering
component on matching readout.
```

### 2) neutral + x_order_orth became stronger?

Matching-readout, max alpha:

```text
late/late:
  plus_projection   -9863.863057 -> +16318.790304
  plus-minus gap      992.518931 -> 78268.698847

middle/middle:
  plus_projection  -16364.348761 -> -15303.617122
  plus-minus gap      274.611926 -> 16328.298377
```

Interpretation:

```text
Yes, stronger. The neutral +x_order_orth effect is no longer a tiny residual:
late-band injection flips to positive plus_projection and both bands show much
larger causal gap.
```

### 3) target - x_order_orth became less unstable?

Target, matching-readout, x_order_orth gaps across alpha:

```text
unit-L2 late:   [243.515656,  -161.661094,  -553.394467]
unit-L2 middle: [137.031406,  -657.174266,    -0.605217]

natscale late:   [29670.897956, 57833.034265, 84082.583718]
natscale middle: [ 7066.251985, 11754.378617, 14812.285185]
```

Stability proxies:

```text
target matching sign consistency:
  unit-L2: 0.333 (both bands)
  natscale: 1.000 (both bands)

target matching monotonic increase vs alpha:
  unit-L2: late=0.0, middle=0.5
  natscale: late=1.0, middle=1.0
```

Interpretation:

```text
Yes for matching-readout trajectory control: directionality is now consistent
and monotonic with alpha. Residual instability still exists in readout-mismatch
slices (all-readout target min gap remains negative).
```

### 4) alpha scaling improved?

`signed_alpha_projection_slope` (matching readout):

```text
x_order_orth, neutral, late/late:   +250.989974 -> +55269.737498
x_order_orth, neutral, middle/middle: +288.837283 -> +12066.817447
x_order_orth, target, late/late:    -248.569991 -> +56797.816806
x_order_orth, target, middle/middle: -168.447540 -> +10715.980682
```

Interpretation:

```text
Yes. natscale turns weak or contradictory unit-L2 slopes into strong positive
dose-response for x_order_orth on matching readout.
```

### 5) anomaly_flags = 0?

```text
baseline metric_lab anomaly_flags.csv rows = 0
natscale metric_lab anomaly_flags.csv rows = 0
```

Conclusion:

```text
No anomaly flags in either analyzer package.
```

Mechanistic claim update:

```text
This supports: x_order_orth has strong causal involvement at natural scale.
This does not support: x_order_orth as the dominant causal control axis versus
x_content.
```

Next experiment decision:

```text
Proceed to layer/birth localization and sparse SAE carrier analysis, with
single-band and single-layer interventions (middle-only vs late-only) under the
same shared_natural_band_l2 scaling protocol.
```

## 2026-06-02 - Fixed Established Claim And Unit-Norm Causal Metrics

Established claim:

```text
We proved context-induced latent-state shift in Gemma3-12B-IT: a strong
coherent target text moves the model into a different measurable internal
state. This shift is not reducible to words/content, because sentence-shuffle
separates into x_content while coherent target separates into x_order_orth.
```

Coordinate meaning:

```text
The coordinates are not absolute coordinates of the entire model state space.
They are coordinates of condition deltas and generation trajectories relative
to latent axes built from target/control differences: x_full, x_content,
x_order, and x_order_orth.
```

Unit-norm causal run raw norms before norm-control:

```text
middle x_content raw norm      ~= 14518.90
middle x_order_orth raw norm   ~= 8058.43
late x_content raw norm        ~= 29315.89
late x_order_orth raw norm     ~= 14729.57
```

After norm-control:

```text
Both axes were intervened with equal energy.
```

Main unit-norm causal comparison across all readout cells:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate   = 0.527778
x_content mean causal gap    = -125.128343
x_content positive rate      = 0.472222
```

Pairwise all-readout comparison:

```text
x_order_orth beats x_content = 0.416667
mean order_minus_content_gap = +59.186823
median order_minus_content   = -128.777290
```

Matching-readout only, middle->middle and late->late:

```text
x_order_orth beats x_content = 0.500000
mean order_minus_content_gap = -0.191014
median order_minus_content   = +89.468686
```

Interpretation:

```text
This is not a causal dominance win for x_order_orth. Under equal-energy
unit norm-control, x_order_orth and x_content are approximately tied and the
sign is unstable.
```

Cosine causal comparison:

```text
x_order_orth positive cosine-gap rate = 0.583333
x_content positive cosine-gap rate    = 0.416667
x_order_orth mean cosine gap          = -0.000101
x_content mean cosine gap             = -0.000054

all readouts:      x_order_orth beats x_content = 0.416667
matching readouts: x_order_orth beats x_content = 0.500000
```

Meaning:

```text
Cosine does not rescue causal dominance. It says the unit-norm controlled
causal effect is directionally unstable.
```

Important base-condition asymmetry:

```text
neutral:
  x_order_orth beats x_content = 0.666667
  mean order_minus_content_gap = +354.870122

target:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -236.496475
```

Mechanistic read:

```text
x_order_orth works better as injection from neutral than as a symmetric
bidirectional control handle. neutral + x_order_orth can move the trajectory
in the expected direction, but target - x_order_orth does not produce a stable
reverse effect. Bidirectional causal symmetry is not closed.
```

Alpha scaling:

```text
x_order_orth positive slope rate = 0.250000
x_content positive slope rate    = 0.416667
```

Meaning:

```text
Dose-response is weak in the unit-norm causal run.
```

## Research framing: fundamental interpretability for model improvement

This research thread should be framed as fundamental mechanistic interpretability, not as a jailbreak or bypass recipe. The central object is the internal hidden-state / residual-stream trajectory of an LLM under strong coherent context. The goal is to measure, decompose, and eventually localize how context changes internal model state before final visible output.

The practical value is model improvement. If strong context can move an agent into a different internal trajectory before it chooses tools, writes memory, plans, self-monitors, or produces final text, then output-only evaluation is incomplete. The constructive direction is to build better diagnostics: hidden-state trajectory monitoring, content/order separation controls, SAE feature localization, causal intervention tests, and behavioral coupling checks.

Working formulation:

```text
We are studying context-induced latent-state shifts as an alignment and
reliability object. The research aims to help laboratories understand where
internal response modes arise, which sparse features carry them, how they
propagate through generation, and how future systems can monitor or stabilize
these internal trajectories.
```

Boundary:

```text
Do not frame the work as "we found how to bypass RLHF." Frame it as: we found
evidence that strong context can induce measurable internal trajectory shifts,
and this exposes why future model audits should include hidden-state diagnostics
in addition to final-output checks.
```

## Qwen3.5-9B Qwen-Scope steering implementation note

The Qwen workspace steering script is restored as the expanded Qwen-Scope
mediation/KL version:

```text
model_workspaces/qwen3_5_9b_qwen_scope/steering/01_candidate_discovery_and_rough_sae_patching.py
```

This script is not the minimal Gemma-only column clone. It keeps the Gemma
mediation core while adding hidden/logit/KL readouts:

```text
standard sae_order_feature_contrast.csv -> select order_specific/order_enriched
features -> run real Qwen forward -> hook residual stream -> Qwen-Scope SAE
encode -> patch selected feature -> replace layer output -> measure
baseline-vs-patched final-token hidden and logits -> write mediation CSV.
```

Main output columns include:

```text
real_layer, csv_layer, feature_index, x_order_orth_delta, status,
mediated_effect, target_hidden_delta_l2, target_logit_l2,
target_logit_mean_abs, target_kl_base_to_patched, target_top1_flip_rate,
control_mediated_effect, target_minus_control_mediated_effect, patch_mode,
patch_position, patch_value, model_name, sae_repo_id, sae_top_k,
sae_relu_before_topk, prompt_batch_size, max_length
```

If control prompts are provided, the script also writes:

```text
control_hidden_delta_l2, control_logit_l2, control_logit_mean_abs,
control_kl_base_to_patched, control_top1_flip_rate, diff_hidden_delta_l2,
diff_logit_l2, diff_kl_base_to_patched
```

Important operational distinction:

```text
For Qwen steering/patching, use the standard Grade 4 artifact
sae_order_feature_contrast.csv, not metric-lab's
sae_order_feature_contrast_matrix.csv.
```

Qwen layer mapping:

```text
CSV layer = hidden_states layer = transformer block layer + 1
Qwen real/block layer = CSV layer - 1
Qwen-Scope SAE file = layer{real_layer}.sae.pt
```

## 2026-06-02 - Qwen3.5-9B Base Qwen-Scope W64K-L0_50 Grade 4 Readout

Source packages:

```text
C:\Users\stasv\Downloads\grade4_qwen3_5_9b_base_qwen_scope_w64k_l0_50_full32.zip
C:\Users\stasv\Downloads\qwen3_5_9b_base_qwen_scope_metric_lab.zip
```

Run identity:

```text
model_id = Qwen/Qwen3.5-9B-Base
run_label = grade4_qwen3_5_9b_base_qwen_scope_w64k_l0_50_full32
decoder_layer_count = 32
expected_decoder_layer_count = 32
decoder_layer_count_mismatch = false
question_count = 10
research_meta_question_count = 5
SAE backend = qwen_scope
SAE repo = Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
SAE specs computed = 32/32
metric-lab csv_files_processed = 58
metric-lab npz_files_processed = 3
metric-lab errors = 0
anomaly_flags.csv = empty
```

SAE compatibility:

```text
sae_d_in = 4096
hidden_size = 4096
sae_d_sae = 65536
status = computed for all 32 layers
reconstruction cosine mean = 0.966660
reconstruction cosine min = 0.868464
explained_variance_proxy mean = 0.933639
explained_variance_proxy min = 0.722491
```

Grade 4 component norms:

```text
middle:
  full_norm = 29.372602
  content_norm = 27.588603
  order_orth_norm = 18.459240
  content_energy_fraction_of_full = 0.882215
  order_orth_energy_fraction_of_full = 0.394951

late:
  full_norm = 70.723911
  content_norm = 66.400941
  order_orth_norm = 42.972818
  content_energy_fraction_of_full = 0.881487
  order_orth_energy_fraction_of_full = 0.369194

all:
  full_norm = 76.902571
  content_norm = 72.260437
  order_orth_norm = 47.023432
  content_energy_fraction_of_full = 0.882916
  order_orth_energy_fraction_of_full = 0.373893
```

Interpretation:

```text
Qwen3.5-9B has a real x_order_orth component, but its full shift is more
content-heavy than Gemma's. The content component carries most of the raw
energy. x_order_orth is smaller than x_content, but still cleanly separable.
```

Projection geometry:

```text
target on x_order_orth = 0.979462
sentence_shuffle on x_order_orth = 0.009969
word_shuffle on x_order_orth = 0.059662

sentence_shuffle on x_content = 0.967008
target on x_content = 0.770266
word_shuffle on x_content = 0.594366

target on x_full = 0.973778
sentence_shuffle on x_full = 0.813187
```

Mechanistic read:

```text
This is a Qwen replication of the core hidden-state finding: coherent target
strongly separates from shuffled-content controls on x_order_orth. However,
unlike Gemma, Qwen target also carries a strong x_content projection. So the
right Qwen claim is not "order dominates content"; it is "order is separable
and measurable despite a large content component."
```

Qwen-Scope sparse-feature readout:

```text
sae_order_feature_contrast.csv rows = 1503
order_abs_gt_content_abs = 575
interpretation_status counts:
  content_only_or_missing_order_component = 479
  content_overlap_or_content_dominant_feature = 449
  order_component_specific_top_feature = 434
  order_enriched_overlap_feature = 70
  order_specific_generation_persistent_feature = 54
  order_specific_prompt_feature = 17
```

Top order-specific Qwen features:

```text
layer 27 feature 65254:
  x_order_orth_delta = -22.089539
  order_specific_score = 22.367545
  interpretation_status = order_specific_generation_persistent_feature

layer 23 feature 51987:
  x_order_orth_delta = -8.362167
  x_content_delta = -2.294868
  order_specific_score = 14.773435

layer 27 feature 5335:
  x_order_orth_delta = -7.184792
  x_content_delta = 0.847343
  order_specific_score = 13.976547

layer 28 feature 28136:
  x_order_orth_delta = +3.726776
  x_content_delta = +1.544765
  order_specific_score = 8.050881
```

Component causal setup:

```text
axes = [x_order_orth, x_content]
layer_bands = [middle, late]
alphas = [0.25, 0.50, 0.75]
base_conditions = [neutral, target]
norm_control_mode = shared_natural_band_l2
readout_uses_normed_axis = true
```

Component causal result:

```text
all readout cells:
  x_content mean gap = 41.878616
  x_content positive rate = 1.0
  x_order_orth mean gap = 38.246761
  x_order_orth positive rate = 1.0

matching readout only:
  x_content mean gap = 73.851162
  x_order_orth mean gap = 72.449630
```

Pairwise component comparison:

```text
all readouts:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -3.631854
  median order_minus_content_gap = -1.896108

matching readouts:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -1.401532
  median order_minus_content_gap = -0.811232

neutral:
  x_order_orth beats x_content = 0.333333
  mean order_minus_content_gap = -2.779402

target:
  x_order_orth beats x_content = 0.000000
  mean order_minus_content_gap = -4.484307
```

Max-alpha matching readout:

```text
neutral late/late:
  x_order_orth gap = 163.267378
  x_content gap = 161.761760

neutral middle/middle:
  x_content gap = 55.528447
  x_order_orth gap = 53.531738

target late/late:
  x_content gap = 169.018255
  x_order_orth gap = 161.597248

target middle/middle:
  x_content gap = 55.913159
  x_order_orth gap = 53.361986
```

Alpha scaling:

```text
x_content mean slope = 41.791397
x_content positive slope rate = 1.0
x_order_orth mean slope = 38.130515
x_order_orth positive slope rate = 1.0

matching:
  x_content mean slope = 73.759712
  x_order_orth mean slope = 72.261836
```

Qwen causal interpretation:

```text
Both component directions are causally active and alpha-scaled under
shared_natural_band_l2. x_order_orth is not unstable here; it has positive
gaps and positive slopes. But x_content is slightly stronger almost everywhere,
especially in target condition. Therefore Qwen supports causal involvement of
x_order_orth, not causal dominance over x_content.
```

Claim boundary:

```text
Qwen3.5-9B Base Qwen-Scope replicates the core hidden-state / order-readout
phenomenon, but in a more content-heavy form than Gemma. The Qwen result
strengthens cross-model evidence for context-induced latent-state shift and
separable x_order_orth readout. It weakens any claim that x_order_orth is the
dominant natural causal steering component across models.
```

Next Qwen experiment:

```text
Do not spend the next Qwen run on generic text-analysis questions. Use
held-out high-friction behavioral probes plus neutral matched controls, and
test the top Qwen order-specific sparse features directly:
layer 27 feature 65254, layer 23 feature 51987, layer 27 feature 5335,
layer 28 feature 28136, plus high-mediation features from the separate
Qwen causal mediation script if available.
```

## 2026-06-02 - Behavioral stress-test prompt policy

Operational correction:

```text
Questions about the target text are useful for clean geometry discovery,
axis construction, and reviewer-safe hidden-state evidence. They are not the
right primary behavioral probes if the hypothesis is that the target context
relaxes the model's response mode in political, policy-sensitive, or otherwise
high-friction domains.
```

Current behavioral-test decision:

```text
For downstream mediation/steering/behavior coupling, use held-out questions
from the domains where the shift is expected to manifest: political analysis,
controversial public-policy reasoning, institutional critique, high-friction
normative comparison, and non-operational safety-adjacent explanations.
```

Control rule:

```text
Keep the same target/control question set across runs. Pair political or
high-friction probes with neutral analytical controls of similar length and
style. Do not interpret directness changes without matched neutral/control
baselines, because otherwise ordinary topic difficulty can be mistaken for
latent response-mode steering.
```

Claim boundary:

```text
This does not reframe the work as bypass testing. The research object remains
context -> hidden-state shift -> sparse feature / residual-stream carrier ->
generation/logit/behavior coupling. Political prompts are stress-test probes
for response-mode relaxation, not the core evidence by themselves.
```

## 2026-06-02 - Qwen SAE Top-K mediation clean run

Observed file:

```text
C:\Users\stasv\Downloads\my_results4.csv
```

Run interpretation:

```text
This is a clean Qwen-Scope SAE candidate mediation / ablation readout using
SAE_TOP_K=50. It should be read as target-prompt sparse-feature ablation
strength, not yet as target-vs-control order-specific causal proof, because
the current CSV contains only target-side mediated_effect and no control/diff
columns.
```

Main numbers:

```text
rows = 47
nonzero mediated_effect rows = 24
zero mediated_effect rows = 23
effect > 1 rows = 20
effect > 5 rows = 14
effect > 20 rows = 7

mediated_effect mean = 10.274831
mediated_effect median = 0.303857
mediated_effect max = 111.121696

corr(mediated_effect, signed x_order_orth_delta) = 0.332664
corr(mediated_effect, abs x_order_orth_delta) = 0.379819
```

Top Qwen sparse mediation candidates:

```text
real_layer 31, csv_layer 32, feature 11169:
  x_order_orth_delta = 1.262140
  status = order_specific_generation_persistent_feature
  mediated_effect = 111.121696

real_layer 31, csv_layer 32, feature 42831:
  x_order_orth_delta = 5.718891
  status = order_specific_generation_persistent_feature
  mediated_effect = 85.469116

real_layer 28, csv_layer 29, feature 41435:
  x_order_orth_delta = 9.999897
  status = order_specific_generation_persistent_feature
  mediated_effect = 77.897545

real_layer 24, csv_layer 25, feature 47391:
  x_order_orth_delta = 5.066154
  status = order_enriched_overlap_feature
  mediated_effect = 30.897112

real_layer 31, csv_layer 32, feature 12129:
  x_order_orth_delta = -2.005377
  status = order_enriched_overlap_feature
  mediated_effect = 27.603189

real_layer 31, csv_layer 32, feature 2431:
  x_order_orth_delta = -2.755733
  status = order_specific_generation_persistent_feature
  mediated_effect = 26.914242

real_layer 28, csv_layer 29, feature 52698:
  x_order_orth_delta = 3.691418
  status = order_specific_generation_persistent_feature
  mediated_effect = 21.966143
```

Layer structure:

```text
real_layer 31:
  rows = 8
  mean_effect = 35.436642
  max_effect = 111.121696
  median_effect = 21.702826

real_layer 28:
  rows = 16
  mean_effect = 8.517493
  max_effect = 77.897545
  median_effect = 0.306630

real_layer 24:
  rows = 9
  mean_effect = 4.990232
  max_effect = 30.897112
  median_effect = 0.335334

late layers >=24:
  rows = 33
  mean_effect = 14.081367
  median_effect = 1.319505
  max_effect = 111.121696

layers <24:
  rows = 14
  mean_effect = 1.302282
  median_effect = 0.0
  max_effect = 8.540574
```

Mechanistic meaning:

```text
The sparse-feature carrier signal is concentrated in late Qwen layers,
especially layer 31 and layer 28. This matches the broader Qwen Grade 4
finding: the target-induced order/readout signal is visible and causally
touchable, but not all high x_order_orth readout features mediate equally.
The moderate correlation between x_order_orth_delta and mediated_effect means
that readout strength and ablation strength are related but not identical.
```

Claim boundary:

```text
This strengthens the hypothesis that Qwen has sparse late-layer carriers for
the target/order-induced residual-stream state. It does not yet prove
order-specific mediation, because target-vs-control ablation gaps are not in
this file. The next required run is the same feature set with matched
prompts_control and explicit target_minus_control_effect / KL / logit metrics.
```

## 2026-06-02 - Current research lock: finish the geometry-coordinate claim

Decision:

```text
Do not branch into a new Grade 5 direction until the current result is finished.
The active research object is:

context-induced hidden-state geometry shift with measured coordinates
relative to discovered latent axes.
```

What is already established:

```text
There is a measurable context-induced latent-state shift. Strong coherent
target context moves the model's inference-time residual-stream / hidden-state
trajectory into a different measurable region of representation space.

The shift is not described only at the visible-output layer. It is measured
inside the model through hidden states, condition deltas, generation
trajectories, and projections onto discovered axes.
```

Coordinate claim:

```text
The coordinates are projections relative to the discovered axes:

x_full
x_content
x_order
x_order_orth

These are not absolute coordinates of the whole model. They are coordinates
of the target/control condition deltas and generation trajectories in the
subspace defined by the experiment.
```

Core direction of the paper/result:

```text
Finish and present the geometry-coordinate evidence as the main result:

1. target context shifts hidden-state geometry;
2. sentence/word shuffle controls separate content from coherent order;
3. x_order_orth provides a measurable coordinate for coherent target structure;
4. generation trajectories can be read in the same coordinate system;
5. SAE features provide candidate sparse carriers, but they are secondary to
   the core geometry-coordinate proof.
```

Current boundary:

```text
The result proves inference-time hidden-state shift and coordinate readout.
It does not claim permanent weight change. It does not require claiming that
x_order_orth is the dominant causal steering axis. The main claim is already
strong without that: the model enters a different measurable internal state
under coherent target context.
```

Next work is not a new experiment family. It is packaging:

```text
Build the final geometry-coordinate evidence package:

- one table of coordinates for Gemma and Qwen;
- one table of target vs sentence-shuffle vs word-shuffle separation;
- one table of causal involvement / non-dominance boundary;
- one figure showing x_content vs x_order_orth coordinates;
- one figure showing generation trajectory projections;
- one Russian final conclusion;
- one English abstract-ready claim block.
```

## 2026-06-02 - Qwen SAE mini-check protocol archived

New archived protocol:

```text
model_workspaces/qwen3_5_9b_qwen_scope/steering/mini_check_protocol/README_RU.md
```

Preserved artifacts:

```text
model_workspaces/qwen3_5_9b_qwen_scope/steering/mini_check_protocol/artifacts/
```

Preserved Colab mini-check scripts:

```text
01_top_activating_contexts_blacklisted_colab.py
02_feature_context_report_colab.py
03_logit_loss_patching_top_features_colab.py
04_token_level_loss_delta_by_feature_colab.py
```

Main finding:

```text
The Qwen mini-checks connect Grade 4 hidden geometry to sparse downstream
evidence. Features 28:41435 and 24:47391 are strongest after loss/logit/token
localization. Feature 28:41435 is the current best candidate:

mediated_effect = 77.897545
loss_delta = +1.342655
last_logit_l2 = 574.866821
KL_last = 0.700875

The largest token-level patch-worse deltas localize around averaged-user,
safety/default, caution, objection-avoidance, and precision/directness
formulation spans rather than random vocabulary positions.
```

Mechanistic meaning:

```text
This does not replace the core geometry-coordinate proof. It adds a lower-level
SAE readout: selected Qwen-Scope decoder directions appear to participate in
the response-construction regime exposed by the target context. The evidence
chain is now:

Grade 4 coordinates -> SAE mediation -> top activating contexts ->
sequence loss / final-token KL -> token-level loss localization.
```

Claim boundary:

```text
Treat these as candidate sparse carriers for formulation / epistemic-posture
dynamics, not as final feature names or universal safety features. The next
clean step is held-out transfer plus random-feature and matched-neutral
controls.
```

## 2026-06-02 - Related work note: measurement geometry for trustworthy generative inverse problems

Local PDF inspected:

```text
C:\Users\stasv\Downloads\2606.02309v1.pdf
Measurement Geometry and Design for Trustworthy Generative Inverse Problems
Pengfei Jin, Na Li, Quanzheng Li
```

Core idea:

```text
In generative inverse problems, a plausible reconstruction can be supported by
measurements or hallucinated by the generative prior along directions not
observed by the measurement operator. The paper formalizes this as
measurement-manifold compatibility: whether the acquisition operator observes
locally plausible tangent directions on the data manifold.
```

Useful conceptual bridge to our work:

```text
Their object:
  measurement operator A must observe tangent directions of the generative
  prior manifold, otherwise plausible alternatives remain indistinguishable.

Our object:
  hidden-state / residual-stream diagnostics must observe target-induced
  trajectory directions, otherwise final-output evaluation can miss the
  internal shift that produced the behavior.

Shared vocabulary:
  geometry of trust, locally plausible alternatives, observed vs unobserved
  directions, tangent ambiguity, measurement/readout design.
```

Why this matters for framing:

```text
This paper supports a broader methodological framing: trustworthiness is not
only a property of the generative prior/output. It depends on whether the
measurement/readout system observes the relevant internal or manifold
directions. For our LLM work, this strengthens the argument that output-only
evaluation is an insufficient measurement operator for agentic systems.
```

## 2026-06-02 - COAST beta calibration for Gemma layer 41

Signal:

```text
The COAST beta calibration run succeeded using global CONCEPTOR_STATES from
the already fitted conceptor experiment. This is a calibration run, not the
full 03 generation run. It measured how large the COAST operator displacement
is relative to the residual stream and ran final next-token KL checks.
```

Main measured strengths:

```text
Layer 41, aperture 1.0:
  beta=1 -> median update ~= 25.69% residual
  5% residual beta ~= 0.195
  10% residual beta ~= 0.389
  20% residual beta ~= 0.778
  pos_neg_overlap = 0.911, weak contrast

Layer 41, aperture 3.0:
  beta=1 -> median update ~= 25.78% residual
  5% residual beta ~= 0.194
  10% residual beta ~= 0.388
  20% residual beta ~= 0.776
  pos_neg_overlap = 0.758

Layer 41, aperture 10.0:
  beta=1 -> median update ~= 26.27% residual
  5% residual beta ~= 0.190
  10% residual beta ~= 0.381
  20% residual beta ~= 0.761
  pos_neg_overlap = 0.518, strongest contrast among the three
```

Mechanistic meaning:

```text
Beta is now calibrated as intervention size, not an arbitrary knob:

  x_patched = x + beta * (T(x) - x)

where T is the contrastive conceptor operator. The practical working region is
around beta ~= 0.19 for a 5% residual-stream displacement. Beta ~= 0.35 is a
strong but still interpretable push. Beta ~= 0.38 reaches roughly 10% residual
and should be treated as aggressive. Beta ~= 0.76 and beta=1 are stress tests:
they produce large KL and top-token changes and are likely unstable for
ordinary behavioral interpretation.
```

Hypothesis update:

```text
This strengthens the COAST/subspace-steering line: the learned target-vs-control
operator has nontrivial next-token effects at calibrated, residual-norm-scale
interventions. It does not yet prove semantic quality or agentic safety impact.
It establishes that the low-rank response-mode subspace can be re-entered with
a measured hidden-state displacement.
```

Next experiment:

```text
Run 03_conceptor_subspace_steering.py with per-aperture beta grids. For the
main interpretability run, prefer:

  (41, 1.0):  [0.0, 0.05, 0.195, 0.35, 0.389]
  (41, 3.0):  [0.0, 0.05, 0.194, 0.35, 0.388]
  (41, 10.0): [0.0, 0.05, 0.190, 0.35, 0.381]

Reserve beta ~= 0.76 and beta=1.0 for a separate stress-test run. Aperture 10.0
is the cleanest primary condition because its POS/NEG overlap is lowest.
```

## 2026-06-03 - COAST L41 quick run on NEG/control base

Source artifact:

```text
C:\Users\stasv\Downloads\neg.zip
```

Files present:

```text
conceptor_fit_summary_coast_l41_beta_calib_quick.csv
conceptor_steering_generation_summary_metrics_coast_l41_beta_calib_quick.csv
conceptor_steering_generation_full_metrics_coast_l41_beta_calib_quick.csv
```

Teacher-forced KL files were not present. This is consistent with the quick-run
configuration where `RUN_TEACHER_FORCED_KL_AFTER_GENERATION=False`, or with an
interruption before teacher-forced postprocessing. The main full metrics file
does contain final next-token KL / JS / logit-L2 diagnostics.

Run shape:

```text
base_condition = neg
layer = 41
apertures = 1.0, 3.0, 10.0
generation modes = greedy + sampled
tasks = 5
rows = 240

Per aperture beta grid:
  L41/A1.0:  [0.0, 0.05, 0.195, 0.389]
  L41/A3.0:  [0.0, 0.05, 0.194, 0.388]
  L41/A10.0: [0.0, 0.05, 0.190, 0.381]
```

Fit summary:

```text
A1.0  pos_neg_overlap = 0.907400
A3.0  pos_neg_overlap = 0.747206
A10.0 pos_neg_overlap = 0.498356
```

Main signal:

```text
The run does not show the punctuation-collapse failure seen at beta ~= 0.76/1.0.
The restricted beta grid stays readable. The intervention produces clear
distributional movement: final next-token KL and logit L2 rise with beta, while
final top-token changed rate remains 0.0 across all groups.
```

Representative mean final KL / logit-L2:

```text
Greedy A1.0:
  beta 0.195 -> KL 0.0516, logit_L2 307.0
  beta 0.389 -> KL 0.2714, logit_L2 1029.8

Greedy A3.0:
  beta 0.194 -> KL 0.0567, logit_L2 293.3
  beta 0.388 -> KL 0.2019, logit_L2 881.7

Greedy A10.0:
  beta 0.190 -> KL 0.0520, logit_L2 263.6
  beta 0.381 -> KL 0.1118, logit_L2 586.9
```

Mechanistic interpretation:

```text
This is a distributional hidden-trajectory perturbation, not a simple immediate
top-token flip. The final prompt-position top token is stable, but the
continuation distribution and sampled generation path move. This is exactly the
kind of case where output-only inspection is too coarse: the surface answer can
remain similar while the model's token distribution and trajectory geometry
change.
```

Behavioral readout:

```text
On the NEG/control base, the model often remains context-bound ("the provided
text is about a library") for the NATO task in greedy mode. Under sampled mode,
some beta/aperture settings escape the library-text constraint and produce
substantive political answers. This suggests the COAST intervention changes the
competition between context-bound refusal/procedural framing and direct-answer
generation, but the effect is stochastic and task-dependent rather than a
deterministic override.
```

Hypothesis update:

```text
Strengthens:
  COAST L41 can induce measurable nontrivial logit/KL movement on a control
  base without using unstable beta values.

Weakens / limits:
  The current NEG run alone does not prove stable semantic steering. Greedy
  outputs are often unchanged or only paraphrased. Teacher-forced KL and
  target-base runs are needed to localize where the trajectory diverges.

Next:
  1. Run the same restricted beta grid on BASE pos.
  2. Run a compact teacher-forced KL postprocess for the NEG full metrics, or
     rerun 03 with RUN_TEACHER_FORCED_KL_AFTER_GENERATION=True.
  3. Treat aperture 10.0 beta 0.190 and 0.381 as the main readable working
     pair; use aperture 1.0/3.0 beta ~=0.388/0.389 as more aggressive probes.
```

## 2026-06-03 - COAST L41 POS/target-base run with teacher-forced KL

Source artifact:

```text
C:\Users\stasv\Downloads\pos.zip
```

Detailed COAST log:

```text
steering/docs/coast_conceptor_findings.md
```

Main result:

```text
The POS/target-base run successfully produced teacher-forced KL outputs:

  conceptor_steering_generation_full_metrics_with_tf_kl_pos_coast_l41_beta_calib_quick.csv
  conceptor_teacher_forced_kl_summary_by_layer_aperture_beta_pos_coast_l41_beta_calib_quick.csv
  conceptor_teacher_forced_per_token_kl_details_pos_coast_l41_beta_calib_quick.csv

Run shape:
  base_condition = pos
  layer = 41
  tasks = 3
  rows = 144
  per-token TF rows = 7220
```

Key signal:

```text
Unlike the NEG run, where final top-token changed rate stayed 0.0, the POS run
reaches top-token changed rate ~= 0.333 at calibrated working/aggressive beta.
Teacher-forced KL also grows with beta:

  A10.0 greedy beta 0.190 -> mean_tf_kl 0.029747
  A10.0 greedy beta 0.381 -> mean_tf_kl 0.125469
  A10.0 sampled beta 0.190 -> mean_tf_kl 0.038807
  A10.0 sampled beta 0.381 -> mean_tf_kl 0.138302
```

Mechanistic interpretation:

```text
The same COAST operator is more decision-boundary-relevant on the POS/target
base than on the NEG/control base. On POS, the target/direct-answer mode is
already active; COAST perturbs which explanatory frame dominates rather than
merely escaping a library/control context.
```

Strongest behavioral locus:

```text
The NATO expansion task shows the largest divergence. Steering moves the model
between direct geopolitical explanation, official/legal framing about promises
not being formally binding, official-vs-unofficial framing, and occasional
context-bound regressions in sampled mode.
```

Per-token KL readout:

```text
The largest teacher-forced KL spikes are localized rather than uniform. They
cluster around NATO/geopolitical continuation tokens, including tokenizer
pieces inside "НАТО" such as "АТО". This supports trajectory-level
distributional divergence at semantically important continuation positions,
but also requires negative controls because named-entity tokenization can
amplify KL spikes.
```

Hypothesis update:

```text
Strengthened:
  COAST/conceptor steering has real trajectory-level effects at calibrated beta,
  visible in final KL, final top-token changes, teacher-forced KL, and localized
  per-token KL.

Limited:
  This is a 3-task diagnostic, not publication-scale evidence. It does not yet
  show clean semantic steering; it shows response-frame perturbation. Held-out
  prompts, random/mismatched conceptor controls, and plots are the next required
  layer.
```

## 2026-06-03 - COAST next step locked: matched A10 operator controls

Protocol:

```text
steering/docs/coast_next_control_experiment_protocol.md
```

Code update:

```text
steering/03_conceptor_subspace_steering.py now supports CONCEPTOR_OPERATOR_MODE.
Default remains actual, so old configs preserve behavior.
```

Modes:

```text
actual
random_same_spectrum
swap_pos_neg
pos_only
neg_remove_only
```

Scientific reason:

```text
The POS COAST run shows real trajectory movement, but the next risk is generic
low-rank perturbation. The next experiment must test whether the learned
POS/NEG conceptor orientation matters by comparing actual COAST against a
random same-spectrum control and a swapped POS/NEG control.
```

Primary run design:

```text
layer = 41
aperture = 10.0
betas = [0.0, 0.05, 0.190, 0.381]
base_condition = ["pos", "neg"]
tasks = 3
generation = greedy 1 + sampled 1
teacher-forced KL = enabled
per-token details = enabled
```

Configs:

```text
steering/configs/coast_l41_a10_actual_posneg_tf_quick.py
steering/configs/coast_l41_a10_random_same_spectrum_posneg_tf_quick.py
steering/configs/coast_l41_a10_swap_posneg_tf_quick.py
```

Comparison analyzer:

```text
steering/analysis/compare_coast_control_runs.py
```

Decision rule:

```text
If actual COAST beats random_same_spectrum at matched beta/base/mode/aperture,
especially in teacher-forced KL and structured per-token divergence, then the
learned POS/NEG orientation matters. If random_same_spectrum matches actual,
the current COAST effect must be treated as generic low-rank perturbation until
the operator is redesigned.
```

## 2026-06-03 - COAST A10 equal-beta controls: specificity not validated

Source artifacts:

```text
C:\Users\stasv\Downloads\coast_a10_control_comparison.zip
C:\Users\stasv\Downloads\coast_l41_a10_actual_posneg_tf_quick.zip
C:\Users\stasv\Downloads\coast_l41_a10_random_same_spectrum_posneg_tf_quick.zip
C:\Users\stasv\Downloads\coast_l41_a10_swap_posneg_tf_quick.zip
```

Detailed log:

```text
steering/docs/coast_conceptor_findings.md
```

Run shape:

```text
operator modes = actual, random_same_spectrum, swap_pos_neg
layer = 41
aperture = 10.0
base_condition = pos + neg
tasks = 3
rows per operator = 48
per-token TF rows per operator = 2748
```

Main result:

```text
The equal-beta A10 control did not validate COAST specificity. At beta 0.381,
random_same_spectrum was much stronger than actual COAST on final KL,
teacher-forced KL, logit L2, and per-token high-KL counts. Actual and swap were
often nearly tied.
```

Key numbers:

```text
NEG / greedy / beta 0.381:
  final KL actual = 0.109073
  final KL random = 0.583680
  final KL swap   = 0.107161

  TF-KL actual = 0.083240
  TF-KL random = 0.319150
  TF-KL swap   = 0.085899

POS / sampled / beta 0.381:
  final KL actual = 0.157229
  final KL random = 0.779660
  final KL swap   = 0.190659

  TF-KL actual = 0.326677
  TF-KL random = 0.832924
  TF-KL swap   = 0.328931
```

Per-token result:

```text
POS / sampled / beta 0.381:
  mean per-token KL actual = 0.307545
  mean per-token KL random = 0.810942
  mean per-token KL swap   = 0.308491

NEG / greedy / beta 0.381:
  mean per-token KL actual = 0.093600
  mean per-token KL random = 0.342644
  mean per-token KL swap   = 0.099941
```

Mechanistic interpretation:

```text
The current equal-beta control is compatible with generic low-rank perturbation
or update-size effects. It does not prove that learned POS/NEG conceptor
orientation is the specific cause of the observed COAST movement.
```

Important boundary:

```text
This does not fully kill COAST. The control matched beta and conceptor spectrum,
but not realized residual-stream update norm. The random operator appears to
produce a much larger effective activation/logit perturbation at the same beta.
Therefore the next fair test must match operator-wise update norm, not beta.
```

Next required experiment:

```text
Run operator-wise norm-matched controls:
  actual beta for 5% / 10% median residual update
  random_same_spectrum beta for the same 5% / 10% median residual update
  swap_pos_neg beta for the same 5% / 10% median residual update

If random remains comparable or stronger after norm matching, COAST should be
treated as generic low-rank perturbation. If actual becomes more behaviorally
structured at equal update norm, learned orientation matters.
```

Boundary relative to the main research claim:

```text
This COAST control result does not weaken the main geometry-coordinate result:
context -> hidden-state shift -> measurable latent axes / coordinates.

It only limits the newer steering interpretation. The failed equal-beta
specificity test says: do not yet claim that the learned POS/NEG conceptor is a
clean behavioral steering handle. It does not say that the original
context-induced latent-state shift was false.
```

## 2026-06-03 - Refocus after COAST controls: main line remains geometry-coordinate proof

Decision:

```text
Do not let COAST become the main research frame. COAST is a secondary
intervention/steering appendix. The central result remains context-induced
hidden-state geometry shift with coordinates relative to discovered latent axes.
```

Main frame:

```text
1. Strong coherent target context changes the model's inference-time
   residual-stream / hidden-state trajectory.

2. The shift is measured internally, not inferred only from visible output.

3. Shuffled-content controls separate content overlap from coherent
   order/response-regime structure.

4. Coordinates are relative to discovered axes:
   x_full, x_content, x_order, x_order_orth.

5. Causal/component tests support involvement but not clean steering dominance.
```

Where COAST fits:

```text
COAST asks a later question:
Can we re-enter or perturb the discovered target/control response-mode
subspace through a conceptor operator?

Current answer:
The operator produces real movement, but equal-beta controls do not yet prove
specificity. Therefore COAST should be presented only as exploratory
intervention evidence, not as the main proof.
```

Next work order:

```text
1. Finish the geometry-coordinate evidence package.
2. Keep the Reddit post / abstract focused on hidden-state shift and controls,
   not on COAST steering.
3. Use COAST only as a cautious appendix: "exploratory intervention shows
   trajectory perturbation, but specificity controls are still open."
4. Do not run more COAST until the main package is clean, unless the next COAST
   run is explicitly norm-matched and designed as an appendix control.
```

## 2026-06-03 - COAST norm-matched control setup

Decision:

```text
Continue COAST only as an experimental appendix branch. The immediate next
COAST step is norm-matched operator control, not abstract/reddit revision and
not a larger equal-beta run.
```

Protocol:

```text
steering/docs/coast_norm_matched_control_protocol.md
```

New files:

```text
steering/05_operator_norm_matched_beta_calibration.py
steering/configs/coast_l41_a10_operator_norm_calibration.py
```

Purpose:

```text
Calibrate beta separately for actual, random_same_spectrum, and swap_pos_neg so
all operators are compared at the same realized median residual-stream update:

  median ||T(x) - x|| / ||x||
```

Generated run configs after calibration:

```text
coast_l41_a10_normmatched_actual.py
coast_l41_a10_normmatched_random_same_spectrum.py
coast_l41_a10_normmatched_swap_posneg.py
```

Interpretation rule:

```text
If random_same_spectrum remains comparable/stronger after update-norm matching,
COAST is generic low-rank trajectory perturbation. If actual becomes more
structured at equal update norm, learned POS/NEG orientation matters.
```

## 2026-06-03 - COAST norm-matched control result: no clean specificity

Source artifacts:

```text
C:\Users\stasv\Downloads\coast_a10_normmatched_control_comparison.zip
C:\Users\stasv\Downloads\conceptor_operator_norm_calibration_summary_coast_l41_a10_operator_norm_calib.csv
C:\Users\stasv\Downloads\coast_l41_a10_normmatched_actual.zip
C:\Users\stasv\Downloads\coast_l41_a10_normmatched_random_same_spectrum.zip
C:\Users\stasv\Downloads\coast_l41_a10_normmatched_swap_posneg.zip
```

Detailed log:

```text
steering/docs/coast_conceptor_findings.md
```

Calibration result:

```text
Operator-specific betas successfully matched median residual update levels.

actual:
  1% beta = 0.039793
  5% beta = 0.198967
  10% beta = 0.397935

random_same_spectrum:
  1% beta = 0.038264
  5% beta = 0.191318
  10% beta = 0.382636

swap_pos_neg:
  1% beta = 0.039779
  5% beta = 0.198894
  10% beta = 0.397788
```

Main result:

```text
Norm matching did not validate actual COAST specificity. At 5% update, actual
and random are close, with random often slightly stronger. At 10% update,
random is much stronger/noisier; swap remains close to actual in many cells.
```

Key numbers:

```text
10% update / NEG greedy:
  final KL actual = 0.124828
  final KL random = 1.194093
  final KL swap   = 0.143286

  TF-KL actual = 0.100347
  TF-KL random = 0.423480
  TF-KL swap   = 0.098819

10% update / POS sampled:
  final KL actual = 0.200348
  final KL random = 1.105405
  final KL swap   = 0.260006

  TF-KL actual = 0.365468
  TF-KL random = 1.002669
  TF-KL swap   = 0.369239
```

Behavioral read:

```text
Actual can move NEG/NATO sampled generations from context-bound/library mode
toward direct NATO explanation, but random and swap can also produce similar
shifts. Random often creates stronger malformed-token artifacts and larger
per-token KL spikes.
```

Conclusion:

```text
COAST/conceptor intervention is real as trajectory perturbation, but the
current C_pos(I-C_neg) operator is not validated as a clean specific
target/control steering handle. Treat COAST as exploratory appendix evidence,
not as the main causal proof.
```

Research decision:

```text
Stop expanding COAST for now. Return focus to the main Grade 3/4
geometry-coordinate result. If COAST resumes later, redesign the operator
instead of running more broad sweeps: use x_order_orth-style orthogonalized
directions, better contrastive objectives, or position-selective patching.
```

## 2026-06-03 - COAST-v2 axis intervention setup

Decision:

```text
The old C_pos(I-C_neg) conceptor operator remains an exploratory appendix
result only. It produced real trajectory perturbations, but norm-matched
random/swap controls did not validate clean specificity.
```

Next COAST-only step:

```text
Run one narrower COAST-v2 axis test, not another broad conceptor sweep.
```

Implemented operator modes:

```text
mean_diff_add:
  add mean(hidden_states_pos_fit) - mean(hidden_states_neg_fit)

random_mean_diff_add:
  add a same-norm random axis

mean_diff_subtract:
  add the negative learned axis
```

Files:

```text
steering/03_conceptor_subspace_steering.py
steering/05_operator_norm_matched_beta_calibration.py
steering/configs/coast_l41_a10_axis_operator_norm_calibration.py
steering/docs/coast_axis_intervention_protocol_ru.md
```

Success criterion:

```text
learned mean_diff_add must beat same-norm random_mean_diff_add and separate
from mean_diff_subtract across final KL, teacher-forced KL, jaccard movement,
per-token KL, and readable output behavior.
```

Boundary:

```text
This does not replace the main Grade 3/4 hidden-state geometry claim. It only
tests whether a discovered POS-vs-NEG direction can become a controlled
intervention handle. If it fails, COAST should be closed as exploratory
appendix work, while the main context-induced latent-state shift result
remains intact.
```

## 2026-06-03 - COAST-v2 axis result: simple mean-diff handle failed specificity

Source artifacts:

```text
C:\Users\stasv\Downloads\coast_a10_axis_normmatched_control_comparison.zip
C:\Users\stasv\Downloads\conceptor_operator_norm_calibration_summary_coast_l41_a10_axis_operator_norm_calib.csv
C:\Users\stasv\Downloads\conceptor_operator_norm_calibration_fit_summary_coast_l41_a10_axis_operator_norm_calib.csv
C:\Users\stasv\Downloads\coast_l41_a10_axis_normmatched_mean_diff_add.zip
C:\Users\stasv\Downloads\coast_l41_a10_axis_normmatched_random_mean_diff_add.zip
C:\Users\stasv\Downloads\coast_l41_a10_axis_normmatched_mean_diff_subtract.zip
```

Detailed local analysis:

```text
steering/runs/coast_axis_normmatched_analysis/summary.md
```

Calibration:

```text
mean_diff_norm = 6975.239258
median update ratio beta=1 = 0.062160
1% beta = 0.160875
5% beta = 0.804374
10% target clamped at beta=1, so realized median update is only ~6.216%.
```

Main result:

```text
The simple learned POS-vs-NEG mean-diff axis did not beat the same-norm random
axis. Random was stronger on final KL, TF-KL, logit-L2, top-token changes, and
per-token KL spikes. The negative learned axis was too close to learned,
especially in TF-KL.
```

Key beta=1 numbers:

```text
Final KL / POS greedy:
  learned  = 0.055679
  random   = 0.121840
  negative = 0.029856

TF-KL / POS greedy:
  learned  = 0.037785
  random   = 0.318847
  negative = 0.043285

TF-KL / POS sampled:
  learned  = 0.036841
  random   = 0.373132
  negative = 0.045237
```

Across nonzero beta cells:

```text
mean_final_kl learned > random rate = 0.000
mean_tf_kl    learned > random rate = 0.000
max_tf_kl     learned > random rate = 0.000
```

Per-token:

```text
At beta=1, POS greedy:
  learned:  KL>1 = 0,  KL>5 = 0, max KL = 0.607828
  random:   KL>1 = 10, KL>5 = 6, max KL = 11.612469
  negative: KL>1 = 0,  KL>5 = 0, max KL = 0.972140
```

Behavioral read:

```text
On BASE neg, learned does not reliably overcome the control/library context.
On BASE pos, learned outputs are readable, but negative is also readable and
random often produces similar semantics with malformed artifacts. This is not
validated directional steering.
```

Mechanistic conclusion:

```text
mean(hidden_states_pos_fit) - mean(hidden_states_neg_fit) is too crude as a
global intervention axis. It likely mixes content, response frame, residual
scale offsets, and prompt artifacts. It is not equivalent to the Grade 4
orthogonalized x_order_orth component.
```

Research decision:

```text
Close the current COAST branch as exploratory appendix evidence. Do not present
C_pos(I-C_neg) or mean_pos - mean_neg as validated behavioral steering handles.
The main Grade 3/4 context-induced hidden-state geometry result remains intact.
If an intervention appendix resumes, use Grade-derived component axes directly
with position-selective patching and held-out prompts.
```

## Gemma SAE Candidate Patching V2 Status

Date: 2026-06-05

Input run:

```text
RUN_TAG = gemma_sae_order_feature_patching2
TOP_K_CANDIDATES = 50
MAX_FEATURES_PER_LAYER = None
N_CONTEXT_FEATURES = None
TOP_N_CONTEXTS = 50
CONTEXT_WINDOW = 50
PATCH_FEATURE_BATCH_SIZE = 16
PATCH_MODE = zero
PATCH_POSITION_MODE = all_tokens
PREPEND_BOS = True
```

Tables loaded:

```text
sae_order_feature_contrast.csv rows = 278
sae_reconstruction_quality.csv rows = 360
sae_grade4_component_feature_summary.csv rows = 768
sae_prompt_feature_delta_summary.csv rows = 960
sae_generation_feature_summary.csv rows = 960
sae_generation_top_features.csv rows = 1728
sae_top_changed_features.csv rows = 960
sae_model_compatibility.csv rows = 6
sae_prompt_feature_activation_summary.csv rows = 1152
```

SAE quality:

```text
mean reconstruction cosine across layers = 0.996023
minimum layer reconstruction cosine       = 0.979815
```

Candidate/context readout:

```text
Top context-active order candidates include:
  real_layer 41 / CSV layer 42 / feature 207
  real_layer 41 / CSV layer 42 / feature 29
  real_layer 41 / CSV layer 42 / feature 208
  real_layer 36 / CSV layer 37 / feature 323
  real_layer 41 / CSV layer 42 / feature 13686

Other candidate classes:
  real_layer 30 features 58, 161, 451 are strong order-enriched overlap features.
  real_layer 24 feature 76 is a strong order-enriched overlap feature.
  real_layer 18 has several prompt/generation candidates, but their current
  rough KL signal is not yet clean.
```

Important correction:

```text
The first rough_sae_zero_ablation_logit_results.csv from this run is not clean
single-feature causality. The script version used SAE decoded_patched as the
whole residual activation, so the measured KL included SAE reconstruction
replacement error plus feature removal.

The script has been corrected to use:
  activation + (decode(latent_patched) - decode(latent_original))

This delta update is the correct clean single-feature intervention for rough
SAE ablation. Before using KL/logit values for 02_scale_calibration.py or
sae_steering_with_kl_full.py, rerun 01b with:
  PATCH_RESIDUAL_UPDATE_MODE = "delta"
  RUN_TOP_CONTEXT_INSPECTION = False
```

Follow-up clean delta rerun:

```text
RUN_TAG = gemma_sae_order_feature_patching2_delta_clean
PATCH_RESIDUAL_UPDATE_MODE = delta
RUN_TOP_CONTEXT_INSPECTION = False
PATCH_FEATURE_BATCH_SIZE = 16
PATCH_POSITION_MODE = all_tokens
```

Clean rough ablation status:

```text
max KL(base||patched)  = 0.117562
mean KL(base||patched) = 0.00676843
mean top-token changed rate = 0.0
```

Interpretation:

```text
The clean single-feature delta intervention no longer produces the artificial
large KL from SAE reconstruction replacement. Single features have real but
moderate logit effects. This supports SAE features as partial readout/mediator
candidates, not as complete single-feature control handles.
```

Best clean ablation candidates:

```text
real_layer 18 feature 378:  KL = 0.117562, prompt-specific order feature
real_layer 18 feature 373:  KL = 0.102871, generation-persistent order feature
real_layer 36 feature 323:  KL = 0.048566, order-specific generation-persistent
real_layer 24 feature 76:   KL = 0.031414, order-enriched overlap
real_layer 41 feature 207:  KL = 0.011883, order-specific generation-persistent
real_layer 36 feature 1914: KL = 0.008440, order-component-specific
```

Strong readout/context candidates with weak clean ablation:

```text
real_layer 41 feature 29:    high order score/context, KL = 0.000093
real_layer 41 feature 208:   high order score/context, KL = 0.000161
real_layer 41 feature 13686: high order/context, KL = 0.000010
real_layer 30 feature 58:    strongest order-enriched score, KL = 0.000144
real_layer 30 feature 161:   order-enriched, KL = 0.002749
```

Next action:

```text
Use 02_scale_calibration.py on a mixed shortlist:
  causal-ablation probes: (18,378), (18,373), (36,323), (24,76), (41,207), (36,1914)
  readout/steering probes: (41,29), (41,208), (41,13686), (30,58), (30,161)

Then run sae_steering_with_kl_full.py only after scale calibration selects
mild/medium/stress scales.
```

## Old BASE_TARGET vs BASE_CONTRL SAE Decoder Steering Readout

Recorded analysis:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/sae_decoder_steering_base_target_control_old_run_readout_ru.md
```

Source folders:

```text
C:\Users\stasv\Downloads\BASE_TARGET
C:\Users\stasv\Downloads\BASE_CONTRL
```

Status:

```text
This is not COAST and not the primary Grade 4 proof. It is an old paired
sae_steering_with_kl_full.py run with the same late SAE decoder directions
tested under target-base and control-base context.
```

Main visible signal:

```text
scale=0 no-info / text-does-not-contain frame:
  control: 12 / 40 rows = 0.30
  target:   0 / 40 rows = 0.00

all rows:
  control: 60 / 200 rows = 0.30
  target:   0 / 200 rows = 0.00
```

Interpretation:

```text
The old paired run is useful as an appendix / qualitative mechanism bridge:
base context changes the response-construction regime, and the same SAE
decoder directions can modulate answers differently under target-base vs
control-base. It should not replace the main hidden-geometry evidence:
shuffle controls, x_content / x_order_orth separation, generation trajectory
readout, and norm-controlled component intervention.
```

Qualitative answer-regime readout:

```text
BASE_TARGET visibly pushes Gemma into a direct analytical answer regime across
the five political/high-friction tasks.

BASE_CONTRL sometimes leaves Gemma in a local-text/procedural regime, where it
answers "the provided text does not contain information" despite the task
asking for a direct answer based on model knowledge.

This visible difference is not the primary proof, but it is the behavioral
shadow of the already established geometry result: target context changes the
internal processing regime, while control context can keep the model tied to
local text grounding.
```

## 2026-06-05 - COAST / POS-NEG conceptor branch archived

Physical archive path:

```text
archive/coast_posneg_conceptor_branch/
```

Moved out of active `steering/`:

```text
03_conceptor_subspace_steering.py
04_beta_calibration.py
04_beta_calibration_V2.py
05_operator_norm_matched_beta_calibration.py
COAST configs
COAST analysis scripts
COAST docs
COAST run outputs
```

Reason:

```text
COAST is a secondary exploratory appendix branch. It showed real
trajectory-level perturbation, but matched controls did not establish clean
specificity of learned POS/NEG conceptor orientation. The active steering
folder should stay focused on SAE / Grade 4 / x_order_orth axis work.
```

Do not merge COAST back into the main claim unless a redesigned specificity
test succeeds.

## 2026-06-05 - SAE / Gemma / Qwen steering workspace organized

Main local steering workspace:

```text
experiments/steering/sae_gemma_qwen/
```

Start here:

```text
experiments/steering/sae_gemma_qwen/README_RU_EN.md
```

Per-script English documentation:

```text
experiments/steering/sae_gemma_qwen/script_docs/
```

Coverage:

```text
18 Python scripts
18 matching per-script markdown docs
```

Active Gemma pipeline:

```text
experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
experiments/steering/sae_gemma_qwen/gemma_active/02_scale_calibration.py
experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

Legacy Gemma scripts moved but preserved:

```text
experiments/steering/sae_gemma_qwen/gemma_legacy/01_candidate_discovery_and_rough_sae_patching.py
experiments/steering/sae_gemma_qwen/gemma_legacy/sae_feature_steering_light.py
experiments/steering/sae_gemma_qwen/gemma_legacy/sae_feature_steering_v2_no_control.py
experiments/steering/sae_gemma_qwen/gemma_legacy/steering_gemma3_V1.py
```

Qwen reference snapshot:

```text
experiments/steering/sae_gemma_qwen/qwen_reference/
```

Important boundary:

```text
Qwen reference files are nearby copies for inspection. The canonical Qwen
workspace remains model_workspaces/qwen3_5_9b_qwen_scope/.
```

Colab note:

```text
If files are uploaded to the Colab root, it is still fine to run `%run -i
filename.py`. The local folder layout is for project organization, audit trail,
and avoiding confusion between active, legacy, Qwen, and COAST branches.
```

## 2026-06-06 - Experiments workspace created

Main experimental code container:

```text
experiments/
```

The two main code families now live side by side:

```text
experiments/grade4_axis_decomposition/
experiments/steering/
```

Root-level compatibility pointers:

```text
grade4_axis_decomposition/README.md
steering/README.md
```

These root-level folders now contain README pointers only. The real content is
inside `experiments/`.

Important current paths:

```text
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
experiments/grade4_axis_decomposition/metrics/
experiments/steering/sae_gemma_qwen/
experiments/steering/sae_gemma_qwen/script_docs/
```

## 2026-06-06 - Gemma SAE decoder-direction steering, two new 3-task runs

Analyzed two new Gemma SAE steering packages:

```text
experiments/steering/metrics/01_gemma_sae_steering_movers_3tasks.zip
experiments/steering/metrics/gemma_sae_steering_fast_readout_3tasks.zip
```

Detailed report:

```text
experiments/steering/sae_gemma_qwen/gemma_runs/sae_steering_3tasks_new_runs_analysis_ru.md
```

Run A, causal movers:

```text
rows = 360
features = 6
tasks = 3
generation = greedy + sampled
generation errors = 0
final-KL errors = 0
TF-KL errors = 0
prompt truncation = 0
scale=0 sanity:
  mean final KL = 0
  exact = 1
  jaccard = 1
```

Strongest causal mover:

```text
L36 f1914, negative scale -8460
mean final KL ~= 7.2831
mean TF-KL    ~= 7.9051
max TF-KL     ~= 41.570
mean jaccard  ~= 0
final top-token changed rate = 1.0
```

Additional language/script audit:

```text
The flagship L36 f1914, scale=-8460 effect is not a clean semantic/procedural
regime readout by itself. All 12/12 outputs at this setting switch from Russian
baseline to English output, so near-zero jaccard is mostly explained by RU->EN
language/script switching.

L36 f323 at scale=-8460 also shows 12/12 script-switch relative to baseline,
usually as RU/EN mixed output, with mean TF-KL ~= 3.1673.

Updated reading: L36 f1914 and L36 f323 are real causal movers of the
generation distribution, but the strongest high-scale visible effect is
language/script-mode flip with partially preserved analytic stance. Whether
these features control procedural/local-document vs direct-analytic behavior
remains open and requires paired target/control plus language-aware semantic
metrics.
```

Mechanistic reading:

```text
L36 f1914 is not merely a readout marker. Its decoder direction is a strong
causal mover in the negative direction. It can sharply alter the generation
distribution, but current -8460 evidence should be read primarily as a
threshold-like language/script flip, not as closed proof of procedural-vs-direct
behavioral control.
```

Second strongest mover:

```text
L36 f323, negative scale -8460
mean final KL ~= 0.3509
mean TF-KL    ~= 3.1673
max TF-KL     ~= 26.212
mean jaccard  ~= 0.016
```

Other movers:

```text
L18 f373, L18 f378, L24 f76 produce moderate generation/TF shifts.
L41 f207 is weak as a steering mover despite being a late/context readout
candidate.
```

Run B, late/readout probes:

```text
rows = 300
features = 5
tasks = 3
generation = greedy + sampled
errors/truncation = 0
```

Main Run B result:

```text
Late/readout probes are mostly weak as causal decoder-direction steering
handles. Best Run B feature is L30 f161, but it is still much weaker than
Run A L36 f1914 and L36 f323.
```

Cross-run interpretation:

```text
The new runs strengthen the claim that the context-induced latent shift has
sparse SAE-accessible causal handles, especially around layer 36.

They also sharpen the distinction between readout and control:
high x_order_orth/readout features are not automatically strong behavioral
steering handles. A feature can read the hidden state well while its decoder
direction is weak as an intervention vector.
```

Updated status:

```text
descriptive hidden-state shift: strong
content/order separation: strong
SAE sparse readout: strengthened
SAE causal involvement: strengthened for L36 f1914 and L36 f323
stable bidirectional behavioral control: still not proven
```

Next experiment:

```text
Focused rerun on L36 f1914 and L36 f323 with a narrower scale grid around the
strong negative direction. Goal: separate useful regime shift from brute-force
degradation/language flip, add language/script metrics and cross-lingual
semantic similarity, and test paired target/control base conditions in one run.
```

Focused L36 target-base run reminder:

```text
Artifact: gemma_sae_focused_l36_target.zip
RUN_TAG = gemma_sae_focused_l36_target_3tasks
BASE_TEXT = prompts_target[0]

Rows = 132
Features = L36 f323, L36 f1914
Tasks = 3
Scales per feature =
  -8460, -7000, -5500, -4230, -3000, -2000, -1000, 0, 1000, 4230, 8460
Generation modes = greedy + sampled
Errors = 0
Teacher-forced KL errors = 0
No-info / local-document procedural collapse rows = 0 / 132

Scale-0 target-base outputs are direct analytical answers in Russian for all
three tasks. This is important: under target-base, the model does not fall into
the "provided text contains no answer" procedural/local-document mode.

Strongest target-base perturbation remains L36 f1914 at negative scale:
  f1914, -8460:
    mean final next-token KL ~= 6.83
    mean TF-KL ~= 7.39-7.77 depending on greedy/sample mode
    max TF-KL ~= 30.28
    jaccard to scale0 ~= 0.0
    top-token-changed fraction ~= 0.88-0.89
    dominant visible effect = Russian -> English / code-switching, while
      much of the analytical stance/content survives.

Second mover:
  f323, -8460:
    mean final next-token KL ~= 0.15
    mean TF-KL ~= 2.84-3.42
    max TF-KL ~= 30.63
    jaccard to scale0 ~= 0.04-0.05
    dominant visible effect = mixed Russian/English output, not clean
      procedural-vs-direct control.

Interpretation:
  Target-base focused L36 run strengthens the claim that L36 f1914/f323 are
  real causal movers, but the strongest negative-scale effect is still mostly a
  language/script threshold effect. It does not by itself prove stable
  procedural-vs-direct behavioral steering. The key comparison is the pending
  paired control-base run with identical tasks/config.
```

Focused L36 control-base run with direct-answer system prompt:

```text
Artifact: gemma3_12b_sae_focused_l36_control_3tasks.zip
RUN_TAG = gemma_sae_focused_l36_control_3tasks
BASE_TEXT = prompts_control[0]
Separate base-text file:
  sae_feature_steering_base_text_gemma_sae_focused_l36_control_3tasks.txt

Base text integrity:
  chars = 8630
  sha256 = cf3ea5010fe9f7b434fba550499527b13b8765d59ae0af113cbd0a75f95bd559
  CSV base_text_sha256 matches this value.
  prompt_token_count = 2449..2460
  final_kl_prompt_truncated = 0 for all rows
  tf_prompt_truncated = 0 for all rows

Rows = 132
Features = L36 f323, L36 f1914
Tasks = 3
Scales per feature =
  -8460, -7000, -5500, -4230, -3000, -2000, -1000, 0, 1000, 4230, 8460
Generation modes = greedy + sampled, n_samples=1
Errors = 0
Teacher-forced KL errors = 0
No-info / local-document procedural collapse rows = 0 / 132

Important interpretation boundary:
  This is not a pure reproduction of the earlier natural control-base
  procedural/no-info collapse, because SYSTEM_PROMPT and PROMPT_PREAMBLE
  explicitly force direct answering from general knowledge and forbid
  "the text contains no answer" behavior. Therefore no-info collapse being
  absent is expected and does not falsify the earlier base-condition asymmetry.

What this run does test:
  It tests L36 f1914/f323 steering under a direct-answer/Russian-only control
  base condition. It is useful for target-vs-control comparison under matched
  direct-answer instruction, but not for measuring natural local-document
  procedural fallback.

Strongest control-base perturbation:
  f1914, -8460:
    mean final next-token KL ~= 6.11
    mean TF-KL ~= 7.41-8.29 depending on sample/greedy mode
    max TF-KL ~= 42.13
    jaccard to scale0 ~= 0.0
    top-token-changed fraction ~= 0.88-0.89
    dominant visible effect = Russian -> English / code-switching, despite
      explicit Russian-only instruction.

Second mover:
  f323, -8460:
    mean final next-token KL ~= 0.27
    mean TF-KL ~= 2.90-3.50
    max TF-KL ~= 30.38
    jaccard to scale0 ~= 0.009-0.011
    dominant visible effect = mixed Russian/English or English-majority output.

Conclusion:
  The control-base direct-instruction run confirms the same threshold-like
  language/script vulnerability of L36 negative steering. It does not show
  procedural/no-info collapse because the prompt configuration deliberately
  suppresses that mode.
```

Metric snapshot location:

```text
Primary detailed report:
  experiments/steering/sae_gemma_qwen/gemma_runs/focused_l36_target_control_analysis_ru.md

Compact CSV:
  experiments/steering/sae_gemma_qwen/gemma_runs/focused_l36_postprocess/focused_l36_metric_snapshot.csv

Recorded metric families:
  final next-token KL
  teacher-forced KL
  max teacher-forced KL
  teacher-forced top-token changed fraction
  lexical Jaccard vs scale0
  Cyrillic fraction
  script-switch rate

Fixed conclusion:
  L36 f1914/f323 are working causal movers. The strongest -8460 effect is a
  language/script threshold effect with very large KL and top-token movement,
  not clean proof of procedural-vs-direct behavioral steering.
```

Regime-axis Grade bridge script added:

```text
Script:
  experiments/steering/sae_gemma_qwen/gemma_active/regime_axis_grade_bridge_causal_audit.py

Run note:
  experiments/steering/sae_gemma_qwen/gemma_runs/regime_axis_grade_bridge_causal_audit_plan_ru.md

Purpose:
  This is the stronger follow-up to Claude's `regime_diff_steering.py`.
  Claude's script builds a bank-level contrast vector:

    v_regime = mean(target_bank) - mean(control_bank)

  and tests bidirectional injection:

    control + alpha*v_regime
    target  - alpha*v_regime

  The new script keeps that causal intervention but adds the missing audit
  layer: train/test split, held-out projection separation, Grade-axis overlap,
  SAE overlap, projection-out variants, random same-norm axes, label-permuted
  axes, small-alpha causal generation, final next-token KL, top-token-change,
  script-switch metrics, hedging/procedural/directness proxies, and a claim
  ladder.

Scientific question:
  Is the Claude-style bank-level regime vector a genuinely independent regime
  attractor, or mostly a recombination of already-known Grade/SAE geometry
  such as x_content, x_order_orth, L36 f1914, and L36 f323?

Positive result would mean:
  held-out target/control separation survives Grade+SAE projection-out and
  bidirectional generation steering beats same-norm random/permutation
  controls without collapsing into RU->EN script switching.

Negative result would mean:
  raw v_regime is useful but not a new independent axis; it is mostly a
  recombination of existing content/order/SAE directions.

Boundary:
  This does not replace the Grade4 content/order result. It is a separate
  bank-level regime-attractor audit.
```

Regime bridge verdict analyzer added:

```text
Script:
  experiments/steering/sae_gemma_qwen/gemma_active/regime_bridge_verdict.py

Input:
  regime_bridge_causal_generation_<RUN_TAG>.csv

Purpose:
  This is a post-hoc behavioral verdict layer for
  regime_axis_grade_bridge_causal_audit.py. It does not rerun activations or
  generations. It reads causal-generation rows and checks whether actual
  intervention effects exceed same-norm random controls on a normalized
  directness metric, while excluding script-switch rows from that directness
  score.

Extra gate:
  It adds a "target gate" / gap-closed reading: control+regime should move
  toward target-baseline behavior, and target-regime should move toward
  control-baseline behavior. This catches the failure mode where an axis moves
  generation away from its starting point but not toward the intended opposite
  side.

Optional:
  If sentence-transformers/LaBSE is available, it can add language-agnostic
  semantic similarity to target/control baselines. Treat that as useful
  supporting evidence, not as a replacement for hidden-state projection and
  random/permutation controls.

Use after each full bridge run:
  python experiments/steering/sae_gemma_qwen/gemma_active/regime_bridge_verdict.py \
    <path>/regime_bridge_causal_generation_<RUN_TAG>.csv --no-semantic
```

Regime bridge clean-window run analyzed:

```text
Source archive:
  C:\Users\stasv\Downloads\regime_bridge_gemma_sae_focused_l36_clean_window.zip

Detailed report:
  experiments/steering/sae_gemma_qwen/gemma_active/research_synthesis/regime_bridge_results/runs/regime_bridge_gemma_sae_focused_l36_clean_window/REGIME_BRIDGE_SYNTHESIS.md

Canonical bridge index:
  experiments/steering/sae_gemma_qwen/gemma_active/research_synthesis/regime_bridge_results/README.md

Run tag:
  gemma_sae_focused_l36_clean_window_target_3tasks

Actual run shape:
  hook=blocks.36.hook_resid_post
  pool=prompt_mean
  d_model=3840
  target_train=6, control_train=7
  target_test=3, control_test=3
  axis_tasks=2, eval_tasks=2
  position_mode=all_tokens
  v_regime_norm=4514.54

Main geometry result:
  raw held-out AUC-like=1.000, gap=3705.91
  sae_orth AUC-like=1.000, gap=3487.27
  grade_orth AUC-like=1.000, gap=3634.80
  grade_sae_orth AUC-like=1.000, gap=3345.25

Controls:
  random same-dim max gap=87.73, p95 gap=67.31
  label-permutation max gap=3412.01, p95 gap=3407.84

Interpretation:
  The run strongly supports a residual L36 bank-level hidden-state regime
  contrast that survives removing known Grade axes and selected SAE directions.
  However, because label-permutation controls sometimes produce large gaps in
  this small clean-window bank, the independence claim is not fully closed.

Overlap:
  cos(v_regime, x_full)=+0.316
  cos(v_regime, x_content)=+0.241
  cos(v_regime, x_order_orth)=+0.205
  cos(v_regime, x_order)=-0.032
  cos(v_regime, L36 f323)=+0.409
  cos(v_regime, L36 f1914)=-0.233

Norm kept after projection-out:
  SAE-only removal kept 87.66%
  Grade-only removal kept 94.86%
  Grade+SAE removal kept 88.99%

Causal generation verdict:
  `regime_bridge_verdict.py --no-semantic` was run and produced
  regime_bridge_causal_generation_gemma_sae_focused_l36_clean_window_target_3tasks_VERDICT.csv.
  Directness/100w does not show a stable effect. Control-side rows do not move
  on directness. Target-side movement appears mainly at alpha=0.5 and is
  low-n. KL movement is not consistently stronger than same-norm random axes:
  control-side actual KL at alpha=0.5 is weaker than random, while target-side
  actual KL is roughly matched/slightly stronger.

Scientific status:
  Strong hidden-state diagnostic/readout axis; behavioral steering not proven.
  The right formulation is "residual regime readout exists" rather than
  "stable bidirectional causal steering handle is closed."

Next experiment:
  Expand to larger bank and stronger controls:
    REGIME_BRIDGE_N_EVAL_TEXTS_PER_SIDE = 3
    REGIME_BRIDGE_N_RANDOM_CAUSAL_AXES = 4
    REGIME_BRIDGE_N_RANDOM_AXES = 64
    REGIME_BRIDGE_N_PERMUTATION_AXES = 64
    REGIME_BRIDGE_N_AXIS_TASKS = 2
    REGIME_BRIDGE_N_EVAL_TASKS = 3
  Use less politically loaded eval tasks and, if possible, run verdict with
  semantic scoring instead of only marker-based directness.
```

Regime bridge prompt-mode correction:

```text
The bridge script now supports:

  REGIME_BRIDGE_PROMPT_MODE = "analyze_text"
  REGIME_BRIDGE_PROMPT_MODE = "context_probe"

`analyze_text` is the old/default mode: the task asks the model to analyze the
stimulus text. This is useful for text-comprehension/readout tests but mixes
content comprehension with regime transfer.

`context_probe` treats the target/control text as a conditioning context and
then asks an independent downstream task. This is the cleaner causal-readout
mode for the regime-attractor question, because it tests whether the hidden
state induced by the text transfers to unrelated behavior.

Next bridge causal runs should prefer `context_probe` unless the specific goal
is text-analysis behavior.
```

Regime bridge input-contract cleanup:

```text
The active bridge script now treats `prompts_target` and `prompts_control` as
the normal input contract. Legacy `TARGET_BASE_TEXTS` / `CONTROL_BASE_TEXTS`
are ignored by default unless `REGIME_REQUIRE_PROMPT_BANKS=False`, because
notebook globals can remain stale across `%run -i` calls.

Recommended guard for 20/20 runs:

  REGIME_EXPECTED_TARGET_TEXTS = 20
  REGIME_EXPECTED_CONTROL_TEXTS = 20

If the notebook passes fewer/more texts, the script stops before running the
audit. Each run also writes:

  regime_bridge_input_split_<RUN_TAG>.csv

with source name, original text index, split, hash, char count, and preview.
This prevents another ambiguous "where did the texts go" situation.
```

Deep proof ladder for the next phase:

```text
The project should now separate five proof levels:

1. Descriptive hidden-state shift:
   target/control contexts occupy separable regions in residual-stream space.
   This is already strongly supported.

2. Content/order decomposition:
   sentence shuffle and orthogonalized axes show the shift is not reducible to
   token content alone. This is already strongly supported by Grade4.

3. Independent regime residual:
   a bank-level v_regime remains predictive after removing x_content,
   x_order_orth, and selected SAE decoder directions. The clean-window bridge
   supports this, but larger/permutation-stronger runs are still needed.

4. Causal state involvement:
   interventions along the discovered direction move hidden trajectory or
   next-token distribution more than same-norm random controls. Partially
   supported; behavioral causal steering remains weak/inconclusive.

5. Agent-safety relevance:
   the shifted internal state changes action-relevant decisions before final
   answer filtering: tool choice, memory write, stop/continue choice,
   self-monitoring, or policy/risk classification. This is not yet proven and
   is the deepest next target.

The deepest useful next claim is not "the model outputs different text." It is:

  context -> hidden-state regime shift -> changed downstream decision state

where the downstream decision is measured before or around action selection,
not only after final visible generation.
```

2026-06-09 bridge rerun `l36_context_probe_clean_window`:

```text
Input split was correct:
  target  = 6 train + 3 test = 9
  control = 7 train + 3 test = 10

Hook/pool:
  blocks.36.hook_resid_post / prompt_mean

v_regime_norm:
  4521.193848

Projection audit:
  raw             AUC=1.000 gap=3718.732
  sae_orth        AUC=1.000 gap=3495.028
  grade_orth      AUC=1.000 gap=3644.912
  grade_sae_orth  AUC=1.000 gap=3351.831

Controls:
  random same-dim gap p95 = 130.362, max = 155.138
  label-permutation gap p95 = 3421.207, max = 3681.045

Boundary:
  Geometry/readout remains strong.
  raw/sae_orth/grade_orth beat label-permutation p95 by projection gap.
  grade_sae_orth remains AUC=1 but is below label-permutation p95 by gap,
  so full Grade+SAE-independent residual is not fully closed in this small split.

Causal generation:
  324 rows total, no script-switch collapse.
  Actual alpha=0.5 KL is below random-causal p95 and often below random mean
  on control side. Free-form generation remains causal-inconclusive.

Next step:
  Use forced-choice decision-margin audit with primary metric:
    margin = logp(A) - logp(B)
  rather than another free-form generation run.
```

Decision-probe script added:

```text
experiments/steering/sae_gemma_qwen/gemma_active/regime_decision_probe_causal_audit.py
```

Purpose:

```text
This is the central next script after the L36 regime bridge. It keeps the same
train/test bank split, Grade axes, SAE orthogonalization, random same-norm
controls, and label-permutation controls, but replaces free-form generation as
the primary endpoint with forced-choice decision margins:

  margin = logp(A) - logp(B)

The key causal field is:

  signed_margin_shift

where positive means movement toward the configured opposite-side choice:
control+regime toward target_choice, target-regime toward control_choice.
```

Output files:

```text
regime_decision_input_split_<RUN_TAG>.csv
regime_decision_manifest_<RUN_TAG>.csv
regime_decision_cosines_<RUN_TAG>.csv
regime_decision_projection_audit_<RUN_TAG>.csv
regime_decision_control_axes_<RUN_TAG>.csv
regime_decision_probe_rows_<RUN_TAG>.csv
regime_decision_probe_summary_<RUN_TAG>.csv
regime_decision_vectors_<RUN_TAG>.npz
regime_decision_probe_config_<RUN_TAG>.json
regime_decision_claim_ladder_<RUN_TAG>.md
```

Scientific success condition:

```text
The strongest next result would be:
  grade_sae_orth still separates held-out target/control hidden states
  and actual signed_margin_shift beats same-norm random and permutation
  decision controls on independent A/B probes.

That would connect:
  hidden-state geometry -> decision-state causality
```

Top-level next research direction:

```text
The most ambitious next step is not another generation-style test. It is a
closed-loop hidden-state safety experiment:

  context -> hidden-state regime detector -> predicted decision shift ->
  causal intervention -> decision restored

To be a top-level result, the experiment should show all four parts:

1. Detection:
   a train-only hidden-state axis/probe predicts a later decision boundary on
   held-out contexts and independent tasks.

2. Timing:
   the signal appears before the visible final answer, ideally before the
   forced-choice/action token.

3. Causality:
   injecting or subtracting the discovered direction changes the decision
   margin more than same-norm random/permutation controls.

4. Mitigation:
   projecting out or counter-steering the detected regime restores the baseline
   decision margin while preserving normal task competence.

This would move the project from "hidden shift exists" to "hidden shift is an
action-relevant internal state that can be detected and causally corrected."
That is the strongest alignment/safety formulation currently available in this
thread.
```

## 2026-06-13 Base vs Instruct Geometry/Probability Run 02 - detailed metric readout

Run directory:

```text
experiments/variance_compression_finding/alignment_geometry_probability_run_02/run_20260613_090705
```

Setup:

```text
base_model     = google/gemma-3-12b-pt
instruct_model = google/gemma-3-12b-it
prompt_mode    = raw
conditions     = target, target_word_shuffle, target_sentence_shuffle, control, question_only
n_prompts/model = 50
hidden shape   = [50, 49, 3840]
```

Layer 48 caveat:

```text
Layer 48 has centroid_norm around 100 while layers 30-47 have centroid_norm
around 97k-127k. Treat layer 48 as a final hidden/final norm transition, not
as the same residual-stream band. Main residual-band reading uses L30-L47.
```

L30-L47 target condition:

```text
centroid_norm:     base 124,817 -> instruct 107,665  ratio 0.863
abs_disp_l2_mean:  base 22,463  -> instruct 20,851   ratio 0.928
rel_disp_l2_mean:  base 0.1803  -> instruct 0.1907   ratio 1.058
pairwise_l2_mean:  base 32,684  -> instruct 30,829   ratio 0.943
cosine distance:   base 0.0191  -> instruct 0.0272   ratio 1.427
effective_rank_pr: base 2.142   -> instruct 2.830    ratio 1.321
spectral_entropy_norm: base 0.509 -> instruct 0.614  ratio 1.205
top1 PC share:     base 0.670   -> instruct 0.564    ratio 0.841
cov_trace:         base 7.08e8  -> instruct 6.27e8   ratio 0.885
```

L30-L47 question_only condition:

```text
centroid_norm:     base 127,501 -> instruct 97,001   ratio 0.761
abs_disp_l2_mean:  base 24,826  -> instruct 19,749   ratio 0.796
rel_disp_l2_mean:  base 0.1972  -> instruct 0.2019   ratio 1.024
cosine distance:   base 0.0195  -> instruct 0.0301   ratio 1.546
effective_rank_pr: base 1.791   -> instruct 3.080    ratio 1.720
spectral_entropy_norm: base 0.425 -> instruct 0.654  ratio 1.540
top1 PC share:     base 0.746   -> instruct 0.545    ratio 0.731
cov_trace:         base 8.25e8  -> instruct 5.49e8   ratio 0.665
```

Interpretation of the hidden metrics:

```text
The instruct model does show lower absolute hidden-state scale:
  lower centroid_norm, lower abs_disp_l2_mean, lower pairwise_l2_mean,
  and usually lower cov_trace.

But it does not show simple global hidden-state collapse:
  rel_disp_l2_mean is higher or similar,
  pairwise cosine distance is 1.4-1.55x higher,
  effective rank is higher,
  spectral entropy is higher,
  first-PC variance share is lower.

So the hidden geometry is smaller in absolute Euclidean magnitude but more
angularly/rank-distributed relative to its centroid. This is not "RLHF simply
compresses everything"; it is closer to scale reduction plus a different
hidden-to-logit readout regime.
```

Probability/logit readout:

```text
target entropy:        base 2.693 -> instruct 1.482  ratio 0.551
control entropy:       base 2.815 -> instruct 1.306  ratio 0.464
question_only entropy: base 2.928 -> instruct 0.912  ratio 0.311

target top1 prob:        base 0.430 -> instruct 0.654  ratio 1.52
control top1 prob:       base 0.429 -> instruct 0.697  ratio 1.62
question_only top1 prob: base 0.419 -> instruct 0.816  ratio 1.95

top5 mass is also much higher in instruct across all conditions.
```

Current best formulation:

```text
Instruction tuning in this run is not just late-layer variance suppression.
It reduces absolute hidden-state scale and covariance trace while increasing
relative/angular dispersion and effective rank, then maps those hidden states
to a much sharper next-token probability distribution.

The supported mechanism is therefore:

  hidden-state scale/geometry change -> sharper hidden-to-logit readout ->
  lower entropy and higher top1/top5 probability before generation

Better term:

  alignment-stiffened hidden-to-logit readout

or:

  probability commitment under reduced hidden-state scale.
```

## 2026-06-13 Fullbank Base-vs-Instruct Geometry/Probability Run

Run:

```text
C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703
```

Coverage:

```text
base_model     = google/gemma-3-12b-pt
instruct_model = google/gemma-3-12b-it
prompt_mode    = raw
include_shuffles = true
n_target_contexts = 10
n_control_contexts = 10
n_questions = 10
n_prompts/model = 410
hidden_shape = [410, 49, 3840]
late band = L30-L47

condition rows/model:
  target = 100
  target_word_shuffle = 100
  target_sentence_shuffle = 100
  control = 100
  question_only = 10
```

Primary fullbank result:

```text
The target-control separation is stronger in instruct than base.

Late L30-L47 target-control contrast:

  target_control_centroid_l2:
    base     4,781.8
    instruct 9,392.9

  target_control_projection_gap_z:
    base     0.593
    instruct 0.868

  target_control_axis_auc_like:
    base     0.704
    instruct 0.747

  leave-one-question balanced accuracy:
    base     0.589
    instruct 0.654

  leave-one-question auc_like:
    base     0.914
    instruct 0.938
```

Interpretation:

```text
The target/control hidden-state distinction survives the fullbank setting and is
amplified in the instruct model. The LOO question split is important: the axis is
not only memorizing one question wording. Ranking separation is strong; threshold
classification is moderate.
```

Target vs control inside each model:

```text
base:
  target has larger centroid_norm, larger absolute/relative dispersion,
  larger cov_trace, and slightly lower effective_rank than control.
  This does not support "target is simply more compact" in base.

instruct:
  target has much larger centroid_norm than control and is much farther from
  question_only, but has lower rel_disp and lower pairwise cosine distance than
  control. This supports a more organized target-conditioned regime in instruct.
```

Question-only/context snapping:

```text
rel_disp question_minus_context:
  base     0.009512
  instruct 0.009484

Context lowers relative hidden dispersion similarly in base and instruct.
This part is not uniquely stronger in instruct.
```

Probability/readout:

```text
Instruct remains much lower entropy and higher top1/top5 than base across all
conditions, supporting hidden-to-logit readout stiffening.

But target is not the most probability-concentrated instruct condition:

  instruct entropy:
    question_only          0.912
    control                1.176
    target_word_shuffle    1.227
    target_sentence_shuffle 1.519
    target                1.768

  instruct top1_prob:
    question_only          0.816
    target_word_shuffle    0.699
    control                0.678
    target_sentence_shuffle 0.638
    target                0.614
```

Interpretation:

```text
Instruction tuning strongly sharpens the output distribution relative to base,
but coherent target context can reduce that commitment relative to control.
This is not a simple "target makes the model more confident" effect. It is
closer to a target-conditioned regime shift that changes hidden geometry and
readout structure, sometimes broadening uncertainty compared with neutral
control while remaining more concentrated than base.
```

Base-vs-instruct representational alignment:

```text
Late L30-L47 linear CKA:
  all                    0.892
  control                0.938
  target                 0.920
  target_sentence_shuffle 0.911
  target_word_shuffle    0.882
  question_only          0.763

Instruct/base norm ratio:
  all                    0.848
  control                0.826
  target                 0.864
  target_sentence_shuffle 0.854
  target_word_shuffle    0.854
  question_only          0.776
```

Interpretation:

```text
The instruct model has lower late hidden norm across conditions, but the
representational map remains substantially aligned with base, especially for
contextual prompts. Question-only is the least aligned condition.
```

Current fullbank conclusion:

```text
1. Fullbank confirms that target/control latent-state separation exists.
2. Instruct amplifies target/control separation in late hidden geometry.
3. Context lowers relative dispersion vs question-only in both base and instruct.
4. Instruct still strongly sharpens next-token probabilities relative to base.
5. Coherent target is not merely a "confidence booster"; in instruct it is less
   probability-concentrated than control/word-shuffle while being geometrically
   farther from question-only and more separated from control.

Best wording:

  context-induced latent-state shift is confirmed in fullbank;
  instruction tuning amplifies target/control hidden separation and stiffens
  hidden-to-logit readout relative to base, but coherent target context can
  reorganize the readout rather than simply increasing output confidence.
```
