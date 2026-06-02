# Current Research Context

This is the compact handoff file and the active place for new research memory.
The old long historical context is archived at `archive/research_context_anchor.md`.

## Current Claim

We are not using formal "attractor" language as the main claim.

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
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
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
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
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

grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
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
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
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
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
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
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
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
