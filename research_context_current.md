# Current Research Context

This is the compact handoff file. Use it before reading the large historical
`research_context_anchor.md`.

## Current Claim

We are not using formal "attractor" language as the main claim.

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
