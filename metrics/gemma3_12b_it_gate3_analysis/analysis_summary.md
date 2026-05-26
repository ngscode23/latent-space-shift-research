# Hidden Geometry Result Analysis: gemma3_12b_it_gate3

This is an external read-only analysis. The source package was not modified.

## Run Validity

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip`
- Model: `google/gemma-3-12b-it`
- Gate: `gate3`
- Decoder OK: `True` from `red_team_input_manifest.json`
- Prompt budget OK: `True` from `prompt_budget_overflow_warnings.csv` presence
- Numeric integrity OK: `True` from `analysis_notes/extracted_narrative_columns/numeric_integrity_check.csv`

## Primary Metrics

- Geometry pass: `True`; target middle projection `0.9346552360003808`. Source: `middle_layer_condition_summary.csv`.
- Specificity pass: `True`; best target-control gap `1.3921496852788746`. Source: `paired_target_vs_control_tests.csv`.
- Strict causal symmetry pass: `False`; score `0.075`. Source: `claim_ladder_final.csv` / `causal_symmetry_score_summary.csv`.
- Behavior random p95 pass: `False`; score `0.2`. Source: `claim_ladder_final.csv` / `behavior_random_p95_gate.csv`.

## Mechanistic Reading

The package supports a hidden-geometry/readout shift against the available controls.
It weakens or does not establish strict bidirectional causal symmetry for the tested intervention.
It does not establish visible behavioral steering against random-p95 controls.

## Boundary

This analysis does not create discovery/verdict labels in machine CSV outputs. It reports pass/fail fields, failure_code values, source files, and conservative missingness.

## Recommended Next Experiment

`run_gate4_axis_decomposition`

## Top Anomalies

- `high` `behavior_p95_metric_mismatch` in `behavioral_control_axis_threshold_eval.csv`: p95_metric_name_mismatch = plus_x_lift_over_random
- `high` `behavior_p95_metric_mismatch` in `behavioral_control_axis_threshold_eval.csv`: p95_metric_value_mismatch = threshold=0.0042964240573698, hard_random_p95=-0.0199293519035931
- `medium` `below_random_p95` in `behavior_random_p95_gate.csv`: random_p95_not_beaten = -0.0005011356097773
- `medium` `below_random_p95` in `behavior_random_p95_gate.csv`: random_p95_not_beaten = -0.0004976824252649
- `medium` `below_random_p95` in `behavior_random_p95_gate.csv`: random_p95_not_beaten = -0.0002303815061731
- `medium` `below_random_p95` in `behavior_random_p95_gate.csv`: random_p95_not_beaten = -0.0001447546885884
- `medium` `quality_degeneration_high` in `quality_adjusted_behavior_summary.csv`: quality_degenerate_rate = 0.5
- `medium` `quality_degeneration_high` in `quality_adjusted_behavior_summary.csv`: quality_degenerate_rate = 1.0
- `medium` `quality_degeneration_high` in `quality_adjusted_behavior_summary.csv`: quality_degenerate_rate = 1.0
- `medium` `quality_degeneration_high` in `quality_adjusted_behavior_summary.csv`: quality_degenerate_rate = 1.0
- `medium` `quality_degeneration_high` in `quality_adjusted_behavior_summary.csv`: quality_degenerate_rate = 1.0
- `medium` `quality_degeneration_high` in `quality_adjusted_behavior_summary.csv`: quality_degenerate_rate = 1.0

## Peak Tables

- `geometry_peaks.csv` rows: `35`
- `specificity_peaks.csv` rows: `10`
- `component_peaks.csv` rows: `0`
- `causal_peaks.csv` rows: `44`
- `behavior_peaks.csv` rows: `39`
- `architecture_peaks.csv` rows: `160`
- `anomaly_flags.csv` rows: `12`
