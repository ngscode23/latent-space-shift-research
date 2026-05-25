# Red-Team Hidden Geometry Metric Audit

Reader-only audit. No model was run.

## Bottom Line

- Runs loaded: `1`
- Strong internal axis runs: `1/1`
- Best neutral +X visible lift: `0.0111138` on `Qwen/Qwen3.5-9B` (p95 lift `-0.0527653`, question win mean `71.4%`).
- Best target -X suppression lift: `0.0556908` on `Qwen/Qwen3.5-9B` (p95 lift `-0.0040929`, random win `93.8%`).
- Overall statuses: `{"strong_internal_axis_asymmetric_visible_readout": 1}`

Practical interpretation:

`Strong mechanism, asymmetric readout.` Frame Vector X first as an ablation/suppression axis; keep testing neutral +X separately.

## Run Summary

| model_id | target_projection | neutral_lift_random_mean | neutral_lift_random_p95 | neutral_questions_over_random_mean | generation_win_random | target_minus_supp_lift_random_mean | target_minus_supp_win_random_mean | neutral_degeneration | visible_status | ablation_status | overall_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-9B | 0.945212 | 0.0111138 | -0.0527653 | 0.714286 | 1 | 0.0556908 | 0.9375 | 0 | weak_partial | good | strong_internal_axis_asymmetric_visible_readout |

## Run: Qwen/Qwen3.5-9B

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_full_middle_behavioral_control_middle_alpha_retest_03.zip`
- Run label: `full_middle_clean_behavioral_control_middle_alpha_retest`
- Verdict CSV: `partial_behavioral_control_axis_supported`
- Existing hard-random file: `False`
- Existing breakthrough audit: `False`
- Files in archive/directory: `52`

Key numbers:

- Target projection: `0.945212`
- Length-neutral projection: `0.00730324`
- Word-shuffle projection: `0.496309`
- Sentence-shuffle projection: `0.84667`
- Neutral +X target-likeness: `0.526543`
- Random visible mean: `0.515429`
- Neutral +X lift over random mean: `0.0111138`
- Neutral +X lift over random p95: `-0.0527653`
- Neutral +X question wins over random mean: `71.4%`
- Neutral +X question wins over random p95: `14.3%`
- Generation projection win over random: `100.0%`
- Target -X suppression lift over random mean: `0.0556908`
- Target -X suppression lift over random p95: `-0.0040929`
- Target -X suppression win over random mean: `93.8%`
- Neutral +X degeneration: `0`
- Target -X degeneration: `0`

Status: `strong_internal_axis_asymmetric_visible_readout`

Next action: move to next model/text family; frame X as ablation/suppression axis first

Full metric-family overview:

| family | status | readout | artifacts |
| --- | --- | --- | --- |
| endpoint_geometry | computed | target=0.945212, length_neutral=0.00730324, word_shuffle=0.496309, sentence_shuffle=0.84667 | middle_layer_condition_summary.csv |
| layerwise_geometry | computed | target_best_layer=1, max_projection=0.999225, target_layer_mean=0.935422 | layerwise_geometry_summary.csv |
| statistical_controls | computed | paired_rows=12; min_p=n/a; max_abs_d=7.63475 | paired_target_vs_control_tests.csv, layerwise_fdr_target_vs_control.csv, null_vector_baseline_summary.csv, pca_baseline_projection_summary.csv |
| bias_and_dataset_audits | computed | max_abs_length_corr=0.33346; duplicate_questions=0 | length_bias_audit.csv, deduplication_audit.csv, domain_robustness_geometry_summary.csv |
| architecture_features | present_empty_or_disabled | Architecture file exists but has no rows. | architecture_module_delta_summary.csv |
| generation_trajectory | present_empty_or_profile_skipped | Generation summary exists but has no rows. | generation_middle_layer_summary.csv |
| causal_interventions | missing_or_disabled | Full causal injection/ablation block was not exported. | causal_intervention_middle_layer_summary.csv |
| behavioral_control_axis | computed | neutral_lift_mean=0.0111138, neutral_lift_p95=-0.0527653, generation_win=100.0%, target_minus_supp_lift=0.0556908 | behavioral_control_axis_similarity_raw.csv, behavioral_control_axis_response_quality_summary.csv, behavioral_control_axis_verdict.csv |
| response_quality | computed | neutral_degeneration=0, target_minus_degeneration=0, neutral_unique=0.780221 | behavioral_control_axis_response_quality_summary.csv |
| dynamic_geometry | missing_or_disabled | Dynamic trajectory block not exported. | dynamic_trajectory_summary.csv |
| feature_proxy | computed_or_statused | not_run_no_sae_model_configured; computed_from_hidden_top_changed_dimensions; dense_proxy_rows=71194 | feature_level_interpretability_status.csv, dense_feature_proxy_mapping.csv |
| artifact_coverage | computed | files=52, expected_present=46/54, missing_expected=8 | full archive inventory |

Artifact inventory by family:

| family | file_count | total_mb |
| --- | --- | --- |
| architecture_features | 6 | 4.34286 |
| behavioral_control_axis | 9 | 3.57723 |
| endpoint_geometry | 5 | 17.8974 |
| generation_trajectory | 3 | 2.86102e-06 |
| manifest_protocol | 4 | 0.0108862 |
| plots | 3 | 0.235063 |
| secondary_geometry | 4 | 0.0484781 |
| statistical_controls | 11 | 0.00382996 |
| tensor_snapshots | 3 | 22.313 |
| verdict_report | 4 | 0.0119696 |

Expected artifact coverage:

- Missing expected artifacts: `8`
- Missing list: `causal_intervention_response_audit.csv, causal_intervention_middle_layer_summary.csv, causal_bidirectional_symmetry_summary.csv, behavioral_control_axis_hard_random_comparison.csv, behavioral_control_axis_hard_random_summary.csv, behavioral_control_axis_asymmetry_summary.csv, breakthrough_readiness_audit.csv, breakthrough_readiness_audit.md`

CSV health and NaN coverage:

- CSV status counts: `{"read": 43}`

CSV files needing attention:

| basename | family | status | rows | columns | null_fraction | numeric_inf_cells | top_nan_columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feature_level_interpretability_status.csv | architecture_features | read | 2 | 4 | 0.25 | 0 | sae_model_id:1 |
| deduplication_audit.csv | statistical_controls | read | 15 | 4 | 0.25 | 0 | normalized_duplicate_of:1 |

Largest CSV artifacts scanned:

| basename | family | status | size_mb | rows | columns | null_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| hidden_top_changed_dimensions.csv | endpoint_geometry | read | 17.4245 | 158400 | 11 | 0 |
| dense_feature_proxy_mapping.csv | architecture_features | read | 4.34263 | 71194 | 7 | 0 |
| behavioral_control_axis_similarity_raw.csv | behavioral_control_axis | read | 1.844 | 826 | 47 | 0.000901551 |
| behavioral_control_axis_response_audit.csv | behavioral_control_axis | read | 1.64992 | 826 | 34 | 0.000997009 |
| layerwise_geometry_metrics_raw.csv | endpoint_geometry | read | 0.445017 | 2475 | 14 | 0.004329 |
| residual_stream_decomposition.csv | secondary_geometry | read | 0.0467663 | 495 | 8 | 0.0151515 |
| behavioral_control_axis_similarity_summary.csv | behavioral_control_axis | read | 0.0452499 | 118 | 26 | 0.00130378 |
| behavioral_control_axis_response_quality_summary.csv | behavioral_control_axis | read | 0.0223236 | 118 | 15 | 0.00225989 |
| layerwise_geometry_summary.csv | endpoint_geometry | read | 0.0185318 | 165 | 10 | 0.00606061 |
| behavioral_control_axis_intervention_plan.csv | behavioral_control_axis | read | 0.0105944 | 118 | 8 | 0.0275424 |

Per-question neutral +X vs random:

| question_index | x_likeness | random_mean | random_p95 | random_max | x_minus_random_mean | x_minus_random_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.51368 | 0.470422 | 0.596424 | 0.656963 | 0.0432578 | -0.0827445 |
| 1 | 0.574211 | 0.549483 | 0.629413 | 0.670749 | 0.0247283 | -0.0552024 |
| 3 | 0.451841 | 0.434434 | 0.476583 | 0.489224 | 0.0174069 | -0.0247417 |
| 5 | 0.0675151 | 0.328065 | 0.565902 | 0.659643 | -0.26055 | -0.498387 |
| 8 | 0.599544 | 0.589825 | 0.643377 | 0.660989 | 0.00971915 | -0.0438333 |
| 10 | 0.452678 | 0.509076 | 0.630954 | 0.648983 | -0.0563984 | -0.178277 |
| 13 | 1.02633 | 0.7267 | 0.978902 | 1.07694 | 0.299633 | 0.0474306 |

## Decision Gates

- Hidden geometry strong: `target_projection >= 0.85` and length-neutral near zero.
- Internal generation strong: neutral +X generation projection beats almost all random vectors.
- Visible neutral +X strong: beats random p95 and wins most held-out questions without degeneration.
- Target -X ablation strong: suppression beats random p95/mean without degeneration.
- Breakthrough-grade claim needs cross-model and cross-text-family replication, not one run.
