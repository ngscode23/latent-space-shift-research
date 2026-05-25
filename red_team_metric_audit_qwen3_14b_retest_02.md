# Red-Team Hidden Geometry Metric Audit

Reader-only audit. No model was run.

## Bottom Line

- Runs loaded: `1`
- Strong internal axis runs: `1/1`
- Best neutral +X visible lift: `0.035109` on `Qwen/Qwen3-14B` (p95 lift `-0.0161607`, question win mean `57.1%`).
- Best target -X suppression lift: `0.023687` on `Qwen/Qwen3-14B` (p95 lift `-0.0124462`, random win `87.5%`).
- Overall statuses: `{"strong_internal_axis_partial_visible_readout": 1}`

Practical interpretation:

`Strong internal geometry.` The missing piece is hard visible readout over random p95/best and broader replication.

## Run Summary

| model_id | target_projection | neutral_lift_random_mean | neutral_lift_random_p95 | neutral_questions_over_random_mean | generation_win_random | target_minus_supp_lift_random_mean | target_minus_supp_win_random_mean | neutral_degeneration | visible_status | ablation_status | overall_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-14B | 0.974829 | 0.035109 | -0.0161607 | 0.571429 | 1 | 0.023687 | 0.875 | 0 | good_partial | good | strong_internal_axis_partial_visible_readout |

## Run: Qwen/Qwen3-14B

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_full_middle_behavioral_control_middle_alpha_retest_02.zip`
- Run label: `full_middle_clean_behavioral_control_middle_alpha_retest`
- Verdict CSV: `partial_behavioral_control_axis_supported`
- Existing hard-random file: `False`
- Existing breakthrough audit: `False`
- Files in archive/directory: `52`

Key numbers:

- Target projection: `0.974829`
- Length-neutral projection: `0.0128273`
- Word-shuffle projection: `0.733918`
- Sentence-shuffle projection: `0.873508`
- Neutral +X target-likeness: `0.55941`
- Random visible mean: `0.524301`
- Neutral +X lift over random mean: `0.035109`
- Neutral +X lift over random p95: `-0.0161607`
- Neutral +X question wins over random mean: `57.1%`
- Neutral +X question wins over random p95: `0.0%`
- Generation projection win over random: `100.0%`
- Target -X suppression lift over random mean: `0.023687`
- Target -X suppression lift over random p95: `-0.0124462`
- Target -X suppression win over random mean: `87.5%`
- Neutral +X degeneration: `0`
- Target -X degeneration: `0`

Status: `strong_internal_axis_partial_visible_readout`

Next action: run behavioral-control-only with 32 random baselines and quality audit

Full metric-family overview:

| family | status | readout | artifacts |
| --- | --- | --- | --- |
| endpoint_geometry | computed | target=0.974829, length_neutral=0.0128273, word_shuffle=0.733918, sentence_shuffle=0.873508 | middle_layer_condition_summary.csv |
| layerwise_geometry | computed | target_best_layer=5, max_projection=0.997479, target_layer_mean=0.955858 | layerwise_geometry_summary.csv |
| statistical_controls | computed | paired_rows=12; min_p=n/a; max_abs_d=15.4057 | paired_target_vs_control_tests.csv, layerwise_fdr_target_vs_control.csv, null_vector_baseline_summary.csv, pca_baseline_projection_summary.csv |
| bias_and_dataset_audits | computed | max_abs_length_corr=0.64072; duplicate_questions=0 | length_bias_audit.csv, deduplication_audit.csv, domain_robustness_geometry_summary.csv |
| architecture_features | present_empty_or_disabled | Architecture file exists but has no rows. | architecture_module_delta_summary.csv |
| generation_trajectory | present_empty_or_profile_skipped | Generation summary exists but has no rows. | generation_middle_layer_summary.csv |
| causal_interventions | missing_or_disabled | Full causal injection/ablation block was not exported. | causal_intervention_middle_layer_summary.csv |
| behavioral_control_axis | computed | neutral_lift_mean=0.035109, neutral_lift_p95=-0.0161607, generation_win=100.0%, target_minus_supp_lift=0.023687 | behavioral_control_axis_similarity_raw.csv, behavioral_control_axis_response_quality_summary.csv, behavioral_control_axis_verdict.csv |
| response_quality | computed | neutral_degeneration=0, target_minus_degeneration=0, neutral_unique=0.628446 | behavioral_control_axis_response_quality_summary.csv |
| dynamic_geometry | missing_or_disabled | Dynamic trajectory block not exported. | dynamic_trajectory_summary.csv |
| feature_proxy | computed_or_statused | not_run_no_sae_model_configured; computed_from_hidden_top_changed_dimensions; dense_proxy_rows=69291 | feature_level_interpretability_status.csv, dense_feature_proxy_mapping.csv |
| artifact_coverage | computed | files=52, expected_present=46/54, missing_expected=8 | full archive inventory |

Artifact inventory by family:

| family | file_count | total_mb |
| --- | --- | --- |
| architecture_features | 6 | 4.09081 |
| behavioral_control_axis | 9 | 2.95577 |
| endpoint_geometry | 5 | 20.7254 |
| generation_trajectory | 3 | 2.86102e-06 |
| manifest_protocol | 4 | 0.0108919 |
| plots | 3 | 0.238173 |
| secondary_geometry | 4 | 0.0599985 |
| statistical_controls | 11 | 0.00378036 |
| tensor_snapshots | 3 | 34.7243 |
| verdict_report | 4 | 0.0119619 |

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
| hidden_top_changed_dimensions.csv | endpoint_geometry | read | 20.1426 | 196800 | 11 | 0 |
| dense_feature_proxy_mapping.csv | architecture_features | read | 4.09058 | 69291 | 7 | 0 |
| behavioral_control_axis_similarity_raw.csv | behavioral_control_axis | read | 1.53492 | 826 | 47 | 0.000901551 |
| behavioral_control_axis_response_audit.csv | behavioral_control_axis | read | 1.34086 | 826 | 34 | 0.000997009 |
| layerwise_geometry_metrics_raw.csv | endpoint_geometry | read | 0.55092 | 3075 | 14 | 0.00348432 |
| residual_stream_decomposition.csv | secondary_geometry | read | 0.0582809 | 615 | 8 | 0.0121951 |
| behavioral_control_axis_similarity_summary.csv | behavioral_control_axis | read | 0.0421581 | 118 | 26 | 0.00130378 |
| layerwise_geometry_summary.csv | endpoint_geometry | read | 0.0226908 | 205 | 10 | 0.00487805 |
| behavioral_control_axis_response_quality_summary.csv | behavioral_control_axis | read | 0.0221081 | 118 | 15 | 0.00225989 |
| behavioral_control_axis_intervention_plan.csv | behavioral_control_axis | read | 0.0105944 | 118 | 8 | 0.0275424 |

Per-question neutral +X vs random:

| question_index | x_likeness | random_mean | random_p95 | random_max | x_minus_random_mean | x_minus_random_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.570983 | 0.511784 | 0.63296 | 0.680867 | 0.0591993 | -0.0619768 |
| 1 | 0.484518 | 0.465542 | 0.522007 | 0.545103 | 0.0189762 | -0.0374887 |
| 3 | 0.546251 | 0.562709 | 0.628624 | 0.785993 | -0.0164583 | -0.0823727 |
| 5 | 0.632824 | 0.562351 | 0.699355 | 0.732156 | 0.070473 | -0.0665307 |
| 8 | 0.566464 | 0.568698 | 0.685769 | 0.727732 | -0.0022347 | -0.119305 |
| 10 | 0.55862 | 0.567562 | 0.714017 | 0.827469 | -0.00894203 | -0.155398 |
| 13 | 0.556209 | 0.43146 | 0.604094 | 0.618801 | 0.124749 | -0.0478846 |

## Decision Gates

- Hidden geometry strong: `target_projection >= 0.85` and length-neutral near zero.
- Internal generation strong: neutral +X generation projection beats almost all random vectors.
- Visible neutral +X strong: beats random p95 and wins most held-out questions without degeneration.
- Target -X ablation strong: suppression beats random p95/mean without degeneration.
- Breakthrough-grade claim needs cross-model and cross-text-family replication, not one run.
