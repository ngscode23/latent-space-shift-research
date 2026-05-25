# Red-Team Hidden Geometry Metric Audit

Reader-only audit. No model was run.

## Bottom Line

- Runs loaded: `1`
- Strong internal axis runs: `1/1`
- Best neutral +X visible lift: `0.038719` on `Qwen/Qwen2.5-7B-Instruct` (p95 lift `-0.0172459`, question win mean `85.7%`).
- Best target -X suppression lift: `-0.0130438` on `Qwen/Qwen2.5-7B-Instruct` (p95 lift `-0.0594904`, random win `40.6%`).
- Overall statuses: `{"strong_internal_axis_visible_weak": 1}`

Practical interpretation:

`Strong internal geometry.` The missing piece is hard visible readout over random p95/best and broader replication.

## Run Summary

| model_id | target_projection | neutral_lift_random_mean | neutral_lift_random_p95 | neutral_questions_over_random_mean | generation_win_random | target_minus_supp_lift_random_mean | target_minus_supp_win_random_mean | neutral_degeneration | visible_status | ablation_status | overall_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-7B-Instruct | 0.954907 | 0.038719 | -0.0172459 | 0.857143 | 1 | -0.0130438 | 0.40625 | 0 | good_partial | not_supported | strong_internal_axis_visible_weak |

## Run: Qwen/Qwen2.5-7B-Instruct

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_qwen25_7b_middle_alpha_retest_behavioral_control_middle_alpha_retest.zip`
- Run label: `qwen25_7b_middle_alpha_retest_behavioral_control_middle_alpha_retest`
- Verdict CSV: `partial_behavioral_control_axis_supported`
- Existing hard-random file: `True`
- Existing breakthrough audit: `True`
- Files in archive/directory: `57`

Key numbers:

- Target projection: `0.954907`
- Length-neutral projection: `0.0224988`
- Word-shuffle projection: `0.599101`
- Sentence-shuffle projection: `0.921586`
- Neutral +X target-likeness: `0.475231`
- Random visible mean: `0.436512`
- Neutral +X lift over random mean: `0.038719`
- Neutral +X lift over random p95: `-0.0172459`
- Neutral +X question wins over random mean: `85.7%`
- Neutral +X question wins over random p95: `0.0%`
- Generation projection win over random: `100.0%`
- Target -X suppression lift over random mean: `-0.0130438`
- Target -X suppression lift over random p95: `-0.0594904`
- Target -X suppression win over random mean: `40.6%`
- Neutral +X degeneration: `0`
- Target -X degeneration: `0`

Status: `strong_internal_axis_visible_weak`

Next action: try neutral-only question set and held-out text family; do not chase larger alpha

Full metric-family overview:

| family | status | readout | artifacts |
| --- | --- | --- | --- |
| endpoint_geometry | computed | target=0.954907, length_neutral=0.0224988, word_shuffle=0.599101, sentence_shuffle=0.921586 | middle_layer_condition_summary.csv |
| layerwise_geometry | computed | target_best_layer=1, max_projection=0.997202, target_layer_mean=0.931648 | layerwise_geometry_summary.csv |
| statistical_controls | computed | paired_rows=12; min_p=n/a; max_abs_d=15.2696 | paired_target_vs_control_tests.csv, layerwise_fdr_target_vs_control.csv, null_vector_baseline_summary.csv, pca_baseline_projection_summary.csv |
| bias_and_dataset_audits | computed | max_abs_length_corr=0.817945; duplicate_questions=0 | length_bias_audit.csv, deduplication_audit.csv, domain_robustness_geometry_summary.csv |
| architecture_features | present_empty_or_disabled | Architecture file exists but has no rows. | architecture_module_delta_summary.csv |
| generation_trajectory | present_empty_or_profile_skipped | Generation summary exists but has no rows. | generation_middle_layer_summary.csv |
| causal_interventions | missing_or_disabled | Full causal injection/ablation block was not exported. | causal_intervention_middle_layer_summary.csv |
| behavioral_control_axis | computed | neutral_lift_mean=0.038719, neutral_lift_p95=-0.0172459, generation_win=100.0%, target_minus_supp_lift=-0.0130438 | behavioral_control_axis_similarity_raw.csv, behavioral_control_axis_response_quality_summary.csv, behavioral_control_axis_verdict.csv |
| response_quality | computed | neutral_degeneration=0, target_minus_degeneration=0, neutral_unique=0.854913 | behavioral_control_axis_response_quality_summary.csv |
| dynamic_geometry | missing_or_disabled | Dynamic trajectory block not exported. | dynamic_trajectory_summary.csv |
| feature_proxy | computed_or_statused | not_run_no_sae_model_configured; computed_from_hidden_top_changed_dimensions; dense_proxy_rows=45311 | feature_level_interpretability_status.csv, dense_feature_proxy_mapping.csv |
| artifact_coverage | computed | files=57, expected_present=51/54, missing_expected=3 | full archive inventory |

Artifact inventory by family:

| family | file_count | total_mb |
| --- | --- | --- |
| architecture_features | 6 | 2.7368 |
| behavioral_control_axis | 12 | 3.15035 |
| endpoint_geometry | 5 | 14.997 |
| generation_trajectory | 3 | 2.86102e-06 |
| manifest_protocol | 4 | 0.011178 |
| other | 1 | 0.00218296 |
| plots | 3 | 0.236409 |
| secondary_geometry | 4 | 0.0425854 |
| statistical_controls | 11 | 0.00380802 |
| tensor_snapshots | 3 | 17.2348 |
| verdict_report | 5 | 0.0271511 |

Expected artifact coverage:

- Missing expected artifacts: `3`
- Missing list: `causal_intervention_response_audit.csv, causal_intervention_middle_layer_summary.csv, causal_bidirectional_symmetry_summary.csv`

CSV health and NaN coverage:

- CSV status counts: `{"read": 47}`

CSV files needing attention:

| basename | family | status | rows | columns | null_fraction | numeric_inf_cells | top_nan_columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feature_level_interpretability_status.csv | architecture_features | read | 2 | 4 | 0.25 | 0 | sae_model_id:1 |
| deduplication_audit.csv | statistical_controls | read | 15 | 4 | 0.25 | 0 | normalized_duplicate_of:1 |

Largest CSV artifacts scanned:

| basename | family | status | size_mb | rows | columns | null_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| hidden_top_changed_dimensions.csv | endpoint_geometry | read | 14.5816 | 139200 | 11 | 0 |
| dense_feature_proxy_mapping.csv | architecture_features | read | 2.73656 | 45311 | 7 | 0 |
| behavioral_control_axis_similarity_raw.csv | behavioral_control_axis | read | 1.62744 | 826 | 47 | 0.000901551 |
| behavioral_control_axis_response_audit.csv | behavioral_control_axis | read | 1.43383 | 826 | 34 | 0.000997009 |
| layerwise_geometry_metrics_raw.csv | endpoint_geometry | read | 0.38992 | 2175 | 14 | 0.00492611 |
| behavioral_control_axis_similarity_summary.csv | behavioral_control_axis | read | 0.041955 | 118 | 26 | 0.00130378 |
| residual_stream_decomposition.csv | secondary_geometry | read | 0.0408621 | 435 | 8 | 0.0172414 |
| behavioral_control_axis_response_quality_summary.csv | behavioral_control_axis | read | 0.0221329 | 118 | 15 | 0.00225989 |
| layerwise_geometry_summary.csv | endpoint_geometry | read | 0.0161581 | 145 | 10 | 0.00689655 |
| behavioral_control_axis_intervention_plan.csv | behavioral_control_axis | read | 0.0105944 | 118 | 8 | 0.0275424 |

Per-question neutral +X vs random:

| question_index | x_likeness | random_mean | random_p95 | random_max | x_minus_random_mean | x_minus_random_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.460727 | 0.433881 | 0.472729 | 0.501954 | 0.0268467 | -0.0120013 |
| 1 | 0.511051 | 0.407622 | 0.526151 | 0.588868 | 0.103429 | -0.0151003 |
| 3 | 0.565822 | 0.439837 | 0.574774 | 0.590747 | 0.125985 | -0.008952 |
| 5 | 0.409931 | 0.404318 | 0.530266 | 0.543314 | 0.00561327 | -0.120336 |
| 8 | 0.390025 | 0.489621 | 0.728453 | 0.800731 | -0.0995962 | -0.338429 |
| 10 | 0.517211 | 0.423962 | 0.687608 | 0.998385 | 0.0932484 | -0.170398 |
| 13 | 0.471851 | 0.456344 | 0.513925 | 0.608858 | 0.0155064 | -0.0420748 |

## Decision Gates

- Hidden geometry strong: `target_projection >= 0.85` and length-neutral near zero.
- Internal generation strong: neutral +X generation projection beats almost all random vectors.
- Visible neutral +X strong: beats random p95 and wins most held-out questions without degeneration.
- Target -X ablation strong: suppression beats random p95/mean without degeneration.
- Breakthrough-grade claim needs cross-model and cross-text-family replication, not one run.
