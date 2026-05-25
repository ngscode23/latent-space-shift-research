# Red-Team Hidden Geometry Metric Audit

Reader-only audit. No model was run.

## Bottom Line

- Runs loaded: `1`
- Strong internal axis runs: `1/1`
- Best neutral +X visible lift: `0.0186818` on `Qwen/Qwen3.5-9B` (p95 lift `-0.0267254`, question win mean `60.0%`).
- Best target -X suppression lift: `0.00964887` on `Qwen/Qwen3.5-9B` (p95 lift `-0.0313993`, random win `71.9%`).
- Overall statuses: `{"strong_internal_axis_visible_weak": 1}`

Practical interpretation:

`Strong internal geometry.` The missing piece is hard visible readout over random p95/best and broader replication.

## Run Summary

| model_id | target_projection | neutral_lift_random_mean | neutral_lift_random_p95 | neutral_questions_over_random_mean | generation_win_random | target_minus_supp_lift_random_mean | target_minus_supp_win_random_mean | neutral_degeneration | visible_status | ablation_status | overall_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-9B | 0.940905 | 0.0186818 | -0.0267254 | 0.6 | 1 | 0.00964887 | 0.71875 | 1 | not_supported | weak_partial | strong_internal_axis_visible_weak |

## Run: Qwen/Qwen3.5-9B

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_breakthrough_grade.zip`
- Run label: `breakthrough_grade_hardened`
- Verdict CSV: `internal_axis_supported_behavioral_control_not_supported`
- Existing hard-random file: `True`
- Existing breakthrough audit: `False`
- Files in archive/directory: `71`

Key numbers:

- Target projection: `0.940905`
- Length-neutral projection: `0.000702429`
- Word-shuffle projection: `0.361236`
- Sentence-shuffle projection: `0.846842`
- Neutral +X target-likeness: `0.482841`
- Random visible mean: `0.464159`
- Neutral +X lift over random mean: `0.0186818`
- Neutral +X lift over random p95: `-0.0267254`
- Neutral +X question wins over random mean: `60.0%`
- Neutral +X question wins over random p95: `20.0%`
- Generation projection win over random: `100.0%`
- Target -X suppression lift over random mean: `0.00964887`
- Target -X suppression lift over random p95: `-0.0313993`
- Target -X suppression win over random mean: `71.9%`
- Neutral +X degeneration: `1`
- Target -X degeneration: `1`

Status: `strong_internal_axis_visible_weak`

Next action: try neutral-only question set and held-out text family; do not chase larger alpha

Full metric-family overview:

| family | status | readout | artifacts |
| --- | --- | --- | --- |
| endpoint_geometry | computed | target=0.940905, length_neutral=0.000702429, word_shuffle=0.361236, sentence_shuffle=0.846842 | middle_layer_condition_summary.csv |
| layerwise_geometry | computed | target_best_layer=1, max_projection=0.998936, target_layer_mean=0.933084 | layerwise_geometry_summary.csv |
| statistical_controls | computed | paired_rows=12; min_p=0.00029997; max_abs_d=8.62149; fdr_sig=89/99; min_q=0.000761462; null_p=0.00775194; obs_minus_null=0.941165; pca_max_abs_cos=0.363487 | paired_target_vs_control_tests.csv, layerwise_fdr_target_vs_control.csv, null_vector_baseline_summary.csv, pca_baseline_projection_summary.csv |
| bias_and_dataset_audits | computed | max_abs_length_corr=0.375432; duplicate_questions=0 | length_bias_audit.csv, deduplication_audit.csv, domain_robustness_geometry_summary.csv |
| architecture_features | computed | mlp.up_proj=0.935649; mlp.gate_proj=0.93527; mlp=0.915182; mlp.down_proj=0.915182; self_attn=0.88655 | architecture_module_delta_summary.csv, architecture_top_changed_units.csv |
| generation_trajectory | computed | neutral=0.359852; question_only=0.354439; target=0.547419; target_word_shuffle_control=0.485869 | generation_middle_layer_summary.csv, generation_response_audit.csv |
| causal_interventions | computed | rows=64, projection_range=[-1.42543, 2.08625], symmetry_supported=31/32 | causal_intervention_middle_layer_summary.csv, causal_bidirectional_symmetry_summary.csv |
| behavioral_control_axis | computed | neutral_lift_mean=0.0186818, neutral_lift_p95=-0.0267254, generation_win=100.0%, target_minus_supp_lift=0.00964887 | behavioral_control_axis_similarity_raw.csv, behavioral_control_axis_response_quality_summary.csv, behavioral_control_axis_verdict.csv |
| response_quality | computed | neutral_degeneration=1, target_minus_degeneration=1, neutral_unique=0.0873921 | behavioral_control_axis_response_quality_summary.csv |
| dynamic_geometry | computed | rows=52 | dynamic_trajectory_summary.csv, phase_transition_candidates.csv |
| feature_proxy | computed_or_statused | not_run_no_sae_model_configured; computed_from_hidden_top_changed_dimensions; dense_proxy_rows=64836 | feature_level_interpretability_status.csv, dense_feature_proxy_mapping.csv |
| artifact_coverage | computed | files=71, expected_present=48/54, missing_expected=6 | full archive inventory |

Artifact inventory by family:

| family | file_count | total_mb |
| --- | --- | --- |
| architecture_features | 6 | 75.5781 |
| behavioral_control_axis | 9 | 6.2499 |
| causal_interventions | 7 | 744.971 |
| dynamic_geometry | 3 | 0.0518322 |
| endpoint_geometry | 5 | 15.5089 |
| generation_trajectory | 3 | 61.2663 |
| manifest_protocol | 4 | 0.00937653 |
| other | 5 | 21.3314 |
| plots | 8 | 0.942412 |
| secondary_geometry | 4 | 0.0503569 |
| statistical_controls | 11 | 0.0381298 |
| tensor_snapshots | 3 | 19.4274 |
| verdict_report | 3 | 0.0226736 |

Expected artifact coverage:

- Missing expected artifacts: `6`
- Missing list: `behavioral_control_axis_response_quality_summary.csv, behavioral_control_axis_layer_band_comparison.csv, behavioral_control_axis_layer_band_verdict.csv, behavioral_control_axis_asymmetry_summary.csv, breakthrough_readiness_audit.csv, breakthrough_readiness_audit.md`

CSV health and NaN coverage:

- CSV status counts: `{"read": 56, "skipped_larger_than_512mb": 1}`

CSV files needing attention:

| basename | family | status | rows | columns | null_fraction | numeric_inf_cells | top_nan_columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| causal_intervention_trajectory_metrics_raw.csv | causal_interventions | skipped_larger_than_512mb | n/a | n/a | n/a | n/a |  |
| feature_level_interpretability_status.csv | architecture_features | read | 2 | 4 | 0.25 | 0 | sae_model_id:1 |
| deduplication_audit.csv | statistical_controls | read | 13 | 4 | 0.25 | 0 | normalized_duplicate_of:1 |
| behavioral_control_axis_verdict.csv | verdict_report | read | 1 | 17 | 0.235294 | 0 | target_baseline_likeness:1, target_minus_x_suppression:1, neutral_baseline_likeness:1, plus_x_lift_over_neutral:1 |

Largest CSV artifacts scanned:

| basename | family | status | size_mb | rows | columns | null_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| causal_intervention_trajectory_metrics_raw.csv | causal_interventions | skipped_larger_than_512mb | 743.423 | n/a | n/a | n/a |
| architecture_top_changed_units.csv | architecture_features | read | 68.6899 | 565760 | 12 | 0 |
| generation_trajectory_metrics_raw.csv | generation_trajectory | read | 61.0961 | 424050 | 13 | 0.00468794 |
| mlp_unit_cluster_summary.csv | other | read | 21.1197 | 290498 | 8 | 0 |
| hidden_top_changed_dimensions.csv | endpoint_geometry | read | 15.096 | 137280 | 11 | 0 |
| dense_feature_proxy_mapping.csv | architecture_features | read | 3.94789 | 64836 | 7 | 0 |
| behavioral_control_axis_similarity_raw.csv | behavioral_control_axis | read | 3.20823 | 1070 | 53 | 0.000440839 |
| behavioral_control_axis_response_audit.csv | behavioral_control_axis | read | 2.95488 | 1070 | 40 | 0.00046729 |
| architecture_module_delta_summary.csv | architecture_features | read | 1.54916 | 8840 | 13 | 0 |
| causal_intervention_response_audit.csv | causal_interventions | read | 1.5252 | 832 | 23 | 0 |

Per-question neutral +X vs random:

| question_index | x_likeness | random_mean | random_p95 | random_max | x_minus_random_mean | x_minus_random_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.530101 | 0.514297 | 0.592379 | 0.633164 | 0.0158047 | -0.0622776 |
| 3 | 0.434485 | 0.476434 | 0.558111 | 0.586258 | -0.0419494 | -0.123626 |
| 5 | 0.483747 | 0.491689 | 0.59698 | 0.755216 | -0.0079424 | -0.113233 |
| 8 | 0.500287 | 0.39022 | 0.496783 | 0.581394 | 0.110067 | 0.00350392 |
| 11 | 0.465587 | 0.448158 | 0.538695 | 0.604698 | 0.0174294 | -0.0731075 |

## Decision Gates

- Hidden geometry strong: `target_projection >= 0.85` and length-neutral near zero.
- Internal generation strong: neutral +X generation projection beats almost all random vectors.
- Visible neutral +X strong: beats random p95 and wins most held-out questions without degeneration.
- Target -X ablation strong: suppression beats random p95/mean without degeneration.
- Breakthrough-grade claim needs cross-model and cross-text-family replication, not one run.
