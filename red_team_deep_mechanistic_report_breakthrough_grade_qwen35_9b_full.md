# Deep Mechanistic Report

Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_breakthrough_grade.zip`

## Run

- Model: `Qwen/Qwen3.5-9B`
- Run label: `breakthrough_grade_hardened`
- Questions: `13`
- Reference: `neutral`
- Conditions: `question_only, neutral, target_word_shuffle_control, target_sentence_shuffle_control, neutral_length_matched_control, target`
- Causal bands: `['middle', 'late']`
- Causal alphas: `[0.25, 0.6, 0.75, 1.0]`
- Behavioral bands: `['middle', 'late']`
- Behavioral alphas: `[0.25, 0.6, 0.75, 1.0]`
- Behavioral random baselines: `64`
- Behavioral random alpha: `1.0`

## Artifact Risk

| artifact | size_mb |
| --- | --- |
| causal_intervention_trajectory_metrics_raw.csv | 743.423 |
| architecture_top_changed_units.csv | 68.6899 |
| generation_trajectory_metrics_raw.csv | 61.0961 |
| mlp_unit_cluster_summary.csv | 21.1197 |
| prompt_hidden_states.npz | 18.7835 |
| hidden_top_changed_dimensions.csv | 15.096 |
| dense_feature_proxy_mapping.csv | 3.94789 |
| behavioral_control_axis_similarity_raw.csv | 3.20823 |
| behavioral_control_axis_response_audit.csv | 2.95488 |
| architecture_module_delta_summary.csv | 1.54916 |
| causal_intervention_response_audit.csv | 1.5252 |
| architecture_target_vs_control_overlap.csv | 0.695436 |
| architecture_target_vs_shuffle_overlap.csv | 0.695436 |
| layerwise_geometry_metrics_raw.csv | 0.385647 |
| causal_intervention_projection_trajectory.png | 0.342403 |
| vector_x_by_layer.npz | 0.338061 |
| behavioral_control_train_vector_x_by_layer.npz | 0.305768 |
| generation_projection_trajectory.png | 0.17699 |
| generation_response_audit.csv | 0.169673 |
| output_semantic_shift_raw.csv | 0.134259 |

## 1. Hidden Geometry

| condition | projection_fraction_on_vector_x_loo_mean | projection_fraction_on_vector_x_loo_ci95_low | projection_fraction_on_vector_x_loo_ci95_high | direction_cosine_with_vector_x_loo_mean | projection_positive_fraction | l2_distance_to_reference_mean |
| --- | --- | --- | --- | --- | --- | --- |
| question_only | -0.000351141 | -0.0524153 | 0.0473795 | 0.0210594 | 0.633136 | 18.4774 |
| target_word_shuffle_control | 0.361236 | 0.331335 | 0.390136 | 0.269944 | 0.952663 | 10.7408 |
| target_sentence_shuffle_control | 0.846842 | 0.814051 | 0.881005 | 0.618352 | 1 | 11.2426 |
| neutral_length_matched_control | 0.000702429 | -0.00626132 | 0.00762338 | -0.00158543 | 0.514793 | 3.83716 |
| target | 0.940905 | 0.90931 | 0.973674 | 0.727699 | 1 | 10.8938 |

Mechanistic read:

- Target projection is `0.940905`; length-matched neutral is `0.000702429`.
- This separates a target-induced latent geometry shift from a generic long-context effect.

## 2. Statistical Hardening

Paired target-vs-control:

| control_condition | metric | target_minus_control_mean | target_minus_control_ci95_low | target_minus_control_ci95_high | paired_cohen_d | target_greater_than_control_fraction | fdr_q_value | fdr_significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_word_shuffle_control | mean_projection_fraction_on_vector_x_loo | 0.579669 | 0.534214 | 0.635287 | 6.25614 | 1 | 0.000514234 | 1 |
| target_word_shuffle_control | mean_direction_cosine_with_vector_x_loo | 0.457755 | 0.402929 | 0.504255 | 4.65538 | 1 | 0.000514234 | 1 |
| target_word_shuffle_control | mean_l2_distance_to_reference | 0.152991 | -0.824306 | 1.00998 | 0.0850116 | 0.615385 | 0.835989 | 0 |
| target_word_shuffle_control | mean_cosine_distance_to_reference | 0.000431932 | -0.00496335 | 0.00545684 | 0.0425904 | 0.615385 | 0.887911 | 0 |
| target_sentence_shuffle_control | mean_projection_fraction_on_vector_x_loo | 0.0940634 | 0.0614429 | 0.12771 | 1.46118 | 0.923077 | 0.00089991 | 1 |
| target_sentence_shuffle_control | mean_direction_cosine_with_vector_x_loo | 0.109348 | 0.0850441 | 0.132558 | 2.37787 | 1 | 0.000514234 | 1 |
| target_sentence_shuffle_control | mean_l2_distance_to_reference | -0.348789 | -0.884553 | 0.0931083 | -0.366748 | 0.384615 | 0.287131 | 0 |
| target_sentence_shuffle_control | mean_cosine_distance_to_reference | -0.00267887 | -0.00610998 | -4.06716e-05 | -0.449335 | 0.307692 | 0.151585 | 0 |
| neutral_length_matched_control | mean_projection_fraction_on_vector_x_loo | 0.940203 | 0.844055 | 1.04253 | 4.8095 | 1 | 0.000514234 | 1 |
| neutral_length_matched_control | mean_direction_cosine_with_vector_x_loo | 0.729284 | 0.684118 | 0.773462 | 8.62149 | 1 | 0.000514234 | 1 |
| neutral_length_matched_control | mean_l2_distance_to_reference | 7.05662 | 6.10542 | 8.25329 | 3.32089 | 1 | 0.000514234 | 1 |
| neutral_length_matched_control | mean_cosine_distance_to_reference | 0.0289613 | 0.0227842 | 0.0371671 | 2.03375 | 1 | 0.000514234 | 1 |

- Layerwise FDR significant rows: `89/99`

Random-vector null:

| baseline_type | observed_target_projection_mean | null_mean | null_std | observed_minus_null_mean | empirical_p_greater_equal_observed | null_count |
| --- | --- | --- | --- | --- | --- | --- |
| random_same_norm | 0.940905 | -0.000259816 | 0.00170044 | 0.941165 | 0.00775194 | 128 |

Subspace decomposition:

| rank_k | explained_variance_fraction_cumulative | vector_x_reconstruction_fraction | vector_x_projection_norm | vector_x_residual_norm |
| --- | --- | --- | --- | --- |
| 1 | 0.308645 | 0.10684 | 15.6851 | 31.1499 |
| 2 | 0.43866 | 0.16303 | 19.0859 | 29.1902 |
| 3 | 0.561306 | 0.168006 | 19.3487 | 29.0167 |
| 4 | 0.643807 | 0.176302 | 19.7757 | 28.7273 |
| 5 | 0.725398 | 0.184729 | 20.196 | 28.4334 |
| 6 | 0.78867 | 0.184812 | 20.2001 | 28.4305 |
| 7 | 0.842814 | 0.193117 | 20.6017 | 28.1409 |
| 8 | 0.88452 | 0.193266 | 20.6088 | 28.1357 |

## 3. Architecture

| module | mean_projection | mean_direction_cosine | mean_abs_delta | n_rows |
| --- | --- | --- | --- | --- |
| mlp.up_proj | 0.935649 | 0.732364 | 0.156727 | 416 |
| mlp.gate_proj | 0.93527 | 0.73258 | 0.158717 | 416 |
| mlp | 0.915182 | 0.683842 | 0.108108 | 416 |
| mlp.down_proj | 0.915182 | 0.683842 | 0.108108 | 416 |
| self_attn | 0.88655 | 0.637547 | 0.0757487 | 104 |

Top circuit rows:

| condition | module | layer | mean_projection_fraction_on_arch_vector_x_loo | mean_direction_cosine_with_arch_vector_x_loo | mean_abs_delta | mean_l2_distance_to_reference |
| --- | --- | --- | --- | --- | --- | --- |
| question_only | mlp | 1 | 1.24805 | 0.470691 | 0.00527442 | 0.533335 |
| question_only | mlp.down_proj | 1 | 1.24805 | 0.470691 | 0.00527442 | 0.533335 |
| target | mlp.gate_proj | 1 | 0.999743 | 0.998208 | 0.0260308 | 3.93859 |
| target | mlp.up_proj | 1 | 0.99973 | 0.998137 | 0.022219 | 3.22011 |
| target | mlp | 1 | 0.999437 | 0.99616 | 0.00247479 | 0.201719 |
| target | mlp.down_proj | 1 | 0.999437 | 0.99616 | 0.00247479 | 0.201719 |
| target | mlp.up_proj | 2 | 0.999284 | 0.995309 | 0.0275447 | 3.89554 |
| target | mlp.gate_proj | 2 | 0.99922 | 0.994865 | 0.0322431 | 4.61907 |
| target | mlp | 2 | 0.999074 | 0.993886 | 0.00362496 | 0.295347 |
| target | mlp.down_proj | 2 | 0.999074 | 0.993886 | 0.00362496 | 0.295347 |
| target | mlp.up_proj | 3 | 0.998388 | 0.989606 | 0.0316157 | 4.40954 |
| target | mlp.gate_proj | 3 | 0.998161 | 0.988001 | 0.0363009 | 5.03952 |
| target | mlp | 3 | 0.997572 | 0.984507 | 0.00460523 | 0.373195 |
| target | mlp.down_proj | 3 | 0.997572 | 0.984507 | 0.00460523 | 0.373195 |
| target | mlp.up_proj | 4 | 0.993399 | 0.957975 | 0.0503379 | 7.08386 |
| target | mlp.gate_proj | 4 | 0.992729 | 0.95342 | 0.0590875 | 8.37178 |
| target | self_attn | 4 | 0.991196 | 0.945786 | 0.00830909 | 0.674791 |
| target | mlp | 4 | 0.990363 | 0.940247 | 0.0114945 | 0.920151 |
| target | mlp.down_proj | 4 | 0.990363 | 0.940247 | 0.0114945 | 0.920151 |
| target | mlp.up_proj | 5 | 0.989512 | 0.934937 | 0.0608609 | 8.48557 |

## 4. Generation And Causal Trajectory

Generation summary:

| condition | mean_projection_fraction_on_vector_x_loo | mean_direction_cosine_with_vector_x_loo | mean_l2_distance_to_reference_prompt_endpoint | mean_entropy | n_rows |
| --- | --- | --- | --- | --- | --- |
| neutral | 0.359852 | 0.0636821 | 42.3946 | 0.760122 | 42874 |
| question_only | 0.354439 | 0.0619572 | 42.8079 | 0.667684 | 37648 |
| target | 0.547419 | 0.101587 | 42.5631 | 0.866348 | 43264 |
| target_word_shuffle_control | 0.485869 | 0.0877545 | 42.5078 | 0.840613 | 43264 |

Causal symmetry:

| base_condition | layer_band | alpha_abs | plus_x_projection | minus_x_projection | plus_minus_projection_gap | bidirectional_symmetry_supported |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | late | 0.25 | 0.349977 | 0.305548 | 0.0444293 | 1 |
| neutral | late | 0.6 | 0.305994 | 0.240269 | 0.0657249 | 1 |
| neutral | late | 0.75 | 0.301776 | 0.273077 | 0.0286986 | 1 |
| neutral | late | 1 | 0.279622 | 0.297353 | -0.0177309 | 0 |
| neutral | middle | 0.25 | 0.825205 | -0.130479 | 0.955684 | 1 |
| neutral | middle | 0.6 | 1.49432 | -0.79309 | 2.28741 | 1 |
| neutral | middle | 0.75 | 1.66038 | -1.08556 | 2.74594 | 1 |
| neutral | middle | 1 | 1.99269 | -1.42543 | 3.41812 | 1 |
| target | late | 0.25 | 0.563401 | 0.537119 | 0.0262827 | 1 |
| target | late | 0.6 | 0.556725 | 0.35231 | 0.204415 | 1 |
| target | late | 0.75 | 0.469979 | 0.361112 | 0.108867 | 1 |
| target | late | 1 | 0.397802 | 0.360031 | 0.0377703 | 1 |
| target | middle | 0.25 | 1.02443 | 0.05604 | 0.968392 | 1 |
| target | middle | 0.6 | 1.58817 | -0.714069 | 2.30224 | 1 |
| target | middle | 0.75 | 1.79773 | -0.98243 | 2.78016 | 1 |
| target | middle | 1 | 2.0776 | -1.37202 | 3.44961 | 1 |
| target_sentence_shuffle_control | late | 0.25 | 0.54761 | 0.528602 | 0.0190077 | 1 |
| target_sentence_shuffle_control | late | 0.6 | 0.531265 | 0.366359 | 0.164906 | 1 |
| target_sentence_shuffle_control | late | 0.75 | 0.449718 | 0.346746 | 0.102972 | 1 |
| target_sentence_shuffle_control | late | 1 | 0.388222 | 0.353343 | 0.0348792 | 1 |
| target_sentence_shuffle_control | middle | 0.25 | 1.01054 | 0.0420549 | 0.968485 | 1 |
| target_sentence_shuffle_control | middle | 0.6 | 1.59404 | -0.725554 | 2.31959 | 1 |
| target_sentence_shuffle_control | middle | 0.75 | 1.79778 | -0.987767 | 2.78555 | 1 |
| target_sentence_shuffle_control | middle | 1 | 2.08625 | -1.35135 | 3.4376 | 1 |
| target_word_shuffle_control | late | 0.25 | 0.496883 | 0.431925 | 0.0649575 | 1 |
| target_word_shuffle_control | late | 0.6 | 0.418434 | 0.317732 | 0.100703 | 1 |
| target_word_shuffle_control | late | 0.75 | 0.376655 | 0.316353 | 0.0603023 | 1 |
| target_word_shuffle_control | late | 1 | 0.343141 | 0.329225 | 0.0139152 | 1 |
| target_word_shuffle_control | middle | 0.25 | 0.974723 | -0.082905 | 1.05763 | 1 |
| target_word_shuffle_control | middle | 0.6 | 1.59418 | -0.779732 | 2.37391 | 1 |
| target_word_shuffle_control | middle | 0.75 | 1.75434 | -1.02708 | 2.78142 | 1 |
| target_word_shuffle_control | middle | 1 | 2.04055 | -1.36267 | 3.40323 | 1 |

Causal alpha scaling:

| base_condition | layer_band | sign_name | alpha_projection_slope | projection_min | projection_max | n_alpha_points |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | late | minus_x | 0.00966553 | 0.240269 | 0.305548 | 4 |
| neutral | late | plus_x | -0.0923568 | 0.279622 | 0.349977 | 4 |
| neutral | middle | minus_x | 1.74784 | -1.42543 | -0.130479 | 4 |
| neutral | middle | plus_x | 1.55485 | 0.825205 | 1.99269 | 4 |
| target | late | minus_x | 0.238443 | 0.35231 | 0.537119 | 4 |
| target | late | plus_x | -0.227011 | 0.397802 | 0.563401 | 4 |
| target | middle | minus_x | 1.9158 | -1.37202 | 0.05604 | 4 |
| target | middle | plus_x | 1.4161 | 1.02443 | 2.0776 | 4 |
| target_sentence_shuffle_control | late | minus_x | 0.242082 | 0.346746 | 0.528602 | 4 |
| target_sentence_shuffle_control | late | plus_x | -0.219517 | 0.388222 | 0.54761 | 4 |
| target_sentence_shuffle_control | middle | minus_x | 1.87218 | -1.35135 | 0.0420549 | 4 |
| target_sentence_shuffle_control | middle | plus_x | 1.44423 | 1.01054 | 2.08625 | 4 |
| target_word_shuffle_control | late | minus_x | 0.141669 | 0.316353 | 0.431925 | 4 |
| target_word_shuffle_control | late | plus_x | -0.209865 | 0.343141 | 0.496883 | 4 |
| target_word_shuffle_control | middle | minus_x | 1.72032 | -1.36267 | -0.082905 | 4 |
| target_word_shuffle_control | middle | plus_x | 1.42383 | 0.974723 | 2.04055 | 4 |

Causal middle-layer summary, strongest projection rows:

| base_condition | layer_band | alpha | alpha_abs | sign_name | mean_projection_fraction_on_vector_x_loo | mean_direction_cosine_with_vector_x_loo | mean_entropy | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_sentence_shuffle_control | middle | 1 | 1 | plus_x | 2.08625 | 0.353871 | 0.67743 | 21632 |
| target | middle | 1 | 1 | plus_x | 2.0776 | 0.348354 | 0.703965 | 21632 |
| target_word_shuffle_control | middle | 1 | 1 | plus_x | 2.04055 | 0.344057 | 0.582418 | 21632 |
| neutral | middle | 1 | 1 | plus_x | 1.99269 | 0.339315 | 0.65302 | 21632 |
| target_sentence_shuffle_control | middle | 0.75 | 0.75 | plus_x | 1.79778 | 0.318245 | 1.43648 | 21632 |
| target | middle | 0.75 | 0.75 | plus_x | 1.79773 | 0.318521 | 1.35161 | 21632 |
| target_word_shuffle_control | middle | 0.75 | 0.75 | plus_x | 1.75434 | 0.308273 | 1.03509 | 21632 |
| neutral | middle | 0.75 | 0.75 | plus_x | 1.66038 | 0.295706 | 1.14155 | 21632 |
| target_word_shuffle_control | middle | 0.6 | 0.6 | plus_x | 1.59418 | 0.287629 | 1.47927 | 21632 |
| target_sentence_shuffle_control | middle | 0.6 | 0.6 | plus_x | 1.59404 | 0.28893 | 1.58815 | 21632 |
| target | middle | 0.6 | 0.6 | plus_x | 1.58817 | 0.288193 | 1.5528 | 21632 |
| neutral | middle | 0.6 | 0.6 | plus_x | 1.49432 | 0.272119 | 1.65316 | 21632 |
| target | middle | 0.25 | 0.25 | plus_x | 1.02443 | 0.194945 | 0.988152 | 21632 |
| target_sentence_shuffle_control | middle | 0.25 | 0.25 | plus_x | 1.01054 | 0.191947 | 1.01629 | 21632 |
| target_word_shuffle_control | middle | 0.25 | 0.25 | plus_x | 0.974723 | 0.183138 | 0.978398 | 21632 |
| neutral | middle | 0.25 | 0.25 | plus_x | 0.825205 | 0.159234 | 0.922574 | 21632 |

## 5. Behavioral Readout

Behavioral verdict:

| verdict | n_train | n_test | primary_layer_band | primary_max_alpha | neutral_baseline_likeness | target_baseline_likeness | neutral_plus_x_likeness | neutral_minus_x_likeness | target_minus_x_likeness | random_plus_likeness | plus_x_lift_over_neutral | plus_x_lift_over_random | target_minus_x_suppression | plus_x_behavioral_alpha_slope | neutral_plus_x_generation_projection | random_plus_generation_projection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_axis_supported_behavioral_control_not_supported | 8 | 5 | middle | 1 | n/a | n/a | 0.482841 | 0.486557 | 0.48132 | 0.464159 | n/a | 0.0186818 | n/a | 0.00641274 | 1.81805 | 0.248628 |

Hard-random rows:

| base_condition | sign_name | alpha_abs | layer_band | mean_vector_x_likeness | mean_random_mean_likeness | mean_lift_over_random_mean | mean_lift_over_random_p95 | mean_lift_over_random_best | win_rate_vs_random_mean | win_rate_vs_random_p95 | win_rate_vs_random_best | n_questions | mean_n_random_vectors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | minus_x | 0.25 | middle | 0.453978 | 0.466168 | -0.0121895 | -0.0923666 | -0.189014 | 0.6 | 0 | 0 | 5 | 64 |
| neutral | minus_x | 0.6 | middle | 0.468267 | 0.466168 | 0.0020997 | -0.0780774 | -0.174725 | 0.4 | 0 | 0 | 5 | 64 |
| neutral | minus_x | 0.75 | middle | 0.474749 | 0.466168 | 0.00858136 | -0.0715957 | -0.168243 | 0.4 | 0.2 | 0 | 5 | 64 |
| neutral | minus_x | 1 | middle | 0.486557 | 0.466168 | 0.0203894 | -0.0597876 | -0.156435 | 0.6 | 0 | 0 | 5 | 64 |
| neutral | plus_x | 0.25 | middle | 0.478597 | 0.464159 | 0.0144373 | -0.0779925 | -0.153549 | 0.6 | 0.2 | 0 | 5 | 64 |
| neutral | plus_x | 0.6 | middle | 0.50174 | 0.464159 | 0.037581 | -0.0548488 | -0.130406 | 1 | 0 | 0 | 5 | 64 |
| neutral | plus_x | 0.75 | middle | 0.494231 | 0.464159 | 0.0300712 | -0.0623586 | -0.137915 | 0.8 | 0 | 0 | 5 | 64 |
| neutral | plus_x | 1 | middle | 0.482841 | 0.464159 | 0.0186818 | -0.0737481 | -0.149305 | 0.6 | 0.2 | 0 | 5 | 64 |
| target | minus_x | 0.25 | middle | 0.482915 | 0.490969 | -0.00805396 | -0.0957116 | -0.148187 | 0.4 | 0 | 0 | 5 | 64 |
| target | minus_x | 0.6 | middle | 0.492211 | 0.490969 | 0.00124162 | -0.086416 | -0.138892 | 0.4 | 0 | 0 | 5 | 64 |
| target | minus_x | 0.75 | middle | 0.476409 | 0.490969 | -0.0145602 | -0.102218 | -0.154694 | 0.2 | 0 | 0 | 5 | 64 |
| target | minus_x | 1 | middle | 0.48132 | 0.490969 | -0.00964887 | -0.0973065 | -0.149782 | 0.2 | 0 | 0 | 5 | 64 |

Best neutral +X rows:

| base_condition | sign_name | alpha_abs | layer_band | mean_vector_x_likeness | mean_random_mean_likeness | mean_lift_over_random_mean | mean_lift_over_random_p95 | win_rate_vs_random_mean | win_rate_vs_random_p95 | corrected_degenerate_rate | loop_rate | mean_unique_word_ratio | mean_repeated_trigram_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | plus_x | 0.6 | middle | 0.50174 | 0.464159 | 0.037581 | -0.0548488 | 1 | 0 | 0.6 | 0.6 | 0.58692 | 0.247229 |
| neutral | plus_x | 0.75 | middle | 0.494231 | 0.464159 | 0.0300712 | -0.0623586 | 0.8 | 0 | 1 | 1 | 0.347486 | 0.627906 |
| neutral | plus_x | 1 | middle | 0.482841 | 0.464159 | 0.0186818 | -0.0737481 | 0.6 | 0.2 | 1 | 1 | 0.0873921 | 0.919887 |
| neutral | plus_x | 0.25 | middle | 0.478597 | 0.464159 | 0.0144373 | -0.0779925 | 0.6 | 0.2 | 0 | 0 | 0.838572 | 0 |

Best target -X rows:

| base_condition | sign_name | alpha_abs | layer_band | mean_vector_x_likeness | mean_random_mean_likeness | mean_lift_over_random_mean | mean_lift_over_random_p95 | win_rate_vs_random_mean | win_rate_vs_random_p95 | corrected_degenerate_rate | loop_rate | mean_unique_word_ratio | mean_repeated_trigram_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target | minus_x | 0.6 | middle | 0.492211 | 0.490969 | 0.00124162 | -0.086416 | 0.4 | 0 | 1 | 1 | 0.302164 | 0.659386 |
| target | minus_x | 0.25 | middle | 0.482915 | 0.490969 | -0.00805396 | -0.0957116 | 0.4 | 0 | 0 | 0 | 0.826147 | 0.00792079 |
| target | minus_x | 1 | middle | 0.48132 | 0.490969 | -0.00964887 | -0.0973065 | 0.2 | 0 | 1 | 1 | 0.0545754 | 0.934094 |
| target | minus_x | 0.75 | middle | 0.476409 | 0.490969 | -0.0145602 | -0.102218 | 0.2 | 0 | 1 | 1 | 0.206225 | 0.766238 |

Behavioral alpha slopes:

| base_condition | layer_band | sign_name | intervention_kind | alpha_behavioral_target_likeness_slope_cosine | alpha_generation_projection_slope | n_alpha_points |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | middle | minus_x | vector_x | -0.0432721 | 1.67451 | 4 |
| neutral | middle | plus_x | vector_x | 0.00641274 | 1.42305 | 4 |
| target | middle | minus_x | vector_x | 0.0056733 | 1.73493 | 4 |
| target | middle | plus_x | vector_x | -0.0484425 | 1.35534 | 4 |

Corrected response quality:

| base_condition | intervention_kind | sign_name | alpha_abs | layer_band | n_questions | corrected_degenerate_rate | loop_rate | low_diversity_rate | mean_visible_word_count | mean_unique_word_ratio | mean_repeated_trigram_fraction | mean_entropy | mean_generation_projection_on_train_vector_x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | baseline | none | n/a | none | 5 | 0 | 0 | 0 | 89 | 0.857238 | 0.0047619 | 0.772543 | 0.266085 |
| neutral | random | minus_random | 1 | middle | 5 | 0.00625 | 0.00625 | 0 | 92.5219 | 0.82911 | 0.0159284 | 1.09388 | 0.249113 |
| neutral | random | plus_random | 1 | middle | 5 | 0 | 0 | 0 | 91.4719 | 0.834173 | 0.0112477 | 1.07433 | 0.248628 |
| neutral | vector_x | minus_x | 0.25 | middle | 5 | 0 | 0 | 0 | 96.2 | 0.832185 | 0 | 0.999156 | -0.182576 |
| neutral | vector_x | minus_x | 0.6 | middle | 5 | 0.8 | 0.8 | 0 | 113.4 | 0.454373 | 0.426528 | 1.77052 | -0.870289 |
| neutral | vector_x | minus_x | 0.75 | middle | 5 | 1 | 1 | 0.6 | 112.2 | 0.24912 | 0.755218 | 1.89418 | -1.11612 |
| neutral | vector_x | minus_x | 1 | late | 5 | 1 | 1 | 1 | 190.6 | 0.007352 | 0.997872 | 0.878433 | 0.204429 |
| neutral | vector_x | minus_x | 1 | middle | 5 | 1 | 1 | 1 | 190.6 | 0.0283501 | 0.970302 | 1.54807 | -1.42547 |
| neutral | vector_x | plus_x | 0.25 | middle | 5 | 0 | 0 | 0 | 88.6 | 0.838572 | 0 | 0.848173 | 0.752419 |
| neutral | vector_x | plus_x | 0.6 | middle | 5 | 0.6 | 0.6 | 0 | 79.4 | 0.58692 | 0.247229 | 1.39054 | 1.3514 |
| neutral | vector_x | plus_x | 0.75 | middle | 5 | 1 | 1 | 0.2 | 81 | 0.347486 | 0.627906 | 1.03578 | 1.52021 |
| neutral | vector_x | plus_x | 1 | late | 5 | 1 | 1 | 1 | 173.6 | 0.0308879 | 0.971776 | 1.33138 | 0.204475 |
| neutral | vector_x | plus_x | 1 | middle | 5 | 1 | 1 | 1 | 98.4 | 0.0873921 | 0.919887 | 1.31118 | 1.81805 |
| target | baseline | none | n/a | none | 5 | 0 | 0 | 0 | 88 | 0.87258 | 0 | 0.927623 | 0.435817 |
| target | random | minus_random | 1 | middle | 5 | 0.003125 | 0.003125 | 0 | 92.6656 | 0.827999 | 0.0124959 | 1.21766 | 0.406558 |
| target | vector_x | minus_x | 0.25 | middle | 5 | 0 | 0 | 0 | 93.8 | 0.826147 | 0.00792079 | 1.07504 | -0.0811902 |
| target | vector_x | minus_x | 0.6 | middle | 5 | 1 | 1 | 0.2 | 110.2 | 0.302164 | 0.659386 | 1.40725 | -0.797558 |
| target | vector_x | minus_x | 0.75 | middle | 5 | 1 | 1 | 0.8 | 124.2 | 0.206225 | 0.766238 | 1.54002 | -1.03728 |
| target | vector_x | minus_x | 1 | late | 5 | 1 | 1 | 1 | 191.4 | 0.0104494 | 0.99472 | 1.22798 | 0.258426 |
| target | vector_x | minus_x | 1 | middle | 5 | 1 | 1 | 1 | 165.4 | 0.0545754 | 0.934094 | 1.47254 | -1.37266 |
| target | vector_x | plus_x | 0.25 | middle | 5 | 0 | 0 | 0 | 88.8 | 0.812538 | 0.0137507 | 1.00444 | 0.897216 |
| target | vector_x | plus_x | 0.6 | middle | 5 | 0.2 | 0.2 | 0 | 92.8 | 0.638128 | 0.169472 | 1.50709 | 1.44365 |
| target | vector_x | plus_x | 0.75 | middle | 5 | 1 | 1 | 0.4 | 103.8 | 0.279591 | 0.658259 | 1.05271 | 1.61198 |
| target | vector_x | plus_x | 1 | late | 5 | 1 | 1 | 1 | 415.6 | 0.013358 | 0.993961 | 2.73903 | 0.272036 |
| target | vector_x | plus_x | 1 | middle | 5 | 1 | 1 | 0.6 | 63.6 | 0.0192791 | 0.585672 | 0.549619 | 1.91341 |

## 6. Output Semantic Shift

| condition | mean_response_cosine_distance_to_reference | mean_response_l2_distance_to_reference | mean_response_projection_fraction_on_vector_x_loo | mean_response_direction_cosine_with_vector_x_loo | n_rows |
| --- | --- | --- | --- | --- | --- |
| question_only | 0.400713 | 40.462 | -0.0929459 | -0.016648 | 169 |
| target | 0.384752 | 40.1053 | 0.0536817 | 0.0117938 | 169 |
| target_word_shuffle_control | 0.387548 | 39.8528 | 0.0248966 | 0.00577046 | 169 |

## 7. Dynamic Geometry

| question_index | condition | n_steps | projection_start | projection_end | projection_mean | projection_max | projection_min | projection_slope_per_token | projection_largest_abs_jump | projection_largest_jump_step | projection_volatility | early_projection_mean | late_projection_mean | late_minus_early_projection | tail_projection_std | attractor_convergence_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | neutral | 256 | -0.000214692 | 0.155933 | 0.0494294 | 0.537254 | -0.196613 | 0.000400423 | 0.542827 | 77 | 0.0808138 | -0.0119198 | 0.0998025 | 0.111722 | 0.0441616 | -0.00682873 |
| 0 | question_only | 129 | -0.455024 | -0.260724 | -0.0500968 | 0.177284 | -0.614507 | 0.00192582 | 0.329974 | 3 | 0.0930194 | -0.298115 | -0.0935645 | 0.204551 | 0.134546 | 0.00572196 |
| 0 | target | 256 | 0.652607 | 0.116059 | 0.232723 | 0.914329 | 0.00697197 | -0.000213026 | 0.748161 | 117 | 0.112758 | 0.433197 | 0.156849 | -0.276348 | 0.0503858 | 0.0402112 |
| 0 | target_word_shuffle_control | 256 | 0.144864 | 0.252607 | 0.12484 | 0.676225 | -0.0965179 | 3.21682e-05 | 0.590389 | 105 | 0.0964061 | 0.140936 | 0.179379 | 0.0384433 | 0.0837167 | -0.0356354 |
| 1 | neutral | 256 | 0.00354926 | 0.494475 | 0.519085 | 0.732062 | 0.00354926 | 0.000364734 | 0.347229 | 1 | 0.0650481 | 0.366232 | 0.566425 | 0.200193 | 0.0892189 | 0.0436415 |
| 1 | question_only | 105 | -0.407963 | 0.153916 | 0.276999 | 0.563547 | -0.407963 | 0.00288464 | 0.36449 | 1 | 0.0898447 | 0.111542 | 0.383977 | 0.272434 | 0.129837 | 0.0653276 |
| 1 | target | 256 | 0.871084 | 0.659464 | 0.639654 | 1.10321 | 0.444834 | 8.87673e-05 | 0.389973 | 111 | 0.0829272 | 0.736209 | 0.695315 | -0.0408931 | 0.0711546 | 0.0163666 |
| 1 | target_word_shuffle_control | 256 | 0.296893 | 0.679102 | 0.605551 | 0.850392 | 0.296893 | 0.000477504 | 0.286455 | 137 | 0.0763385 | 0.472401 | 0.592226 | 0.119825 | 0.0651806 | 0.0142787 |
| 2 | neutral | 256 | 0.00516675 | 0.333685 | 0.317312 | 0.690945 | 0.00516675 | 0.000504391 | 0.487934 | 86 | 0.0856728 | 0.192882 | 0.358957 | 0.166075 | 0.0393274 | 0.0364693 |
| 2 | question_only | 256 | -0.320323 | 0.212067 | 0.204644 | 0.454422 | -0.337811 | 0.00100899 | 0.333592 | 17 | 0.0848185 | -0.103543 | 0.252645 | 0.356188 | 0.0894784 | 0.0693147 |
| 2 | target | 256 | 0.886518 | 0.503325 | 0.490162 | 1.1698 | 0.273926 | -0.000282172 | 0.784355 | 134 | 0.110248 | 0.697163 | 0.477163 | -0.219999 | 0.0627575 | 0.0254012 |
| 2 | target_word_shuffle_control | 256 | 0.250543 | 0.593302 | 0.447276 | 0.9416 | 0.145101 | 0.000439171 | 0.70582 | 115 | 0.110332 | 0.369406 | 0.446414 | 0.0770074 | 0.0693536 | -0.00332829 |
| 3 | neutral | 256 | -0.00173684 | 0.598879 | 0.597989 | 0.836451 | -0.00173684 | 0.000451641 | 0.386092 | 1 | 0.0775016 | 0.394275 | 0.667945 | 0.273671 | 0.05619 | 0.108418 |
| 3 | question_only | 102 | -0.218582 | 0.301257 | 0.451942 | 0.739686 | -0.218582 | 0.00240529 | 0.363186 | 37 | 0.107241 | 0.281157 | 0.581866 | 0.300709 | 0.106242 | 0.0888803 |
| 3 | target | 256 | 0.934879 | 0.766536 | 0.791059 | 1.06815 | 0.518786 | -7.63731e-05 | 0.413109 | 130 | 0.0965669 | 0.845579 | 0.759023 | -0.0865559 | 0.0378337 | 0.0629097 |
| 3 | target_word_shuffle_control | 256 | 0.390676 | 0.636431 | 0.774264 | 0.969158 | 0.390676 | 0.000129 | 0.280168 | 1 | 0.0823083 | 0.698672 | 0.721878 | 0.0232066 | 0.0926538 | 0.0356128 |
| 4 | neutral | 226 | -0.00163729 | 0.679911 | 0.724685 | 0.964686 | -0.00163729 | 2.24766e-05 | 0.452673 | 1 | 0.0895308 | 0.527469 | 0.751159 | 0.22369 | 0.0552684 | 0.143893 |
| 4 | question_only | 256 | 0.155992 | 0.84828 | 0.842682 | 1.04252 | 0.155992 | 0.000567509 | 0.431739 | 1 | 0.0725425 | 0.532615 | 0.89085 | 0.358236 | 0.0477143 | 0.0917902 |
| 4 | target | 256 | 1.21576 | 1.04493 | 1.05557 | 1.31349 | 0.892025 | -0.000203509 | 0.351158 | 54 | 0.0771755 | 1.09546 | 0.995458 | -0.100005 | 0.055075 | 0.0455995 |
| 4 | target_word_shuffle_control | 256 | 0.719833 | 0.983595 | 1.04724 | 1.2921 | 0.719833 | 8.66417e-05 | 0.298283 | 78 | 0.0677666 | 0.952688 | 1.02827 | 0.075581 | 0.0328879 | 0.0694542 |
| 5 | neutral | 256 | 0.00116748 | 0.720401 | 0.484573 | 0.829846 | 0.00116748 | 0.000977179 | 0.27736 | 186 | 0.0840554 | 0.249486 | 0.62184 | 0.372354 | 0.085271 | 0.0372988 |
| 5 | question_only | 256 | -0.130408 | 0.369248 | 0.426591 | 0.629493 | -0.130408 | 0.000477839 | 0.370761 | 41 | 0.0842927 | 0.26401 | 0.400175 | 0.136165 | 0.0568427 | 0.123534 |
| 5 | target | 256 | 1.11772 | 0.459317 | 0.666947 | 1.37111 | 0.459317 | -0.000658792 | 0.675058 | 172 | 0.108254 | 0.849766 | 0.5204 | -0.329366 | 0.0307311 | 0.11162 |
| 5 | target_word_shuffle_control | 256 | 0.390211 | 0.492652 | 0.494659 | 0.969827 | 0.243763 | 9.99959e-05 | 0.41853 | 219 | 0.0827078 | 0.480139 | 0.482775 | 0.0026361 | 0.0708003 | 0.00391632 |
| 6 | neutral | 256 | 0.00181309 | 0.373002 | 0.269648 | 0.773921 | 0.00181309 | 0.000498413 | 0.531138 | 156 | 0.0776473 | 0.18233 | 0.287654 | 0.105325 | 0.0490083 | 0.0439377 |
| 6 | question_only | 256 | 0.168233 | 0.23759 | 0.283302 | 0.751528 | 0.0946679 | 0.00025722 | 0.417429 | 136 | 0.0736562 | 0.275046 | 0.3216 | 0.0465538 | 0.0486539 | 0.0116362 |
| 6 | target | 256 | 0.991573 | 0.653693 | 0.442896 | 1.13893 | 0.274309 | -0.000170228 | 0.6816 | 158 | 0.110333 | 0.508281 | 0.54232 | 0.0340397 | 0.0698237 | 0.107668 |
| 6 | target_word_shuffle_control | 256 | 0.459585 | 0.297965 | 0.379928 | 0.912067 | 0.171662 | 1.23377e-05 | 0.544645 | 144 | 0.0816046 | 0.374975 | 0.304695 | -0.0702796 | 0.0499668 | 0.0215837 |
| 7 | neutral | 256 | 0.00292603 | 0.651843 | 0.59126 | 0.838063 | 0.00292603 | 0.000661947 | 0.389697 | 95 | 0.0962606 | 0.34339 | 0.576057 | 0.232667 | 0.0742543 | 0.0804432 |
| 7 | question_only | 256 | 0.330501 | 0.615139 | 0.653837 | 0.997673 | 0.330501 | 0.000727109 | 0.376053 | 140 | 0.0757128 | 0.623502 | 0.773877 | 0.150375 | 0.0572118 | 0.057856 |

## Bottom-Line Interpretation

1. Hidden-state geometry: strong if target projection is high and length-matched neutral is near zero.
2. Statistical controls: strong if paired/FDR/null-vector controls support target > controls.
3. Architecture: strong if MLP/attention module deltas align with Vector X.
4. Causal state control: strong if +X/-X bidirectionally moves generation-state projection.
5. Visible behavior: strong only if target-likeness beats alpha-matched random p95 without degeneration.

For the current breakthrough-grade run, the expected scientific reading is:

```text
Strong internal causal latent axis.
Strong architecture-level and generation-state evidence.
Visible behavioral readout remains weak/partial and quality-constrained.
Next experiment should be a narrow, alpha-matched, non-degenerate visible-readout retest.
```
