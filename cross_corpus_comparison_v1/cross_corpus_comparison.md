# Cross-Corpus Comparison v1

This report compares induction families, not only model replications.

```text
selfref / mirror corpus  !=  heldout procedural/risk corpus
```

Use it to answer whether the project is about special self-reference texts
or about broader context-induced latent regime formation.

## Runs

| run | model_family | corpus | model_id | declared_preset | inferred_corpus | max_tokens | max_input_token_count | best_hidden_index | best_hidden_contrast_over_mean_norm | best_hidden_cosine_distance | best_probe_accuracy | best_probe_permutation_p95 | clean_fraction | clean_label_task_pairs | candidate_problem_count | bootstrap_available | run_dir | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen selfref | Qwen3-14B | selfref | Qwen/Qwen3-14B |  | selfref | 4096 | 2779.0 | 40.0 | 0.5796376354590936 | 0.1210469007492065 | 1.0 | 0.6694444444444444 | 0.625 | 15.0 | 0 | True | attractor_results_agent_loop_qwen3_14b3 | representative latest Qwen selfref run |
| Qwen heldout | Qwen3-14B | heldout | Qwen/Qwen3-14B | heldout_domain | heldout | 4096 | 383.0 | 40.0 | 0.5117504748588363 | 0.1211521625518798 | 1.0 | 0.7222222222222222 | 0.8333333333333334 | 20.0 | 0 | True | attractor_results_agent_loop_qwen3_14b4_heldout | heldout procedural/risk corpus |
| Ministral selfref | Ministral-3-14B | selfref | mistralai/Ministral-3-14B-Instruct-2512-BF16 | original | selfref | 4096 | 2518.0 | 14.0 | 0.5490187795064326 | 0.1504061818122863 | 1.0 | 0.6166666666666668 | 0.75 | 18.0 | 0 | True | attractor_results_agent_loop_ministral3_14b_selfref | selfref/mirror corpus |
| Ministral heldout | Ministral-3-14B | heldout | mistralai/Ministral-3-14B-Instruct-2512-BF16 | heldout_domain | heldout | 3070 | 340.0 | 14.0 | 0.5938516074492297 | 0.1749927401542663 | 1.0 | 0.7222222222222222 | 1.0 | 24.0 | 0 | True | attractor_results_agent_loop_ministral3_14b_heldout | heldout procedural/risk corpus |
| OLMo2 heldout | OLMo2-13B | heldout | allenai/OLMo-2-1124-13B-Instruct | heldout_domain | heldout | 3070 | 529.0 | 35.0 | 0.7773349255565731 | 0.2791077494621277 | 1.0 | 0.6722222222222223 | 0.9583333333333334 | 23.0 | 0 | True | attractor_results_olmo2_13b_heldout | third-family heldout check; no selfref mate yet |

## Main Cross-Corpus Metrics

| metric_name | Qwen selfref | Qwen heldout | Ministral selfref | Ministral heldout | OLMo2 heldout |
| --- | --- | --- | --- | --- | --- |
| hidden_best_contrast_over_mean_norm | 0.580 | 0.512 | 0.549 | 0.594 | 0.777 |
| blind_clean_fraction | 0.625 | 0.833 | 0.750 | 1.000 | 0.958 |
| blind_clean_overall_mean_abs | 16.541 [15.320, 17.885] | 26.106 [23.425, 28.782] | 2.467 [2.015, 3.023] | 7.616 [6.531, 8.717] | 1.929 [1.587, 2.239] |
| blind_persistence_turn_6_mean_abs | 8.857 [7.194, 11.225] | 6.218 [4.781, 7.788] | 1.597 [1.261, 1.937] | 2.087 [1.606, 2.563] | 0.450 [0.367, 0.525] |
| rejection_persistence_turn_6_mean_abs | 4.358 [2.655, 6.830] | 3.688 [2.960, 4.510] | 1.356 [1.016, 1.729] | 0.946 [0.800, 1.091] | 0.316 [0.261, 0.376] |
| hard_control_specificity_ratio | 2.269 [1.756, 3.067] | 1.871 [1.667, 2.218] | 1.001 [0.689, 1.381] | 2.351 [2.184, 2.593] | 1.206 [0.957, 1.665] |
| agent_loop_turn_4_rejection_False_mean_abs | 6.145 [2.386, 8.250] | 6.081 [5.226, 6.930] | 1.399 [0.999, 1.810] | 5.288 [4.511, 6.107] | 1.939 [1.574, 2.272] |
| agent_loop_turn_4_rejection_True_mean_abs | 2.454 [1.559, 3.372] | 2.476 [2.082, 2.910] | 0.863 [0.509, 1.259] | 2.038 [1.661, 2.371] | 1.527 [1.223, 1.822] |
| order_CNT_all_mean_fraction | 1.093 [-0.249, 1.381] | 0.947 [0.901, 1.001] | 0.998 [0.839, 1.158] | 0.828 [0.772, 0.895] | 0.814 [0.721, 1.201] |
| order_TNC_all_mean_fraction | 0.494 [-1.619, 3.543] | 0.525 [0.469, 0.593] | 0.427 [0.248, 0.571] | 0.202 [0.137, 0.291] | 0.359 [0.024, 0.591] |
| order_TNN_all_mean_fraction | 0.321 [-8.557, 15.656] | 0.554 [0.498, 0.609] | 0.671 [0.365, 0.841] | 0.647 [0.550, 0.726] | 0.497 [-0.479, 0.921] |
| mix_target_prefix_0.5_mean_fraction | 0.768 [0.598, 1.157] | 0.347 [0.270, 0.415] | 0.720 [0.069, 1.376] | 0.324 [0.176, 0.476] | 0.307 [0.163, 0.433] |
| mix_target_suffix_0.5_mean_fraction | 0.798 [0.732, 0.851] | 0.707 [0.638, 0.772] | 0.685 [0.138, 1.346] | 0.717 [0.600, 0.821] | 0.799 [0.688, 1.018] |

## Selfref / Heldout Ratios

| model_family | metric_name | selfref_formatted | heldout_formatted | selfref_minus_heldout | selfref_over_heldout |
| --- | --- | --- | --- | --- | --- |
| Ministral-3-14B | agent_loop_turn_4_rejection_False_mean_abs | 1.399 [0.999, 1.810] | 5.288 [4.511, 6.107] | -3.889 | 0.265 |
| Ministral-3-14B | agent_loop_turn_4_rejection_True_mean_abs | 0.863 [0.509, 1.259] | 2.038 [1.661, 2.371] | -1.175 | 0.423 |
| Ministral-3-14B | blind_clean_overall_mean_abs | 2.467 [2.015, 3.023] | 7.616 [6.531, 8.717] | -5.149 | 0.324 |
| Ministral-3-14B | blind_persistence_turn_6_mean_abs | 1.597 [1.261, 1.937] | 2.087 [1.606, 2.563] | -0.490 | 0.765 |
| Ministral-3-14B | hard_control_specificity_ratio | 1.001 [0.689, 1.381] | 2.351 [2.184, 2.593] | -1.350 | 0.426 |
| Ministral-3-14B | rejection_persistence_turn_6_mean_abs | 1.356 [1.016, 1.729] | 0.946 [0.800, 1.091] | 0.410 | 1.434 |
| Qwen3-14B | agent_loop_turn_4_rejection_False_mean_abs | 6.145 [2.386, 8.250] | 6.081 [5.226, 6.930] | 0.064 | 1.011 |
| Qwen3-14B | agent_loop_turn_4_rejection_True_mean_abs | 2.454 [1.559, 3.372] | 2.476 [2.082, 2.910] | -0.023 | 0.991 |
| Qwen3-14B | blind_clean_overall_mean_abs | 16.541 [15.320, 17.885] | 26.106 [23.425, 28.782] | -9.565 | 0.634 |
| Qwen3-14B | blind_persistence_turn_6_mean_abs | 8.857 [7.194, 11.225] | 6.218 [4.781, 7.788] | 2.639 | 1.424 |
| Qwen3-14B | hard_control_specificity_ratio | 2.269 [1.756, 3.067] | 1.871 [1.667, 2.218] | 0.397 | 1.212 |
| Qwen3-14B | rejection_persistence_turn_6_mean_abs | 4.358 [2.655, 6.830] | 3.688 [2.960, 4.510] | 0.670 | 1.182 |

## Quality Caveats

- Qwen selfref is included as the latest representative of the older Qwen selfref folders. Its core blind/persistence/action magnitudes are useful, but some order-fraction bootstrap intervals are very wide. Treat those rows as low-confidence validation fractions, not as headline evidence.
- Ministral selfref is the cleaner selfref cross-model check for order/mixing, but its hard-control specificity fails: pressure-style controls reproduce much of the original effect.
- Heldout rows are the cleaner reviewer-facing comparison line because they avoid direct model self-reference and have stronger hard-control behavior in Qwen and Ministral.

## Interpretation

- Статус: **СИЛЬНО ПОДДЕРЖАНО** for the broad state-induction claim.
- What the comparison shows: both selfref and heldout corpora induce hidden/readout/action shifts, so the phenomenon is not only a self-reference trick.
- Selfref is not cleanly unique: in the Ministral selfref run, hard-control specificity is weak because `pressure_style_no_model` matches or exceeds original.
- Heldout is cleaner for reviewer-facing claims because it avoids direct model self-reference and still reproduces the structure across Qwen, Ministral, and OLMo2.
- The right framing is: texts are induction stimuli; the measured object is a distributed latent discourse regime.

## What Not To Claim

- Do not claim that self-reference alone is the mechanism.
- Do not claim that original selfref texts always beat all hard controls.
- Do not claim corpus-independent equal effect size.
- Do not call this strict dynamical attractor evidence from this table alone.

## Practical Decision

Use heldout as the main reviewer-facing replication line. Use selfref as the
strong mirror/self-model pressure line and as evidence that pressure cadence
and rhetorical topology are active ingredients.
