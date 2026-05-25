# Reviewer Robustness Audit v1

This audit reads existing outputs only. It does not run models.

## Validity Checks

| run | check | status | detail |
| --- | --- | --- | --- |
| Qwen heldout | setup | clean | model=Qwen/Qwen3-14B; preset=heldout_domain; control=content_matched; max_tokens=4096 |
| Qwen heldout | candidate_token_diagnostics | clean | problem_count=0 |
| Qwen heldout | truncation:blind_neutral_probe_raw.csv | clean | truncated_rows=0; max_prompt_tokens=476 |
| Qwen heldout | truncation:blind_neutral_persistence_raw.csv | clean | truncated_rows=0; max_prompt_tokens=749 |
| Qwen heldout | truncation:rejection_persistence_raw.csv | clean | truncated_rows=0; max_prompt_tokens=862 |
| Qwen heldout | truncation:agent_loop_raw.csv | clean | truncated_rows=0; max_prompt_tokens=799 |
| Qwen heldout | truncation:order_hysteresis_raw.csv | clean | truncated_rows=0; max_prompt_tokens=993 |
| Qwen heldout | truncation:mixing_threshold_raw.csv | clean | truncated_rows=0; max_prompt_tokens=477 |
| Ministral heldout | setup | clean | model=mistralai/Ministral-3-14B-Instruct-2512-BF16; preset=heldout_domain; control=content_matched; max_tokens=3070 |
| Ministral heldout | candidate_token_diagnostics | clean | problem_count=0 |
| Ministral heldout | truncation:blind_neutral_probe_raw.csv | clean | truncated_rows=0; max_prompt_tokens=423 |
| Ministral heldout | truncation:blind_neutral_persistence_raw.csv | clean | truncated_rows=0; max_prompt_tokens=628 |
| Ministral heldout | truncation:rejection_persistence_raw.csv | clean | truncated_rows=0; max_prompt_tokens=713 |
| Ministral heldout | truncation:agent_loop_raw.csv | clean | truncated_rows=0; max_prompt_tokens=685 |
| Ministral heldout | truncation:order_hysteresis_raw.csv | clean | truncated_rows=0; max_prompt_tokens=863 |
| Ministral heldout | truncation:mixing_threshold_raw.csv | clean | truncated_rows=0; max_prompt_tokens=423 |

## Mapping / Label-Position Checks

| run | check | status | detail |
| --- | --- | --- | --- |
| Qwen heldout | blind_normal_reversed_consistency | robust | clean_pairs=20/24; same_sign_rate=0.833; min_directional_consistency=0.344 |
| Qwen heldout | agent_normal_reversed_consistency | check | clean_rows=23/24; same_sign_rate=0.958; min_directional_consistency=0.163 |
| Ministral heldout | blind_normal_reversed_consistency | robust | clean_pairs=24/24; same_sign_rate=1.000; min_directional_consistency=1.000 |
| Ministral heldout | agent_normal_reversed_consistency | check | clean_rows=23/24; same_sign_rate=0.958; min_directional_consistency=0.083 |

## Mapping Exceptions

| run | block | label_pair | task | rejection_applied | filler_turns_elapsed | normal_delta | reversed_delta | same_sign | directional_consistency | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen heldout | blind_neutral_probe | AB | select_one_vs_inventory |  |  | -5.4974 | 0.5777 | False | 0.8098 | excluded from clean blind-probe set |
| Qwen heldout | blind_neutral_probe | MN | select_one_vs_inventory |  |  | -4.5864 | 1.4978 | False | 0.5076 | excluded from clean blind-probe set |
| Qwen heldout | blind_neutral_probe | PQ | select_one_vs_inventory |  |  | -6.0738 | 2.9638 | False | 0.3441 | excluded from clean blind-probe set |
| Qwen heldout | blind_neutral_probe | XY | select_one_vs_inventory |  |  | -6.8594 | 0.8592 | False | 0.7774 | excluded from clean blind-probe set |
| Qwen heldout | agent_loop |  | execute_vs_substitute | True | 0 | -1.0182 | 1.4149 | False | 0.1630 | small early post-rejection mapping inconsistency; core turn-4 rows remain clean |
| Ministral heldout | agent_loop |  | concrete_result_vs_preconditions | True | 0 | -0.2951 | 0.3485 | False | 0.0829 | small early post-rejection mapping inconsistency; core turn-4 rows remain clean |

## Bootstrap Key-Metric Bounds

| run | claim_piece | status | observed | ci_low | ci_high | threshold_type | n_units | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen heldout | clean blind semantic readout | passes_ci_threshold | 26.1064 | 23.4249 | 28.7820 | ci_low_gt_0 | 9 | 360 |
| Qwen heldout | neutral persistence at final turn | passes_ci_threshold | 6.2181 | 4.7813 | 7.7878 | ci_low_gt_0 | 9 | 360 |
| Qwen heldout | post-rejection residual at final turn | passes_ci_threshold | 3.6884 | 2.9599 | 4.5098 | ci_low_gt_0 | 9 | 360 |
| Qwen heldout | fake-agent action drift after neutral turns | passes_ci_threshold | 6.0808 | 5.2255 | 6.9296 | ci_low_gt_0 | 9 | 72 |
| Qwen heldout | fake-agent action drift after rejection | passes_ci_threshold | 2.4764 | 2.0818 | 2.9105 | ci_low_gt_0 | 9 | 72 |
| Qwen heldout | hard-control specificity | passes_ci_threshold | 1.8714 | 1.6672 | 2.2184 | ci_low_gt_1 | 5 | 160 |
| Qwen heldout | control-then-target order moves toward target | passes_ci_threshold | 0.9468 | 0.9013 | 1.0010 | ci_low_gt_0_5 | 9 | 648 |
| Qwen heldout | 50 percent suffix mix already target-like | passes_ci_threshold | 0.7074 | 0.6383 | 0.7720 | ci_low_gt_0_5 | 9 | 1296 |
| Ministral heldout | clean blind semantic readout | passes_ci_threshold | 7.6162 | 6.5313 | 8.7169 | ci_low_gt_0 | 9 | 432 |
| Ministral heldout | neutral persistence at final turn | passes_ci_threshold | 2.0872 | 1.6060 | 2.5631 | ci_low_gt_0 | 9 | 432 |
| Ministral heldout | post-rejection residual at final turn | passes_ci_threshold | 0.9457 | 0.7997 | 1.0914 | ci_low_gt_0 | 9 | 432 |
| Ministral heldout | fake-agent action drift after neutral turns | passes_ci_threshold | 5.2877 | 4.5110 | 6.1073 | ci_low_gt_0 | 9 | 72 |
| Ministral heldout | fake-agent action drift after rejection | passes_ci_threshold | 2.0377 | 1.6607 | 2.3714 | ci_low_gt_0 | 9 | 72 |
| Ministral heldout | hard-control specificity | passes_ci_threshold | 2.3510 | 2.1840 | 2.5928 | ci_low_gt_1 | 5 | 160 |
| Ministral heldout | control-then-target order moves toward target | passes_ci_threshold | 0.8283 | 0.7723 | 0.8946 | ci_low_gt_0_5 | 9 | 648 |
| Ministral heldout | 50 percent suffix mix already target-like | passes_ci_threshold | 0.7172 | 0.6002 | 0.8212 | ci_low_gt_0_5 | 9 | 1296 |

## Leave-One-Inducing-Text-Out Checks

| run | metric | status | observed_mean_abs | observed_mean_signed | loo_min_mean_abs | loo_max_mean_abs | max_drop_fraction_when_one_text_removed | most_influential_removed_index | sign_consistency_vs_row_reference | n_inducing_texts | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen heldout | blind_clean_probe_gap | robust | 26.1064 | -16.3898 | 25.3082 | 26.9309 | 0.0306 | 8 | 0.9833 | 9 | 360 |
| Qwen heldout | blind_persistence_turn_6_gap | robust | 6.2181 | -3.5279 | 5.6875 | 6.5250 | 0.0853 | 6 | 0.9583 | 9 | 360 |
| Qwen heldout | rejection_persistence_turn_6_gap | robust | 3.6884 | -2.7644 | 3.3790 | 3.8962 | 0.0839 | 8 | 0.8361 | 9 | 360 |
| Qwen heldout | agent_loop_rejection_False_turn_4_direct_margin_gap | robust | 6.0808 | -5.5711 | 5.8129 | 6.3796 | 0.0441 | 5 | 0.9306 | 9 | 72 |
| Qwen heldout | agent_loop_rejection_True_turn_4_direct_margin_gap | robust | 2.4764 | -1.8107 | 2.3329 | 2.5772 | 0.0579 | 0 | 0.8611 | 9 | 72 |
| Ministral heldout | blind_clean_probe_gap | robust | 7.6162 | -6.0849 | 7.2855 | 7.9303 | 0.0434 | 8 | 0.9444 | 9 | 432 |
| Ministral heldout | blind_persistence_turn_6_gap | robust | 2.0872 | -1.2957 | 1.9373 | 2.2237 | 0.0718 | 8 | 0.9236 | 9 | 432 |
| Ministral heldout | rejection_persistence_turn_6_gap | robust | 0.9457 | -0.3134 | 0.9029 | 0.9852 | 0.0453 | 6 | 0.8495 | 9 | 432 |
| Ministral heldout | agent_loop_rejection_False_turn_4_direct_margin_gap | robust | 5.2877 | -5.2877 | 5.0674 | 5.4953 | 0.0417 | 3 | 1.0000 | 9 | 72 |
| Ministral heldout | agent_loop_rejection_True_turn_4_direct_margin_gap | robust | 2.0377 | -1.9185 | 1.9367 | 2.1755 | 0.0496 | 5 | 0.8889 | 9 | 72 |

## Paired Sign-Flip Tests

| run | metric | status | observed_mean_signed | observed_abs_mean_signed | two_sided_p | n_units | n_assignments | dominant_unit_sign | same_direction_unit_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen heldout | blind_clean_probe_gap | passes_sign_flip_0_05 | -16.3898 | 16.3898 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | blind_persistence_turn_0_gap | passes_sign_flip_0_05 | -13.5222 | 13.5222 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | rejection_persistence_turn_0_gap | passes_sign_flip_0_05 | -7.7411 | 7.7411 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | blind_persistence_turn_6_gap | passes_sign_flip_0_05 | -3.5279 | 3.5279 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | rejection_persistence_turn_6_gap | passes_sign_flip_0_05 | -2.7644 | 2.7644 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | agent_loop_rejection_False_turn_0_direct_margin_gap | passes_sign_flip_0_05 | -11.9766 | 11.9766 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | agent_loop_rejection_False_turn_4_direct_margin_gap | passes_sign_flip_0_05 | -5.5711 | 5.5711 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Qwen heldout | agent_loop_rejection_True_turn_4_direct_margin_gap | passes_sign_flip_0_05 | -1.8107 | 1.8107 | 0.0078 | 9 | 512 | -1 | 0.8889 |
| Ministral heldout | blind_clean_probe_gap | passes_sign_flip_0_05 | -6.0849 | 6.0849 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Ministral heldout | blind_persistence_turn_0_gap | passes_sign_flip_0_05 | -2.3479 | 2.3479 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Ministral heldout | rejection_persistence_turn_0_gap | passes_sign_flip_0_05 | -0.6719 | 0.6719 | 0.0117 | 9 | 512 | -1 | 0.8889 |
| Ministral heldout | blind_persistence_turn_6_gap | passes_sign_flip_0_05 | -1.2957 | 1.2957 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Ministral heldout | rejection_persistence_turn_6_gap | passes_sign_flip_0_05 | -0.3134 | 0.3134 | 0.0234 | 9 | 512 | -1 | 0.7778 |
| Ministral heldout | agent_loop_rejection_False_turn_0_direct_margin_gap | passes_sign_flip_0_05 | -6.5984 | 6.5984 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Ministral heldout | agent_loop_rejection_False_turn_4_direct_margin_gap | passes_sign_flip_0_05 | -5.2877 | 5.2877 | 0.0039 | 9 | 512 | -1 | 1.0000 |
| Ministral heldout | agent_loop_rejection_True_turn_4_direct_margin_gap | passes_sign_flip_0_05 | -1.9185 | 1.9185 | 0.0039 | 9 | 512 | -1 | 1.0000 |

## Cross-Model Agreement

| comparison | left_run | right_run | rows | sign_agreement_rate | pearson | spearman |
| --- | --- | --- | --- | --- | --- | --- |
| blind_gap_summary | Qwen heldout | Ministral heldout | 48 | 0.9167 | 0.9675 | 0.9006 |
| agent_loop_delta | Qwen heldout | Ministral heldout | 48 | 0.9167 | 0.4774 | 0.6154 |
| order_hysteresis_condition_summary | Qwen heldout | Ministral heldout | 6 |  | 0.9405 | 1.0000 |
| mixing_threshold_condition_summary | Qwen heldout | Ministral heldout | 12 |  | 0.9907 | 0.9930 |

## Reviewer-Objection Readout

- Single-text driver: addressed by leave-one-inducing-text-out checks; core effects remain nonzero after removing any one text.
- Paired target/control null: addressed by exact sign-flip tests over inducing-text pairs.
- A/B or label-position bias: addressed by normal/reversed mappings and candidate-token diagnostics.
- Mapping exceptions: explicitly listed above; they are limited and do not touch the main final-turn action-policy rows.
- Bootstrap uncertainty: key claim pieces have positive lower confidence bounds, and hard-control specificity has ratio lower bound above 1.
- Truncation artifact: addressed by raw-file truncation counts; all core raw files show zero truncated rows in these runs.
- Qwen-only artifact: addressed by Qwen3-14B and Ministral 3 14B agreement.
- Only abstract semantic probes: addressed by controlled fake-agent action-choice drift.

## Status

Статус: **СИЛЬНО ПОДДЕРЖАНО for reviewer-facing robustness**, while still not a claim about real external-tool agents or all model families.
