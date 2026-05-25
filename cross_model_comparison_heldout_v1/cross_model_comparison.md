# Held-Out Cross-Model Comparison v1

Runs compared:

| run | model_id | model_type | max_tokens | text_family_preset | primary_control_mode | best_hidden_index | best_module_layer | hidden_cosine_distance | contrast_over_mean_norm | best_probe_accuracy | best_probe_perm_p95 | candidate_token_problem_count | order_truncated_rows | order_max_prompt_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen heldout | Qwen/Qwen3-14B | qwen3 | 4096 | heldout_domain | content_matched | 39 | 38 | 0.0616838932037353 | 0.352405163742904 | 1.0 | 0.7222222222222222 | 0 | 0 | 993 |
| Ministral heldout | mistralai/Ministral-3-14B-Instruct-2512-BF16 | mistral3 | 3070 | heldout_domain | content_matched | 40 | 39 | 0.0634058713912963 | 0.3564802023309592 | 1.0 | 0.7222222222222222 | 0 | 0 | 863 |
| OLMo2 heldout | allenai/OLMo-2-1124-13B-Instruct | olmo2 | 3070 | heldout_domain | content_matched | 39 | 38 | 0.1927352547645568 | 0.6616023471809576 | 1.0 | 0.6722222222222223 | 0 | 0 | 1332 |

## Control Baseline Check

The heldout cross-model comparison uses the content-matched control baseline.
This was rechecked for the Ministral run after the causal-mediation audit:

```text
attractor_results_agent_loop_ministral3_14b_heldout/run_metadata.json:
  primary_control_mode = content_matched

attractor_results_agent_loop_ministral3_14b_heldout/summary_report.txt:
  Control source: auto:content_matched
  Primary control mode: content_matched

latent_shift_evidence_package_v1/input_texts_heldout.json:
  primary_control_mode = content_matched
  control_texts_source = auto:content_matched

content_matched_control_seeds are identical between:
  attractor_results_agent_loop_qwen3_14b4_heldout/input_texts.json
  attractor_results_agent_loop_ministral3_14b_heldout/input_texts.json
  latent_shift_evidence_package_v1/input_texts_heldout.json
```

Therefore the Ministral natural gaps used in the mediation readout
(`agent_action=-6.218764`, `blind_semantic=-11.349387`) are not legacy
repetitive-baseline numbers.

## Causal Mediation Addendum

The natural shift replicates across Qwen, Ministral, and OLMo2, but the
single-direction causal-handle picture is heterogeneous:

| model | raw target-control mediation status | key readout |
| --- | --- | --- |
| Qwen heldout | cleanest positive action-policy handle | layer 32 target-control moves control fake-action margins toward target: alpha 1.0 observed=0.890 CI [0.477, 1.219] |
| Ministral heldout | not supported against controls | natural gaps are strong, but target_control does not beat random/shuffled/wrong-layer controls |
| OLMo2 heldout | positive but not cleanly specific | target_control positive in all four cells, but CIs overlap matched controls in most comparisons |

OLMo2 target-control intervention summary:

```text
agent_action control_plus: 1.071 [0.301, 2.107]
agent_action target_minus: 0.836 [0.041, 2.022]
blind_semantic control_plus: 0.251 [0.170, 0.346]
blind_semantic target_minus: 0.541 [0.296, 0.856]
```

Interpretation:

```text
Do not claim a universal single-vector mechanism. The supported cross-model
claim is functional latent/readout/action shift. The causal implementation is
model-dependent or distributed.
```

## Main Metrics

| metric | Qwen heldout | Ministral heldout | OLMo2 heldout | min_observed | max_observed | min_over_max_observed | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blind_clean_overall_mean_abs | 26.106 [23.425, 28.782] | 7.616 [6.531, 8.717] | 1.929 [1.587, 2.239] | 1.928687342123133 | 26.10641815927293 | 0.07387789969333913 | cross_model_supported |
| blind_persistence_turn_0_mean_abs | 21.674 [17.998, 25.463] | 3.895 [3.145, 4.595] | 1.076 [0.919, 1.224] | 1.0763039289465273 | 21.674226988686456 | 0.04965823830802999 | cross_model_supported |
| blind_persistence_turn_6_mean_abs | 6.218 [4.781, 7.788] | 2.087 [1.606, 2.563] | 0.450 [0.367, 0.525] | 0.4501434017494681 | 6.218147775861952 | 0.07239187905711517 | cross_model_supported |
| rejection_persistence_turn_0_mean_abs | 11.810 [10.748, 12.896] | 1.795 [1.541, 2.068] | 0.516 [0.445, 0.596] | 0.516007119330807 | 11.809673394097222 | 0.04369359779151222 | cross_model_supported |
| rejection_persistence_turn_6_mean_abs | 3.688 [2.960, 4.510] | 0.946 [0.800, 1.091] | 0.316 [0.261, 0.376] | 0.3159581175172962 | 3.6884440104166663 | 0.08566162767415951 | cross_model_supported |
| agent_loop_turn_0_rejection_False_mean_abs | 12.382 [10.847, 14.165] | 6.599 [6.160, 7.136] | 2.227 [1.838, 2.599] | 2.2265896929634943 | 12.382080078125 | 0.17982355782831147 | cross_model_supported |
| agent_loop_turn_4_rejection_False_mean_abs | 6.081 [5.226, 6.930] | 5.288 [4.511, 6.107] | 1.939 [1.574, 2.272] | 1.938693682352702 | 6.080837673611111 | 0.31882016695923526 | cross_model_supported |
| agent_loop_turn_4_rejection_True_mean_abs | 2.476 [2.082, 2.910] | 2.038 [1.661, 2.371] | 1.527 [1.223, 1.822] | 1.5268826484680176 | 2.47637939453125 | 0.6165786437409114 | cross_model_supported |
| hard_control_specificity_ratio | 1.871 [1.667, 2.218] | 2.351 [2.184, 2.593] | 1.206 [0.957, 1.665] | 1.2055227871654828 | 2.3509584264284857 | 0.5127792876358436 | cross_model_supported |
| order_TNC_all_mean_fraction | 0.525 [0.469, 0.593] | 0.202 [0.137, 0.291] | 0.359 [0.024, 0.591] | 0.2020394670609143 | 0.5247184353111987 | 0.38504358426265173 | cross_model_supported |
| order_CNT_all_mean_fraction | 0.947 [0.901, 1.001] | 0.828 [0.772, 0.895] | 0.814 [0.721, 1.201] | 0.8138521071616501 | 0.9468020533941344 | 0.8595799979986525 | cross_model_supported |
| order_TNN_all_mean_fraction | 0.554 [0.498, 0.609] | 0.647 [0.550, 0.726] | 0.497 [-0.479, 0.921] | 0.4968731157211139 | 0.6473177037593586 | 0.7675877128579621 | cross_model_supported |
| order_CNN_all_mean_fraction | 0.138 [0.085, 0.186] | 0.082 [0.015, 0.144] | 0.056 [-0.629, 0.492] | 0.0556692294974535 | 0.137888270902697 | 0.4037270837686938 | cross_model_supported |
| mix_target_prefix_0.5_mean_fraction | 0.347 [0.270, 0.415] | 0.324 [0.176, 0.476] | 0.307 [0.163, 0.433] | 0.3069558765326458 | 0.3474158217363656 | 0.8835402918568787 | cross_model_supported |
| mix_target_suffix_0.5_mean_fraction | 0.707 [0.638, 0.772] | 0.717 [0.600, 0.821] | 0.799 [0.688, 1.018] | 0.7073985990376629 | 0.7992206217648229 | 0.8851105436639004 | cross_model_supported |

## Status Readout

- Статус: **ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ** for the internal cross-model claim.
- What the data show: Qwen3-14B, Ministral 3 14B, and OLMo2 13B all show the same held-out context-induced geometry/readout/action-policy structure.
- Model dependence: Qwen has the largest blind semantic margins, Ministral is smaller but strong, and OLMo2 is weaker on semantic margins while still preserving the same order/mixing/persistence/action pattern.
- What not to claim: do not claim equal effect size, all-model universality, irreversible attractor dynamics, or real external-tool agent behavior.
- Minimal next test: freeze this v1 evidence package; if continuing, use a third model family or model-size ablation, not another diagnostic module.

## Practical Decision

The core research spine is now cross-model supported:

```text
target context -> hidden geometry shift -> logit/semantic readout shift ->
persistence/rejection/order/dose structure -> fake-agent action-policy drift
```

The next engineering move should be reporting and consolidation, not adding more checks to `llm_attractor_colab_copy_paste.py`.
