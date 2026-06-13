# Base vs Instruct Geometry/Probability Audit

Base model: `google/gemma-3-12b-pt`
Instruct model: `google/gemma-3-12b-it`
Prompt mode: `raw`

## Main Tables

- `hidden_dispersion_by_layer.csv`: hidden compression metrics per model/condition/layer.
- `logit_metrics_by_prompt.csv`: next-token probability/logit metrics per prompt.
- `logit_metrics_summary.csv`: probability/logit metrics averaged by condition.
- `base_vs_instruct_layer_compare.csv`: per-layer instruct-base geometry deltas.
- `late_band_summary.csv`: late-band geometry plus logit comparison.
- `context_snapping_summary.csv`: question-only vs context compression by model.
- `readout_stiffness_summary.csv`: probability concentration normalized by late hidden-state scale.

## Reading Rules

- `centroid_norm_instruct_minus_base < 0`: instruct has lower absolute late hidden-state scale.
- `rel_disp_l2_mean_instruct_minus_base < 0`: instruct has lower relative hidden dispersion than base.
- `effective_rank_pr_instruct_minus_base > 0`: instruct uses more effective hidden dimensions.
- `question_minus_context_rel_disp` large: context collapses hidden-state spread relative to question-only.
- `next_token_entropy_mean_instruct_minus_base < 0`: instruct has more concentrated next-token distribution.
- `top1_prob_mean_instruct_minus_base > 0`: instruct is more top-token concentrated.
- `top1_per_rel_disp_instruct_over_base > 1`: instruct has more top-token concentration per unit relative hidden dispersion.

## Context Snapping Snapshot

| model_tag | late_lo | late_hi | question_only_rel_disp | context_mean_rel_disp | question_minus_context_rel_disp | target_rel_disp | control_rel_disp | control_minus_target_rel_disp | question_only_effective_rank | context_mean_effective_rank | target_effective_rank | control_effective_rank | question_only_logit_entropy | context_mean_logit_entropy | question_minus_context_logit_entropy | question_only_top1_prob | context_mean_top1_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 30 | 47 | 0.197161 | 0.180032 | 0.0171288 | 0.180301 | 0.182264 | 0.00196223 | 1.79107 | 2.12722 | 2.14241 | 2.16317 | 2.92827 | 2.76762 | 0.160652 | 0.418598 | 0.432188 |
| instruct | 30 | 47 | 0.201851 | 0.189782 | 0.0120696 | 0.190733 | 0.201785 | 0.0110526 | 3.07996 | 2.85643 | 2.83042 | 2.52218 | 0.911895 | 1.28325 | -0.371358 | 0.816352 | 0.690032 |
| instruct_minus_base | 30 | 47 | 0.00469007 | 0.00974933 | -0.00505926 | 0.0104312 | 0.0195216 | 0.00909039 | 1.28889 | 0.729213 | 0.688006 | 0.359007 | -2.01638 | -1.48437 | -0.53201 | 0.397754 | 0.257844 |

## Late Band Snapshot

| condition | condition_family | centroid_norm_instruct_minus_base | rel_disp_l2_mean_instruct_minus_base | effective_rank_pr_instruct_minus_base | next_token_entropy_mean_instruct_minus_base | top1_prob_mean_instruct_minus_base |
| --- | --- | --- | --- | --- | --- | --- |
| control | control | -18402.6 | 0.0195216 | 0.359007 | -1.5097 | 0.267723 |
| question_only | question_only | -30499.6 | 0.00469007 | 1.28889 | -2.01638 | 0.397754 |
| target | target | -17152.1 | 0.0104312 | 0.688006 | -1.21031 | 0.224031 |
| target_sentence_shuffle | target_shuffle | -18393.1 | 0.00576268 | 0.805689 | -1.36924 | 0.258391 |
| target_word_shuffle | target_shuffle | -22051.2 | 0.00328188 | 1.06415 | -1.84822 | 0.28123 |

## Readout Stiffness Snapshot

| condition | condition_family | entropy_reduction_base_minus_instruct | top1_prob_gain_instruct_minus_base | top1_per_rel_disp_instruct_over_base | inverse_entropy_per_rel_disp_instruct_over_base |
| --- | --- | --- | --- | --- | --- |
| control | control | 1.5097 | 0.267723 | 1.46672 | 1.94765 |
| question_only | question_only | 2.01638 | 0.397754 | 1.90489 | 3.13658 |
| target | target | 1.21031 | 0.224031 | 1.43798 | 1.71713 |
| target_sentence_shuffle | target_shuffle | 1.36924 | 0.258391 | 1.55365 | 1.95356 |
| target_word_shuffle | target_shuffle | 1.84822 | 0.28123 | 1.60737 | 2.80158 |