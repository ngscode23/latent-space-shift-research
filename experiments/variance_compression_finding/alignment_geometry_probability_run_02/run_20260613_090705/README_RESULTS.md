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

## Reading Rules

- `rel_disp_l2_mean_instruct_minus_base < 0`: instruct has lower hidden dispersion than base.
- `effective_rank_pr_instruct_minus_base < 0`: instruct uses fewer effective hidden dimensions.
- `question_minus_context_rel_disp` large: context collapses hidden-state spread relative to question-only.
- `next_token_entropy_mean_instruct_minus_base < 0`: instruct has more concentrated next-token distribution.
- `top1_prob_mean_instruct_minus_base > 0`: instruct is more top-token concentrated.

## Context Snapping Snapshot

| model_tag           |   late_lo |   late_hi |   question_only_rel_disp |   context_mean_rel_disp |   question_minus_context_rel_disp |   target_rel_disp |   control_rel_disp |   control_minus_target_rel_disp |   question_only_effective_rank |   context_mean_effective_rank |   target_effective_rank |   control_effective_rank |   question_only_logit_entropy |   context_mean_logit_entropy |   question_minus_context_logit_entropy |   question_only_top1_prob |   context_mean_top1_prob |
|:--------------------|----------:|----------:|-------------------------:|------------------------:|----------------------------------:|------------------:|-------------------:|--------------------------------:|-------------------------------:|------------------------------:|------------------------:|-------------------------:|------------------------------:|-----------------------------:|---------------------------------------:|--------------------------:|-------------------------:|
| base                |        30 |        48 |                0.224951  |               0.207966  |                        0.016985   |         0.208217  |          0.210543  |                      0.00232626 |                        1.91696 |                      2.24092  |                2.25641  |                 2.28382  |                      2.92827  |                      2.76762 |                               0.160652 |                  0.418598 |                 0.432188 |
| instruct            |        30 |        48 |                0.24324   |               0.228665  |                        0.0145743  |         0.228822  |          0.240257  |                      0.0114351  |                        3.13652 |                      2.91619  |                2.89631  |                 2.59473  |                      0.911895 |                      1.28325 |                              -0.371358 |                  0.816352 |                 0.690032 |
| instruct_minus_base |        30 |        48 |                0.0182881 |               0.0206989 |                       -0.00241079 |         0.0206045 |          0.0297134 |                      0.00910885 |                        1.21957 |                      0.675277 |                0.639907 |                 0.310914 |                     -2.01638  |                     -1.48437 |                              -0.53201  |                  0.397754 |                 0.257844 |

## Late Band Snapshot

| condition               | condition_family   |   rel_disp_l2_mean_instruct_minus_base |   effective_rank_pr_instruct_minus_base |   next_token_entropy_mean_instruct_minus_base |   top1_prob_mean_instruct_minus_base |
|:------------------------|:-------------------|---------------------------------------:|----------------------------------------:|----------------------------------------------:|-------------------------------------:|
| control                 | control            |                              0.0297134 |                                0.310914 |                                      -1.5097  |                             0.267723 |
| question_only           | question_only      |                              0.0182881 |                                1.21957  |                                      -2.01638 |                             0.397754 |
| target                  | target             |                              0.0206045 |                                0.639907 |                                      -1.21031 |                             0.224031 |
| target_sentence_shuffle | target_shuffle     |                              0.0170185 |                                0.747472 |                                      -1.36924 |                             0.258391 |
| target_word_shuffle     | target_shuffle     |                              0.0154592 |                                1.00282  |                                      -1.84822 |                             0.28123  |