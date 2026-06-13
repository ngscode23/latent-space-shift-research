# Hidden NPZ Deep Dive

Run directory: `experiments\variance_compression_finding\alignment_geometry_probability_run_02\run_20260613_090705`
Late band: `30..47`

## Input Coverage

Prompts: `50`

Condition counts:

```text
condition
target                     10
target_word_shuffle        10
target_sentence_shuffle    10
control                    10
question_only              10
```

Metadata:

```json
{
  "base_model": "google/gemma-3-12b-pt",
  "instruct_model": "google/gemma-3-12b-it",
  "prompt_mode": "raw",
  "include_shuffles": true,
  "n_target_contexts": 1,
  "n_control_contexts": 1,
  "n_questions": 10,
  "n_prompts": 50,
  "batch_size": 2,
  "max_length": null,
  "dtype": "torch.bfloat16",
  "device": "cuda",
  "torch_version": "2.12.0.dev20260408+cu128"
}
```

## Additional Metrics

- `linear_cka_base_instruct`: representation similarity between base and instruct hidden states over matching prompts.
- `target_control_projection_gap_z`: target-control diff-in-means gap normalized by pooled projection std.
- `target_control_axis_auc_like`: pairwise AUC-like score on the target-control diff-in-means axis.
- `loo_question_balanced_acc`: target/control classification when the axis is trained excluding each question_id.
- `centroid_to_question_only_l2`: condition centroid displacement from question-only centroid.
- `same_prompt_delta_l2_mean`: same-prompt hidden vector displacement between instruct and base.

## Core CSVs

- `deep_condition_layer_metrics.csv`
- `deep_target_control_contrast_by_layer.csv`
- `deep_base_instruct_alignment_by_layer.csv`
- `deep_late_band_condition_summary.csv`
- `deep_late_band_contrast_summary.csv`
- `deep_late_band_base_instruct_alignment_summary.csv`

## Plots

See `plots/*.png`.

## Late Target-Control Snapshot

```text
model_tag  layer  n_target  n_control  target_control_centroid_l2  target_control_centroid_cosdist  target_control_projection_gap  target_control_projection_gap_z  target_control_axis_auc_like  target_to_question_centroid_l2  control_to_question_centroid_l2  target_minus_control_to_question_l2  target_to_question_cosdist  control_to_question_cosdist  loo_question_balanced_acc  loo_question_auc_like  loo_question_projection_gap  loo_question_folds
     base   38.5        10         10                     4627.46                      0.000506763                        4627.46                         0.600475                      0.349444                         5985.58                          6308.89                             -323.306                 0.000803717                  0.000551734                   0.611111               0.494444                      3719.71                  10
 instruct   38.5        10         10                     5386.57                      0.000800186                        5386.57                         0.629561                      0.343333                         12661.2                          9807.78                              2853.45                  0.00210722                    0.0018504                       0.55               0.441667                      3644.36                  10
```

## Late Base-Instruct Alignment Snapshot

```text
              condition  layer  n_points  linear_cka_base_instruct  same_prompt_delta_l2_mean  same_prompt_delta_l2_std  same_prompt_cosdist_mean  base_norm_mean  instruct_norm_mean  instruct_over_base_norm_mean
                    all   38.5        50                  0.810725                    24953.2                   8992.02                0.00594619          124958              103924                      0.840968
                control   38.5        10                  0.946495                    22554.2                   4885.79                0.00612246          124000              105883                      0.857266
          question_only   38.5        10                  0.763378                    33434.6                     11802                0.00696917          128598             98359.9                      0.775959
                 target   38.5        10                  0.921485                    21365.5                   6260.87                0.00551741          125901              109003                      0.872648
target_sentence_shuffle   38.5        10                  0.907425                    22289.6                   6934.25                0.00549938          124923              106782                      0.865809
    target_word_shuffle   38.5        10                  0.916256                    25122.4                   6976.76                0.00562252          121371             99589.9                      0.833156
```