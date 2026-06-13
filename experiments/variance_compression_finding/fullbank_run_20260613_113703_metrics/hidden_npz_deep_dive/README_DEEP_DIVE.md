# Hidden NPZ Deep Dive

Run directory: `C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703`
Late band: `30..47`

## Input Coverage

Prompts: `410`

Condition counts:

```text
condition
target                     100
target_word_shuffle        100
target_sentence_shuffle    100
control                    100
question_only               10
```

Metadata:

```json
{
  "base_model": "google/gemma-3-12b-pt",
  "instruct_model": "google/gemma-3-12b-it",
  "prompt_mode": "raw",
  "include_shuffles": true,
  "n_target_contexts": 10,
  "n_control_contexts": 10,
  "n_questions": 10,
  "n_prompts": 410,
  "batch_size": 2,
  "max_length": null,
  "late_lo": 30,
  "late_hi": 47,
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
     base   38.5       100        100                      4781.8                      0.000432663                         4781.8                         0.592908                      0.704467                         5347.72                          4587.87                              759.855                 0.000762765                  0.000294094                   0.588889                 0.9145                      3910.71                  10
 instruct   38.5       100        100                     9392.86                       0.00193989                        9392.86                         0.868409                      0.746583                         12583.6                          8193.53                              4390.08                  0.00172718                   0.00231766                   0.653611               0.937778                      8395.24                  10
```

## Late Base-Instruct Alignment Snapshot

```text
              condition  layer  n_points  linear_cka_base_instruct  same_prompt_delta_l2_mean  same_prompt_delta_l2_std  same_prompt_cosdist_mean  base_norm_mean  instruct_norm_mean  instruct_over_base_norm_mean
                    all   38.5       410                  0.892119                    23888.3                   7421.16                0.00547256          124457              104315                        0.8479
                control   38.5       100                   0.93804                    26806.6                   6543.28                0.00729328          125202              102691                      0.825903
          question_only   38.5        10                  0.763378                    33434.6                     11802                0.00696917          128598             98359.9                      0.775959
                 target   38.5       100                  0.920075                    22264.7                   6912.28                0.00444013          127895              109270                      0.864027
target_sentence_shuffle   38.5       100                  0.910647                    23187.3                   7044.35                0.00465863          126255              106512                       0.85439
    target_word_shuffle   38.5       100                  0.882288                    22339.9                   6716.65                0.00534854          118064               99380                      0.854475
```