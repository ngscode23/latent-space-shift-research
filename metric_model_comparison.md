# Cross-Run Core Diagnostics Comparison

| model_id | blind_clean_pairs | blind_clean_fraction | blind_mean_abs_gap | blind_persistence_last_gap | blind_persistence_last_retention | hard_original | hard_best_control | hard_best_control_gap | hard_specificity_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-14B | 13.00 | 0.5417 | 20.79 | 10.43 | 0.4868 | 17.13 | pressure_style_no_model | 8.4440 | 2.0291 |
| Qwen/Qwen3.5-27B | 22.00 | 0.9167 | 1.2075 | 0.9933 | 0.9075 | 1.1961 | dry_summary_same_topic | 1.4250 | 0.8394 |

## Readout
- Compare `contrast_over_mean_norm` against `blind_mean_abs_gap`: this separates hidden displacement from semantic expression.
- Compare `blind_persistence_last_retention` against `rejection_persistence_last_retention`: this separates passive context persistence from explicit-rejection persistence.
- Compare `hard_specificity_ratio`: values above 1 mean the original profile beats tested controls; values below 1 mean controls explain much of the semantic readout.
- Compare `semantic_projection_fraction`: this is the current bridge between late hidden contrast and clean blind semantic readout geometry.
