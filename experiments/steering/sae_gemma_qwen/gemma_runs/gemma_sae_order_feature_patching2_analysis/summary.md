# Gemma SAE order feature patching2 analysis

## High-level status
- All 9 SAE tables loaded from the raw `_natscale` zip.
- Selected candidates: 50
- Context rows: 2500
- Reconstruction cosine mean across layers: 0.996023
- Reconstruction cosine min across layers: 0.979815

## Important caveat
The submitted `rough_sae_zero_ablation_logit_results.csv` is reconstruction-contaminated: the script version used `decoded_patched` as the whole residual activation. That measures SAE reconstruction replacement plus feature removal. It is not a clean single-feature ablation. The script has been fixed to use `activation + (decode(patched) - decode(original))`; rerun rough patching before using KL for scale calibration.

## Selected candidates by status
| interpretation_status | n_features |
| --- | --- |
| order_component_specific_top_feature | 27 |
| order_specific_generation_persistent_feature | 11 |
| order_enriched_overlap_feature | 8 |
| order_specific_prompt_feature | 4 |

## Selected candidates by layer
| layer | real_layer | n_features | mean_score | max_score | mean_abs_order | n_with_content |
| --- | --- | --- | --- | --- | --- | --- |
| 19 | 18 | 7 | 282.436 | 459.115 | 162.164 | 7 |
| 25 | 24 | 3 | 304.899 | 608.737 | 195.729 | 1 |
| 31 | 30 | 8 | 475.672 | 1155.073 | 337.458 | 4 |
| 37 | 36 | 12 | 210.061 | 603.446 | 186.763 | 1 |
| 42 | 41 | 20 | 351.589 | 953.868 | 297.247 | 1 |

## Strongest context activations among selected features
| selected_rank | layer | real_layer | feature_index | max_context_activation | n_positive_contexts | x_order_orth_component_delta | x_content_component_delta | order_specific_score | interpretation_status | top_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 42 | 41 | 207 | 60284.504 | 50 | 194.855 |  | 382.023 | order_specific_generation_persistent_feature | зыва |  потому |  отвечает | до |  цен |  вре | сен |  присут |
| 3 | 42 | 41 | 29 | 32982.055 | 50 | 363.967 |  | 953.868 | order_specific_generation_persistent_feature |  нужно |  отвечает | дель |  модель |  ответы | ние |  должна | дель |
| 8 | 42 | 41 | 208 | 31069.729 | 50 | 495.494 |  | 567.167 | order_specific_generation_persistent_feature |  отвечает |  под |  под |  под | ет |  под | ёт |  на |
| 7 | 37 | 36 | 323 | 18499.918 | 50 | 244.137 |  | 603.446 | order_specific_generation_persistent_feature | шается |  под |  риск |  отвечает | ции | тельство | ет | сти |
| 9 | 42 | 41 | 180 | 8610.612 | 3 | 553.049 | 389.999 | 533.646 | order_enriched_overlap_feature | зыва |  потому |  не | Мо | дель |  встреча | ет |  здесь |
| 6 | 25 | 24 | 76 | 7890.342 | 4 | 281.224 | -144.213 | 608.737 | order_enriched_overlap_feature | зыва |  потому | мо |  потому | Мо | дель |  встреча | ет |
| 43 | 42 | 41 | 1188 | 7690.265 | 50 | 150.831 |  | 150.831 | order_component_specific_top_feature |  бы |  бы |  бы |  должна |  бы |  бы |  можно |  должна |
| 15 | 42 | 41 | 11773 | 5427.854 | 45 | -358.085 |  | 380.197 | order_specific_generation_persistent_feature |  можно | шается | жется | мое |  требует |  стать |  не | ное |
| 5 | 42 | 41 | 13686 | 5312.672 | 50 | 441.245 |  | 649.620 | order_specific_generation_persistent_feature | ким |  стала |  о |  её | льный | чение |  описание |  состоит |
| 22 | 42 | 41 | 1161 | 5088.543 | 50 | -239.833 |  | 239.833 | order_component_specific_top_feature |  как | что | что |  это |  здесь |  нужно | как | что |
| 38 | 19 | 18 | 427 | 4708.980 | 50 | -114.392 | -16.482 | 166.474 | order_specific_prompt_feature | го | сен |  самом |  самом |  оказа |  ощу |  П | шен |
| 29 | 37 | 36 | 1914 | 3382.937 | 50 | -196.214 |  | 196.214 | order_component_specific_top_feature |  мог |  бы |  никто |  мог |  не |  кто |  говорить |  бы |
| 18 | 42 | 41 | 11558 | 3191.251 | 11 | 306.686 |  | 306.686 | order_component_specific_top_feature |  ж |  ж |  ж |  не | ется |  не |  Не | Мо |
| 40 | 42 | 41 | 9705 | 3005.944 | 50 | 155.397 |  | 155.397 | order_component_specific_top_feature |  всех |  что | ться |  вывод |  к |  этот |  Её |  смысле |
| 33 | 42 | 41 | 15698 | 2669.942 | 5 | 176.328 |  | 176.328 | order_component_specific_top_feature | го |  тя |  кого |  заб |  что | Мо | дель |  встреча |
| 32 | 42 | 41 | 3123 | 2322.655 | 7 | 177.119 |  | 177.119 | order_component_specific_top_feature |  настолько |  её |  ей | сть |  который |  что |  что | Мо |
| 45 | 37 | 36 | 5970 | 2307.666 | 1 | 131.358 |  | 147.406 | order_specific_generation_persistent_feature |  Ч | Мо | дель |  встреча | ет |  здесь |  не |  крити |
| 24 | 19 | 18 | 373 | 2228.229 | 50 | 108.621 | -35.628 | 228.763 | order_specific_generation_persistent_feature | зыва |  потому |  бы |  потому | кс |  потому |  потому |  не |
| 25 | 42 | 41 | 1282 | 2225.808 | 1 | 225.550 |  | 225.550 | order_component_specific_top_feature | вет | Мо | дель |  встреча | ет |  здесь |  не |  крити |
| 36 | 37 | 36 | 5831 | 1678.183 | 2 | 170.408 |  | 170.408 | order_component_specific_top_feature |  за |  за | Мо | дель |  встреча | ет |  здесь |  не |

## Old patching top rows, not clean feature causality
| selected_rank | layer | real_layer | feature_index | kl_base_to_patched_mean | logit_l2_mean | top_token_changed_rate | base_top_token_first | patched_top_token_first | interpretation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | 19 | 18 | 427 | 5.834 | 868.880 | 1.000 | <end_of_turn> | И | order_specific_prompt_feature |
| 12 | 19 | 18 | 397 | 5.371 | 866.854 | 1.000 | <end_of_turn> | И | order_specific_generation_persistent_feature |
| 17 | 19 | 18 | 1162 | 5.176 | 867.148 | 1.000 | <end_of_turn> | И | order_enriched_overlap_feature |
| 35 | 19 | 18 | 463 | 5.096 | 862.567 | 1.000 | <end_of_turn> | И | order_specific_prompt_feature |
| 23 | 19 | 18 | 4507 | 5.092 | 865.286 | 1.000 | <end_of_turn> | И | order_specific_prompt_feature |
| 24 | 19 | 18 | 373 | 4.446 | 863.542 | 1.000 | <end_of_turn> | И | order_specific_generation_persistent_feature |
| 13 | 19 | 18 | 378 | 4.143 | 895.673 | 1.000 | <end_of_turn> | И | order_specific_prompt_feature |
| 39 | 25 | 24 | 1795 | 1.201 | 872.030 | 0.000 | <end_of_turn> | <end_of_turn> | order_component_specific_top_feature |
| 49 | 25 | 24 | 10622 | 1.080 | 872.862 | 0.000 | <end_of_turn> | <end_of_turn> | order_component_specific_top_feature |
| 14 | 42 | 41 | 207 | 0.310 | 1328.232 | 0.000 | <end_of_turn> | <end_of_turn> | order_specific_generation_persistent_feature |
| 6 | 25 | 24 | 76 | 0.202 | 981.145 | 0.000 | <end_of_turn> | <end_of_turn> | order_enriched_overlap_feature |
| 42 | 37 | 36 | 9566 | 0.099 | 1099.274 | 0.000 | <end_of_turn> | <end_of_turn> | order_component_specific_top_feature |
| 37 | 37 | 36 | 1102 | 0.098 | 1099.390 | 0.000 | <end_of_turn> | <end_of_turn> | order_enriched_overlap_feature |
| 41 | 37 | 36 | 12008 | 0.098 | 1103.350 | 0.000 | <end_of_turn> | <end_of_turn> | order_component_specific_top_feature |
| 21 | 37 | 36 | 4735 | 0.098 | 1103.350 | 0.000 | <end_of_turn> | <end_of_turn> | order_component_specific_top_feature |

## Recommended immediate rerun
Run the fixed script with `PATCH_RESIDUAL_UPDATE_MODE = "delta"` and skip context inspection to save time. Use a new RUN_TAG. Keep `PATCH_PROMPTS = prompts_target`.
