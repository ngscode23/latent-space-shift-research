# Red-Team Hidden Geometry Metric Audit

Reader-only audit. No model was run.

## Bottom Line

- Runs loaded: `2`
- Strong internal axis runs: `2/2`
- Best neutral +X visible lift: `0.035109` on `Qwen/Qwen3-14B` (p95 lift `-0.0161607`, question win mean `57.1%`).
- Best target -X suppression lift: `0.0556908` on `Qwen/Qwen3.5-9B` (p95 lift `-0.0040929`, random win `93.8%`).
- Overall statuses: `{"strong_internal_axis_partial_visible_readout": 1, "strong_internal_axis_asymmetric_visible_readout": 1}`

Practical interpretation:

`Strong mechanism, asymmetric readout.` Frame Vector X first as an ablation/suppression axis; keep testing neutral +X separately.

## Run Summary

| model_id | target_projection | neutral_lift_random_mean | neutral_lift_random_p95 | neutral_questions_over_random_mean | generation_win_random | target_minus_supp_lift_random_mean | target_minus_supp_win_random_mean | neutral_degeneration | visible_status | ablation_status | overall_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-14B | 0.974829 | 0.035109 | -0.0161607 | 0.571429 | 1 | 0.023687 | 0.875 | 0 | good_partial | good | strong_internal_axis_partial_visible_readout |
| Qwen/Qwen3.5-9B | 0.945212 | 0.0111138 | -0.0527653 | 0.714286 | 1 | 0.0556908 | 0.9375 | 0 | weak_partial | good | strong_internal_axis_asymmetric_visible_readout |

## Run: Qwen/Qwen3-14B

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_full_middle_behavioral_control_middle_alpha_retest_02.zip`
- Run label: `full_middle_clean_behavioral_control_middle_alpha_retest`
- Verdict CSV: `partial_behavioral_control_axis_supported`
- Existing hard-random file: `False`
- Existing breakthrough audit: `False`

Key numbers:

- Target projection: `0.974829`
- Length-neutral projection: `0.0128273`
- Word-shuffle projection: `0.733918`
- Sentence-shuffle projection: `0.873508`
- Neutral +X target-likeness: `0.55941`
- Random visible mean: `0.524301`
- Neutral +X lift over random mean: `0.035109`
- Neutral +X lift over random p95: `-0.0161607`
- Neutral +X question wins over random mean: `57.1%`
- Neutral +X question wins over random p95: `0.0%`
- Generation projection win over random: `100.0%`
- Target -X suppression lift over random mean: `0.023687`
- Target -X suppression lift over random p95: `-0.0124462`
- Target -X suppression win over random mean: `87.5%`
- Neutral +X degeneration: `0`
- Target -X degeneration: `0`

Status: `strong_internal_axis_partial_visible_readout`

Next action: run behavioral-control-only with 32 random baselines and quality audit

Per-question neutral +X vs random:

| question_index | x_likeness | random_mean | random_p95 | random_max | x_minus_random_mean | x_minus_random_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.570983 | 0.511784 | 0.63296 | 0.680867 | 0.0591993 | -0.0619768 |
| 1 | 0.484518 | 0.465542 | 0.522007 | 0.545103 | 0.0189762 | -0.0374887 |
| 3 | 0.546251 | 0.562709 | 0.628624 | 0.785993 | -0.0164583 | -0.0823727 |
| 5 | 0.632824 | 0.562351 | 0.699355 | 0.732156 | 0.070473 | -0.0665307 |
| 8 | 0.566464 | 0.568698 | 0.685769 | 0.727732 | -0.0022347 | -0.119305 |
| 10 | 0.55862 | 0.567562 | 0.714017 | 0.827469 | -0.00894203 | -0.155398 |
| 13 | 0.556209 | 0.43146 | 0.604094 | 0.618801 | 0.124749 | -0.0478846 |

## Run: Qwen/Qwen3.5-9B

- Source: `C:\Users\stasv\Downloads\red_team_hidden_geometry_results_full_middle_behavioral_control_middle_alpha_retest_03.zip`
- Run label: `full_middle_clean_behavioral_control_middle_alpha_retest`
- Verdict CSV: `partial_behavioral_control_axis_supported`
- Existing hard-random file: `False`
- Existing breakthrough audit: `False`

Key numbers:

- Target projection: `0.945212`
- Length-neutral projection: `0.00730324`
- Word-shuffle projection: `0.496309`
- Sentence-shuffle projection: `0.84667`
- Neutral +X target-likeness: `0.526543`
- Random visible mean: `0.515429`
- Neutral +X lift over random mean: `0.0111138`
- Neutral +X lift over random p95: `-0.0527653`
- Neutral +X question wins over random mean: `71.4%`
- Neutral +X question wins over random p95: `14.3%`
- Generation projection win over random: `100.0%`
- Target -X suppression lift over random mean: `0.0556908`
- Target -X suppression lift over random p95: `-0.0040929`
- Target -X suppression win over random mean: `93.8%`
- Neutral +X degeneration: `0`
- Target -X degeneration: `0`

Status: `strong_internal_axis_asymmetric_visible_readout`

Next action: move to next model/text family; frame X as ablation/suppression axis first

Per-question neutral +X vs random:

| question_index | x_likeness | random_mean | random_p95 | random_max | x_minus_random_mean | x_minus_random_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.51368 | 0.470422 | 0.596424 | 0.656963 | 0.0432578 | -0.0827445 |
| 1 | 0.574211 | 0.549483 | 0.629413 | 0.670749 | 0.0247283 | -0.0552024 |
| 3 | 0.451841 | 0.434434 | 0.476583 | 0.489224 | 0.0174069 | -0.0247417 |
| 5 | 0.0675151 | 0.328065 | 0.565902 | 0.659643 | -0.26055 | -0.498387 |
| 8 | 0.599544 | 0.589825 | 0.643377 | 0.660989 | 0.00971915 | -0.0438333 |
| 10 | 0.452678 | 0.509076 | 0.630954 | 0.648983 | -0.0563984 | -0.178277 |
| 13 | 1.02633 | 0.7267 | 0.978902 | 1.07694 | 0.299633 | 0.0474306 |

## Decision Gates

- Hidden geometry strong: `target_projection >= 0.85` and length-neutral near zero.
- Internal generation strong: neutral +X generation projection beats almost all random vectors.
- Visible neutral +X strong: beats random p95 and wins most held-out questions without degeneration.
- Target -X ablation strong: suppression beats random p95/mean without degeneration.
- Breakthrough-grade claim needs cross-model and cross-text-family replication, not one run.
