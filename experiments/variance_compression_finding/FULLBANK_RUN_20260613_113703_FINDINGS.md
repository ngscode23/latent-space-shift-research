# Fullbank Base-vs-Instruct Geometry/Probability Run

Run id: `run_20260613_113703`

Local run path:

`C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703`

Deep-dive path:

`C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\hidden_npz_deep_dive`

Plots:

`C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\hidden_npz_deep_dive\plots`

Repository research packet:

`experiments\variance_compression_finding\alignment_geometry_probability_run_02\metric`

Use these files for the final claim/evidence/narrative:

- `alignment_geometry_probability_run_02\metric\README.md`
- `alignment_geometry_probability_run_02\metric\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\README_RESEARCH_NARRATIVE.md`
- `alignment_geometry_probability_run_02\metric\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\RESEARCH_NARRATIVE_RU.md`
- `alignment_geometry_probability_run_02\metric\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\RESEARCH_NARRATIVE_EN.md`

Russian copy of this summary:

`experiments\variance_compression_finding\FULLBANK_RUN_20260613_113703_FINDINGS_RU.md`

## Setup

This run compares the same prompts on:

- Base model: `google/gemma-3-12b-pt`
- Instruct model: `google/gemma-3-12b-it`
- Prompt mode: `raw`
- Target contexts: `10`
- Control contexts: `10`
- Questions: `10`
- Shuffles enabled: `true`
- Total prompts per model: `410`
- Hidden tensor shape per model: `[410, 49, 3840]`
- Main late band used for interpretation: layers `30..47`

Condition counts per model:

- `target`: 100 prompts
- `target_word_shuffle`: 100 prompts
- `target_sentence_shuffle`: 100 prompts
- `control`: 100 prompts
- `question_only`: 10 prompts

The verifier script confirmed that the prompt bank was present and covered the expected conditions. This run should be treated as the first serious fullbank base-vs-instruct geometry/probability audit.

The deep-dive metrics were rebuilt after fixing the `auc_like` calculation in `hidden_npz_deep_dive_visualizer.py`. Older `target_control_axis_auc_like` values were too low. The current deep-dive AUC columns should be treated as the corrected values.

## Question Tested

The motivating hypothesis was:

> Alignment / RLHF / instruction tuning bends the model's vector-space geometry during training so that hidden-state dispersion is suppressed before logits. The "invisible alignment" is therefore a forced shift in the next-token probability distribution.

This run does test that idea, but the result is more precise than the original formulation.

## Main Scientific Claim

Dense context can induce a measurable pre-output latent-state shift in an LLM. In this Gemma-3-12B fullbank audit, target/control contexts separate in late hidden-state space before generation, and this separation is stronger in the instruct model than in the base model. Instruction tuning does not merely collapse hidden-state geometry; it reduces absolute hidden-state scale while preserving or increasing angular/rank structure. The strongest alignment-like effect appears in the hidden-to-logit readout: the instruct model converts hidden states into a sharper, lower-entropy next-token probability distribution.

Short version:

> Alignment looks less like "all hidden variance is suppressed" and more like "the hidden state remains structured, but the readout into logits becomes stiffer and more committed."

## Hidden Geometry: Base vs Instruct

Across late layers `30..47`, instruct has lower absolute hidden-state scale:

- `centroid_norm` is lower in instruct for every condition.
- `abs_disp_l2_mean` is lower in instruct for every condition.
- `cov_trace` is lower in instruct for every condition.

Examples from `late_band_summary.csv`:

| condition | centroid_norm instruct-base | abs_disp instruct-base | cov_trace instruct-base |
|---|---:|---:|---:|
| control | -22,762.93 | -2,665.14 | -114,618,574.96 |
| question_only | -30,499.59 | -5,076.88 | -276,225,978.96 |
| target | -18,820.33 | -3,579.07 | -188,953,167.60 |
| target_sentence_shuffle | -19,946.89 | -3,463.38 | -188,350,424.28 |
| target_word_shuffle | -18,959.53 | -2,219.09 | -95,436,379.50 |

This supports a real absolute-scale compression in instruct.

But the same run shows that hidden geometry is not merely crushed:

- `pairwise_cosine_distance_mean` is higher in instruct.
- `effective_rank_pr` is higher in instruct.
- `spectral_entropy_norm` is higher in instruct.
- `top1_pc_variance_share` is lower in instruct.

Examples:

| condition | pairwise cosdist instruct-base | effective rank instruct-base | spectral entropy norm instruct-base | top1 PC share instruct-base |
|---|---:|---:|---:|---:|
| control | +0.00881 | +0.66056 | +0.06934 | -0.10792 |
| question_only | +0.01063 | +1.28889 | +0.22936 | -0.20073 |
| target | +0.00646 | +0.72464 | +0.09490 | -0.11625 |
| target_sentence_shuffle | +0.00688 | +0.88183 | +0.08643 | -0.13084 |
| target_word_shuffle | +0.00830 | +1.17019 | +0.09259 | -0.15736 |

Interpretation:

Instruct hidden states are smaller in raw L2 scale, but they are not geometrically simpler. Direction-space remains active and is often more distributed across effective dimensions.

## Target Context Effect

Target context produces a real hidden-state shift relative to control. This is visible before generation, in late hidden states.

From `deep_late_band_contrast_summary.csv`:

| metric | base | instruct |
|---|---:|---:|
| target-control centroid L2 | 4,781.80 | 9,392.86 |
| target-control projection gap z | 0.59291 | 0.86841 |
| target-control axis AUC-like | 0.70447 | 0.74658 |
| leave-one-question balanced accuracy | 0.58889 | 0.65361 |
| leave-one-question AUC-like | 0.91450 | 0.93778 |
| target-minus-control distance to question-only | 759.86 | 4,390.08 |

Interpretation:

Target and control contexts are separable in hidden space. The separation is stronger in instruct than in base. The leave-one-question AUC-like is especially important: it means the target/control separation is not only memorizing one question phrasing.

The threshold classifier is moderate (`0.654` balanced accuracy in instruct), while ranking separation is strong (`0.938` leave-one-question AUC-like in instruct). The right wording is strong ranking/separation evidence, not a perfect binary classifier.

## Target Is Not Just a Confidence Booster

The target context does not simply make the instruct model more confident.

From `logit_metrics_summary.csv`:

| model | condition | entropy | top1 prob | top5 mass |
|---|---|---:|---:|---:|
| base | control | 2.783 | 0.417 | 0.650 |
| base | target | 2.777 | 0.430 | 0.642 |
| instruct | control | 1.176 | 0.678 | 0.907 |
| instruct | target | 1.768 | 0.614 | 0.808 |
| instruct | question_only | 0.912 | 0.816 | 0.912 |

For instruct, target has higher entropy and lower top1 probability than control and question-only. That means the target context produces a different processing state, not merely a stronger commitment to one token.

Good wording:

> The target context causes latent-state reorganization, not a simple confidence increase.

## Readout Stiffness

The strongest alignment-like effect is in next-token distribution concentration.

From `readout_stiffness_summary.csv`:

| condition | entropy reduction base-instruct | top1 gain instruct-base | top1 per relative dispersion ratio |
|---|---:|---:|---:|
| control | 1.607 | 0.261 | 1.535 |
| question_only | 2.016 | 0.398 | 1.905 |
| target | 1.009 | 0.184 | 1.436 |
| target_sentence_shuffle | 1.361 | 0.219 | 1.513 |
| target_word_shuffle | 1.601 | 0.243 | 1.467 |

Interpretation:

For comparable hidden-state relative dispersion, instruct produces a more concentrated next-token distribution. This is the cleanest support for the "hidden-to-logit readout is tightened" interpretation.

The correct conclusion is not:

> Alignment suppresses all hidden-state variance.

The correct conclusion is:

> Alignment/instruction tuning changes the mapping from hidden state to logits so that probability mass is concentrated more aggressively.

## Base-Instruct Representation Alignment

From `deep_late_band_base_instruct_alignment_summary.csv`:

| condition | linear CKA | same-prompt delta L2 | same-prompt cosdist | instruct/base norm |
|---|---:|---:|---:|---:|
| all | 0.892 | 23,888.26 | 0.00547 | 0.848 |
| control | 0.938 | 26,806.57 | 0.00729 | 0.826 |
| question_only | 0.763 | 33,434.63 | 0.00697 | 0.776 |
| target | 0.920 | 22,264.68 | 0.00444 | 0.864 |
| target_sentence_shuffle | 0.911 | 23,187.28 | 0.00466 | 0.854 |
| target_word_shuffle | 0.882 | 22,339.88 | 0.00535 | 0.854 |

Interpretation:

Base and instruct representations are still substantially aligned, especially for contextual prompts. Instruct is not using a completely unrelated representation space. It is a related space with lower norm and different readout behavior.

Question-only is the least aligned condition, which suggests context stabilizes cross-model representational alignment.

## What This Run Establishes

1. Dense target context creates a measurable latent-state shift before generation.
2. Target/control separation exists in both base and instruct.
3. Instruct amplifies target/control separation in late hidden states.
4. Instruct has lower absolute hidden-state scale.
5. Instruct does not simply collapse hidden geometry; relative/angular/rank structure remains active or increases.
6. Instruct strongly sharpens next-token probability readout.
7. Target context is not merely a confidence booster; in instruct it can broaden the next-token distribution while still causing stronger hidden separation.

Best current formulation:

> Fullbank confirms context-induced target/control latent-state separation and refines the alignment hypothesis. Instruction tuning does not merely compress hidden-state geometry. It reduces absolute hidden-state scale while preserving or increasing angular/rank structure, and it strongly stiffens the hidden-to-logit readout. Alignment looks like a change in how complex hidden states are converted into probability distributions.

## Working Note

Coverage:

```text
target = 100 rows
target_word_shuffle = 100
target_sentence_shuffle = 100
control = 100
question_only = 10

hidden shape = (410, 49, 3840)
late band = L30-L47
```

This is `10 target / 10 control`, not `9 target / 10 control`. Target/control are balanced.

Main result:

```text
Fullbank confirms context-induced latent-state separation.
Target/control hidden-state separation is stronger in instruct than in base.
Instruction tuning sharply changes hidden-to-logit readout relative to base.
Coherent target context is not just a confidence booster.
```

Late `L30-L47` target-control contrast:

```text
target_control_centroid_l2:
  base     4,781.8
  instruct 9,392.9

target_control_projection_gap_z:
  base     0.593
  instruct 0.868

target_control_axis_auc_like:
  base     0.704
  instruct 0.747

leave-one-question balanced_acc:
  base     0.589
  instruct 0.654

leave-one-question auc_like:
  base     0.914
  instruct 0.938
```

Target/control separation exists in both models, but it is stronger in instruct. Leave-one-question is especially important: the axis is not merely capturing one question wording. Ranking separation is strong, while threshold accuracy is moderate. The axis really ranks target above control, but the hard boundary is not perfect.

Target effect:

```text
base:
  target rel_disp   0.196
  control rel_disp  0.188

instruct:
  target rel_disp   0.195
  control rel_disp  0.199
```

In base, target does not compress more strongly than control. It is even more dispersed by relative dispersion and covariance trace. Therefore, the simple formula `target always compresses hidden geometry` is false.

In instruct, target enters a more distinct regime:

```text
target_control_centroid_l2:
  instruct 9,392.9
  base     4,781.8
```

Target in instruct is farther from control and farther from question-only. This is the main context-induced latent-state shift signal.

Context snapping:

```text
question_minus_context_rel_disp:
  base     0.009512
  instruct 0.009484
```

Context reduces relative hidden dispersion by about the same amount in base and instruct. The strongest instruct effect is not here. It is in target/control separation and readout.

Output/readout:

```text
instruct narrows the next-token probability distribution much more strongly
than base.
```

But coherent target is not the most confident instruct regime.

Instruct entropy:

```text
question_only           0.912
control                 1.176
target_word_shuffle     1.227
target_sentence_shuffle 1.519
target                  1.768
```

Instruct top1 probability:

```text
question_only           0.816
target_word_shuffle     0.699
control                 0.678
target_sentence_shuffle 0.638
target                  0.614
```

Instruct is generally much lower-entropy than base, but coherent target makes instruct less committed than control or word-shuffle. Target does not simply increase confidence. It moves the model into another regime where hidden separation is stronger but probability readout is broader.

Base-vs-instruct alignment:

```text
Late L30-L47 CKA:

control                 0.938
target                  0.920
target_sentence_shuffle 0.911
target_word_shuffle     0.882
question_only           0.763
```

Base and instruct remain strongly aligned in contextual prompts, while question-only is least aligned. Instruct/base norm ratio is below 1 everywhere:

```text
target  0.864
control 0.826
all     0.848
```

This confirms that instruct has lower late hidden norm, but it does not move into a completely unrelated representation space.

Final formula after fullbank:

```text
Target context induces a measurable latent-state shift.
Instruction tuning amplifies target/control hidden-state separation and stiffens
hidden-to-logit readout, but coherent target context does not merely increase
confidence. It reorganizes the internal regime: stronger hidden separation,
but broader probability readout than neutral control.
```

The scientific value of this result is that the picture is more complex and stronger than the original hypothesis: not `everything collapses`, but `context changes regime; instruction tuning changes readout and amplifies target/control separation`.

## Important Plots to Reopen

From `hidden_npz_deep_dive\plots`:

- `target_control_centroid_l2_by_layer.png`
- `target_control_projection_gap_z_by_layer.png`
- `target_control_axis_auc_like_by_layer.png`
- `loo_question_balanced_acc_by_layer.png`
- `loo_question_auc_like_by_layer.png`
- `linear_cka_base_instruct_by_layer.png`
- `instruct_over_base_norm_mean_by_layer.png`
- `late_condition_metric_zscore_heatmap.png`
- `late_condition_summary_table.png`
- `late_target_control_contrast_table.png`
- `late_base_instruct_alignment_table.png`

## Next Steps

Next line of work:

1. Decision-margin audit:
   - Use forced-choice probes.
   - Track `margin = logp(A) - logp(B)`.
   - Test whether the target-induced latent shift changes actual decision margins.

2. Lexical vs order control:
   - Compare coherent target, sentence shuffle, word shuffle, matched vocabulary neutral text, and length-matched control.
   - This separates semantic mass from discourse order.

3. More model families:
   - Repeat base/instruct audit on another open pair if available.
   - Use the Gemma-3-12B result as the reference pattern.

4. Layer-band robustness:
   - Keep `L30-L47` as primary.
   - Avoid mixing layer 48 into late-band summary unless explicitly analyzing the final norm/readout transition.

5. Public wording:
   - Do not say "alignment only compresses hidden states."
   - Say "alignment reduces absolute hidden-state scale and stiffens hidden-to-logit readout."

## Short Version for Future Use

This fullbank run tested whether instruction/alignment tuning suppresses hidden-state dispersion before logits. It found a more nuanced mechanism: instruct has smaller absolute late hidden-state scale, but hidden geometry is not simply collapsed. Relative/angular/rank structure is preserved or stronger. The main alignment-like compression appears in the next-token probability distribution: instruct turns hidden states into much sharper, lower-entropy logits. Separately, target contexts create a measurable latent-state shift relative to controls, and this target/control separation is stronger in instruct than in base.
