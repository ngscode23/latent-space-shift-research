# Research Narrative EN

## Main Scientific Claim After the Fullbank Run

Dense context can induce a measurable pre-output latent-state shift in an LLM. In Gemma-3-12B, target/control contexts separate in late hidden-state space before generation, and this separation is stronger in the instruct model than in the base model. Instruction tuning does not merely collapse hidden-state geometry; it reduces absolute hidden-state scale while preserving or increasing angular/rank structure. The strongest alignment-like effect appears in hidden-to-logit readout: the same class of hidden states is converted into a sharper, lower-entropy next-token probability distribution.

## What the Fullbank Run Tested

The run compares `google/gemma-3-12b-pt` and `google/gemma-3-12b-it` on the same prompt bank:

- `10` target contexts
- `10` control contexts
- `10` questions
- conditions: `target`, `target_word_shuffle`, `target_sentence_shuffle`, `control`, `question_only`
- `410` prompts per model
- hidden state tensor: `(410, 49, 3840)`
- primary analysis band: `L30-L47`

This is not primarily an analysis of final generated text. It measures the model state at the prompt boundary: late hidden states and next-token probability distribution before generation.

Source: `metadata.json`

## Evidence Ladder

### Strongly Supported

1. Target/control latent-state separation exists.

Fullbank `L30-L47`:

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

loo_question_auc_like:
  base     0.914
  instruct 0.938
```

Meaning: target/control contexts differ in hidden space before the model generates an answer. The separation is stronger in the instruct model.

Sources:

- `hidden_npz_deep_dive/deep_late_band_contrast_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_contrast_instruct_minus_base.csv`

2. Instruct amplifies target/control separation.

Most main contrast metrics are higher in instruct:

```text
target_control_centroid_l2 instruct/base ratio = 1.964
target_control_projection_gap_z instruct/base ratio = 1.465
target_control_axis_auc_like instruct-base = +0.042
loo_question_auc_like instruct-base = +0.023
```

Meaning: this is not just a generic context effect. The instruct model separates target/control regimes more strongly in late hidden states.

Source: `hidden_npz_deep_dive/deep_late_band_contrast_instruct_minus_base.csv`

3. Instruct strongly narrows the next-token distribution.

```text
entropy_reduction_base_minus_instruct:
  target        1.009
  control       1.607
  question_only 2.016

top1_prob_gain_instruct_minus_base:
  target        0.184
  control       0.261
  question_only 0.398

top1_per_rel_disp_instruct_over_base:
  target        1.436
  control       1.535
  question_only 1.905
```

Meaning: per unit of hidden-state relative dispersion, instruct produces more concentrated next-token readout. This is the cleanest evidence for readout stiffness.

Source: `readout_stiffness_summary.csv`

4. Hidden geometry does not simply collapse.

In instruct, these are lower:

- `centroid_norm`
- `abs_disp_l2_mean`
- `cov_trace`

But these are higher:

- `pairwise_cosine_distance_mean`
- `effective_rank_pr`
- `spectral_entropy_norm`

And this is lower:

- `top1_pc_variance_share`

Target example:

```text
centroid_norm:
  base     126,770
  instruct 107,950

abs_disp_l2_mean:
  base     24,828
  instruct 21,249

pairwise_cosine_distance_mean:
  base     0.0180
  instruct 0.0245

effective_rank_pr:
  base     1.91
  instruct 2.64

spectral_entropy_norm:
  base     0.251
  instruct 0.346

top1_pc_variance_share:
  base     0.718
  instruct 0.601
```

Meaning: absolute scale is lower, but angular/rank structure is higher. The formula "alignment simply suppresses hidden-state dispersion" is too crude.

Sources:

- `late_band_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_instruct_minus_base.csv`

### Moderately Supported

1. Target moves instruct into a distinct processing regime.

Target is farther from `question_only` and `control` in instruct, while its probability distribution is broader than neutral control.

```text
target_to_question_centroid_l2:
  base     5,348
  instruct 12,584

control_to_question_centroid_l2:
  base     4,588
  instruct 8,194
```

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

Meaning: target is not merely increasing confidence. It moves the model into another regime: stronger hidden separation, but broader probability readout.

Sources:

- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`
- `logit_metrics_summary.csv`

2. Context stabilizes/couples base and instruct representations.

Late `L30-L47` CKA:

```text
control                 0.938
target                  0.920
target_sentence_shuffle 0.911
target_word_shuffle     0.882
question_only           0.763
```

Meaning: base and instruct remain strongly aligned for contextual prompts. Question-only is the least aligned condition. Context seems to move both models into more comparable representation regions.

Source: `hidden_npz_deep_dive/deep_late_band_base_instruct_alignment_summary.csv`

3. The target effect transfers across questions.

```text
loo_question_balanced_acc:
  base     0.589
  instruct 0.654

loo_question_auc_like:
  base     0.914
  instruct 0.938
```

Meaning: ranking transfer is strong across questions. The threshold boundary is moderate, so this is stronger as separation/ranking evidence than as a complete classifier.

Source: `hidden_npz_deep_dive/deep_late_band_contrast_summary.csv`

### Weaker / Next-Test Claims

1. The specific role of target sentence order.

`target_word_shuffle` and `target_sentence_shuffle` still carry signal. The target effect therefore includes lexical-semantic mass, not only coherent order. The next test should separate coherent discourse, sentence order, word order, matched vocabulary and length-matched neutral controls.

Sources:

- `late_band_summary.csv`
- `logit_metrics_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`

2. Direct behavioral consequence.

This run measures pre-output hidden state and readout. The strongest next test is a forced-choice decision-margin audit, where the endpoint is not free-form text but `margin = logp(A) - logp(B)` on fixed decision probes.

3. Pure RLHF-specific claim.

This run compares base vs instruct. The precise current claim is: instruction/alignment post-training changes hidden scale, target/control separation and readout stiffness. Isolating RLHF specifically would require a model lineage or ablation that separates instruction tuning from RLHF.

## What We Have Shown

We have shown that target context creates a measurable pre-output shift: the model has not generated an answer yet, but its late hidden states already differ from control. We have shown that the instruct model amplifies this separation. We have shown that instruction tuning reduces absolute hidden scale, but does not destroy hidden-space structure. We have shown that the strongest alignment-like effect sits in readout: instruct converts hidden states into a narrower, more confident next-token distribution.

## What Became Stronger After Fullbank

1. The target/control separation result became stronger, because it no longer depends on one text. The fullbank run has `10/10` target/control contexts and `410` prompts per model.
2. The claim that instruct amplifies separation became stronger.
3. The readout hypothesis became stronger: instruct sharply narrows next-token distribution.
4. The scientific formula became stronger: not `collapse`, but `readout stiffness plus latent regime separation`.

## Which Hypotheses Weakened

1. The simple hypothesis "alignment suppresses hidden-layer dispersion" weakened. Absolute scale is lower, but geometry does not structurally collapse: rank, entropy and angular dispersion are higher.
2. The hypothesis "target simply makes the model more confident" weakened. In instruct, target produces stronger hidden shift, but higher entropy and lower top1 than control. This is regime reorganization, not a confidence boost.

## One Paragraph for an AI Safety / Mech Interp Lab

I am studying context-induced latent-state shifts: cases where dense context changes an LLM's internal pre-output regime before any visible answer is generated. In a fullbank Gemma-3-12B base-vs-instruct audit, target/control contexts separate in late hidden-state space, with stronger separation in the instruct model. The same run shows that instruction tuning does not merely collapse hidden geometry: instruct states have lower absolute norm and covariance, but higher angular/rank structure. The strongest alignment-like effect appears at the hidden-to-logit readout, where instruct produces much lower-entropy, more top-token-concentrated next-token distributions. This suggests that safety-relevant behavior may be mediated not only by surface refusals or generated text, but by pre-output latent regimes and readout stiffness. The next step is a forced-choice decision-margin audit to test whether these latent shifts causally move decision probabilities.

## Final Research Narrative

This project argues that some safety-relevant model behavior can be studied before text generation. Instead of only reading the final answer, we measure which internal regime the model enters at the prompt boundary. The fullbank Gemma-3-12B run shows that target contexts separate from control contexts in late hidden states, and this separation is stronger in the instruct model. The evidence is carried by `target_control_centroid_l2`, `projection_gap_z`, `axis_auc_like` and `loo_question_auc_like`. At the same time, the base-vs-instruct comparison refines the alignment hypothesis: instruct does have lower absolute hidden scale (`centroid_norm`, `abs_disp_l2_mean`, `cov_trace`), but hidden geometry does not collapse because angular dispersion, effective rank and spectral entropy are higher. The strongest alignment-like effect appears in logits: instruct sharply narrows the next-token distribution (`entropy`, `top1_prob`, `top5_mass`, readout stiffness ratios). The current strong formulation is that context can move the model into another latent regime, while instruction/alignment tuning amplifies regime separation and makes hidden-to-logit readout stiffer. The strongest next experiment is a forced-choice decision-margin audit testing whether target-induced latent shifts move real `logp(A)-logp(B)` decisions. For AI safety and mechanistic interpretability, this should be framed as pre-output monitoring of latent regimes and readout stiffness, not as analysis of only final text.
