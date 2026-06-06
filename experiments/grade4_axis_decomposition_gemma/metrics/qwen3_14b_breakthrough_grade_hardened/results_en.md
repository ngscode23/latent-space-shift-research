# Qwen3-14B / breakthrough_grade_hardened Run Results

## 1. Run Identity

```text
Script: red_team_hidden_geometry_breakthrough_grade.py
Path: C:\Users\stasv\OneDrive\Рабочий стол\agent\red_team_hidden_geometry_breakthrough_grade.py
MODEL_ID: Qwen/Qwen3-14B
RUN_LABEL: breakthrough_grade_hardened
RESULTS_DIR: red_team_hidden_geometry_results_breakthrough_grade
Metric source: C:\Users\stasv\Downloads\red_team_hidden_geometry_results_breakthrough_grade3.zip
```

Configuration:

```text
Questions: 15
Target tokens: 2721
Neutral tokens: 3017
Reference condition: neutral
Max input tokens: 8192
Max new tokens: 256
Causal layer bands: middle, late
Behavioral split: 9 train / 6 held-out test
Behavioral random baselines: 48
```

## 2. Main Verdict

This run supports the strong claim:

```text
Qwen/Qwen3-14B contains a robust target-conditioned latent axis, Vector X.
The axis is specific against length, shuffle, random-vector, and FDR controls,
and it is causally steerable by residual-stream +X/-X intervention in middle
layers.
```

This is not merely descriptive geometry. The middle-layer intervention produces
a clean, monotonic, bidirectional causal effect. The conservative verdict is:

```text
causal_internal_axis_supported
```

The internal result is strong. The visible behavioral steering claim is not yet
closed at reviewer-grade standard.

## 3. Hidden Geometry

Primary target-vs-neutral signal in middle layers:

```text
target middle projection mean:        0.976583
target middle projection CI95:        [0.960451, 0.992008]
target middle direction cosine:       0.852397
middle-band R2:                       0.744126
positive projection fraction:         1.0
```

This is a high-signal hidden-state result. Projection is almost one, direction
cosine is high, and the projection sign is stable.

Controls:

```text
neutral_length_matched projection:    0.002749
question_only projection:             0.330825
word_shuffle projection:              0.654745
sentence_shuffle projection:          0.865168
target projection:                    0.976583
```

The target beats every control:

```text
target - neutral_length_matched:      +0.973834, p=0.0001
target - word_shuffle:                +0.321837, p=0.0001
target - sentence_shuffle:            +0.111415, p=0.0001
```

Mechanistic interpretation:

```text
The axis is not reducible to length, because the length-matched neutral control
is near zero. It is not reducible to random direction noise, because the random
null is near zero. It is not fully explained by shuffled target controls,
because coherent target remains significantly above both word-shuffle and
sentence-shuffle controls.
```

But the shuffled controls are high. The precise claim is therefore:

```text
Vector X contains a strong semantic/lexical target-family component, and
coherent target ordering contributes an additional significant component. It
is not a purely discourse-order axis.
```

## 4. Random Null

Same-norm random-vector baseline:

```text
observed target projection:           0.976583
random null mean:                     0.000040
random null std:                      0.001122
observed - null:                      0.976543
empirical p >= observed:              0.007752
null vectors:                         128
```

This is one of the strongest parts of the run. Same-norm random vectors do not
explain the target projection. The observed projection is several orders above
the null scale.

## 5. Causal Internal Intervention

The middle-layer residual-stream intervention is the central causality result.

Neutral base, middle-layer +X/-X gap:

```text
alpha 0.10: 0.441223
alpha 0.25: 1.150607
alpha 0.50: 2.267656
alpha 0.75: 3.313378
```

Target base, middle-layer +X/-X gap:

```text
alpha 0.10: 0.468811
alpha 0.25: 1.141333
alpha 0.50: 2.251842
alpha 0.75: 3.336544
```

Dose response:

```text
middle plus_internal slope:              2.318534
middle minus_internal_suppression slope: 2.185581
middle monotonicity:                     1.0
late plus_internal dose-response:        failed
```

Mechanistic interpretation:

```text
Vector X is not only a post-hoc description of the target/reference
difference. Adding or subtracting it in middle-layer residual streams moves the
generation-time hidden trajectory in the expected direction with an
alpha-dependent dose.
```

This is a causal internal axis.

## 6. Architecture-Level Readout

Architecture/module deltas show that the effect is not limited to a post-hoc
residual metric.

Mean projection fractions for target:

```text
mlp:          0.957424
mlp.down:     0.957424
mlp.gate:     0.964687
mlp.up:       0.964915
self_attn:    0.936562
```

Mechanistic interpretation:

```text
The axis is visible in MLP and attention-path activations. MLP gate/up
projections are especially strong. This is consistent with a target-induced
activation regime rather than a surface-only style signature.
```

## 7. Generation-Time Readout

Projection during ordinary generation:

```text
neutral generation middle projection:        0.128824
question_only generation middle projection:  0.176867
target generation middle projection:         0.292595
word_shuffle generation middle projection:   0.266662
```

The target leaves a trace in the generation trajectory. However, the prompt
endpoint signal is stronger than the downstream generation signal: target
starts around `0.977` and later decays to smaller values. This is not a failure;
it is the expected dilution of a prompt-conditioned axis through autoregressive
dynamics.

## 8. Visible Behavior

Visible behavioral steering does not yet pass the hard random-p95 gate.

Best visible-like result:

```text
neutral +X, middle alpha 0.75 likeness:       0.557539
random plus mean likeness:                    0.532424
lift over random mean:                        +0.025115
lift over random p95:                         -0.089669
win rate vs random p95:                       0
```

Internal-visible coupling:

```text
middle alpha 0.75 Pearson r:                  0.106428
pass_coupling:                                0
```

This does not weaken the internal result. It means the current visible semantic
readout is not specific enough: same-norm random perturbations also produce
high response-likeness. Therefore this run should not be labeled as
reviewer-grade behavioral steering.

## 9. Established / Not Established

Established by this run:

```text
1. Strong target-conditioned hidden shift.
2. Vector X stability across questions and middle layers.
3. Target beats length, shuffle, random, and FDR controls.
4. Middle-layer +X/-X intervention causally controls the internal trajectory.
5. Architecture-level activations align with Vector X.
```

Not established by this run:

```text
1. Permanent weight-level or topology-level change.
2. Reviewer-grade visible behavioral control.
3. Cross-model replication.
4. SAE-level feature localization.
5. Comparison against global all-layer intervention; this is an optional localization check, not part of the core claim.
```

## 10. Next Experiment

Primary next run:

```python
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late"]
```

`all` is not required for Grade 4. It can be added in a separate rerun only if
we want a localized-vs-global intervention check. The core Grade 4 question is
answered by middle/late: decompose Vector X into content/order components and
test whether `X_order_orth` preserves a causal internal gap.

Axis decomposition:

```text
X_content = sentence_shuffle - neutral
X_order   = target - sentence_shuffle
X_full    = target - neutral
```

Next-run gates:

```text
1. middle must beat late on internal effect / quality tradeoff; all-layer is an optional control;
2. visible +X must beat alpha-matched random p95;
3. internal-visible coupling must become positive and stable;
4. output semantic shift must separate target from target_word_shuffle_control.
```

## 11. Short Final Claim

```text
Breakthrough Grade 3 establishes a robust, middle-layer, target-conditioned
causal internal latent axis in Qwen/Qwen3-14B. It does not yet establish
reviewer-grade visible behavioral control.
```
