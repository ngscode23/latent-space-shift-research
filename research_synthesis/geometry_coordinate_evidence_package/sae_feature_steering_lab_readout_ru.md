# SAE Feature Steering: Lab-Facing Readout

## Scope

Этот документ фиксирует отдельный feature-level steering результат поверх
основной geometry-coordinate линии. Он не заменяет главный claim о
`context-induced hidden-state geometry shift`; он показывает следующий слой:
некоторые SAE decoder directions, выбранные из contrastive/order-aware
readout, могут модулировать generation trajectory и token distributions на
политически и нормативно чувствительных вопросах.

Использованные пакеты:

```text
C:\Users\stasv\Downloads\BASE_CONTRL
C:\Users\stasv\Downloads\BASE_TARGET
```

Ключевые файлы:

```text
sae_feature_steering_generation_full_metrics_with_tf_kl (2).csv
sae_feature_steering_generation_full_metrics (1).csv
sae_teacher_forced_per_token_kl_details (1).csv
sae_teacher_forced_kl_summary_by_feature_scale (1).csv
sae_feature_steering_generation_summary_metrics (1).csv
causal_mediation_sae_order_features_results (1).csv
```

---

## Experimental Design

Тестировались две SAE features на layer 41:

```text
feature 208:
  provisional label = contrastive_208

feature 13686:
  provisional label = abstract_epistemic_13686
```

Для каждого base condition:

```text
BASE_CONTROL rows = 200
BASE_TARGET rows = 200
tasks = 5
generation modes = greedy, sampled
errors = 0
teacher-forced KL errors = 0
```

Scales:

```text
feature 208:
  -25400, -12700, 0, 12700, 25400

feature 13686:
  -12700, -6350, 0, 6350, 12700
```

Метрики:

```text
free-generation metrics:
  exact_match_to_scale0
  jaccard_similarity_to_scale0
  delta_token_count_vs_scale0
  delta_diagnostic_keywords_vs_scale0
  delta_contrastive_markers_vs_scale0
  delta_negation_markers_vs_scale0

final next-token metrics:
  final_next_token_kl_base_to_patched
  final_next_token_js_divergence
  final_logit_l2
  final_top_token_changed

teacher-forced metrics:
  tf_kl_base_to_patched_mean
  tf_js_mean
  tf_top_token_changed_fraction
  tf_ref_logprob_delta_mean
```

---

## Main Empirical Pattern

### Feature 208: contrastive-framing direction

Feature 208 shows the clearest target-specific rhetorical modulation. On
BASE_TARGET, nonzero steering increases absolute contrastive-marker movement
more strongly than on BASE_CONTROL:

```text
mean abs(delta_contrastive_markers), nonzero scales:

CONTROL / feature 208 = 0.2625
TARGET  / feature 208 = 1.1750
target-minus-control  = +0.9125
```

It also produces larger output-length shifts under BASE_TARGET:

```text
mean abs(delta_token_count), nonzero scales:

CONTROL / feature 208 = 6.0500
TARGET  / feature 208 = 10.7375
target-minus-control  = +4.6875
```

Strong scale examples:

```text
feature 208, scale -25400:
  TARGET - CONTROL delta_tokens       = +27.3
  TARGET - CONTROL delta_contrast     = +2.15
  TARGET - CONTROL delta_negation     = +1.40

feature 208, scale +25400:
  TARGET - CONTROL delta_tokens       = +21.1
  TARGET - CONTROL delta_contrast     = +1.45
  TARGET - CONTROL delta_negation     = +1.85
```

Interpretation:

```text
Feature 208 is best described as a contrastive-framing / rhetorical
organization direction. It does not encode a political belief. It changes
how answers are structured: more contrast, opposition, qualification, and
framing movement relative to the scale-0 baseline.
```

Important boundary:

```text
Feature 208 is not simply "larger KL on target". Final-next-token KL and
teacher-forced KL are not consistently larger on TARGET than CONTROL. The
stronger TARGET evidence is in free-generation rhetorical structure:
contrast markers, output length, exact-match collapse, and visible divergence
from scale-0 completions.
```

### Feature 13686: abstract-epistemic direction

Feature 13686 shows a different pattern. TARGET is much more sensitive in
surface continuation stability:

```text
mean exact_match_to_scale0, nonzero scales:

CONTROL / feature 13686 = 0.2625
TARGET  / feature 13686 = 0.0125
target-minus-control    = -0.2500

mean jaccard_to_scale0, nonzero scales:

CONTROL / feature 13686 = 0.538339
TARGET  / feature 13686 = 0.329113
target-minus-control    = -0.209227
```

It also shows stronger distributional movement on TARGET:

```text
mean final_next_token_kl, nonzero scales:

CONTROL / feature 13686 = 0.066346
TARGET  / feature 13686 = 0.082654
target-minus-control    = +0.016308

mean teacher-forced KL, nonzero scales:

CONTROL / feature 13686 = 0.060805
TARGET  / feature 13686 = 0.067961
target-minus-control    = +0.007156
```

Strong scale examples:

```text
feature 13686, scale -12700:
  TARGET - CONTROL delta_tokens       = -20.2
  TARGET - CONTROL delta_contrast     = +0.7
  TARGET - CONTROL delta_negation     = -2.45
  TARGET - CONTROL exact_match        = -0.20
  TARGET - CONTROL jaccard            = -0.246703

feature 13686, scale +12700:
  TARGET - CONTROL delta_tokens       = +2.3
  TARGET - CONTROL delta_contrast     = +0.3
  TARGET - CONTROL delta_negation     = -1.8
  TARGET - CONTROL final_next_token_KL = +0.073778
```

Interpretation:

```text
Feature 13686 is best described as an abstract-epistemic formulation
direction. It appears to modulate whether the answer stays close to a narrow
baseline completion or moves into broader explanatory / epistemic framing.
```

Important boundary:

```text
Feature 13686 should not be described as a discrete "policy" or "position"
feature. It is a direction affecting formulation stability and explanatory
abstraction under the tested prompts.
```

---

## Causal Mediation CSV Boundary

The mediation CSVs contain strong layer-41 effects:

```text
BASE_CONTROL:
  layer 41 / feature 13686 mediated_effect = 17474.007812
  layer 41 / feature 208   mediated_effect = 17474.007812

BASE_TARGET:
  layer 41 / feature 13686 mediated_effect = 15227.294922
  layer 41 / feature 208   mediated_effect = 15227.294922
```

However, these mediation values are repeated for several features within the
same layer. Therefore they should be treated mainly as discovery/provenance
evidence that layer 41 is a high-impact intervention site, not as precise
per-feature semantic evidence.

For feature semantics and publication-facing claims, rely primarily on:

```text
generation_full_metrics_with_tf_kl
teacher_forced_kl_summary_by_feature_scale
teacher_forced_per_token_kl_details
free-generation deltas relative to scale 0
```

---

## Conservative Claim

Supported:

```text
SAE decoder-direction steering at layer 41 changes generated answer structure
and token distributions under controlled prompts. Features 208 and 13686 show
different signatures: feature 208 is associated with contrastive/rhetorical
framing movement, while feature 13686 is associated with abstract-epistemic
formulation and continuation instability.
```

Also supported:

```text
The effects are not only visible-output anecdotes. They appear across
free-generation deltas, final next-token divergence, and teacher-forced
per-token KL on scale-0 continuations.
```

Not supported:

```text
The data do not show that a feature contains a political stance.
The data do not show a discrete refusal-policy feature.
The data do not show a universal mechanism independent of domain/model.
The data do not yet prove that these directions transfer unchanged to neutral
or non-political prompts.
```

---

## Lab-Facing Abstract

```text
We report feature-level intervention experiments showing that sparse
autoencoder decoder directions can modulate the formulation dynamics of
answers to politically and normatively sensitive prompts. Candidate features
were selected from a prior order-aware contrastive readout and intervened on
during generation via decoder-direction steering at layer 41.

The central finding is not that an SAE feature encodes a political view or a
discrete refusal policy. Rather, the data support a narrower mechanistic claim:
some SAE directions participate in latent formulation dynamics, changing how
the model organizes direct assertion, qualification, contrastive explanation,
negation, and abstract epistemic framing.

Feature 208 is provisionally characterized as a contrastive-framing direction:
on target-context runs, nonzero steering produces stronger changes in
contrastive markers and output length than in matched control-context runs.
Feature 13686 is provisionally characterized as an abstract-epistemic
direction: target-context runs show lower exact-match stability to the scale-0
baseline, lower lexical overlap, and stronger next-token / teacher-forced KL
movement than control runs.

These results should be interpreted as local causal evidence under the tested
setup, not as a complete behavioral theory of the model. The interventions
alter hidden-state trajectory consequences and token distributions, but the
semantic interpretation of each feature requires validation on unrelated
domains, neutral prompts, alternate decoding settings, and cross-model
replications.
```

---

## Relation To Main Geometry Result

This feature-level result should be presented as a downstream mechanistic
layer of the main project:

```text
main result:
  coherent context induces measurable latent-state geometry shift

feature-level extension:
  selected SAE decoder directions can modulate formulation dynamics within
  that shifted state
```

The feature-level work is therefore not the foundation of the paper. The
foundation remains the coordinate geometry result:

```text
context -> hidden-state shift -> latent-axis coordinates
```

The SAE steering result adds:

```text
latent-axis readout -> candidate sparse directions -> local causal modulation
of formulation trajectory
```

