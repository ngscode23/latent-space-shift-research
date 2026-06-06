# Final Conclusion: Context-Induced Latent-State Shift in Gemma-3-12B-IT

This document records the current scientific status of the Gemma-3-12B-IT Grade
4 analysis with SAE-res-all-small and the subsequent norm-controlled
component-causal run.

The factual source for the causal norm-control part is the archived run package:

```text
C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41 (1).zip
```

The important artifacts inside that package include:

```text
red_team_input_manifest.json
grade4_axis_component_causal_symmetry_summary.csv
grade4_axis_component_causal_projection_summary.csv
grade4_axis_component_causal_alpha_scaling_summary.csv
grade4_axis_component_causal_rank_summary.csv
grade4_axis_component_norm_summary.csv
grade4_axis_component_causal_response_audit.csv
claim_ladder_final.csv
```

## Main Claim

The main supported hypothesis is:

```text
A strong coherent target text moves Gemma-3-12B-IT into a different measurable
internal state during inference, without changing the model weights.
```

This is not the trivial claim that a prompt can change the final output. The
stronger claim is that a coherent target context changes the internal geometry
of hidden states / residual stream before and during answer generation. In this
sense, the result is a context-induced latent-state shift: the shift is caused
by context, exists inside the model's computation, and is measured through
hidden-state geometry rather than only through the visible response text.

The correct boundary is:

```text
temporary inference-time hidden-state shift
not permanent weight change
not permanent model state
```

## Why This Is Not Just Lexical Or Content Overlap

The coherent target text was compared not only with a neutral control, but also
with shuffled controls:

```text
sentence_shuffle
word_shuffle
```

These controls preserve much of the vocabulary, topic, length, and local
content, but destroy coherent discourse order. If the model were only reacting
to similar words, topic, length, or lexical overlap, the coherent target and
shuffled controls should look geometrically similar. They do not.

This is the central separation result: the model is not merely seeing similar
tokens. It enters a different residual-stream configuration when the target text
is coherent.

## How The Hidden-State Coordinates Were Built

The measurement is a direct hidden-state readout, not an interpretation of the
final answer.

The script runs the same questions under multiple context conditions:

```text
target
neutral
sentence_shuffle
word_shuffle
neutral_length_matched_control
question_only
```

For each prompt, the model is run with:

```text
output_hidden_states=True
```

The script reads the final-token hidden state at every layer. In the script's
terms:

```text
hidden_states[0]  = embedding output
hidden_states[1:] = layer outputs
```

Thus the primary measurement table is not "what answer did the model give", but
"where is the model's internal state at the end of the prompt-context at each
layer".

The axes are constructed from condition differences:

```text
x_full = mean(H_target - H_neutral)
x_content = mean(H_sentence_shuffle - H_neutral)
x_order = mean(H_target - H_sentence_shuffle)
x_order_orth = x_order - proj_x_content(x_order)
```

This is the key methodological step. `x_order_orth` was not manually invented or
selected because it looked good. It is the residual component that remains after
subtracting the content-like direction from the target-vs-shuffle difference.

If a condition strongly projects onto `x_order_orth`, the meaning is not merely
"the text contains similar words". It means that the hidden-state geometry
contains a shift that remains after the content signal has been separated.

## Projection Coordinates

For every condition, layer, and question, the script computes:

```text
delta(condition, layer, question) =
H_condition(layer, question) - H_neutral(layer, question)
```

Then it projects this delta onto a chosen axis:

```text
projection_fraction =
dot(delta, axis) / dot(axis, axis)
```

The script also computes a norm-invariant angular similarity:

```text
direction_cosine =
dot(delta, axis) / (norm(delta) * norm(axis))
```

Projection measures coordinate displacement; cosine checks whether the
direction itself aligns with the axis independent of vector length.

The descriptive result is therefore geometric in a literal sense: it is based
on hidden-state vectors, differences between those vectors, projections onto
constructed internal directions, and trajectory measurements in residual-stream
space.

## Leave-One-Question-Out Readout

To avoid training an axis and testing it on exactly the same question, the
readout uses a leave-one-question-out procedure. For each held-out question, the
axis is recomputed from the other questions, and the held-out question is then
projected onto that axis.

If the target condition still projects stably onto `x_order_orth`, the result is
not simply overfitting to one question. It transfers across questions within the
experimental set.

## Generation Trajectories

After prompt-endpoint geometry, the experiment checks whether the shift persists
during generation.

The model generates an answer autoregressively. Hidden states are recorded
during generation, then the generation trajectory is projected onto the same
axes. This produces metrics such as:

```text
start projection
end projection
late-minus-early projection
mean projection
direction cosine
L2 distance to reference
```

This asks not only where the prompt endpoint lands, but whether the internal
trajectory remains shifted while the model produces the answer.

## SAE Readout

SAE readout adds an interpretability layer. Hidden states and component
directions are read through sparse autoencoder features at selected SAE layers.

The SAE layer is not the primary proof of the coordinate system. The primary
coordinates already come from the dense residual-stream hidden states. SAE is a
sparse-feature lens: if the dense shift is meaningful, part of it should appear
as contrast in sparse features.

## Key Descriptive Grade4 Result

The Grade4 decomposition separates the target and sentence-shuffle controls into
different components:

```text
target on x_order_orth = 0.909026
sentence_shuffle on x_order_orth = -0.069058

sentence_shuffle on x_content = 0.849551
target on x_content = -0.010294
```

The interpretation is direct:

```text
sentence_shuffle contains similar content and moves into x_content.
coherent target barely loads onto x_content but strongly loads onto x_order_orth.
```

Therefore the measured shift cannot honestly be reduced to content similarity.
The model is not simply reacting to a bag of similar words. It is sensitive to
coherence, order, discourse structure, and response mode.

## x_order_orth Is A Large Component, Not A Tiny Residual

`x_order_orth` is not a weak leftover after subtracting content. Its energy
fraction relative to the full component is large:

```text
middle x_order_orth_energy_fraction_of_full = 0.613503
late x_order_orth_energy_fraction_of_full = 0.564123
all x_order_orth_energy_fraction_of_full = 0.575700
```

This means the order/structure/response-mode component carries more than half
of the full target/control shift energy in middle, late, and all-band
representations. It is a large separable geometric component.

## Causal Intervention Logic

After descriptive geometry, the experiment tests whether the discovered
component directions are only readout coordinates or whether intervening along
them can move generation trajectories.

For each component axis, especially `x_order_orth` and `x_content`, the script
adds or subtracts the direction in the residual stream during generation:

```text
residual_state = residual_state + alpha * axis
residual_state = residual_state - alpha * axis
```

The causal readout is:

```text
plus_minus_projection_gap =
projection_after_plus_intervention - projection_after_minus_intervention
```

This is not an accuracy, percentage, or answer-quality score. It is a coordinate
difference in residual-stream readout space. Large values mean large
hidden-space readout shifts. They do not by themselves imply stable behavioral
control.

## Why Norm-Control Was Necessary

The raw-alpha causal comparison had a confound. With:

```text
residual_state = residual_state + alpha * vector
```

the same `alpha` does not imply equal intervention strength if the vectors have
different raw L2 norms.

The raw norms show the issue:

```text
middle x_content raw norm = 14518.902068
middle x_order_orth raw norm = 8058.432071

late x_content raw norm = 29315.891582
late x_order_orth raw norm = 14729.571563
```

`x_content` was roughly 1.8-2.0 times longer than `x_order_orth` in raw
representation. In a raw-alpha setting, `x_content` could look causally stronger
simply because a physically larger vector was added to the residual stream.

The norm-controlled run fixes this by normalizing both axes to L2 norm 1 over
the intervention band before applying the same alpha values.

The manifest records:

```text
model_id = google/gemma-3-12b-it
run_label = grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl

GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_MODE = band_l2
GRADE4_COMPONENT_CAUSAL_READOUT_USES_NORMED_AXIS = True

CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False
```

This was specifically a component-causal norm-control test. It was not a full
behavioral steering run. It tested internal generation trajectories after a
controlled residual-stream intervention.

Technically, norm-control worked:

```text
mean_intervention_axis_band_norm = 1.0
mean_effective_intervention_l2 = alpha_abs
```

## Aggregate Norm-Control Causal Result

The main aggregate result:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate = 0.527778

x_content mean causal gap = -125.128343
x_content positive rate = 0.472222

all readouts: x_order_orth beats x_content = 0.416667
matching readouts only: x_order_orth beats x_content = 0.500000
```

After an equal-energy unit-L2 comparison, `x_order_orth` is not a stable winner
over `x_content`. It has a slightly higher positive rate and a less negative
mean gap, but no pairwise dominance.

This sets the causal boundary:

```text
x_order_orth is not proven as the dominant causal component under unit-L2
norm-control.
```

## Base-Condition Asymmetry

The important structure appears when the result is split by base condition:

```text
neutral: x_order_orth beats x_content = 0.666667
neutral mean order_minus_content_gap = +354.870122

target: x_order_orth beats x_content = 0.166667
target mean order_minus_content_gap = -236.496475
```

In the neutral condition, `x_order_orth` is stronger: adding `x_order_orth` to a
neutral state more often and more strongly moves the generation trajectory than
adding `x_content`.

In the target condition, the picture reverses. Subtracting or symmetrically
intervening along `x_order_orth` from an already target-conditioned state does
not work as a clean mirror switch.

Plainly:

```text
x_order_orth works better as an injection direction from neutral than as a
stable bidirectional handle that cleanly moves into and out of the target-like
state.
```

## Concrete Large-Shift Examples

At `alpha_abs = 0.75`, the rank summary gives visible examples:

```text
neutral late x_order_orth gap = +992.518931
neutral late x_content gap = +356.819982

neutral middle x_order_orth gap = +274.611926
neutral middle x_content gap = +52.812108

target late x_content gap = -51.775190
target late x_order_orth gap = -553.394467

target middle x_content gap = +459.055941
target middle x_order_orth gap = -0.605217
```

The intervention moves internal state, but it does not provide stable
bidirectional control symmetry.

## Alpha Scaling

Alpha scaling confirms this limitation:

```text
x_order_orth signed alpha slope mean = -23.426489
x_order_orth positive slope rate = 0.250000

x_content signed alpha slope mean = -121.248341
x_content positive slope rate = 0.416667
```

The trajectories are sensitive, but signed dose-response is not stable.

## Final Scientific Status

```text
Proven: coherent target text causes a context-induced latent-state shift in
Gemma-3-12B-IT. The shift is measured in hidden-state / residual-stream
geometry, separates from shuffled-content controls, and contains a large
order/structure component x_order_orth.

Supported: causal involvement. Interventions along discovered component
directions change generation trajectories, so the directions are not purely
passive readout coordinates.

Not proven: x_order_orth as a stable bidirectional steering axis or complete
behavioral-control handle.
```

The norm-controlled causal run is an important refinement, not a rollback. The
main established result remains the internal latent-state shift induced by a
coherent target context and separated from content/shuffle controls.

## Next Experiment

The next required test is:

```text
natural-scale norm-controlled component-causal run
```

Unit-L2 intervention was fair by energy, but tiny relative to the natural raw
component norms:

```text
x_order_orth raw norm ~= 8058 in middle band
x_order_orth raw norm ~= 14730 in late band
effective intervention L2 in normctl run = 0.25, 0.50, 0.75
```

The next run should keep fairness between `x_content` and `x_order_orth`, while
restoring a natural-scale magnitude through a shared natural band norm.

## Final Compact Formulation

```text
Coherent target text induces a context-induced latent-state shift in
Gemma-3-12B-IT. This shift is measured in high-dimensional residual-stream
hidden-state geometry, separates from shuffled-content controls, and contains a
large x_order_orth order/structure component. Component interventions support
causal involvement, but x_order_orth is not yet proven to be a stable
bidirectional behavioral steering handle.
```

