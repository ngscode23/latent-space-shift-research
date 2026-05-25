# OLMo2 Heldout Causal Mediation Readout

Date: 2026-05-20

## Setup

```text
model_id = allenai/OLMo-2-1124-13B-Instruct
run_tag = heldout
texts_per_kind = 9
max_tokens = 3070
selected_hidden_indices = [35, 34, 40]
truncated_rows = 0
```

Provenance note:

```text
This run was produced by the pre-patch causal_mediation_v1 script copy in
Colab, so run_metadata.json does not yet include input_texts_path,
input_primary_control_mode, or input_control_texts_source.

Interpret these files as the OLMo2 heldout mediation bundle supplied by the
user. For a final reviewer artifact, rerun with the patched script or keep the
Colab command log next to these files.
```

## Status

```text
СИЛЬНО ПОДДЕРЖАНО:
  OLMo2 target_control interventions move margins in the expected direction in
  all four main cells, with bootstrap lower bounds above zero.

ИНТЕРЕСНО, НО ГРЯЗНО:
  The effect is not cleanly specific to target_control. Confidence intervals
  overlap random_same_norm, shuffled_label, or wrong_layer controls in most
  comparison cells.
```

## What The Data Show

Natural target-control gaps are present but smaller than Qwen/Ministral:

```text
agent_action natural gap mean = -2.170
agent_action mean abs natural gap = 2.205

blind_semantic natural gap mean = -3.268
blind_semantic mean abs natural gap = 3.495
```

The target-control vector moves readouts in the expected direction:

```text
agent_action control_plus:
  target_control hidden_index=40 alpha=1.0 observed=1.071
  CI [0.301, 2.107]

agent_action target_minus:
  target_control hidden_index=40 alpha=1.0 observed=0.836
  CI [0.041, 2.022]

blind_semantic control_plus:
  target_control hidden_index=40 alpha=1.0 observed=0.251
  CI [0.170, 0.346]

blind_semantic target_minus:
  target_control hidden_index=34 alpha=1.0 observed=0.541
  CI [0.296, 0.856]
```

But target_control does not cleanly separate from matched controls:

```text
agent_action control_plus:
  target_control = 1.071 [0.301, 2.107]
  random_same_norm = 0.306 [0.054, 0.590]
  shuffled_label = 0.123 [-0.079, 0.395]
  wrong_layer = 0.592 [0.169, 1.129]
  CI overlap with all three controls.

agent_action target_minus:
  target_control = 0.836 [0.041, 2.022]
  random_same_norm = 0.101 [-0.007, 0.239]
  shuffled_label = 0.405 [-0.101, 1.074]
  wrong_layer = 0.711 [0.164, 1.437]
  CI overlap with all three controls.

blind_semantic control_plus:
  target_control = 0.251 [0.170, 0.346]
  random_same_norm = 0.201 [0.170, 0.234]
  shuffled_label = -0.049 [-0.081, -0.020]
  wrong_layer = 0.148 [0.099, 0.206]
  CI overlap with random_same_norm and wrong_layer; no overlap with shuffled_label.

blind_semantic target_minus:
  target_control = 0.541 [0.296, 0.856]
  random_same_norm = 0.089 [0.041, 0.157]
  shuffled_label = 0.272 [0.183, 0.385]
  wrong_layer = 0.382 [0.223, 0.584]
  no CI overlap with random_same_norm; CI overlap with shuffled_label and wrong_layer.
```

## Mechanistic Interpretation

OLMo2 is not a Ministral-style null result. The target-control vector has a
real directional effect in OLMo2.

But it is not a clean Qwen-style causal-handle replication either:

```text
The intervention effect is positive, but matched controls also move margins.
Wrong-layer and random controls are not inert enough to claim a specific
single-direction mechanism.
```

The current three-model pattern is:

```text
Qwen:
  cleanest single-direction action-policy mediation signal.

Ministral:
  strong natural gaps, but raw target-control vector does not beat controls.

OLMo2:
  positive target-control directional mediation signal, but specificity is
  contaminated by overlapping control intervals.
```

## Claim Impact

Upgrade:

```text
The Qwen single-vector causal result is no longer completely isolated:
OLMo2 shows directional target-control mediation.
```

Do not upgrade to:

```text
Qwen and OLMo2 share a clean single-direction causal handle.
```

Correct wording:

```text
Qwen has the cleanest current single-direction causal-handle evidence.
OLMo2 shows preliminary directional support, but not clean specificity against
matched controls. Ministral does not support the raw-vector handle.
```

## Next Test

The next useful mechanism test is not another natural-gap run.

Run distributed/subspace mediation:

```text
1. Build rank-k target-control subspaces from paired hidden differences or
   margin-trained directions.
2. Fit on 8 text pairs, test on the held-out pair.
3. Compare against random same-rank, shuffled-label same-rank, and wrong-layer
   same-rank controls.
4. Run the same protocol on Qwen, Ministral, and OLMo2.
```

This directly tests whether the mechanism is:

```text
single-direction in Qwen;
distributed/subspace in Ministral;
positive but nonspecific perturbation sensitivity in OLMo2;
or genuinely model-specific across all three.
```
