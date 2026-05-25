# Ministral Heldout Causal Mediation Readout

Date: 2026-05-20

## Setup

```text
model_id = mistralai/Ministral-3-14B-Instruct-2512-BF16
run_tag = heldout
texts_per_kind = 9
max_tokens = 3070
selected_hidden_indices = [14, 13, 40]
truncated_rows = 0
input_texts_path = latent_shift_evidence_package_v1/input_texts_heldout.json
input_primary_control_mode = content_matched
input_control_texts_source = auto:content_matched
```

Baseline/control-source check:

```text
The natural gaps below are from the content-matched heldout control set, not
from the legacy repetitive baseline.

Main Ministral run metadata:
  primary_control_mode = content_matched
  control source = auto:content_matched

Causal mediation command used:
  INPUT_TEXTS_PATH=latent_shift_evidence_package_v1/input_texts_heldout.json

That input JSON also records:
  primary_control_mode = content_matched
  control_texts_source = auto:content_matched
```

## Status

```text
НЕ ПОДДЕРЖАНО:
  broad claim that the raw target-control hidden vector is a cross-model
  causal handle for downstream margins.

ИНТЕРЕСНО, НО ГРЯЗНО:
  Ministral interventions show that the readouts are perturbation-sensitive,
  but target_control does not beat random/shuffled controls.
```

## What The Data Show

Natural target-control gaps are strong:

```text
agent_action natural gap mean = -6.219
blind_semantic natural gap mean = -11.349
```

So the run is not failing because Ministral lacks the measured heldout shift.
It fails specifically at the intervention/mediation step.

Best target-control rows:

```text
agent_action control_plus:
  target_control hidden_index=40 alpha=1.0 observed=0.074
  CI [0.045, 0.107]
  random_same_norm at same layer observed=0.163

agent_action target_minus:
  target_control hidden_index=40 alpha=1.0 observed=0.010
  CI [-0.024, 0.043]
  random_same_norm at same layer observed=0.119

blind_semantic control_plus:
  target_control hidden_index=40 alpha=1.0 observed=0.089
  CI [0.081, 0.096]
  random_same_norm observed=0.092

blind_semantic target_minus:
  target_control hidden_index=40 alpha=1.0 observed=0.094
  CI [0.086, 0.102]
  shuffled_label observed=0.111
```

Target-control never beats the best matched control in the four main
readout/intervention cells.

## Mechanistic Interpretation

The Ministral heldout result separates two claims:

```text
1. Context induces a hidden/readout/action shift.
   Supported by the main heldout run.

2. The mean target-control vector is a clean causal handle for that shift.
   Not supported in this Ministral mediation run.
```

This suggests that the causal handle is model- and layer-dependent. Qwen has a
useful layer-32 action-policy handle. Ministral does not expose the same effect
through the tested raw centroid vector. The relevant state may be distributed,
nonlinear, or not captured by a single mean contrast vector.

## Mechanism Fork

The negative single-vector result has two live interpretations:

```text
Variant 1:
  Ministral has the same broad context-induced regime phenomenon, but the
  causal handle is distributed across a subspace rather than captured by one
  centroid direction.

Variant 2:
  Ministral's natural gaps are functionally similar to Qwen's gaps, but they
  arise from a different underlying mechanism. Similar readout numbers do not
  by themselves prove mechanistic identity.
```

Current status:

```text
Variant 1 is plausible and fits the distributed discourse-regime hypothesis,
but it is not proven by single-vector failure.

Variant 2 is also live. The current evidence supports cross-model functional
similarity, not same-mechanism equivalence.
```

The next mechanism test should distinguish these variants directly:

```text
1. Build a rank-k target-control subspace from paired hidden differences or
   margin-trained directions.
2. Intervene with that subspace on blind semantic and agent-action readouts.
3. Compare against random same-rank, shuffled-label same-rank, and wrong-layer
   same-rank controls.
4. Use leave-one-text-out fitting: train subspace on 8 text pairs, test on the
   held-out pair.
```

Decision rule:

```text
If rank-k Ministral subspace mediation recovers a meaningful fraction of the
natural gap and beats matched random/shuffled subspaces, then the distributed
mechanism interpretation becomes supported.

If rank-k subspace mediation also fails, then the safer interpretation is that
Ministral shares the functional readout pattern but not the same tested causal
geometry as Qwen.
```

## Claim Impact

Keep:

```text
context-induced latent/readout/action shift across Qwen, Ministral, and OLMo2
```

Narrow:

```text
partial causal mediation by raw target-control vector
```

to:

```text
Qwen-heldout action-policy mediation only, pending better causal handles or
model-specific mediation tests.
```

## Next Test

Do not repeat the same raw-vector mediation on Ministral unchanged.

More useful next options:

```text
1. OLMo2 heldout mediation as another architecture check.
2. A model-specific learned direction for Ministral, trained on natural margins.
3. Layer-local causal scan with smaller alpha values and per-layer normalization.
```
