# Grade4 Geometry -> SAE Steering: Unified Readout

## Main Answer

Yes: Grade4 shows that the model changes the coordinates of its inference-time
hidden state inside an experimentally constructed latent-axis space.

Strict formulation:

```text
Grade4 shows that coherent target context induces a measurable shift in the
model's residual-stream / hidden-state geometry. This shift is quantified as
coordinates of condition deltas and generation trajectories projected onto the
discovered axes x_full, x_content, x_order, and x_order_orth.
```

Important boundary:

```text
This is not an absolute coordinate system of the whole model.
This is not a permanent weight change.
This is not a permanent model state.
```

It is a coordinate system inside an experimental subspace defined by
target/control differences.

## What Grade4 Proves

### 1. The target condition has a coordinate shift

Qwen3.5-9B Base / Qwen-Scope Grade4:

```text
target:
  x_full       = 0.973778
  x_content    = 0.770266
  x_order      = 0.397044
  x_order_orth = 0.979462
```

Meaning:

```text
The target condition almost fully projects onto the discovered x_full and
x_order_orth coordinates.
```

Mechanistic interpretation:

```text
The target text does not merely add words to the prompt. It moves the
final-prompt hidden state into a different region of residual-stream
representation space.
```

### 2. This is not merely content or lexical overlap

Sentence-shuffle control:

```text
sentence_shuffle:
  x_content    = 0.967008
  x_order_orth = 0.009969
```

Word-shuffle control:

```text
word_shuffle:
  x_content    = 0.594366
  x_order_orth = 0.059662
```

Question-only:

```text
question_only:
  x_order_orth = -0.305250
```

Main interpretation:

```text
sentence_shuffle preserves content but loses the coherent-order coordinate.
target preserves content and strongly expresses the coherent-order coordinate.
```

This is the key separation result. It shows that coherent target structure is
read by a different internal coordinate than simple word/content overlap.

### 3. Qwen's content-heavy profile does not cancel the shift

Qwen has a high content component:

```text
target x_content = 0.770266
```

But at the same time:

```text
target x_order_orth = 0.979462
sentence_shuffle x_order_orth = 0.009969
```

Correct interpretation:

```text
Qwen is content-heavy, but the coherent-order readout is still cleanly
separable from shuffled content.
```

Qwen is weaker than Gemma as a pure order-dominance case, but strong as a
cross-model replication of hidden-state / order-readout geometry.

## Component Norms: How Much Energy Is Content And Order

Qwen Grade4 component norms:

```text
middle:
  content_energy_fraction_of_full    = 0.882215
  order_orth_energy_fraction_of_full = 0.394951

late:
  content_energy_fraction_of_full    = 0.881487
  order_orth_energy_fraction_of_full = 0.369194

all:
  content_energy_fraction_of_full    = 0.882916
  order_orth_energy_fraction_of_full = 0.373893
```

Interpretation:

```text
Qwen's target-induced state is strongly content-bearing, but a substantial
orthogonal order component remains. That component is measurable and is not
absorbed by sentence-shuffle content.
```

This matters for academic framing. Qwen should not be presented as a clean
order-only model. Its contribution is:

```text
replication with content-heavy geometry
```

## Causal Involvement: What The Grade4 Intervention Block Adds

Qwen component causal result:

```text
all readout cells:
  x_content mean gap    = 41.878616
  x_order_orth mean gap = 38.246761
  positive gap rate     = 1.0 for both
```

Matching readout:

```text
x_content mean gap    = 73.851162
x_order_orth mean gap = 72.449630
```

Max-alpha matching examples:

```text
neutral late/late x_order_orth:
  plus = 196.416635
  minus = 33.149257
  gap = 163.267378

neutral late/late x_content:
  plus = 117.799171
  minus = -43.962590
  gap = 161.761760

target late/late x_order_orth:
  plus = 205.458359
  minus = 43.861111
  gap = 161.597248

target late/late x_content:
  plus = 141.075956
  minus = -27.942299
  gap = 169.018255
```

Strict interpretation:

```text
Both x_content and x_order_orth are causally active under norm-controlled
intervention. x_order_orth is involved, but Qwen does not support a strong
claim that x_order_orth causally dominates x_content.
```

Thus Grade4 gives:

```text
descriptive coordinate proof + causal involvement
```

but it does not require the stronger claim:

```text
complete behavioral control by x_order_orth
```

## SAE Evidence: What Qwen-Scope Adds

SAE health:

```text
SAE specs computed = 32/32
model_id = Qwen/Qwen3.5-9B-Base
hidden_size = 4096
sae_d_in = 4096
sae_d_sae = 65536
top_k = 50
reconstruction cosine mean = 0.966660
explained_variance_proxy mean = 0.933639
```

Top Qwen order-specific candidates from Grade4:

```text
layer 27 feature 65254:
  x_order_orth_delta = -22.089539
  order_specific_score = 22.367545

layer 23 feature 51987:
  x_order_orth_delta = -8.362167
  order_specific_score = 14.773435

layer 27 feature 5335:
  x_order_orth_delta = -7.184792
  order_specific_score = 13.976547

layer 28 feature 28136:
  x_order_orth_delta = 3.726776
  order_specific_score = 8.050881
```

Meaning:

```text
The SAE layer does not prove the geometry shift by itself. The geometry shift
is already supported by hidden-state coordinate projections. SAE adds candidate
sparse carriers: features that may participate in the shifted formulation /
order state.
```

## How This Connects To SAE Steering

Main chain:

```text
Grade4:
  target context -> hidden-state coordinate shift

SAE readout:
  shifted coordinate system -> candidate sparse features

SAE steering:
  candidate decoder directions -> local modulation of formulation trajectory
```

Feature steering does not need to claim that features contain political
positions. The correct statement is:

```text
Some SAE decoder directions participate in formulation dynamics: contrastive
framing, epistemic abstraction, qualification, negation, and continuation
stability.
```

This is downstream evidence. It supports the idea that the shifted latent state
is not merely a metaphor: it has directions/features that can be perturbed and
measured through generation and KL.

## Intuitive Formulation

The intuitive user-level formulation was:

```text
target context moves the model into a region of latent-state space where these
kinds of answers are normal.
```

More academic formulation:

```text
target context shifts the model into a region of residual-stream representation
space where continuations with a different formulation regime are more
latent-compatible with the current generation trajectory.
```

This does not mean the model acquired a belief or a permanent state. It means
the current inference trajectory is parameterized differently.

## Final Unified Claim

```text
Grade4 proves the core geometry result: coherent target context induces a
measurable inference-time hidden-state shift, expressed as coordinates on
latent axes in residual-stream representation space. Shuffled controls show
that this coordinate shift is not reducible to lexical/content overlap. Causal
interventions show involvement of the discovered components, while SAE readouts
and steering experiments identify candidate sparse directions that locally
modulate the formulation trajectory downstream of that shifted state.
```

Short version:

```text
Grade4 supports the main result: coherent target context changes the coordinates
of the model's internal state in hidden-state / residual-stream space.
Sentence-shuffle and word-shuffle controls show that this shift is not reducible
to words. Causal intervention shows involvement of the discovered components,
and SAE steering shows that part of the downstream formulation dynamics can be
locally modulated through sparse decoder directions.
```

