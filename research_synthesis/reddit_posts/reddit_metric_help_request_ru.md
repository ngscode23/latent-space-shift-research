# Reddit Post EN: help interpreting context-induced hidden-state shift metrics

## Title

```text
Feedback wanted: context-induced hidden-state geometry shifts in Gemma/Qwen
```

## Post

```text
Hi everyone,

I am working on a mechanistic interpretability / hidden-state geometry project,
and I would like technical criticism from people who work with residual-stream
geometry, activation analysis, causal interventions, PCA/state-space readouts,
generation trajectories, and SAE-based interpretability.

The question is not whether text can change a model's final output. That is
obvious. The question is whether a strong coherent context can move a model
into a different measurable inference-time hidden-state / residual-stream
region before the final answer is produced, without changing model weights.

The control structure compares:

- question-only / baseline conditions;
- neutral or reference context;
- coherent target context;
- sentence-shuffled target context;
- word-shuffled target context.

The goal is to separate lexical/content overlap from coherent discourse/order
structure. If the effect is only caused by shared words, length, or ordinary
semantic content, then the coherent target and the shuffled controls should
look similar in hidden-state geometry. If coherent discourse order induces a
different internal response mode, then the coherent target should separate from
the shuffled controls along a different latent coordinate.

The current readout constructs layerwise target/control axes:

- x_full: target minus reference;
- x_content: sentence-shuffle minus reference;
- x_order: target minus sentence-shuffle;
- x_order_orth: x_order with the sentence-shuffle/content component removed
  layer by layer.

The reported values are projection coordinates, not absolute latent-space
positions. They measure how much of a discovered target/control direction is
present in a condition delta or generated trajectory.

Current runs:

- Gemma3-12B-IT;
- Qwen3.5-9B-Base with Qwen-Scope SAE.

The central descriptive result is that coherent target context strongly
projects onto the coherent-order residual coordinate, while sentence-shuffled
controls preserve much of the content coordinate but largely lose that
coherent-order coordinate.

Gemma3-12B-IT:

target:
  x_full       = 0.936508
  x_order_orth = 0.909026

sentence shuffle:
  x_content    = 0.849551
  x_order_orth = -0.069058

Qwen3.5-9B-Base:

target:
  x_full       = 0.973778
  x_content    = 0.770266
  x_order_orth = 0.979462

sentence shuffle:
  x_content    = 0.967008
  x_order_orth = 0.009969

word shuffle:
  x_content    = 0.594366
  x_order_orth = 0.059662

This is the main result I want stress-tested: sentence-shuffled controls can
preserve content coordinates while failing to reproduce the coherent-order
residual coordinate of the original target. My current interpretation is that
the model did not merely see similar words; the coherent target appears to move
the residual stream into a different measurable internal configuration.

I also ran component-level causal interventions by injecting positive and
negative versions of discovered directions during generation and reading out
plus/minus trajectory gaps.

Qwen3.5-9B-Base:

all readout cells:
  x_content mean plus/minus gap     = 41.878616
  x_order_orth mean plus/minus gap  = 38.246761
  positive gap rate                 = 1.0 for both

matching readouts:
  x_content mean gap                = 73.851162
  x_order_orth mean gap             = 72.449630

Gemma3-12B-IT:

all readout cells:
  x_content mean plus/minus gap     = 27352.919286
  x_order_orth mean plus/minus gap  = 19284.481823

matching readouts:
  x_content mean gap                = 37883.852822
  x_order_orth mean gap             = 34227.185962

So my causal claim is deliberately narrow: the coherent-order component is
descriptively separable and causally involved in trajectory movement, but I am
not claiming that it is the dominant steering axis over the content component.

I then connected the dense geometry to SAE diagnostics. For Qwen3.5-9B-Base /
Qwen-Scope SAE:

SAE reconstruction cosine mean       = 0.966660
explained-variance proxy mean        = 0.933639
SAE specs computed                   = 32 / 32
hidden size                          = 4096
SAE width                            = 65536
TopK                                 = 50

Candidate features were tested with SAE-delta patching:

  h_patched = h + SAE_decode(a_patched) - SAE_decode(a_original)

So the ablation does not replace the residual stream with a full SAE
reconstruction. It only adds the decoded feature delta.

The strongest current Qwen downstream candidate is:

layer 28 / feature 41435:
  mediated_effect          = 77.897545
  sequence loss_delta      = +1.342655
  final-token logit L2     = 574.866821
  KL(base || patched)      = 0.700875

Second candidate:

layer 24 / feature 47391:
  mediated_effect          = 30.897112
  sequence loss_delta      = +0.140961
  final-token logit L2     = 528.348450
  KL(base || patched)      = 0.529381

Token-level loss localization places large patch-worse deltas around spans
that look related to averaged-recipient framing, safety/default framing,
caution as a default response mode, objection avoidance, and directness /
precision tradeoffs. I am treating these as candidate sparse carriers for a
formulation or epistemic-posture regime, not as final universal feature names.

The AI safety angle is that output-only evaluation may be late. If an agent's
hidden trajectory shifts before planning, tool selection, self-monitoring, or
memory writes, final-answer evaluation may observe the symptom after the
decision state has already happened. For a chat model this is an
interpretability result; for agentic systems it may become a safety-relevant
object.

What I want is a hard critique, not agreement. In particular:

1. Does this target / reference / sentence-shuffle / word-shuffle decomposition
   make sense as a control structure for separating content overlap from
   coherent discourse/order structure?

2. Is projection_fraction(delta, axis) = dot(delta, axis) / dot(axis, axis) a
   reasonable coordinate readout here, or should the geometry be framed
   differently?

3. What controls are still missing to rule out length, rhetorical intensity,
   prompt artifacts, position effects, or ordinary semantic similarity?

4. What are the most likely failure modes in constructing x_order_orth by
   removing the sentence-shuffle/content component layerwise?

5. For causal interventions, what would be the cleanest dose-response /
   sign-symmetry test?

6. For the SAE feature readout, what negative controls would you require?
   Random features? content-heavy features? same-layer matched-norm features?
   features with similar activation frequency?

7. What would make this evidence convincing to a mechanistic interpretability
   audience: activation patching, ablation, steering, held-out prompt transfer,
   lexical-set logit probes, more model families, or something else?

Current claim boundary:

I am claiming that coherent target context can induce a measurable
inference-time hidden-state / residual-stream geometry shift; that this shift
can be read as projection coordinates relative to target/control-derived axes;
that sentence-shuffled controls preserve content signal while largely losing
the coherent-order residual coordinate; and that component interventions and
SAE mini-checks provide partial causal and sparse-carrier evidence.

I am not claiming permanent weight change, universal behavior control, a
complete theory of refusal/safety behavior, or proof that any single SAE
feature has a final universal semantic label.

If anyone is interested, I can share the technical note, scripts, CSV summaries,
PDFs, and metric artifacts. I am mainly looking for hard criticism of the
evidence structure and suggestions for stronger controls.
```

## Short Version

```text
I am looking for hard technical criticism of a hidden-state geometry result.

Question: can a strong coherent context move an LLM into a different measurable
inference-time residual-stream state before the final answer, without changing
weights?

I compare target, reference, sentence-shuffled target, and word-shuffled target
conditions. The key readout constructs target/control axes and measures
projection coordinates. In Gemma3-12B-IT and Qwen3.5-9B-Base, the coherent
target strongly projects onto a coherent-order residual coordinate, while the
sentence-shuffled control preserves content signal but mostly loses that
coordinate.

Gemma:
target x_order_orth = 0.909026
sentence_shuffle x_content = 0.849551
sentence_shuffle x_order_orth = -0.069058

Qwen:
target x_order_orth = 0.979462
sentence_shuffle x_content = 0.967008
sentence_shuffle x_order_orth = 0.009969

Component interventions show causal involvement of both content and
coherent-order directions, without proving coherent-order dominance.
Qwen-Scope SAE mini-checks identify candidate sparse carriers; the strongest
current feature is layer 28 / feature 41435, with mediated_effect = 77.897545,
loss_delta = +1.342655, final-token logit L2 = 574.866821, and KL(base ||
patched) = 0.700875 under ablation.

I want critique on whether this control structure and projection-coordinate
readout are valid, what controls are missing, what SAE negative controls are
needed, and what causal experiment would make the result convincing.
```
