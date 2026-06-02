# GitHub Discussion Draft: Context-Induced Hidden-State Geometry Shifts in LLMs

## Suggested Title

```text
Feedback wanted: context-induced hidden-state geometry shifts and order-controlled latent axes in LLMs
```

## Short Copy-Paste Version

Use this version if the GitHub venue prefers compact discussion posts.

```text
Hi everyone,

I am working on a mechanistic interpretability project about context-induced
hidden-state geometry shifts in language models, and I would like technical
feedback from people who work on activation analysis, causal interventions, and
SAE-based interpretability.

The core question is whether a strong coherent context can move a model into a
different measurable inference-time residual-stream / hidden-state region
before we look at the final visible output.

The experiment builds latent directions from target/control differences and
projects prompt-endpoint hidden states and generation trajectories onto those
directions. The control structure uses target, neutral/reference,
sentence-shuffled target, and word-shuffled target conditions. This separates
the full target-induced shift, the lexical/content component captured by
sentence-shuffled controls, and the remaining coherent-discourse/order component
after content is removed.

Current runs:

- Gemma3-12B-IT
- Qwen3.5-9B-Base with Qwen-Scope SAE

The central readout is that coherent target context strongly projects onto the
coherent-order residual direction, while sentence-shuffled controls preserve
much of the lexical content but largely lose that coordinate.

Example coordinates:

Gemma3-12B-IT:
- target coherent-order residual coordinate: 0.909026
- sentence-shuffle coherent-order residual coordinate: -0.069058
- sentence-shuffle content coordinate: 0.849551

Qwen3.5-9B-Base:
- target coherent-order residual coordinate: 0.979462
- sentence-shuffle coherent-order residual coordinate: 0.009969
- sentence-shuffle content coordinate: 0.967008

So the sentence-shuffle controls preserve content but do not reproduce the same
coherent-order coordinate as the original target context.

I also ran component-level causal interventions and SAE feature-level checks.
For Qwen-Scope SAE, the current strongest downstream candidate is:

- layer 28 / feature 41435
- mediated_effect: 77.897545
- sequence loss_delta under ablation: +1.342655
- final-token logit L2: 574.866821
- KL(base || patched): 0.700875

Token-level loss localization for this and related features concentrates around
spans involving averaged-recipient framing, safety/default framing, caution as a
default response mode, objection avoidance, and directness/precision tradeoffs.

I am treating these as candidate sparse carriers for a formulation /
epistemic-posture regime, not as fully named universal features.

I would especially value criticism on:

1. Does the target / neutral / sentence-shuffle / word-shuffle decomposition
   make sense as a control structure for separating content overlap from
   coherent discourse/order structure?
2. Is the projection-coordinate readout a reasonable way to describe the
   internal state shift?
3. What controls would be necessary to make this evidence publication-grade?
4. What are the most likely failure modes in constructing the order-residual
   direction?
5. What negative controls would you want for the SAE feature-level readout?
6. Is this better framed as mechanistic interpretability, representation
   dynamics, agent-safety evaluation, or something else?

Current claim boundary:

Strong coherent context can induce a measurable inference-time hidden-state /
residual-stream geometry shift. The shift can be read as coordinates relative
to experimentally discovered latent directions, and sentence-shuffle controls
indicate that the coherent-order component is not reducible to lexical content
alone.

I am not claiming permanent weight change, universal behavior control, or that
the coherent-order direction is the dominant causal steering axis.

If anyone is interested, I can share the technical note, scripts, CSV summaries,
and metric artifacts. I am mainly looking for hard criticism of the evidence
structure and suggestions for stronger controls.
```

## Post

Hi everyone,

I am working on a mechanistic interpretability project about **context-induced hidden-state geometry shifts** in language models, and I would like critical feedback from people with experience in activation analysis, causal interventions, and SAE-based interpretability.

The core question is:

```text
Can a strong coherent context move a model into a different measurable
inference-time residual-stream / hidden-state region, before we look at the
final visible output?
```

The experiment builds latent directions from target/control differences and then measures prompt-endpoint hidden states and generation trajectories as projections onto those directions.

The main decomposition separates:

```text
1. the full target-induced shift;
2. the lexical/content component captured by sentence-shuffled controls;
3. the remaining coherent-discourse / order component after the content
   component is removed.
```

In other words, the goal is not to show that “the output changed”. The goal is to test whether a coherent context changes the model’s internal trajectory in residual-stream representation space, and whether that shift can be separated from ordinary lexical/content overlap.

## Current Evidence

I have run this on:

```text
Gemma3-12B-IT
Qwen3.5-9B-Base with Qwen-Scope SAE
```

The strongest descriptive result is that the coherent target context projects strongly onto the coherent-order component, while sentence-shuffled controls preserve much of the lexical content but largely lose that coordinate.

For Qwen3.5-9B-Base, the prompt-endpoint projection readout is:

```text
target:
  full target axis                  = 0.973778
  content / sentence-shuffle axis   = 0.770266
  coherent-order residual axis      = 0.979462

sentence-shuffle control:
  content / sentence-shuffle axis   = 0.967008
  coherent-order residual axis      = 0.009969

word-shuffle control:
  content / sentence-shuffle axis   = 0.594366
  coherent-order residual axis      = 0.059662

question-only:
  coherent-order residual axis      = -0.305250
```

So the sentence-shuffle control keeps a strong content coordinate but does not reproduce the coherent-order coordinate of the original target context.

For Gemma3-12B-IT, the separation is even cleaner:

```text
target:
  full target axis                  = 0.936508
  content / sentence-shuffle axis   = -0.010294
  coherent-order residual axis      = 0.909026

sentence-shuffle control:
  content / sentence-shuffle axis   = 0.849551
  coherent-order residual axis      = -0.069058
```

This is the central result I am trying to stress-test:

```text
coherent target context appears to induce a measurable internal state shift
that is not reducible to content overlap alone.
```

## Causal / Intervention Evidence

I also ran component-level interventions along the discovered directions.

For Qwen3.5-9B-Base:

```text
all readout cells:
  content-axis mean plus/minus gap          = 41.878616
  coherent-order-axis mean plus/minus gap   = 38.246761
  positive gap rate                         = 1.0 for both

matching readouts only:
  content-axis mean gap                     = 73.851162
  coherent-order-axis mean gap              = 72.449630
```

For Gemma3-12B-IT:

```text
content-axis mean plus/minus gap            = 27352.919286
coherent-order-axis mean plus/minus gap     = 19284.481823

matching readouts:
  content-axis mean gap                     = 37883.852822
  coherent-order-axis mean gap              = 34227.185962
```

The causal result is not that the coherent-order direction dominates the content direction. It does not. The current interpretation is narrower:

```text
the coherent-order component is descriptively separable and causally involved,
but not yet established as the dominant steering axis.
```

## SAE Feature-Level Readout

For Qwen3.5-9B-Base / Qwen-Scope SAE:

```text
SAE reconstruction cosine mean       = 0.966660
explained variance proxy mean        = 0.933639
SAE specs computed                   = 32 / 32
hidden size / SAE input dimension    = 4096
SAE feature dimension                = 65536
```

I then tested candidate sparse features from the order-related readout using feature ablation / reconstruction patching.

The current strongest Qwen downstream candidate is:

```text
layer 28 / feature 41435
  mediated_effect          = 77.897545
  sequence loss_delta      = +1.342655
  final-token logit L2     = 574.866821
  KL(base || patched)      = 0.700875
```

A second strong candidate is:

```text
layer 24 / feature 47391
  mediated_effect          = 30.897112
  sequence loss_delta      = +0.140961
  final-token logit L2     = 528.348450
  KL(base || patched)      = 0.529381
```

Token-level loss localization for these features shows the largest patch-worse deltas around spans involving:

```text
averaged-recipient framing
safety/default framing
caution as a default response mode
objection avoidance
directness / precision tradeoffs
```

I am treating these as **candidate sparse carriers for a formulation / epistemic-posture regime**, not as fully named universal features.

## Why I Think This Might Matter

The safety/alignment angle is not only about final text.

For ordinary chat models, this is an interpretability result. For LLM agents, it may become more important because planning, tool use, memory writes, and intermediate commitments can depend on internal trajectories before the final answer is produced.

So the broader question is:

```text
Should internal trajectory shifts be treated as an alignment object,
instead of evaluating only final visible outputs?
```

## What I Would Like Feedback On

I would especially value criticism on the following:

1. Does the target / neutral / sentence-shuffle / word-shuffle decomposition make sense as a control structure for separating content overlap from coherent-order structure?

2. Is the projection-coordinate readout a reasonable way to describe the internal state shift, or should it be framed differently?

3. What controls would be necessary to make this evidence publication-grade?

4. Are there obvious failure modes in the way the order-residual direction is constructed?

5. For the causal interventions, what would be the cleanest test of bidirectional symmetry and dose-response?

6. For the SAE feature-level readout, what would be the right negative controls?

7. Would this be better framed as mechanistic interpretability, agent-safety evaluation, representation dynamics, or something else?

## Current Claim Boundary

The current claim is:

```text
Strong coherent context can induce a measurable inference-time hidden-state /
residual-stream geometry shift. The shift can be read as coordinates relative
to experimentally discovered latent directions, and sentence-shuffle controls
indicate that the coherent-order component is not reducible to lexical content
alone.
```

The current claim is not:

```text
permanent weight change;
universal behavior control;
a complete theory of refusals or safety behavior;
proof that the coherent-order direction is the dominant causal steering axis;
proof that any individual SAE feature has a final universal semantic label.
```

If anyone is interested, I can share the technical note, scripts, CSV summaries, and metric artifacts. I am mainly looking for hard criticism of the evidence structure and suggestions for stronger controls.

Thanks.
