# Read-only audit: abstract / math / claims for latent-state geometry paper draft

Дата: 2026-06-02

Цель этого документа: сверить набросок abstract + mathematical formulation с тем,
что реально делают скрипты, CSV и экспериментальная логика проекта.

Проверенные источники:

```text
model_workspaces/qwen3_5_9b_qwen_scope/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
model_workspaces/qwen3_5_9b_qwen_scope/steering/qwen35_9b_sae_mediation_top_k.py
model_workspaces/qwen3_5_9b_qwen_scope/steering/mini_check_protocol/README_RU.md
research_context_current.md
```

## 1. Главный verdict

Набросок концептуально сильный и хорошо звучит в стиле paper про measurement
geometry. Но математический блок нужно привести к реальной реализации.

Главная поправка:

```text
Реальный Grade 4 не строит одну глобальную Phi(c) и не ортогонализует target
direction против span(sentence_shuffle, word_shuffle).

Реальный Grade 4 строит layerwise axes:

x_full       = mean_q [H_target(q) - H_reference(q)]
x_content    = mean_q [H_sentence_shuffle(q) - H_reference(q)]
x_order      = mean_q [H_target(q) - H_sentence_shuffle(q)]
x_order_orth = x_order with the x_content component removed layerwise
```

Вторая важная поправка:

```text
Координаты в CSV являются projection_fraction:

projection_fraction(delta, axis) = <delta, axis> / <axis, axis>

Это не просто dot product абсолютного hidden state с axis.
```

Третья важная поправка:

```text
Qwen SAE mini-check uses SAE-delta patching:

patched = residual + (decode(acts_patched) - decode(acts_orig))

It does not replace the whole residual stream with an SAE reconstruction.
```

Это надо явно написать, потому что иначе reviewer может подумать, что весь
эффект загрязнён SAE reconstruction error.

## 2. Что в abstract уже хорошо

Сильные и корректные элементы:

```text
1. Main object is not final textual output, but inference-time residual-stream
   / hidden-state trajectory.

2. Target / neutral / sentence-shuffle / word-shuffle controls are the right
   framing.

3. The claim that sentence-shuffle preserves lexical/content material but loses
   the coherent-order coordinate is supported by Gemma and Qwen Grade 4.

4. The agent-safety framing is valid: output-only evaluation is late if planning,
   tool use, memory writes, or intermediate commitments depend on hidden states.

5. SAE features should be framed as candidate sparse carriers, not as final
   universal semantic features.
```

Strong supported one-sentence claim:

```text
Coherent target context induces a measurable inference-time residual-stream
geometry shift, and this shift can be read as coordinates relative to
experimentally derived target/control axes.
```

## 3. Claims that need correction

### 3.1 "Qwen features + steering + teacher-forced KL"

Draft phrase:

```text
In Qwen3.5-9B-Base, selected SAE features ... Steering selected decoder
directions further changes formulation dynamics, as measured by output deltas,
final next-token KL, and teacher-forced per-token KL.
```

Audit:

```text
This mixes two evidence layers.
```

What is actually established for Qwen mini-check:

```text
Qwen:
  - SAE mediation
  - top activating contexts
  - sequence loss_delta
  - final-token logit L2
  - KL(base || patched) at final token
  - token-level loss localization
```

What is established in the separate steering/KL package:

```text
Gemma/Gemma-Scope features 208 and 13686:
  - generation metrics
  - final next-token KL
  - teacher-forced per-token KL
```

Correct wording:

```text
For Qwen, feature ablation provides residual-stream, loss/logit, and
token-localization evidence. In a separate feature-steering setting, selected
decoder directions also change formulation dynamics measured by generation
metrics, final next-token KL, and teacher-forced per-token KL.
```

### 3.2 "Feature alignment by correlation"

Draft formula:

```text
rho_{ell,j} = Corr(A_{ell,j}(c), z_ord(c))
```

Audit:

```text
This is a reasonable future metric, but it is not the main metric currently
computed in the Qwen/Gemma Grade 4 SAE tables.
```

What is actually computed:

```text
sae_order_feature_contrast / order feature tables use component feature deltas:
  x_content_component_delta
  x_order_orth_component_delta
  order_minus_content_abs_component_delta
  order_specific_score
  status labels such as order_specific_generation_persistent_feature
```

Correct wording:

```text
Feature-level readout is based on component activation deltas and ranking
scores, with top-activating contexts and ablation diagnostics used as
downstream validation. Correlation with z_ord can be introduced as an additional
future validation metric, not as the current selection rule.
```

### 3.3 "Localization ratio R_loc"

Draft:

```text
R_loc = sum_{t in S_target} max(delta_t,0) / sum_t max(delta_t,0)
```

Audit:

```text
This is a good proposed metric, but it is not currently computed in the sent
token_level_loss_delta_by_feature.csv. Current localization is table/ranking
based, not an explicit ratio over annotated spans.
```

Correct wording:

```text
The current run reports token-level patch-worse / patch-better rankings. A
future annotated-span metric can summarize these rankings as R_loc.
```

### 3.4 "Lexical-set logit probes"

Draft formulas:

```text
P_r(c) = sum_{v in V_r} p(v | c)
Delta P_r = ...
```

Audit:

```text
These are not currently part of the Grade 4 or Qwen mini-check outputs.
They are a good next experiment.
```

Correct placement:

```text
Future Work / Proposed diagnostics.
```

### 3.5 "Monotonic steering criterion"

Draft:

```text
SpearmanCorr(alpha, Q(alpha))
```

Audit:

```text
Grade 4 component causal alpha scaling exists for component axes. Separate
feature steering also has scale sweeps. But a general monotonicity criterion
over lexical/behavioral metrics is not uniformly established across Qwen SAE
mini-checks.
```

Correct placement:

```text
Use as future criterion for clean steering, not as current completed result.
```

## 4. Corrected mathematical formulation aligned with the scripts

This is the version that should replace the current math block.

### 4.1 Residual-state observations

Let \(M_\theta\) be a transformer language model. For question \(q\), condition
\(c\), layer \(\ell\), and the selected endpoint position, let

```text
h_{\ell}(c,q) in R^d
```

denote the prompt-endpoint residual-stream / hidden-state vector. In the Grade 4
scripts, prompt geometry is primarily computed from endpoint states across all
hidden-state layers, with layer 0 corresponding to the embedding state and model
block layer \(b\) mapping to hidden-state index \(b+1\) for SAE readouts.

For a reference condition \(r\), define the layerwise condition delta:

```text
Delta_ell(c,q; r) = h_ell(c,q) - h_ell(r,q).
```

In the Qwen and Gemma Grade 4 runs discussed here, \(r\) is the neutral/reference
condition when the neutral condition is enabled.

### 4.2 Grade 4 component axes

Let \(s\) denote the sentence-shuffled target condition. The experiment defines
four layerwise axes:

```text
x_full,ell =
  mean_q [ h_ell(target,q) - h_ell(reference,q) ]

x_content,ell =
  mean_q [ h_ell(sentence_shuffle,q) - h_ell(reference,q) ]

x_order,ell =
  mean_q [ h_ell(target,q) - h_ell(sentence_shuffle,q) ]
```

The order-orthogonal axis is computed layerwise by removing the projection of
\(x_order\) onto \(x_content\):

```text
x_order_orth,ell =
  x_order,ell
  -
  ( <x_order,ell, x_content,ell> / <x_content,ell, x_content,ell> )
  x_content,ell.
```

If \(x_content,\ell\) has near-zero norm, the implementation keeps
\(x_order,\ell\) unchanged for that layer.

Important:

```text
word_shuffle is a control condition in the readout, but it is not part of the
current x_order_orth orthogonalization formula. x_order_orth removes the
sentence-shuffle/content component layerwise.
```

### 4.3 Projection coordinates

For an axis \(u_\ell\) and condition delta \(\Delta_\ell(c,q;r)\), the main
coordinate is the projection fraction:

```text
P_ell(c,q; u) =
  <Delta_ell(c,q;r), u_ell> / <u_ell, u_ell>.
```

The script computes this through:

```text
projection_fraction(delta, direction)
```

where

```text
projection_fraction(delta, direction)
  = dot(delta, direction) / dot(direction, direction).
```

The published coordinate for a condition/axis is an aggregate over questions
and selected layers:

```text
P(c; u) = mean_{q, ell in band} P_ell(c,q; u).
```

A separate direction cosine is also computed in some tables:

```text
cos(delta, u) = <delta, u> / (||delta|| ||u||).
```

Do not conflate the two:

```text
projection_fraction says "how much of this axis is present in the delta";
cosine says "how aligned are the two directions regardless of magnitude".
```

### 4.4 Central separation criterion

The core descriptive Grade 4 criterion is:

```text
P(target; x_order_orth) is high,
while
P(sentence_shuffle; x_order_orth) is near zero or much lower,
even when
P(sentence_shuffle; x_content) is high.
```

For Qwen3.5-9B Base:

```text
target:
  x_content    = 0.770266
  x_order_orth = 0.979462

sentence_shuffle:
  x_content    = 0.967008
  x_order_orth = 0.009969
```

For Gemma3-12B-IT:

```text
target:
  x_content    = -0.010294
  x_order_orth = 0.909026

sentence_shuffle:
  x_content    = 0.849551
  x_order_orth = -0.069058
```

This is the mathematically clean claim:

```text
sentence-shuffled controls can preserve content coordinates while failing to
reproduce the coherent-order residual coordinate of the original target.
```

### 4.5 Component causal intervention

For component causal runs, the intervention injects \(+\alpha u\) or
\(-\alpha u\) into selected layer bands during generation. With norm control,
axes are normalized over the intervention band and optionally rescaled to a
shared natural band norm.

The causal readout is not "the model obeys the axis." It is:

```text
The generated trajectory shows a measurable plus/minus projection gap on the
same or selected readout axis.
```

For an intervention axis \(u\), sign \(s in {+1,-1}\), and generated state
\(g_{\ell,t}^{(s)}\), the readout coordinate is:

```text
P_generation(s; u_readout) =
  mean_{t,ell in readout band}
  < g_{ell,t}^{(s)} - h_ell(reference,q), u_readout,ell >
  / < u_readout,ell, u_readout,ell >.
```

The plus/minus causal gap is:

```text
Gap(u) = P_generation(+; u_readout) - P_generation(-; u_readout).
```

The supported causal claim:

```text
x_order_orth is causally involved in trajectory movement.
```

The unsupported stronger claim:

```text
x_order_orth is the dominant causal steering axis over x_content.
```

Qwen and Gemma both show causal involvement; neither cleanly proves
x_order_orth dominance over x_content.

## 5. Corrected SAE formulation aligned with Qwen mini-checks

### 5.1 Qwen-Scope SAE encoding

For a Qwen-Scope SAE at model block layer \(b\), the encoder computes:

```text
pre = h W_enc^T + b_enc
a = TopK(ReLU(pre), k=50)
```

The current Qwen scripts use:

```text
SAE_TOP_K = 50
SAE repo = Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
d_model = 4096
d_sae = 65536
```

The feature activation for feature \(j\) is zero unless \(j\) survives the TopK
gate for that token.

### 5.2 Exact ablation / patching operation

Let \(a = SAEEnc(h)\). For a feature set \(J\), define \(a^{patch}\) by replacing
\(a_j\) with `patch_value` for \(j in J\). The actual Qwen patching operation is:

```text
h_patched =
  h + SAEDec(a_patch) - SAEDec(a).
```

For one feature and a linear decoder, this is equivalent to adding:

```text
(patch_value - a_j) d_j
```

but the implementation should be described in delta form because that is what
prevents whole-SAE reconstruction error from contaminating the patch.

### 5.3 Mediation score actually computed

In `qwen35_9b_sae_mediation_top_k.py`, the main mediated effect is computed on
the selected residual state after patching:

```text
M_{b,j} =
  mean_batch || resid_base,b - resid_patched,b ||_2.
```

In the main top-k mediation run, this is a final-token residual displacement at
the patched model layer for the prompt batch.

Therefore avoid writing:

```text
M = E_{c,t}[ ... all tokens ... ]
```

unless the specific script actually computes all-token mediation.

Correct wording:

```text
Mediation is a residual-stream displacement under feature ablation. In the
current Qwen top-k run it is computed at the selected layer and endpoint
position; downstream mini-checks then inspect all-token contexts and
token-level losses.
```

### 5.4 Downstream loss / logit diagnostics

The Qwen mini-checks compute:

```text
baseline sequence loss
patched sequence loss
loss_delta = patched_loss - baseline_loss
last_logit_l2 = ||logits_patched,last - logits_base,last||_2
KL_last = D_KL(p_base,last || p_patched,last)
token_loss_delta_t = loss_patched,t - loss_base,t
```

Interpretation:

```text
loss_delta > 0:
  ablation makes the observed text harder to predict.

loss_delta < 0:
  ablation makes the observed text easier to predict.

large last_logit_l2 / KL_last:
  the feature changes the token distribution.

large token_loss_delta localized to relevant spans:
  stronger evidence that the feature participates in the hypothesized
  formulation regime rather than causing arbitrary degradation.
```

Current Qwen strongest downstream feature:

```text
layer 28 / feature 41435
mediated_effect = 77.897545
loss_delta = +1.342655
last_logit_l2 = 574.866821
KL_last = 0.700875
```

Second strong feature:

```text
layer 24 / feature 47391
mediated_effect = 30.897112
loss_delta = +0.140961
last_logit_l2 = 528.348450
KL_last = 0.529381
```

## 6. Corrected paper-style abstract

This version keeps the complex style but matches the actual evidence.

```text
Large language models are usually evaluated through final textual outputs, but
many behaviorally relevant decisions may be shaped by inference-time
residual-stream trajectories before the final answer is emitted. A model may
produce a cautious, balanced, or safety-weighted response either because such
language is locally preferred at the output layer, or because preceding context
has moved the model into an internal representation region where that
formulation regime is geometrically favored. This distinction is especially
important for LLM agents, whose planning, tool use, memory writes, and
intermediate commitments may depend on hidden-state trajectories rather than
only on final visible text.

We study this problem from a latent-state geometry perspective. The central
question is whether coherent discourse context induces a measurable
inference-time shift in residual-stream representation space, and whether this
shift can be read as coordinates relative to experimentally derived
target/control axes. The experimental design compares target, neutral,
sentence-shuffled, and word-shuffled contexts. This separates the full
target-induced displacement from a content component captured by
sentence-shuffled controls and from a coherent-order residual obtained after
removing the sentence-shuffle/content component layerwise.

Across Gemma3-12B-IT and Qwen3.5-9B-Base, coherent target contexts strongly
project onto the coherent-order residual coordinate, while sentence-shuffled
controls preserve much of the lexical content but largely lose that coordinate.
This provides a controlled internal readout of context-induced state movement
rather than a final-output-only comparison. Component-level interventions show
that the discovered directions are causally involved in generation trajectories,
although the current evidence does not establish the coherent-order component
as the dominant steering axis over the content component.

We then connect the geometric readout to sparse autoencoder diagnostics. In
Qwen3.5-9B-Base with Qwen-Scope SAE, selected late-layer features show
candidate carrier behavior: they mediate residual-stream displacement, activate
on semantically aligned spans, and produce downstream loss/logit effects under
feature ablation. The strongest current Qwen candidate, layer 28 feature 41435,
produces a sequence loss increase of +1.342655 under ablation, a final-token
logit L2 shift of 574.866821, and KL(base || patched) of 0.700875. Token-level
loss localization places the largest effects around averaged-recipient,
safety/default, caution, objection-avoidance, and directness/precision spans.

The interpretation is local. We do not claim permanent weight change, universal
behavior control, or a model-independent safety mechanism. The evidence
supports a narrower claim: coherent context can induce a measurable
inference-time hidden-state geometry shift; this shift can be separated from
lexical content overlap by order-controlled controls; and selected SAE decoder
directions provide candidate sparse carriers for the resulting
response-framing dynamics under the tested prompt regimes.
```

## 7. Corrected contribution statement

Use this contribution block instead of the current broader version:

```text
1. We introduce an order-controlled latent geometry readout for language
   models. Target, neutral, sentence-shuffled, and word-shuffled conditions
   separate content overlap from coherent discourse/order structure.

2. We define layerwise component axes:
   x_full, x_content, x_order, and x_order_orth. The last is computed by
   removing the layerwise x_content projection from x_order.

3. We show cross-model evidence that coherent target context induces a
   measurable residual-stream coordinate shift: Gemma3-12B-IT shows the
   cleanest content/order separation; Qwen3.5-9B-Base replicates the
   hidden-state/order-readout phenomenon with a more content-heavy profile.

4. We run component-level causal interventions showing causal involvement of
   both content and coherent-order components in generation trajectories,
   while not claiming coherent-order causal dominance.

5. We connect the geometry to SAE feature diagnostics. Qwen-Scope mini-checks
   identify candidate sparse carriers whose ablation affects residual states,
   sequence loss, final-token logits/KL, and token-level losses localized to
   semantically relevant response-framing spans.
```

## 8. Terms to use consistently

Use:

```text
residual-stream representation space
inference-time hidden-state shift
target/control-derived latent axes
projection coordinates
coherent-order residual component
candidate sparse carriers
response-framing dynamics
epistemic-posture / addressee-selection hypothesis
```

Avoid or qualify:

```text
"the model enters a permanent state"              -> no, only inference-time
"universal safety feature"                       -> no, candidate local feature
"x_order_orth is dominant steering axis"          -> not supported
"feature contains political position"             -> no
"the coordinate is absolute latent-space position" -> no, relative projection
"correlation-selected feature"                    -> only if that metric is actually run
```

## 9. Clean final claim

The clean claim that matches code and CSV:

```text
We show that coherent target context induces a measurable inference-time
residual-stream geometry shift in Gemma3-12B-IT and Qwen3.5-9B-Base. This shift
is measured as projection coordinates relative to layerwise target/control axes.
Sentence-shuffled controls preserve lexical/content signal but do not reproduce
the coherent-order residual coordinate of the original target. Component
interventions show causal involvement of the discovered axes in generation
trajectories. Qwen-Scope SAE mini-checks further identify candidate sparse
carriers whose ablation changes residual-stream states, downstream token
distributions, and token-level losses on semantically relevant formulation
spans.
```

