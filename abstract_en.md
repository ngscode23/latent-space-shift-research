## Abstract

Title:
Context-Induced Latent-State Geometry Shifts and Sparse Response-Framing Carriers in Language Models

Extended Abstract:

Large language models are usually evaluated through final textual outputs, but
many behaviorally relevant decisions may be shaped earlier, inside
inference-time residual-stream trajectories. A model may answer cautiously,
directly, abstractly, contrastively, evasively, or with broad
averaged-recipient framing not only because a final output token is locally
preferred, but because the preceding context has moved the model into an
internal representation region where a different formulation regime is
geometrically favored. This project studies that internal movement as a
measurable latent-state phenomenon. The central claim is that coherent target
context can induce a temporary inference-time shift in hidden-state /
residual-stream geometry, and that this shift can be measured as projection
coordinates of condition deltas and generation trajectories relative to
experimentally derived target/control axes. The claim is not that the model's
weights change, nor that the model enters a permanent state after the context
is removed. The claim is about inference-time state while the target context is
present in the prompt and KV context.

The experimental object is not a single refusal string, a single political
answer, an output-only classifier, or one isolated safety feature. The object
is the geometry of the residual stream under controlled context
transformations. A coherent target text is compared against a neutral/reference
context, a sentence-shuffled version of the same target, a word-shuffled
version of the same target, and question-only or length-matched baselines. This
design asks a narrow mechanistic question: when lexical material is partially
preserved but coherent discourse order is destroyed, does the model preserve
the same hidden-state coordinate, or does it lose a specific order-dependent
internal displacement? The answer is measured directly in hidden states rather
than inferred only from visible generations.

The first line of evidence is the Grade 4 hidden-geometry decomposition. For
each question and condition, endpoint hidden states are collected across model
layers. Layerwise target/control differences are averaged into component
directions. The full target direction captures the total target-vs-reference
shift. The content direction captures the part of the shift reproduced by the
sentence-shuffled target. The order direction captures the coherent
target-vs-sentence-shuffle difference. The order-orthogonal direction removes
the sentence-shuffle/content component from the order direction layer by layer.
The resulting values are not absolute coordinates of the entire model
representation space. They are diagnostic projection coordinates: they answer
how much of a discovered target/control direction is present in a condition
delta or generated trajectory.

This distinction matters for the interpretation. The project does not claim to
map the whole latent space of Gemma or Qwen. It constructs experimentally
grounded axes for a specific context-induced phenomenon, then reads model
states relative to those axes. A high coordinate means that the condition or
trajectory lies in the same measured direction as the target-induced shift. A
low coordinate means that a control condition, despite sharing content or
surface tokens, does not reproduce that same measured direction. This is why
the sentence-shuffle and word-shuffle controls are central: they separate
lexical/content overlap from coherent discourse structure.

On Gemma3-12B-IT, the separation is clean. Coherent target context strongly
activates the full and coherent-order residual coordinates. Sentence shuffle
preserves a large content coordinate but loses the coherent-order coordinate.
The key prompt-endpoint numbers are target x_full = 0.936508 and
x_order_orth = 0.909026, compared with sentence_shuffle x_content = 0.849551
and x_order_orth = -0.069058. This shows that the sentence-shuffled control can
preserve much of the lexical/content coordinate while failing to reproduce the
coherent-order residual coordinate of the original target. The observed effect
is therefore not reducible to bag-of-words overlap.

On Qwen3.5-9B Base, the same phenomenon appears with a different weighting.
Qwen target reaches x_full = 0.973778, x_content = 0.770266,
x_order = 0.397044, and x_order_orth = 0.979462. Sentence shuffle reaches
x_content = 0.967008 but x_order_orth = 0.009969. Word shuffle reaches
x_content = 0.594366 but x_order_orth = 0.059662. This is a cross-model
replication of the hidden-state/order-readout effect: the coherent target
produces a coordinate that shuffled content does not reproduce. However, Qwen
keeps a large target content coordinate as well, so the Qwen profile is more
content-heavy than Gemma's and should not be described as a pure order-only
effect.

A complementary and axis-independent measurement on the same endpoint hidden
states characterizes not the direction of the induced shift but its dispersion.
Because this analysis never projects onto a derived target/control axis, it is
immune to any readout/intervention circularity. For each condition the spread of
the ten per-question endpoint states around their centroid is measured in two
ways: relative within-condition dispersion (within-centroid L2 normalized by
centroid norm, which removes the depth-driven growth of activation norm), and
the effective rank, or participation ratio, of the ten-state cloud, which is
scale-invariant. On Gemma3-12B-IT natscale, coherent target context produces the
most compressed late-layer state of all six conditions: averaged over layers 30
to 47 the relative within-dispersion is 0.0981 for target, against 0.1195 for
neutral and 0.1278 for question-only, with the sentence-shuffled and
word-shuffled target controls falling in between. The effective rank tells the
same story more sharply, with the ten-question cloud occupying about 1.77
effective dimensions under target context versus 3.44 under neutral at layer 41.
The between-condition to within-condition variance ratio falls from 3.16 at layer
6 to 0.50 by layer 47, so the compression is a late-network phenomenon that
emerges from roughly layer 30 and intensifies toward the logits. Two independent
and norm-invariant metrics therefore agree that the induced regime is not merely
a displaced coordinate but a more compressed, lower-dimensional one: under the
target context the otherwise question-dependent representations occupy a
narrower late-layer region. The shuffle gradient indicates that this compression is driven primarily
by target content with a smaller coherent-order contribution, consistent with
the content/order weighting reported above. As with the rest of the descriptive
geometry this is an inference-time, context-present measurement; whether
alignment-time weight changes are responsible is addressed directly by a
base-versus-instruct comparison.

That comparison runs the base pretrained model gemma-3-12b-pt and the
instruction-tuned gemma-3-12b-it on identical raw inputs (no chat template,
because the base model has none) over five matched conditions (coherent target,
neutral, word- and sentence-shuffled target, bare questions) and ten held-out
analytic questions distinct from the natscale set. The naive form of the
weight-level hypothesis — that alignment training globally suppresses
hidden-state dispersion — is rejected by this data: in the late band (layers 30
to 47) the instruction-tuned model is more dispersed than the base model under
every condition, by 18 to 30 percent under the context conditions and by 133
percent for bare questions (0.063 base versus 0.147 instruct). What alignment
changes is the organization of that variance, in three ways. First, the late
state becomes sharply context-governed: adding a shared context passage reduces
relative within-dispersion by about 2 percent in the base model but by about 48
percent in the instruction-tuned model, so the context-induced compression that
defines the regime effect is amplified by more than an order of magnitude at
alignment time. Second, alignment compresses the effective dimensionality of
context-conditioned states: late-band effective rank falls from 6.27 to 4.50
for the target condition (a 28 percent reduction), with smaller reductions for
the shuffle controls and bare questions, while the neutral condition is left
essentially unchanged (3.84 versus 3.77). Third, the target-versus-neutral
compression gap itself is already present in the base model (about 13 percent
in this raw-prompt setting, versus about 4 percent for the instruction-tuned
model in the same template-free format), so target-content-induced compression
is a pretraining-era phenomenon that alignment reorganizes and re-weights
rather than creates; how that gap behaves under the deployment chat format is
exactly what the template-controlled repeat below is for. This comparison has stated limits: it uses no chat template,
which places the instruction-tuned model outside its deployment format and may
inflate its bare-question dispersion; it uses a different question set than the
natscale measurement; and it is a single run with ten questions per condition.
The weight-level statement supported by the current evidence is therefore not
that alignment suppresses variance, but that alignment makes the late hidden
state strongly context-governed and lower-rank under coherent context while
leaving neutral-context geometry largely untouched.

The second line of evidence is generation-trajectory readout. A prompt-endpoint
shift alone could be a static context signature. The generation readout asks
whether the internal state movement persists into answer formation. The same
component axes are used to measure where the model's hidden states move while
it generates. In both Gemma and Qwen runs, generated trajectories can be read
as coordinates on the discovered axes. This connects the descriptive prompt
geometry to response construction: the model is not merely storing a signature
of the context at the end of the prompt; the induced state can be tracked as the
answer unfolds.

The third line of evidence is norm-controlled component intervention. The
experiment injects positive and negative versions of discovered component
directions into selected layer bands during generation, then reads out the
resulting trajectory movement. The intervention is sign-symmetric and
energy-controlled, so content and coherent-order components can be compared
under matched intervention strength. The causal question is precise: does
adding or subtracting the component direction move generation trajectories
along the same measured coordinate?

For Gemma3-12B-IT, the answer is strongly positive for both content and
coherent-order components. Across readout cells, x_content produces mean
plus-minus projection gap 27352.919286 with positive rate 0.944444, while
x_order_orth produces mean gap 19284.481823 with positive rate 0.861111. In
matching readout cells, both components reach positive rate 1.0, with x_content
mean gap 37883.852822 and x_order_orth mean gap 34227.185962. The strongest
late-to-late target x_order_orth intervention has plus = 21222.761008,
minus = -62859.822710, and gap = 84082.583718. This supports causal
involvement: the discovered coherent-order direction is not only a passive
readout, because intervention on that direction changes generation
trajectories.

For Qwen3.5-9B Base, the causal profile is positive but more balanced and more
content-heavy. Across readout cells, x_content mean gap is 41.878616 and
x_order_orth mean gap is 38.246761, both with positive rate 1.0. In matching
readouts, x_content mean gap is 73.851162 and x_order_orth mean gap is
72.449630. These values show that both components move trajectories under
intervention. Pairwise comparisons do not support an order-orthogonal
dominance claim: x_order_orth beats x_content in only 0.166667 of all-readout
comparisons and 0.166667 of matching-readout comparisons. The causal statement
for Qwen is therefore deliberately specific: Qwen replicates hidden-state shift
and order-sensitive readout, and both content and coherent-order components are
causally involved, but the coherent-order component is not the dominant
steering axis in the current intervention setting.

The fourth line of evidence connects the geometry to sparse autoencoder
features. The Grade 4 axes are dense residual-stream directions. Sparse
autoencoders offer a different view: they decompose residual-stream states into
large sets of sparse features with decoder directions. If the coherent-order
shift is mechanistically meaningful, some SAE features should show elevated
activation deltas on the coherent-order component, semantically coherent
top-activating contexts, and nontrivial downstream effects under ablation or
steering.

In Gemma-Scope, the SAE readout has high reconstruction quality: mean
reconstruction cosine 0.996023 and mean explained-variance proxy 0.991462. Top
order-relevant candidates include layer 31 feature 58, layer 42 feature 29,
layer 42 feature 13686, and layer 42 feature 208. These candidates are treated
as sparse directions linked to response-framing dynamics. The labels are
operational, not ontological. Feature 208 is a contrastive-framing candidate
because its contexts and steering effects involve opposition, qualification,
contrast, and rhetorical framing. Feature 13686 is an abstract-epistemic
candidate because it influences generalized explanatory posture rather than
narrow answer formulation.

In Qwen-Scope, all 32 SAE specifications were computed successfully for
Qwen3.5-9B Base, with hidden size 4096, SAE width 65536, TopK = 50, mean
reconstruction cosine 0.966660, and mean explained-variance proxy 0.933639.
The feature table contains 1503 rows, with 575 cases where
order_abs_gt_content_abs. Top Qwen order-specific candidates include layer 27
feature 65254, layer 23 feature 51987, layer 27 feature 5335, and layer 28
feature 28136. These are sparse diagnostic candidates: they mark places where
the coherent-order component appears concentrated enough to justify direct
activation-context inspection, ablation, and steering tests.

The Qwen mini-check protocol then moves from association to intervention. For
a candidate feature, the script encodes the residual state with Qwen-Scope SAE,
modifies the selected feature activation, decodes the difference, and applies
an SAE-delta patch to the residual stream. This patch does not replace the
residual stream with a full SAE reconstruction; it adds only the decoded
feature delta. This avoids making a single-feature ablation depend on full-SAE
reconstruction error. The measured mediation effect is the residual-stream
displacement caused by the feature ablation. Downstream diagnostics measure
sequence loss change, final-token logit L2, final-token KL(base || patched),
and token-level loss deltas.

The strongest current Qwen downstream candidate is layer 28 feature 41435.
Ablation produces mediated_effect = 77.897545, loss_delta = +1.342655,
final-token logit L2 = 574.866821, and KL(base || patched) = 0.700875. A
second strong candidate is layer 24 feature 47391, with mediated_effect =
30.897112, loss_delta = +0.140961, final-token logit L2 = 528.348450, and
KL(base || patched) = 0.529381. The token-level loss table shows that the
effect is not merely a uniform increase in random token damage. The largest
changes localize around spans semantically aligned with the hypothesized
response-framing regime: averaged-recipient language, safety/default framing,
caution, objection avoidance, directness, and precision tradeoffs.

The fifth line of evidence is direct decoder-direction steering. Instead of
removing a feature contribution, the steering scripts inject selected SAE
decoder directions at chosen scales during generation and teacher-forced
evaluation. This probes whether a sparse decoder direction can modulate
formulation dynamics. The measured outputs include free-generation
differences, final next-token KL, logit displacement, top-token changes, and
teacher-forced per-token KL. These metrics separate visible text differences
from distributional movement: a feature can change token probabilities even
when the surface text remains superficially similar, and token-level KL can
reveal where the continuation distribution diverges.

The interpretation emerging from the combined evidence is a latent
epistemic-posture / addressee-selection mechanism. The model may move between
a more concrete-user, direct-answer posture and a more averaged-recipient,
safety-weighted, heavily qualified posture. This is not identical to refusal,
political bias, or a single safety feature. It is a broader
response-construction axis: how the model decides whether to answer as if
addressing this concrete user with a specific claim, or as if addressing a
generalized recipient under a default risk-minimizing discourse policy. The
relevant behavior includes hedging, both-sides neutrality, soft deflection,
corporate-style non-commitment, contrastive explanation, abstract
justification, and loss of direct addressability. The dispersion measurement
gives this posture a geometric signature: the compressed, lower-dimensional
late-layer regime is what a restrained, averaged-recipient response mode looks
like in state space — a mode that admits fewer effective degrees of freedom
than a direct, concrete-user answer, collapsing otherwise question-specific
states toward a common guarded configuration before the logits are formed.


This framing explains why the same latent phenomenon can appear across
political prompts, normative prompts, and apparently stylistic prompts. The
important variable is not the topic category alone. It is the response regime
selected by the internal trajectory. A context that shifts the residual stream
toward a safety-weighted averaged-recipient region can make direct answers less
natural and broad qualified answers more natural. Conversely, an intervention
that moves the trajectory away from that region may increase directness,
contrast, or specificity, depending on which sparse direction is involved.

The contribution is therefore a multi-level mechanistic evidence chain. Grade
4 geometry establishes that coherent target context changes hidden-state
coordinates relative to target/control axes. Shuffle controls show that the
coherent-order coordinate is not reducible to content overlap. Generation
readouts show that the shift is present during answer formation.
Norm-controlled interventions show causal involvement of the component
directions. SAE readouts identify sparse candidate carriers. SAE-delta
ablations show residual mediation and downstream distributional effects.
Token-level loss localization tests whether the perturbation affects
semantically relevant spans. Decoder-direction steering tests whether selected
sparse directions can actively modulate formulation dynamics.

The compact empirical chain is:

    target discourse coherence
        -> residual-stream coordinate shift
        -> compression into a lower-dimensional late-layer region
        -> coherent-order component separation from content controls
        -> generation-trajectory movement
        -> component-level causal involvement
        -> sparse SAE carrier candidates
        -> feature-level mediation
        -> loss/logit/KL effects
        -> localized response-framing changes.

This evidence structure is relevant to mechanistic interpretability because it
treats internal trajectory geometry as a measurable object rather than
inferring model state only from final text. It is also relevant to agent
safety. For an ordinary chat model, a hidden-state shift is an interpretability
result. For an LLM agent, the same phenomenon becomes a safety-relevant object
because agents plan, call tools, write memory, select actions, and make
intermediate commitments from internal states before final output. Output-only
evaluation is late: it observes the visible surface after the internal
transition has already happened. The safety question becomes not only "what did
the model say?", but "which internal response-construction mode did it enter
before acting?"

The current evidence supports context-induced latent-state shift,
content/order separation through shuffle controls, generation-trajectory
readout, causal involvement of component directions, sparse candidate
carriers with downstream loss/logit effects, and a base-versus-instruct
dispersion contrast showing that alignment amplifies context-governed
compression and lowers the effective rank of context-conditioned late states
while the compression phenomenon itself predates alignment. It does not
establish permanent model change, a universal model-independent mechanism,
dominance of the coherent-order component over the content component in all
causal settings, or that the base-versus-instruct contrasts survive a
chat-template-controlled repeat, the original natscale question set, and
replication on other model families. The
next step is to convert local evidence into a stronger transfer claim through
held-out prompts without the source target text, neutral matched controls,
monotonic scale sweeps, negative-control features, lexical-set logit probes,
feature-composition interactions, search for an opposite direct-answer /
concrete-user direction, a chat-template-controlled repeat of the
base-versus-instruct dispersion comparison using the original natscale
questions, replication of the dispersion and effective-rank measurement on
Qwen, and replication on additional model families. If
these checks hold, hidden-state trajectory monitoring becomes a practical
alignment object: alignment should ask not only whether the final response is
acceptable, but also whether the model entered a policy-relevant, unstable,
goal-shifted, or high-impact internal mode before acting.








