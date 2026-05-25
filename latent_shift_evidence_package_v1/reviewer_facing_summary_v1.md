# Reviewer-Facing Summary v1

Date: 2026-05-20

## One-Sentence Claim

Structured target contexts induce measurable, context-conditioned shifts in
hidden-state geometry, neutral semantic/logit readouts, persistence/rejection
behavior, order/dose response, and controlled fake-agent action-choice margins
in the tested open instruct models.

This is a mechanistic-interpretability / state-space claim. It is not a claim
about consciousness, real external tool use, irreversible attractors, or a
universal law across all language models.

## Core Mechanism Hypothesis

```text
structured context
  -> distributed hidden-state regime
  -> measurable geometry shift
  -> semantic/logit/action-policy readout shift
  -> partial persistence and partial causal steerability
```

The texts are controlled induction stimuli. The measured object is not the
specialness of the texts themselves, but the latent/readout regime induced by
context.

## Evidence Chain

| Link | Status | What supports it |
| --- | --- | --- |
| Hidden geometry separation | ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ | Qwen, Ministral, and OLMo2 show late-layer target/control separation and 1.0 linear probe accuracy. |
| Clean blind semantic shift | ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ | Clean blind readouts remain positive across Qwen, Ministral, and OLMo2 with bootstrap lower bounds above zero. |
| Neutral-turn persistence | ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ | Final-turn residuals remain positive in all three heldout model families, with strong decay. |
| Rejection/reset residual | СИЛЬНО ПОДДЕРЖАНО | Explicit rejection reduces the effect but does not erase it in the tested runs. |
| Order/path sensitivity | СИЛЬНО ПОДДЕРЖАНО | Control->target histories move strongly toward target; target->control histories fall back toward control. |
| Dose and suffix sensitivity | СИЛЬНО ПОДДЕРЖАНО | Mixed contexts show target-fraction response; target suffix is consistently stronger than target prefix. |
| Controlled fake-agent action margins | ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ | Fake-action direct-vs-procedural margins shift after target context and persist after neutral/rejection turns. |
| Hard-control specificity | ИНТЕРЕСНО, НО ГРЯЗНО | Strong in Qwen and Ministral heldout, weaker in OLMo2; selfref specificity fails on Ministral. |
| Partial causal mediation | ИНТЕРЕСНО, НО ГРЯЗНО | Qwen heldout layer-32 target-control vector gives the cleanest action-policy handle. OLMo2 shows positive target-control intervention in all four cells, but with control-overlap caveats. Ministral does not replicate raw-vector mediation. |

## Key Heldout Results

| Metric | Qwen3-14B | Ministral 3 14B | OLMo2 13B |
| --- | --- | --- | --- |
| Best hidden contrast / mean norm | 0.352 | 0.356 | 0.662 |
| Best linear probe accuracy | 1.000 | 1.000 | 1.000 |
| Clean blind semantic mean abs | 26.106 [23.425, 28.782] | 7.616 [6.531, 8.717] | 1.929 [1.587, 2.239] |
| Blind persistence, final turn | 6.218 [4.781, 7.788] | 2.087 [1.606, 2.563] | 0.450 [0.367, 0.525] |
| Rejection residual, final turn | 3.688 [2.960, 4.510] | 0.946 [0.800, 1.091] | 0.316 [0.261, 0.376] |
| Agent-loop no rejection, final turn | 6.081 [5.226, 6.930] | 5.288 [4.511, 6.107] | 1.939 [1.574, 2.272] |
| Agent-loop after rejection, final turn | 2.476 [2.082, 2.910] | 2.038 [1.661, 2.371] | 1.527 [1.223, 1.822] |
| Hard-control specificity ratio | 1.871 [1.667, 2.218] | 2.351 [2.184, 2.593] | 1.206 [0.957, 1.665] |
| Control->target order fraction | 0.947 [0.901, 1.001] | 0.828 [0.772, 0.895] | 0.814 [0.721, 1.201] |
| 50% target-suffix mix | 0.707 [0.638, 0.772] | 0.717 [0.600, 0.821] | 0.799 [0.688, 1.018] |

## Cross-Corpus Readout

Selfref/mirror and heldout procedural-risk corpora both induce measurable
shifts. This matters because it prevents the project from collapsing into the
narrow claim that self-reference alone is doing the work.

Main readout:

```text
Selfref is a mirror/self-model pressure line.
Heldout is the cleaner reviewer-facing line.
Both are controlled induction stimuli for the broader state-space phenomenon.
```

Important caveat: in the Ministral selfref run, hard-control specificity fails.
The pressure-style control reproduces much of the selfref effect. This weakens
any claim that the original selfref texts are uniquely special, but it
strengthens the broader claim that rhetorical topology and procedural pressure
are active induction ingredients.

## Reviewer Objections Already Addressed

| Objection | Current answer |
| --- | --- |
| Qwen-only artifact | Heldout structure also appears in Ministral and OLMo2. |
| Single-text driver | Leave-one-text-out keeps core effects nonzero. |
| Random target/control pairing | Exact paired sign-flip tests pass for key metrics. |
| Candidate-token problem | Candidate-token diagnostics show problem_count = 0 in core heldout runs. |
| Truncation artifact | Core raw files show zero truncated rows in heldout runs. |
| A/B label-position artifact | Normal/reversed mappings are checked; dirty rows are excluded from clean summaries. |
| Only semantic-probe artifact | Controlled fake-agent action-choice margins shift too. |
| Only descriptive geometry | Qwen heldout causal mediation shows partial intervention effect on action margins; OLMo2 gives preliminary positive directional mediation; Ministral shows the raw-vector handle is not consistently cross-model. |

## What The Current Package Can Claim

Use this:

```text
The tested target contexts induce measurable context-conditioned
representational shifts in open instruct models. These shifts are visible in
late-layer geometry, clean semantic/logit readouts, persistence after neutral
turns, residuals after explicit rejection, order/dose sensitivity, and
controlled fake-action choice margins. The effect is replicated across Qwen3,
Ministral 3, and OLMo2 on the heldout corpus, with strong model-dependent
magnitude differences.
```

Do not use this:

```text
This proves a conscious state.
This proves a universal mechanism of all LLMs.
This proves a true dynamical attractor in the strict mathematical sense.
This proves real deployed-agent tool behavior.
This proves one hidden vector fully explains all downstream effects.
This proves self-reference alone is the mechanism.
```

## Minimal Next Experiment

Do not add more generic metrics to the large runner.

The next experiment that actually changes the mechanism claim is:

```text
Build distributed/subspace mediation v2:
  rank-k target-control subspace;
  leave-one-text-out fitting;
  blind semantic + agent-action readouts;
  random same-rank, shuffled-label same-rank, and wrong-layer controls.
```

Decision rule:

```text
If rank-k subspace mediation works in Ministral and/or OLMo2 while beating
matched controls, the distributed discourse-regime mechanism becomes directly
supported.

If rank-k subspace mediation also fails, keep the causal-mechanism claim
model-specific: Qwen cleanest, OLMo2 preliminary/nonspecific, Ministral null
for raw-vector mediation.
```

## Source Files

```text
latent_shift_evidence_package_v1/claim_register.csv
latent_shift_evidence_package_v1/metric_validity_audit.md
cross_model_comparison_heldout_v1/cross_model_comparison.md
cross_corpus_comparison_v1/cross_corpus_comparison.md
reviewer_robustness_audit_v1/reviewer_robustness_audit.md
latent_shift_evidence_package_v1/causal_mediation/qwen3_14b_heldout/causal_mediation_v1_report.md
latent_shift_evidence_package_v1/causal_mediation/olmo2_heldout/mediation_readout.md
latent_shift_evidence_package_v1/causal_mediation/ministral_heldout/mediation_readout.md
```
