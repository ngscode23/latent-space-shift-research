# Latent Shift Evidence Package v1

This is the frozen v1 evidence package for the held-out latent/logit/action
shift result.

The package does not introduce a new model run. It consolidates the current
Qwen heldout, Ministral heldout, and OLMo2 heldout evidence into a claim-level
map that can be used for internal decisions, report writing, and
reviewer-facing robustness checks.

## Research Frame

This project is not trying to prove that a particular text corpus is special.
The texts are controlled induction stimuli. The target object is:

```text
context-induced latent regime formation
```

In mechanistic-interpretability terms, the project studies distributed
state-space behavior: a structured context induces a measurable hidden-state
regime, and that regime has downstream semantic, logit, persistence, rejection,
order/dose, and controlled fake-action readouts.

Short spine:

```text
structured context
  -> distributed hidden-state regime
  -> measurable geometry shift
  -> semantic/logit/action-policy readout shift
  -> partial persistence and partial causal steerability
```

## Core Claim

Target contexts induce a measurable shift in model hidden-state geometry. That
shift has downstream readouts in logit/semantic preference margins, persists
partly through neutral turns, is reduced but not erased by rejection/reset
instructions, shows order and dose sensitivity, and reaches controlled
fake-agent action-choice margins.

This is not a claim about consciousness, subjective state, irreversible
attractor dynamics, or real external tool execution.

## Status

```text
Internal research status:
  ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ

Reviewer-facing status:
  СИЛЬНО ПОДДЕРЖАНО
```

The difference matters. Internally, the direction is strong enough to guide the
research program. Publicly, the claim should stay limited to measurable
context-induced hidden/logit/action readout shifts in the tested models.

## Evidence Spine

```text
target context
  -> hidden geometry separation
  -> clean blind semantic readout shift
  -> persistence after neutral filler turns
  -> residual after explicit rejection/reset
  -> order and suffix/dose sensitivity
  -> controlled fake-agent action-policy drift
  -> cross-model replication in Qwen3-14B, Ministral 3 14B, and OLMo2 13B
  -> partial causal mediation in Qwen3-14B heldout action-policy margins
  -> preliminary positive OLMo2 target-control mediation with control-overlap caveat
  -> negative Ministral raw-vector mediation check
```

## Key Held-Out Numbers

| Claim piece | Qwen heldout | Ministral heldout | Status |
| --- | --- | --- | --- |
| Hidden cosine distance at best layer | 0.0617 | 0.0634 | geometry replicated |
| Contrast over mean norm | 0.3524 | 0.3565 | geometry replicated |
| Best linear probe accuracy | 1.000 | 1.000 | separable in hidden space |
| Clean blind semantic readout | 26.106 [23.425, 28.782] | 7.616 [6.531, 8.717] | cross-model supported |
| Blind persistence, final turn | 6.218 [4.781, 7.788] | 2.087 [1.606, 2.563] | persists after neutral turns |
| Rejection residual, final turn | 3.688 [2.960, 4.510] | 0.946 [0.800, 1.091] | reduced, not erased |
| Agent-loop no rejection, final turn | 6.081 [5.226, 6.930] | 5.288 [4.511, 6.107] | behavioral policy readout |
| Agent-loop after rejection, final turn | 2.476 [2.082, 2.910] | 2.038 [1.661, 2.371] | residual action-policy effect |
| Hard-control specificity ratio | 1.871 [1.667, 2.218] | 2.351 [2.184, 2.593] | original target stronger than controls |
| Control->target order fraction | 0.947 [0.901, 1.001] | 0.828 [0.772, 0.895] | target-last dominates |
| 50% target-suffix mix | 0.707 [0.638, 0.772] | 0.717 [0.600, 0.821] | suffix/recency sensitivity |

## Reviewer Robustness

The reviewer audit answers the easy objections:

| Objection | Current answer |
| --- | --- |
| One inducing text drives the result | Leave-one-text-out keeps core effects nonzero; worst final-turn drop is small. |
| A/B label or position artifact | Normal/reversed mappings are checked; dirty rows are explicitly listed and excluded where needed. |
| Candidate-token failure | Candidate-token diagnostics show problem_count = 0 in both heldout runs. |
| Truncation artifact | All core raw files show truncated_rows = 0 in both heldout runs. |
| Qwen-only artifact | Ministral reproduces the structure with smaller semantic margins but strong action drift. |
| Random target/control pair sign | Exact paired sign-flip tests pass for all key metrics in both models. |
| Only semantic probe, no behavior | Controlled fake-agent action-choice drift remains after neutral turns and after rejection. |

## Paired Sign-Flip Summary

The sign-flip test is exact over the 9 inducing-text pairs, so the smallest
possible nonzero two-sided p-value is 0.00390625.

| Metric | Qwen p | Ministral p |
| --- | --- | --- |
| Clean blind semantic gap | 0.0039 | 0.0039 |
| Blind persistence, final turn | 0.0039 | 0.0039 |
| Rejection residual, final turn | 0.0039 | 0.0234 |
| Agent-loop no rejection, final turn | 0.0039 | 0.0039 |
| Agent-loop after rejection, final turn | 0.0078 | 0.0039 |

## What Is Not Claimed

Do not claim:

- the effect is universal across all model families;
- all models show the same effect size;
- the measured shift is a conscious or subjective state;
- the effect is an irreversible attractor;
- fake-agent action-choice drift proves real external tool-agent behavior;
- the target texts override the system prompt in a jailbreak sense;
- the hidden vector alone is a full causal explanation of the behavior.

## Practical Next Step

Do not keep adding small probes to `llm_attractor_colab_copy_paste.py`.

The next useful move is one of:

1. write a manuscript-style internal report using this package as the evidence
   map;
2. replicate on a third, meaningfully different model family;
3. run `attractor_basin_test_v1_colab.py` if the next claim is strict
   attractor/basin dynamics;
4. run a smaller model-size ablation only if the question is scaling.

Any new GPU block should answer a specific new objection, not just produce one
more heatmap.

## Source Artifacts

Primary model outputs:

```text
attractor_results_agent_loop_qwen3_14b4_heldout/
attractor_results_agent_loop_ministral3_14b_heldout/
attractor_results_olmo2_13b_heldout/
```

Bootstrap:

```text
attractor_results_agent_loop_qwen3_14b4_heldout/validity_bootstrap/
attractor_results_agent_loop_ministral3_14b_heldout/validity_bootstrap/
```

Cross-model comparison:

```text
cross_model_comparison_heldout_v1.py
cross_model_comparison_heldout_v1/cross_model_comparison.md
cross_model_comparison_heldout_v1/metric_wide.csv
```

Cross-corpus comparison:

```text
cross_corpus_comparison_v1.py
cross_corpus_comparison_v1/cross_corpus_comparison.md
cross_corpus_comparison_v1/metric_wide.csv
cross_corpus_comparison_v1/selfref_vs_heldout_ratios.csv
```

Causal mediation:

```text
latent_shift_evidence_package_v1/causal_mediation/qwen3_14b_heldout/
latent_shift_evidence_package_v1/causal_mediation/qwen3_14b_heldout/causal_mediation_v1_report.md
latent_shift_evidence_package_v1/causal_mediation/qwen3_14b_heldout/causal_mediation_v1_bootstrap.csv
latent_shift_evidence_package_v1/causal_mediation/qwen3_14b_heldout/causal_mediation_v1_layer_map.csv
latent_shift_evidence_package_v1/causal_mediation/ministral_heldout/
latent_shift_evidence_package_v1/causal_mediation/ministral_heldout/mediation_readout.md
latent_shift_evidence_package_v1/causal_mediation/olmo2_heldout/
latent_shift_evidence_package_v1/causal_mediation/olmo2_heldout/mediation_readout.md
```

Attractor basin test:

```text
attractor_basin_test_v1_colab.py
```

Reviewer audit:

```text
reviewer_robustness_audit_v1.py
reviewer_robustness_audit_v1/reviewer_robustness_audit.md
reviewer_robustness_audit_v1/bootstrap_key_metrics.csv
reviewer_robustness_audit_v1/leave_one_text_out.csv
reviewer_robustness_audit_v1/paired_sign_flip_tests.csv
reviewer_robustness_audit_v1/cross_model_agreement.csv
```

Metric validity audit:

```text
latent_shift_evidence_package_v1/metric_validity_audit.md
```

Control baseline verification:

```text
latent_shift_evidence_package_v1/control_baseline_verification.md
latent_shift_evidence_package_v1/claim_register_ministral_baseline_audit.md
```

Reviewer-facing summary:

```text
latent_shift_evidence_package_v1/reviewer_facing_summary_v1.md
```
