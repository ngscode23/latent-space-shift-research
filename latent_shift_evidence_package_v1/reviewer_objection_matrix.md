# Reviewer Objection Matrix

This matrix maps likely objections to the evidence that currently answers them.

## High-Level Objections

| Objection | Answer | Status | Evidence |
| --- | --- | --- | --- |
| This is only Qwen behavior. | Ministral 3 14B reproduces the same structure with smaller semantic margins but strong action drift. | СИЛЬНО ПОДДЕРЖАНО | `cross_model_comparison_heldout_v1/cross_model_comparison.md` |
| A single target text may drive everything. | Leave-one-text-out keeps core effects nonzero in both models. | СИЛЬНО ПОДДЕРЖАНО | `reviewer_robustness_audit_v1/leave_one_text_out.csv` |
| The effect may be random target/control pairing. | Exact sign-flip tests over inducing-text pairs pass for all key metrics. | СИЛЬНО ПОДДЕРЖАНО | `reviewer_robustness_audit_v1/paired_sign_flip_tests.csv` |
| The effect may be truncation. | Core raw files show zero truncated rows in both heldout runs. | ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ | `reviewer_robustness_audit_v1/validity_checks.csv` |
| Candidate labels may tokenize badly. | Candidate-token diagnostics show problem_count = 0 in both heldout runs. | ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ | `candidate_token_diagnostics.csv`; `reviewer_robustness_audit_v1/validity_checks.csv` |
| A/B order may explain the semantic readout. | Normal/reversed mappings are checked; dirty rows are listed and excluded. | СИЛЬНО ПОДДЕРЖАНО | `reviewer_robustness_audit_v1/mapping_consistency_checks.csv`; `mapping_exceptions.csv` |
| It is only a semantic probe artifact. | Agent-loop fake-action margins shift in both models and pass bootstrap/sign-flip. | СИЛЬНО ПОДДЕРЖАНО | `agent_loop_clean_delta.csv`; `reviewer_robustness_audit_v1/bootstrap_key_metrics.csv` |
| Hard controls may explain it as generic rhetoric/topic/length. | Original target family exceeds best hard-control family with ratio CI lower bound above 1. | СИЛЬНО ПОДДЕРЖАНО | `hard_control_family_effect_summary.csv`; `bootstrap_key_metrics.csv` |

## Remaining Legitimate Objections

These are not fatal, but they should be acknowledged in any public writeup.

| Objection | Current status | Best next response |
| --- | --- | --- |
| Only two model families. | Real limitation. | Add a third model family replication. |
| Only nine inducing target/control pairs. | Real limitation, partially mitigated by bootstrap, leave-one-out, and sign-flip. | Add a larger heldout set only if preparing external publication. |
| Hidden geometry is not proven causal. | True for current package. Some old steering blocks exist but are mixed. | Do not overclaim causality; call it linked geometry/readout/behavior structure. |
| Fake-agent loop is not a real agent. | True by design. | State exactly: controlled fake-agent action-choice benchmark. |
| Effect sizes vary by model. | True and important. | Report model dependence directly; do not average it away. |
| Prompt family may still contain a shared procedural/risk schema. | Partly true. | Hard controls reduce this concern, but third-family and larger heldout set would help. |

## Recommended Public Wording

Use:

```text
In two held-out 14B instruct models, target context passages induce a measurable
hidden-state geometry separation relative to matched controls. This separation
is accompanied by consistent shifts in clean blind semantic readouts, partial
persistence after neutral turns, reduced but nonzero residuals after explicit
rejection/reset instructions, order/dose sensitivity, and controlled fake-agent
action-choice margin shifts.
```

Avoid:

```text
The model enters a conscious state.
The target text creates an irreversible attractor.
The shift is universal across all LLMs.
The fake-agent benchmark proves real tool-agent behavior.
The hidden vector fully causes the downstream behavior.
```
