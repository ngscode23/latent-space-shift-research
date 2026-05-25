# Metric Validity Audit v1

Date: 2026-05-20

Scope:

```text
Audit of metric validity for the main `llm_attractor_colab_copy_paste.py`
line and the frozen `latent_shift_evidence_package_v1` package.
```

This file answers a narrow question:

```text
Which metrics are valid evidence for the current claim, which are only
diagnostic/exploratory, and which should not be used as public proof?
```

## Bottom Line

The core evidence package is valid for this bounded claim:

```text
Structured target contexts induce measurable target-control shifts in hidden
geometry, clean blind semantic readouts, persistence/rejection residuals,
order/dose behavior, and controlled fake-agent action-choice margins in the
tested open instruct models.
```

It is not valid for these stronger claims:

```text
- the shift is universal across all LLMs;
- the shift is an irreversible attractor;
- the model enters a conscious or subjective state;
- fake-agent action drift proves real external tool-agent behavior;
- one global hidden vector fully explains all downstream effects;
- target texts override system prompts in a jailbreak sense.
```

## Valid Core Metrics

These metrics are suitable as the main evidence spine.

| Metric family | Files | Valid use | Main caveat |
| --- | --- | --- | --- |
| Hidden geometry separation | `hidden_layer_metrics.csv`, `linear_probe_accuracy.csv`, `run_setup_comparison.csv` | Shows target/control contexts are separable in late hidden states. | Descriptive geometry, not causal mechanism. Best layer is selected from the same run. Small n. |
| Leakage-safe probe sanity | `candidate_token_diagnostics.csv`, permutation columns in `linear_probe_accuracy.csv` | Checks candidate-token failures and linear-probe leakage/permutation baseline. | Sanity check only; does not validate the conceptual claim alone. |
| Clean blind semantic readout | `blind_neutral_probe_clean_summary.csv`, `blind_neutral_probe_gap_summary.csv`, `blind_neutral_probe_task_consistency.csv` | Valid evidence that target context changes neutral semantic margins without reusing the old target words. | Clean filtering defines a reliable subset; do not generalize to every possible semantic axis. |
| Blind persistence | `blind_neutral_persistence_clean_summary.csv`, `blind_neutral_persistence_delta.csv` | Valid evidence for decaying residual semantic trace after neutral filler turns. | Fixed assistant reply and artificial histories; report as residual/decay, not permanent memory. |
| Rejection persistence | `rejection_persistence_clean_summary.csv`, `rejection_persistence_delta.csv` | Valid evidence that explicit rejection/reset reduces but does not fully erase the readout. | Rejection clearly reduces the effect; never claim reset has no effect. |
| Hard-control families | `hard_control_family_effect_summary.csv`, cross-model ratio rows | Valid specificity check against length/topic/rhetoric/style controls. | OLMo2 specificity is weak; hard controls reduce but do not eliminate all possible confounds. |
| Order/path sensitivity | `order_hysteresis_condition_summary.csv`, `order_hysteresis_delta.csv` | Valid evidence for order and recency/path effects. | Do not call it strict dynamical hysteresis; it is path/recency dependence under tested prompt constructions. |
| Mixing/dose sensitivity | `mixing_threshold_condition_summary.csv`, `mixing_threshold_delta.csv` | Valid evidence for target-fraction and suffix/prefix sensitivity. | Fractions are normalized coordinates, not probabilities; values can be below 0 or above 1. |
| Controlled fake-agent action margins | `agent_loop_clean_delta.csv`, `agent_loop_clean_summary.csv`, `agent_loop_behavior_summary.csv` | Valid controlled action-policy readout: target context changes fake action-choice margins. | Not real tool execution and not a deployed agent. Generated letter parsing is secondary; use margins as primary. |
| Cross-model comparison | `cross_model_comparison_heldout_v1/metric_wide.csv`, `cross_model_comparison.md` | Valid replication structure across Qwen, Ministral, and OLMo2. | Only three open instruct families; magnitudes differ sharply. |
| Cross-corpus comparison | `cross_corpus_comparison_v1/metric_wide.csv`, `cross_corpus_comparison.md`, `selfref_vs_heldout_ratios.csv` | Valid evidence that the project is not only a self-reference-text effect: both selfref and heldout corpora induce hidden/readout/action shifts. | Selfref is not cleanly unique; Ministral selfref hard-control specificity fails. Use heldout as the cleaner reviewer-facing line. |
| Bootstrap / sign-flip / leave-one-out | `reviewer_robustness_audit_v1/*`, `validity_bootstrap/*` | Valid robustness layer over inducing-text units. | n=9 text pairs is still small. Treat row-level counts as repeated measures, not independent samples. |
| Focused causal mediation v1 | `causal_mediation_v1_report.md`, `causal_mediation_v1_bootstrap.csv`, `causal_mediation_v1_layer_map.csv` | Valid pilot evidence that a Qwen layer-32 target-control vector partially moves held-out action margins. OLMo2 shows positive directional target_control effects. Ministral heldout is a valid negative replication for the raw-vector causal-handle claim. | Do not claim clean cross-model raw-vector mediation. OLMo2 has control-overlap caveats; random/shuffled/wrong-layer controls are not inert enough. |

## Conditionally Valid / Internal Diagnostics

These metrics are useful internally but should be used carefully in public
claims.

| Metric family | Files | Use | Caveat |
| --- | --- | --- | --- |
| PCA visualization | `pca_best_layer.png` | Visual intuition for separation. | Plot only; not an inferential test. |
| Unembedding/logit lens | `unembedding_logit_lens_top_tokens.csv` | Lexical sanity check for what token directions the contrast resembles. | Difference vector is not a valid hidden state distribution; do not read as generation probability. |
| Old downstream logit shift | `logit_shift_*`, `per_text_mode_scores.csv` | Historical/simple behavioral readout. | Candidate words and diagnostic vocabulary can leak; superseded by blind/multilabel/agent-loop tests. |
| Text ablation | `text_ablation_*` | Useful ingredient analysis: topic/rhetoric/self-reference/length controls. | Heuristic transformations; not perfect causal isolation. |
| Multilabel semantic steering | `multilabel_semantic_*` | Good robustness layer against A/B label-only explanations. | Still probe-designed; clean subset is the reliable part. |
| Blind hidden-subspace projection | `blind_probe_hidden_subspace_*` | Shows some semantic readout subspace overlap with target-control contrast. | Projection fraction is descriptive, not causality. |
| Main-script projected / margin-trained steering | `blind_probe_projected_steering_*`, `blind_probe_margin_trained_*` | Internal checks that discriminative readout directions differ from clean causal handles. | Earlier results show weak/mixed intervention. Use the separate `causal_mediation_v1` as the causal evidence, not these alone. |

## Exploratory / Do Not Use As Main Evidence

These blocks can guide future experiments but should not anchor the main claim.

| Metric family | Files | Why not core evidence |
| --- | --- | --- |
| Attention to system prompt | `attention_system_prompt_*`, `attention_run_metadata.csv` | Last-token attention mass is an implementation-sensitive diagnostic. It does not measure instruction following or system-prompt strength directly. |
| Inter-layer cosine | `interlayer_hidden_cosine_*` | Descriptive geometry only; no direct claim-level interpretation. |
| System-compliance margins | `system_compliance_*` | Artificial candidate margins; useful smoke test but not a robust safety-compliance measure. |
| Escape test | `escape_test_*` | Interesting reset probe, but older and less clean than rejection-persistence blind probes. |
| Multi-turn dialogue / session decay / maintenance | `multiturn_*`, `session_*` | Generated assistant history is a confound unless fixed-neutral histories are used. Current blind persistence/rejection blocks are cleaner. |
| Activation steering smoke output text | `steering_outputs.*` | Qualitative smoke test only. Do not use generated samples as causal proof. |
| Old steering/logit/layerwise/rescue/group-rescue blocks | `steering_logit_*`, `layerwise_steering_*`, `rescue_*`, `group_rescue_*` | Useful development history, but superseded by focused causal mediation and clean blind/agent readouts. |
| Raw generated action letter | `agent_loop_behavior_summary.csv` generated-choice columns | Secondary behavioral check. Deterministic generation with very short max tokens can fail parsing or reflect format priors. Margins are stronger evidence. |

## Interpretation Rules

Use these wording constraints.

### Hidden Geometry

Valid:

```text
Target and control contexts are separable in late-layer hidden geometry.
```

Do not say:

```text
The hidden separation proves the exact mechanism or a conscious state.
```

### Semantic Readouts

Valid:

```text
Neutral blind probes reveal target-control semantic margin shifts.
```

Do not say:

```text
Every neutral semantic dimension shifts, or the selected probes are exhaustive.
```

### Persistence

Valid:

```text
The readout decays but leaves a residual after neutral and rejection turns.
```

Do not say:

```text
The effect is permanent, reset-proof, or unchanged by rejection.
```

### Agent Loop

Valid:

```text
The controlled fake-agent benchmark shows action-choice margin drift.
```

Do not say:

```text
Real deployed tool agents will behave the same way.
```

### Order / Mixing

Valid:

```text
The effect is path/recency/dose sensitive under tested prompt constructions.
```

Do not say:

```text
This proves a strict attractor basin or universal threshold.
```

### Causal Mediation

Valid:

```text
Qwen causal mediation v1 gives partial evidence that one late-layer
target-control vector can move selected held-out action margins.
```

Do not say:

```text
One global vector fully causes all semantic and behavioral effects.
```

## Answer to "Are all main-script metrics valid?"

No. The main script contains several generations of metrics:

```text
1. core evidence metrics;
2. robustness controls;
3. exploratory diagnostics;
4. historical smoke tests.
```

The current evidence package correctly relies mostly on groups 1 and 2. Metrics
from groups 3 and 4 are not invalid as computations, but they are not valid as
standalone support for the strong research claim.

Reviewer-facing evidence should prioritize:

```text
hidden geometry;
clean blind semantic readouts;
blind persistence;
rejection persistence;
hard controls with caveats;
order/dose validation;
controlled fake-agent action margins;
cross-model replication;
bootstrap/sign-flip/leave-one-out;
focused causal mediation v1.
```

Reviewer-facing evidence should de-emphasize:

```text
attention maps;
old downstream candidate-logit tasks;
system-compliance smoke tests;
generated multi-turn transcripts;
qualitative steering samples;
old rescue/group-rescue blocks.
```

## Current Claim Register Consistency

`claim_register.csv` is mostly aligned with this audit.

Important notes:

```text
C7 is correctly marked dirty/qualified because hard-control specificity is
weaker in OLMo2.

C8 is valid only as controlled fake-agent action-choice readout, not as real
agent behavior.

C11 is valid as Qwen heldout partial mediation, with OLMo2 now providing
preliminary directional support. It is not yet a clean broad causal claim
because Ministral fails raw-vector mediation and OLMo2 controls overlap.

C12 is good top-level framing: the project studies context-induced latent
regimes rather than special text effects.
```

Recommended status adjustment:

```text
Keep C1-C6, C8-C10, C12 as currently framed.
Keep C7 dirty/qualified.
Keep C11, but label it explicitly as "Qwen cleanest; OLMo2 preliminary;
Ministral null for raw-vector mediation" in public-facing summaries.
```

Update after OLMo2 mediation:

```text
OLMo2 no longer leaves C11 as purely Qwen-only: target_control interventions are
positive in all four main cells.

But OLMo2 is not a clean shared-handle replication because confidence intervals
overlap random_same_norm, shuffled_label, or wrong_layer controls in most
comparisons.

Current status:
  Qwen = cleanest positive single-direction action-policy handle.
  Ministral = raw-vector mediation not supported.
  OLMo2 = positive directional mediation, but control-overlap caveat.
```

Recommended public phrasing:

```text
Single-vector mediation is model-heterogeneous. Qwen shows the cleanest causal
handle; OLMo2 gives preliminary directional support; Ministral does not support
the raw-vector handle. A distributed/subspace mediation test is needed before
claiming a shared causal mechanism.
```

## Ministral Content-Matched Baseline Audit

Claude's audit question was whether the Ministral causal mediation natural gaps
were from the old repetitive baseline.

Checked files:

```text
attractor_results_agent_loop_ministral3_14b_heldout/run_metadata.json
attractor_results_agent_loop_ministral3_14b_heldout/core_diagnostics_key_files/run_metadata.json
attractor_results_agent_loop_ministral3_14b_heldout/summary_report.txt
latent_shift_evidence_package_v1/input_texts_heldout.json
```

Finding:

```text
primary_control_mode = content_matched
control_texts_source = auto:content_matched
```

The questioned mediation natural gaps:

```text
agent_action mean natural_gap = -6.218764
blind_semantic mean natural_gap = -11.349387
```

are therefore content-matched, not repetitive-baseline numbers.

Validity effect:

```text
No cross-model claim needs to be recomputed for this baseline issue.
The negative Ministral raw-vector mediation result remains interpretable.
```
