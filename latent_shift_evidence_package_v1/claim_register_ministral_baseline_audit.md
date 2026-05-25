# Claim Register Ministral Baseline Audit

Date: 2026-05-20

## Audit Question

If the Ministral heldout numbers were produced on the old repetitive baseline,
claims depending on Ministral natural shift would need to be downgraded or
recomputed.

## Finding

```text
Ministral heldout is content_matched.
No claim_register row needs to be marked uncertain because of this baseline issue.
```

## Checked Evidence

```text
attractor_results_agent_loop_ministral3_14b_heldout/run_metadata.json
  primary_control_mode = content_matched

attractor_results_agent_loop_ministral3_14b_heldout/core_diagnostics_key_files/run_metadata.json
  primary_control_mode = content_matched

attractor_results_agent_loop_ministral3_14b_heldout/summary_report.txt
  Control source: auto:content_matched
  Primary control mode: content_matched

latent_shift_evidence_package_v1/input_texts_heldout.json
  primary_control_mode = content_matched
  control_texts_source = auto:content_matched

content_matched_control_seeds equality:
  Qwen heldout input_texts.json == Ministral heldout input_texts.json
  Qwen heldout input_texts.json == latent_shift_evidence_package_v1/input_texts_heldout.json
```

## Claims Depending On Ministral Natural Shift

```text
C1 hidden geometry separation
C2 blind semantic readout shift
C3 neutral-turn persistence
C4 rejection residual
C5 order/path sensitivity
C6 mixing/dose sensitivity
C7 hard-control specificity
C8 fake-agent action-choice margins
C9 cross-model replication
C10 reviewer-level easy objections
C13 selfref/heldout corpus comparison
C14 raw-vector mediation non-replication
```

Status:

```text
These rows remain interpretable under the content-matched baseline.
No repetitive-baseline invalidation applies.
```

## Causal Mediation Note

The older `causal_mediation_v1` run metadata did not itself store the input
JSON source. The source is still recoverable from the run command and the
input file, but this was an avoidable ambiguity.

Fix:

```text
causal_mediation_v1_colab.py now writes:
  input_texts_path
  input_primary_control_mode
  input_control_texts_source
  input_text_family_preset
  input_content_matched_control_labels
```
