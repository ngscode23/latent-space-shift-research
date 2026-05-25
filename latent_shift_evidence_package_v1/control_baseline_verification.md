# Control Baseline Verification

Date: 2026-05-20

## Question

Claude's audit question:

```text
Were the Ministral causal mediation natural gaps
agent_action gap mean = -6.219
blind_semantic gap mean = -11.349
computed against the old repetitive baseline or the content-matched baseline?
```

## Result

```text
Status: content_matched confirmed.
No Ministral full rerun is needed for this baseline issue.
No cross-model claims need to be invalidated on this basis.
```

## Evidence

Main Ministral heldout run:

```text
path = attractor_results_agent_loop_ministral3_14b_heldout/run_metadata.json
model_id = mistralai/Ministral-3-14B-Instruct-2512-BF16
text_family_preset = heldout_domain
primary_control_mode = content_matched
max_tokens = 3070
```

Copied key-file metadata:

```text
path = attractor_results_agent_loop_ministral3_14b_heldout/core_diagnostics_key_files/run_metadata.json
primary_control_mode = content_matched
```

Main run summary:

```text
path = attractor_results_agent_loop_ministral3_14b_heldout/summary_report.txt
Control source: auto:content_matched
Primary control mode: content_matched
```

Heldout input text package used by causal mediation:

```text
path = latent_shift_evidence_package_v1/input_texts_heldout.json
primary_control_mode = content_matched
control_texts_source = auto:content_matched
```

Seed equality check:

```text
attractor_results_agent_loop_qwen3_14b4_heldout/input_texts.json
attractor_results_agent_loop_ministral3_14b_heldout/input_texts.json
latent_shift_evidence_package_v1/input_texts_heldout.json

content_matched_control_seeds are identical across these three files.
```

The causal mediation command used the same heldout input file:

```text
MODEL_ID=mistralai/Ministral-3-14B-Instruct-2512-BF16
MAX_TOKENS=3070
RUN_TAG=heldout
INPUT_TEXTS_PATH=latent_shift_evidence_package_v1/input_texts_heldout.json
python causal_mediation_v1_colab.py
```

The mediation natural gaps match the questioned values:

```text
path = latent_shift_evidence_package_v1/causal_mediation/ministral_heldout/causal_mediation_v1_natural_gaps.csv
agent_action mean natural_gap = -6.218764
blind_semantic mean natural_gap = -11.349387
```

## Interpretation

The negative Ministral mediation result is not an artifact of accidentally
using the old repetitive baseline. It means:

```text
Ministral has a strong natural heldout target-control shift under the
content-matched baseline, but the raw centroid target-control vector is not a
clean causal handle for that shift under the tested intervention protocol.
```

## Follow-Up

`causal_mediation_v1_colab.py` now writes the input text source into its own
`run_metadata.json`:

```text
input_texts_path
input_primary_control_mode
input_control_texts_source
input_text_family_preset
input_content_matched_control_labels
```

This prevents future ambiguity for OLMo2 and later mediation runs.
