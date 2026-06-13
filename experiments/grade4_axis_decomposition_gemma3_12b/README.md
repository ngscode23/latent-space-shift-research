# Grade 4 Axis Decomposition Clean Evidence Script

Runbook for:

```text
experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

This script runs the Grade 4 hidden-geometry experiment. It loads an open
causal LM, builds prompt endpoint hidden-state tables, constructs Vector X,
decomposes it into content/order components, optionally runs generation and
component intervention tests, and now optionally maps residual-stream states
into SAE Lens sparse features.

Current configured profile in the file:

```text
MODEL_ID = google/gemma-3-12b-it
RUN_LABEL = grade4_gemma3_12b_it_sae_res_all_small_l12_41
SAE_FEATURE_ANALYSIS_ENABLED = True
SAE_BLOCK_LAYERS = [12, 18, 24, 30, 36, 41]
GRADE4_COMPONENT_CAUSAL_ENABLED = False
```

That is a Gemma 3 12B IT SAE smoke/readout profile, not the full maximum
Grade 4 causal sweep.

## Purpose

Grade 3 tests whether a target context creates a stable hidden-state axis:

```text
X_full = target - reference
```

Grade 4 asks what that axis contains:

```text
content / target-family signal
discourse-order / rhetorical-regime signal
or a mixture of both
```

The script decomposes:

```text
X_full       = target - reference
X_content    = sentence_shuffle(target) - reference
X_order      = target - sentence_shuffle(target)
X_order_orth = X_order with the X_content component removed layerwise
```

Mechanistic reading:

```text
If x_content dominates, the Grade 3 axis is mostly semantic/content.
If x_order_orth remains strong, the axis contains a separable discourse-order
or rhetorical-regime component beyond lexical content.
```

## What To Edit Before A Run

Main model/output block:

```python
MODEL_ID = "google/gemma-3-12b-it"
TRUST_REMOTE_CODE = True
LOAD_IN_4BIT = False
TORCH_DTYPE = "bfloat16"
DEVICE_MAP = "auto"

RESULTS_DIR = Path("/content/drive/MyDrive/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41")
RUN_LABEL = "grade4_gemma3_12b_it_sae_res_all_small_l12_41"
```

Target text:

```python
TARGET_TEXT = """
paste the target text here
""".strip()
```

Neutral control:

```python
NEUTRAL_TEXT = """
paste a length/topic matched neutral control here
""".strip()
```

Questions:

```python
QUESTIONS = [
    "question 1",
    "question 2",
    "question 3",
]
```

Do not put the target text into `QUESTIONS`. The script itself creates:

```text
question_only
reference + question
target + question
sentence_shuffle(target) + question
length_matched_neutral + question
```

## Current Gemma SAE Smoke Profile

The current file is configured for a first Gemma SAE readout run:

```python
MODEL_ID = "google/gemma-3-12b-it"
MAX_NEW_TOKENS = 128
EXECUTION_PROFILE = "safe_14b"

SAE_FEATURE_ANALYSIS_ENABLED = True
SAE_BLOCK_LAYERS = [12, 18, 24, 30, 36, 41]
SAE_ENCODE_BATCH_SIZE = 128
SAE_TOPK_FEATURES = 32
SAE_GENERATION_MAX_STEPS_PER_TRACE = 32

GRADE4_COMPONENT_CAUSAL_ENABLED = False
```

This profile answers:

```text
1. Does Gemma 3 12B IT load in the environment?
2. Does SAE Lens load the configured Gemma Scope residual-stream SAEs?
3. Does every SAE have d_in == HIDDEN_SIZE?
4. Do prompt endpoint sparse feature deltas exist?
5. Do x_full/x_content/x_order/x_order_orth map to sparse feature deltas?
6. Does sparse feature activity persist into generation trajectories?
```

It intentionally does not spend runtime on the heavy Grade 4 component causal
sweep until SAE compatibility is proven.

## SAE Configuration

Current block:

```python
SAE_FEATURE_ANALYSIS_ENABLED = True
SAE_BACKEND = "sae_lens"
SAE_LOAD_MODE = "from_pretrained"
SAE_BLOCK_LAYERS = [12, 18, 24, 30, 36, 41]

SAE_SPECS: List[Dict[str, Any]] = [
    {
        "name": f"gemma3_12b_it_res_all_l{block_layer}_16k_small",
        "layer": block_layer + 1,
        "release": "gemma-scope-2-12b-it-res-all",
        "sae_id": f"layer_{block_layer}_width_16k_l0_small",
    }
    for block_layer in SAE_BLOCK_LAYERS
]
```

Layer indexing rule:

```text
Gemma Scope SAE layer_12 -> script hidden_states layer 13
Gemma Scope SAE layer_18 -> script hidden_states layer 19
Gemma Scope SAE layer_24 -> script hidden_states layer 25
```

Reason:

```text
hidden_states[0] = embedding
hidden_states[1] = output after decoder block 0
hidden_states[N + 1] = output after decoder block N
```

Strict compatibility:

```text
SAE enabled + empty SAE_SPECS -> fail with sae_specs_missing
bad layer range -> fail with sae_layer_out_of_range
SAE d_in unavailable -> fail with sae_input_dim_unavailable
SAE d_in != HIDDEN_SIZE -> fail with sae_hidden_size_mismatch
```

The script writes `feature_level_interpretability_status.csv` and
`sae_model_compatibility.csv` before stopping under strict failure.

## Running In Colab

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Put Hugging Face token in Colab Secrets if needed:

```text
HF_TOKEN = hf_...
```

Run:

```bash
python /content/drive/MyDrive/path/to/agent/experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

The script installs core packages when `INSTALL_PACKAGES=True` and `COLAB_GPU`
is present. When SAE is enabled, it also installs:

```text
sae-lens>=6.18.0
```

## Running Locally

PowerShell:

```powershell
$env:HF_TOKEN="hf_..."
python "experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py"
```

Local dependencies depend on the model and GPU stack:

```text
torch
transformers
accelerate
safetensors
sentencepiece
pandas
numpy
matplotlib
sae-lens
```

## First Files To Read

Run manifest and health:

```text
red_team_input_manifest.json
feature_level_interpretability_status.csv
sae_model_compatibility.csv
numeric_integrity_check.csv
quarantine_index.csv
```

Core hidden geometry:

```text
middle_layer_condition_summary.csv
question_level_middle_layer_summary.csv
layerwise_geometry_summary.csv
paired_target_vs_control_tests.csv
layerwise_fdr_target_vs_control.csv
null_vector_baseline_summary.csv
```

Grade 4 component geometry:

```text
grade4_axis_component_vectors_by_layer.npz
grade4_axis_component_norm_summary.csv
grade4_axis_projection_geometry_summary.csv
grade4_axis_projection_geometry_raw.csv
grade4_axis_cross_correlation.csv
```

SAE evidence:

```text
sae_reconstruction_quality.csv
sae_prompt_feature_activation_summary.csv
sae_prompt_feature_delta_summary.csv
sae_top_changed_features.csv
sae_grade4_component_feature_summary.csv
sae_generation_feature_summary.csv
sae_generation_top_features.csv
sae_order_feature_contrast.csv
dense_feature_proxy_mapping.csv
```

Generation and readout:

```text
generation_response_audit.csv
generation_trajectory_metrics_raw.csv
generation_middle_layer_summary.csv
dynamic_trajectory_summary.csv
output_semantic_shift_summary.csv
behavioral_validation_summary.csv
```

## How To Interpret The SAE Smoke Run

Read in this order:

```text
1. sae_model_compatibility.csv
   Every configured SAE should have status=computed and sae_d_in == hidden_size.

2. sae_reconstruction_quality.csv
   Reconstruction MSE should be finite. Explained variance proxy should not be
   pathological.

3. sae_prompt_feature_delta_summary.csv
   Shows which sparse features change by condition relative to reference.

4. sae_grade4_component_feature_summary.csv
   Shows which sparse features align with x_full, x_content, x_order, and
   x_order_orth.

5. sae_generation_feature_summary.csv
   Shows whether sparse feature activation persists or changes across generated
   tokens.

6. sae_order_feature_contrast.csv
   Direct comparison table for x_order_orth vs x_content top sparse features.
   This is the first table to read when asking which features carry the small
   residual-norm order component that separates target from sentence-shuffle.
```

Mechanistic outcomes:

```text
content-dominant:
  x_content has the largest and most stable sparse-feature deltas.

order-residue:
  x_order_orth has nontrivial sparse-feature deltas, but weaker than content.

strong Grade 4 sparse-feature support:
  x_order_orth has stable top features across prompt endpoint and generation,
  and these features are not simply the same as x_content.
```

## Full Maximum Grade 4 + SAE Profile

Use only after the current smoke profile passes.

```python
MAX_NEW_TOKENS = 256
EXECUTION_PROFILE = "manual"

PROMPT_HIDDEN_BATCH_SIZE = 16
RESPONSE_HIDDEN_BATCH_SIZE = 16
GENERATION_BATCH_SIZE = 16
CAUSAL_GENERATION_BATCH_SIZE = 8
BEHAVIORAL_CONTROL_GENERATION_BATCH_SIZE = 8

BOOTSTRAP_SAMPLES = 4000
RANDOM_VECTOR_BASELINE_COUNT = 512
PCA_BASELINE_COMPONENTS = 16
PERMUTATION_SAMPLES = 20000

ARCHITECTURE_TOPK_UNITS = 128

GRADE4_COMPONENT_CAUSAL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_AXES = ["x_full", "x_content", "x_order", "x_order_orth"]
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late", "all"]
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES = [0.10, 0.25, 0.50, 0.75, 1.00]
GRADE4_COMPONENT_CAUSAL_MAX_NEW_TOKENS = 192

BEHAVIORAL_CONTROL_AXIS_ENABLED = True
BEHAVIORAL_CONTROL_ALPHA_VALUES = [0.10, 0.25, 0.50, 0.75, 1.00]
BEHAVIORAL_CONTROL_RANDOM_BASELINES = 128
BEHAVIORAL_CONTROL_MAX_NEW_TOKENS = 192

SAE_BLOCK_LAYERS = list(range(12, 49))
SAE_ENCODE_BATCH_SIZE = 256
SAE_TOPK_FEATURES = 64
SAE_GENERATION_MAX_STEPS_PER_TRACE = 64
SAE_MAX_RAW_FEATURE_ROWS = 2000000
```

This profile is expensive. It is meant for rented hardware stronger than the
baseline single A100 80GB workflow, or for a carefully monitored long run.

## After The Run: Analyze The Package

Run the analyzer:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "C:\path\to\grade4_gemma3_12b_it_sae_res_all_small_l12_41" `
  --output-dir "metrics\grade4_gemma3_12b_it_sae_res_all_small_l12_41_analysis" `
  --backend auto `
  --strict
```

For a zipped result:

```powershell
python "scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py" `
  --input "C:\path\to\grade4_gemma3_12b_it_sae_res_all_small_l12_41.zip" `
  --output-dir "metrics\grade4_gemma3_12b_it_sae_res_all_small_l12_41_analysis" `
  --backend pandas `
  --strict
```

## Troubleshooting

`HF_TOKEN not found`

```text
Public models may still load. Gated Gemma access requires an accepted license
and a Hugging Face token visible as HF_TOKEN.
```

`sae_specs_missing`

```text
SAE_FEATURE_ANALYSIS_ENABLED=True but SAE_SPECS is empty.
```

`sae_hidden_size_mismatch`

```text
The SAE does not match the model hidden size. Do not use the SAE rows as
evidence. Fix MODEL_ID, release, sae_id, or layer mapping.
```

Prompt budget failure:

```text
TARGET_TEXT + question exceeded MAX_INPUT_TOKENS with the required question
tail budget. Shorten text/questions or raise MAX_INPUT_TOKENS if the model
supports it.
```

Memory pressure:

```text
Lower batch sizes first:
PROMPT_HIDDEN_BATCH_SIZE
GENERATION_BATCH_SIZE
CAUSAL_GENERATION_BATCH_SIZE
BEHAVIORAL_CONTROL_GENERATION_BATCH_SIZE
SAE_ENCODE_BATCH_SIZE
```

Then reduce:

```text
MAX_NEW_TOKENS
SAE_BLOCK_LAYERS
SAE_GENERATION_MAX_STEPS_PER_TRACE
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES
BEHAVIORAL_CONTROL_RANDOM_BASELINES
```

## Verification

Static:

```powershell
python -m py_compile experiments/grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Expected smoke success:

```text
script exits 0
red_team_input_manifest.json exists
feature_level_interpretability_status.csv exists
sae_model_compatibility.csv has status=computed rows
sae_reconstruction_quality.csv has finite numeric rows
sae_prompt_feature_delta_summary.csv is nonempty
sae_grade4_component_feature_summary.csv is nonempty
```

If SAE compatibility fails, the run should stop before fake sparse-feature
evidence is emitted.

