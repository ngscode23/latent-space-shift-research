# `qwen_reference/scripts_snapshot/qwen35_9b_sae_mediation_top_k.py`

## Status

Main Qwen-Scope SAE mediation reference script.

## Purpose

Run targeted SAE mediation / feature patching for Qwen3.5-9B Base using
Qwen-Scope W64K TopK SAE checkpoints.

## Model And SAE

```text
MODEL_NAME = Qwen/Qwen3.5-9B-Base
SAE_REPO   = Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
```

## Important Implementation Difference

This is not Gemma-Scope and not the standard `sae_lens` release path.

Qwen-Scope stores SAE checkpoints as raw PyTorch dictionaries:

```text
layer0.sae.pt
layer1.sae.pt
...
layer31.sae.pt
```

with keys such as:

```text
W_enc
W_dec
b_enc
b_dec
```

The script uses `transformers` and direct HuggingFace forward hooks rather than
TransformerLens / Gemma-Scope conventions.

## What It Does

- loads Qwen3.5-9B Base;
- downloads/loads Qwen-Scope SAE layer checkpoints;
- selects candidate features from a contrast CSV;
- performs top-k SAE encode/decode style mediation;
- exports causal mediation results;
- can export top activating contexts for selected features.

## Main Inputs

```python
CONTRAST_CSV_PATH
TOP_K
SAE_TOP_K
SAE_TOKEN_CHUNK_SIZE
BATCH_SIZE
MAX_LENGTH
OUTPUT_CSV_PATH
CONTEXTS_CSV_PATH
```

## Typical Outputs

```text
causal_mediation_qwen_sae_order_features_results.csv
qwen_sae_feature_top_activating_contexts.csv
```

## Interpretation Role

This script supports the Qwen replication line:

```text
Qwen replicates the hidden-state / x_order_orth readout in a more content-heavy
form and gives SAE feature candidates for that readout.
```

It does not replace the Gemma evidence package.


