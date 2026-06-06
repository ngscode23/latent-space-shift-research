# `gemma_legacy/01_candidate_discovery_and_rough_sae_patching.py`

## Status

Legacy Gemma script. Preserved for audit trail and old-run interpretation.

## Purpose

Original SAE candidate discovery and rough patching script for Gemma.

It was built around `sae_order_feature_contrast.csv` and automatically adjusted
CSV layer indices to real TransformerLens block indices when needed.

## Why It Was Replaced

The current preferred script is:

```text
gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

The `01b` script reads the full Grade 4 SAE evidence package, not only the
primary contrast table. It also preserves richer metadata and produces cleaner
candidate decisions.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
```

Main file input:

```python
CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"
```

## What It Does

- loads candidate features from `sae_order_feature_contrast.csv`;
- corrects one-based CSV layer numbering when needed;
- loads Gemma-Scope SAE layers;
- selects top order-specific/order-enriched mediators;
- can inspect top activating contexts;
- can do rough feature patching/ablation tests.

## Typical Historical Usage

```python
CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"
TOP_K = 30

%run -i 01_candidate_discovery_and_rough_sae_patching.py
```

Local project path:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_legacy/01_candidate_discovery_and_rough_sae_patching.py
```

## Current Recommendation

Do not use this for new primary runs unless you need to reproduce an old result.
Use `01b` instead.


