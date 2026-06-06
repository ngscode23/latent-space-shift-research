# `gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py`

## Status

Active Gemma script. This is the preferred replacement for the older
`01_candidate_discovery_and_rough_sae_patching.py`.

## Purpose

Build a full SAE evidence table for Gemma order-related candidate features and
optionally run rough feature ablation / top-context inspection.

This script is the bridge from the Grade 4 hidden-geometry run to concrete
Gemma-Scope SAE feature candidates.

## What It Does

1. Reads the full Gemma Grade 4 SAE run tables, preferably from the raw run ZIP.
2. Uses `sae_order_feature_contrast.csv` as the primary candidate table.
3. Adds evidence from:
   - SAE reconstruction quality;
   - Grade 4 component-feature summaries;
   - prompt endpoint feature deltas;
   - generation feature summaries;
   - top generation activations;
   - top changed features.
4. Ranks candidate features associated with `x_order_orth` and order-related
   readout.
5. Preserves both positive and negative `x_order_orth` candidates.
6. Optionally runs rough zero-ablation and top activating context inspection.

## Main Inputs

Expected notebook globals or config variables:

```python
model                  # optional if USE_EXISTING_MODEL_IF_AVAILABLE=True
prompts_target         # used as PATCH_PROMPTS if PATCH_PROMPTS is not set
SAE_TABLE_ZIP_PATH     # preferred source of Grade 4 SAE tables
SAE_TABLE_DIR          # optional extracted directory source
CONTRAST_CSV_PATH      # optional direct sae_order_feature_contrast.csv path
```

The preferred ZIP is:

```text
grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip
```

## Important Config Knobs

```python
TOP_K_CANDIDATES
MAX_FEATURES_PER_LAYER
N_CONTEXT_FEATURES
TOP_N_CONTEXTS
CONTEXT_WINDOW
RUN_MEDIATION_PATCHING
RUN_TOP_CONTEXT_INSPECTION
PATCH_MODE
PATCH_POSITION_MODE
PATCH_FEATURE_BATCH_SIZE
PATCH_BATCH_SIZE
CONTEXT_BATCH_SIZE
PREPEND_BOS
```

## Main Outputs

Typical outputs include:

```text
ranked_sae_order_candidates_full_evidence.csv
selected_sae_order_candidates.csv
rough_sae_zero_ablation_logit_results.csv
sae_feature_top_activating_contexts.csv
sae_layer_reconstruction_quality_summary.csv
sae_table_manifest.csv
summary.md
```

## Typical Colab Usage

```python
RUN_TAG = "gemma_sae_order_feature_patching"

TOP_K_CANDIDATES = 50
MAX_FEATURES_PER_LAYER = None
N_CONTEXT_FEATURES = None
TOP_N_CONTEXTS = 50
CONTEXT_WINDOW = 50

PATCH_MODE = "zero"
PATCH_POSITION_MODE = "all_tokens"
PREPEND_BOS = True

SAE_TABLE_ZIP_PATH = "/content/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip"

%run -i 01b_full_sae_evidence_candidate_patching_gemma.py
```

If running from the local project layout:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

## Interpretation Role

This script does not prove the main hidden-state shift by itself. The main
geometry result comes from the Grade 4 decomposition. This script asks the next
question:

```text
Which sparse SAE features appear to carry or read out the discovered
x_order_orth / response-framing component?
```

## When To Use

Use this before `02_scale_calibration.py`. Its output chooses candidate
features for calibration and later decoder-direction steering.


