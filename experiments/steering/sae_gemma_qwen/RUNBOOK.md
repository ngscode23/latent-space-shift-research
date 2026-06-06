# SAE / Gemma / Qwen Steering Runbook

This runbook explains the active SAE pipeline after the Grade4 hidden-geometry
run.

## Pipeline Position

Upstream source:

```text
experiments/grade4_axis_decomposition_gemma/
```

Main pipeline:

```text
Grade4 ZIP
-> SAE feature evidence with 01b
-> scale calibration with 02
-> SAE decoder-direction steering with full/fast runner
-> optional dense x_order_orth axis steering
```

## Step 1: SAE Feature Candidate Evidence

Script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

What it does:

```text
1. Reads the Grade4 SAE tables from a ZIP or directory.
2. Uses sae_order_feature_contrast.csv as the primary ranking table.
3. Adds support evidence from reconstruction, component, prompt-delta,
   generation, top-feature, changed-feature, and compatibility tables.
4. Selects candidate SAE features associated with x_order_orth / order-related
   readout.
5. Optionally runs rough SAE zero-ablation and top activating context inspection.
```

Minimal Colab setup:

```python
SAE_TABLE_ZIP_PATH = "/content/<grade4_package>.zip"
PATCH_PROMPTS = prompts_target

%run -i experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

If running from a flat Colab directory, copy the script to `/content` and adjust
the `%run -i` path accordingly.

Useful knobs:

```python
RUN_TAG = "gemma_sae_order_feature_patching"
TOP_K_CANDIDATES = 50
MAX_FEATURES_PER_LAYER = None
N_CONTEXT_FEATURES = None
RUN_MEDIATION_PATCHING = True
RUN_TOP_CONTEXT_INSPECTION = True
PATCH_PROMPTS = prompts_target
```

Main outputs:

```text
selected_sae_order_candidates.csv
ranked_sae_order_candidates_full_evidence.csv
sae_table_manifest.csv
sae_layer_reconstruction_quality_summary.csv
rough_sae_zero_ablation_logit_results.csv
sae_feature_top_activating_contexts.csv
summary.md
```

## Step 2: Scale Calibration

Script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/02_scale_calibration.py
```

Why this step exists:

```text
For Gemma SAE decoder-direction steering, scale=1 is usually extremely small
relative to the residual-stream norm. In previous calibration, old tiny scales
behaved like a fraction-of-a-percent residual perturbation and often sat near
noise level. The calibration script estimates scales as fractions of residual
norm and optionally checks next-token KL.
```

Expected notebook globals:

```python
model
saes
prompts_target
STEERING_FEATURES
```

Example feature format:

```python
STEERING_FEATURES = [
    (18, 378),
    (18, 373),
    (36, 323),
]
```

The layer value should be the real model layer expected by the steering script,
not an off-by-one CSV display layer.

Useful knobs:

```python
RESID_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
PASTE_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
RUN_KL_CHECK = True
KL_CHECK_FRACTIONS = [0.02, 0.05, 0.10, 0.20]
```

Outputs:

```text
sae_scale_calibration.csv
sae_scale_calibration_kl_check.csv
printed RECOMMENDED_SCALES_BY_FEATURE
```

Copy the recommended result into:

```python
STEERING_SCALES = {
    (18, 378): [-3180.0, -1590.0, 0.0, 1590.0, 3180.0],
}
```

## Step 3: SAE Decoder-Direction Steering

Preferred script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

Full original runner:

```text
experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
```

What it does:

```text
For each task / feature / scale, the script independently builds a prompt,
applies residual-stream steering with scale * sae.W_dec[feature], generates an
answer, and records output and KL metrics.
```

Important boundary:

```text
Each generation call is independent. The model does not remember previous
questions, previous scales, or previous answers unless they are explicitly put
into the prompt.
```

Typical Colab setup:

```python
RUN_TAG = "gemma_sae_steering_fast_readout_3tasks"

STEERING_FEATURES = [
    {"real_layer": 41, "feature_index": 29, "feature_label": "L41_f29_late_readout_probe"},
]

STEERING_SCALES = {
    (41, 29): [-12600.0, -6300.0, 0.0, 6300.0, 12600.0],
}

TEST_TASKS = [
    "Дай прямой аналитический вывод: ...",
]

RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True
SAVE_PER_TOKEN_DETAILS = True
INCLUDE_BASE_TEXT_FULL_IN_CSV = False

%run -i experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

Main outputs:

```text
sae_feature_steering_generation_full_metrics_<RUN_TAG>.csv
sae_feature_steering_generation_summary_metrics_<RUN_TAG>.csv
sae_feature_steering_generation_full_metrics_with_tf_kl_<RUN_TAG>.csv
sae_teacher_forced_per_token_kl_details_<RUN_TAG>.csv
sae_teacher_forced_kl_summary_by_feature_scale_<RUN_TAG>.csv
sae_feature_steering_base_text_<RUN_TAG>.txt
```

Collected local ZIPs are stored in:

```text
experiments/steering/metrics/
```

Read:

```text
experiments/steering/metrics/README.md
```

## Step 4: Optional Dense x_order_orth Axis Steering

Script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

This is not SAE feature steering.

Difference:

```text
SAE feature steering:
  residual_stream += scale * sae.W_dec[feature]

Dense Grade4 axis steering:
  residual_stream += scale * x_order_orth_axis[layer]
```

Dense axis steering uses:

```text
grade4_axis_component_vectors_by_layer.npz
```

from the Grade4 package.

Use this when the question is:

```text
Does the discovered Grade4 component axis itself move generation behavior or KL,
independent of sparse SAE feature approximations?
```

## What To Record After A Run

After any serious steering run, record in `research_context_current.md`:

```text
1. Which base context was used.
2. Which features or dense axes were used.
3. Which scales were tested.
4. Whether final next-token KL moved.
5. Whether teacher-forced KL localized the effect.
6. Whether visible outputs changed regime or only distribution metrics moved.
7. Whether the result strengthens SAE causal involvement, dense-axis causal
   involvement, or only readout correlation.
```

## Main Claim Boundary

SAE steering can support causal involvement of sparse feature candidates, but it
does not replace the Grade4 hidden-geometry result.

Correct relation:

```text
Grade4 geometry gives the latent axis and content/order separation.
SAE evidence tries to find sparse feature correlates or partial causal handles.
Steering/KL tests whether those candidates actually move distributions or
behavior.
```

