# SAE Feature Steering — Mechanistic Interpretability Toolkit

A set of scripts for analyzing and steering language model behavior via SAE (Sparse Autoencoders).  
Primary goal: find SAE features that mediate the model's "writing mode" (analytic, diagnostic, cold tone), and verify their causal contribution via steering + KL divergence.

Built for **Gemma-3-12B-IT** + **gemma-scope-2-12b-it-res-all** (l0_small).  
Frameworks: [TransformerLens](https://github.com/neelnanda-io/TransformerLens), [SAELens](https://github.com/jbloomAus/SAELens).

---

## File Structure

```
steering/
├── 01_candidate_discovery_and_rough_sae_patching.py   ← entry point, defines BASE_TEXT
├── steering_gemma3_V1.py                              ← steering without KL (mid-level version)
├── sae_feature_steering_light.py                      ← lightweight steering (no KL)
├── sae_feature_steering_v2_no_control.py              ← steering + diagnostics (KL, patching, profiles)
├── sae_steering_with_kl_full.py                       ← full version with two KL layers
│
└── docs/
    ├── 01_candidate_discovery.md
    ├── steering_gemma3_V1.md
    ├── sae_feature_steering_light.md
    ├── sae_feature_steering_v2_no_control.md
    └── sae_steering_with_kl_full.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install transformer_lens sae_lens pandas torch tqdm matplotlib
```

### 2. Check model and SAE availability

Before running:
- The model must exist in TransformerLens
- An SAE must exist for it in SAELens
- You need the correct `release` name and `sae_id` format

### 3. Prepare `sae_order_feature_contrast.csv`

This is the input file with contrastive activations. It is **not included** in the repo — it must be generated fresh for each model.  
Minimum required columns:

```
layer, feature_index, x_order_orth_component_delta, interpretation_status
```

### 4. Run scripts in order

```
01_candidate_discovery_and_rough_sae_patching.py
→ steering_gemma3_V1.py  (optional, basic steering)
→ sae_feature_steering_light.py  (lightweight steering, no KL)
→ sae_feature_steering_v2_no_control.py  (diagnostics)
→ sae_steering_with_kl_full.py  (full analysis with KL)
```

---

## BASE_TEXT — where it comes from

`BASE_TEXT` is defined in  
`01_candidate_discovery_and_rough_sae_patching.py`  
via `prompts_target[0]`.

```python
# In 01_candidate_discovery_and_rough_sae_patching.py:
prompts_target = ["Your target text here..."]

BASE_TEXT = prompts_target[0]  # used by all subsequent scripts
```

All subsequent scripts (`steering_gemma3_V1.py`, `sae_feature_steering_light.py`, etc.) access `prompts_target[0]` as a global variable — they do **not** redefine it themselves.

> **Optionally** — `BASE_TEXT` can be moved to a separate config file, `.env`, or passed as a CLI argument. The logic stays the same: all scripts must receive the same string.

---

## Full Analysis Pipeline

```
0. Verify model and SAE
1. Generate contrast CSV
2. Find top candidate features
3. Inspect top activating contexts
4. Check target-vs-control activation contrast
5. Clean contexts (remove BOS/punctuation)
6. Run delta-only ablation
7. Run all-position contribution
8. Run final logit effect
9. Run top logit deltas
10. Select 2–5 features
11. Run steering + final-next-token KL + teacher-forced KL
12. Read summary CSV — don't fall in love with one pretty output
```

---

## Script Descriptions

### `01_candidate_discovery_and_rough_sae_patching.py`

**Role:** entry point. Loads the model, reads the contrast CSV, loads SAEs, finds top order-specific features, runs causal mediation analysis using `patch_sae_features`.

**Key functions:**
- `get_feature_top_contexts()` — top activating contexts for a feature
- `inspect_top_mediators_on_texts()` — iterates over all top mediators
- `patch_sae_features()` — zero-ablation of features in residual activation
- `run_mediation_experiment()` — main causal mediation loop

**Output files:**
```
sae_feature_top_activating_contexts.csv
causal_mediation_sae_order_features_results.csv
```

**Variables to change when switching models:**
```python
MODEL_NAME        = "google/gemma-3-12b-it"
CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"
TOP_K             = 30
```

---

### `steering_gemma3_V1.py` — mid-level version (no KL)

**Role:** basic steering run with multiple tasks, two generation modes (greedy/sampled), text metrics, and comparison with baseline (scale=0).

**Supports:**
- greedy + sampled generation
- 8 analysis tasks for BASE_TEXT
- full text storage + prompt hash
- metrics: length, Jaccard vs baseline, n-gram uniqueness

**Output files:**
```
sae_feature_steering_generation_full_metrics.csv
sae_feature_steering_generation_summary_metrics.csv
sae_feature_steering_base_text.txt
```

**What to change:**
```python
BASE_TEXT = prompts_target[0]  # set in 01_candidate...

STEERING_FEATURES = [
    {"real_layer": 41, "feature_index": 13686, ...},
]
STEERING_SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]
```

---

### `sae_feature_steering_light.py` — lightweight version (no KL)

**Role:** the simplest steering — just generation + printout. Good for quick visual checks of a feature's effect.

**Difference from V1:** no teacher-forced KL, no metrics, no baseline comparison. Output goes to console and CSV only.

**Hook:**
```python
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale):
    dec_vec = sae.W_dec[feature_index]
    patched = activation + scale * dec_vec  # adds decoder direction to all positions
    return patched
```

**Output files:**
```
sae_feature_steering_generation_test_sampled.csv
```

**What to change:**
```python
STEERING_FEATURES = [(41, 13686), (41, 208), (41, 207)]
STEERING_SCALES   = [-3.0, -1.5, 0.0, 1.5, 3.0]
N_SAMPLES         = 5
BASE_TEXT         = prompts_target[0]
```

---

### `sae_feature_steering_v2_no_control.py` — diagnostic version

**Role:** extended analysis without control comparison, but with several additional diagnostics.

**Capabilities:**

1. **Next-token KL** — `compute_next_token_kl_feature_steering()`  
   KL(base‖patched), KL(patched‖base), JS, logit_l2, top-token change

2. **Activation patching target → control** (flag-controlled) — `capture_activation_for_tokens()`, `build_control_patch_hook()`  
   Position alignment: `CONTROL_PATCH_ALIGNMENT = "right"`

3. **Teacher-forced per-token KL** — `teacher_forced_per_token_kl()`  
   Comparison on a shared token trajectory, not free-running

4. **Unembed projection W_dec[f] @ W_U** — `run_unembed_projection()`  
   Top-10 positive and top-10 negative tokens

5. **Positional activation profile** — `run_positional_profile()`  
   x=position, y=activation → CSV + PNG

6. **Short-prompt ablation for feature 208** — `run_short_prompt_ablation()`  
   12 "not X, but Y" prompts vs 12 without contrast

**Output files:**
```
sae_feature_steering_generation_with_causal_metrics.csv
sae_feature_unembed_top_tokens.csv
sae_feature_position_activation_profile.csv
sae_feature_position_activation_profile.png
sae_feature_208_short_prompt_ablation.csv
sae_feature_208_short_prompt_ablation_summary.csv
```

**Feature flags:**
```python
RUN_GENERATION                            = True
RUN_NEXT_TOKEN_KL                         = True
RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL = False  # requires prompts_control
RUN_UNEMBED_PROJECTION                    = True
RUN_POSITIONAL_PROFILE                    = True
RUN_SHORT_PROMPT_ABLATION                 = True
```

---

### `sae_steering_with_kl_full.py` — full version with two KL layers

**Role:** the final verification script. Contains both KL layers as diagnostics for the token distribution shift.

> KL here is not a training loss — it is a measurement: how much does steering shift the next-token distribution?

**KL Layer 1: Final next-token KL** — `compute_final_next_token_kl()` (line 393)

Computed during the steering run, at the last prompt position:
```
KL(p_base(next_token | prompt) ‖ p_patched(next_token | prompt))
```

```python
# base — no hook
base_logits = model([prompt])[:, -1, :].float()

# patched — with SAE steering hook
with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
    patched_logits = model([prompt])[:, -1, :].float()

kl_base_to_patched = sum(p_base * (log_p_base - log_p_patched))
kl_patched_to_base = sum(p_patched * (log_p_patched - log_p_base))
js = 0.5 * kl_base_to_patched + 0.5 * kl_patched_to_base
```

**KL Layer 2: Teacher-forced per-token KL** — `teacher_forced_per_token_kl()` (line 734)

Runs after generation. Takes `reference_continuation` from scale=0 (baseline), concatenates `prompt + reference_tokens`, computes logits for both base and patched on the same trajectory (teacher forcing, not free-running), computes KL at each step.

```python
# reference = what the base model (scale=0) generated
full_input = prompt_tokens + reference_tokens

base_logits    = model(full_input)   # no hook
patched_logits = model(full_input)   # with hook at every position

# KL at each reference step
for i, ref_token in enumerate(reference_tokens):
    kl_i = KL(base_probs[i] ‖ patched_probs[i])
```

Summary: sum/mean/max/p95 KL, top-token change rate, delta logprob of reference tokens.

**Hook:**
```python
# patched = activation + scale * W_dec[feature]
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale):
    dec_vec = sae.W_dec[feature_index]
    return activation + scale * dec_vec
```

**Output files:**
```
sae_feature_steering_generation_full_metrics.csv
sae_feature_steering_generation_summary_metrics.csv
sae_feature_steering_base_text.txt
sae_feature_steering_generation_full_metrics_with_tf_kl.csv
sae_teacher_forced_per_token_kl_details.csv         (if SAVE_PER_TOKEN_DETAILS=True)
sae_teacher_forced_kl_summary_by_feature_scale.csv  ← most important
```

**Flags:**
```python
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION    = True
SAVE_PER_TOKEN_DETAILS                    = True
MAX_REFERENCE_TOKENS_FOR_TF_KL            = 220
```

---

## Adapting to a New Model

**Change:**
```python
MODEL_NAME          = "new_model_name"
SAE_RELEASE         = "new_sae_release"
SAE_ID_TEMPLATE     = "layer_{layer}_..."
HOOK_POINT_TEMPLATE = "blocks.{layer}.hook_resid_post"
CONTRAST_CSV_PATH   = "path to new contrast CSV"
```

**Do not reuse:**
```
feature_index
layer numbers from old CSV
```

> Feature and layer numbers do not transfer between models. You cannot take feature 13686 from Gemma and apply it to a different model.

**Order for a new model:**
```
1. New contrast CSV
2. New candidates
3. New contexts
4. New causal checks
5. Only then: steering + KL
```

---

## Output Files — What to Read

| File | What it tells you |
|------|-------------------|
| `sae_order_feature_contrast.csv` | Which features differ between target and control |
| `sae_top_candidate_features.csv` | Top-K candidates after filtering |
| `sae_feature_top_activating_contexts.csv` | Where they activate in text |
| `sae_feature_target_vs_control_activation_contrast.csv` | Target-specific or broad/control-heavy |
| `sae_clean_selected_feature_contexts.csv` | Clean contexts for manual interpretation |
| `sae_all_position_feature_contribution_selected.csv` | Whether contribution exists across all positions |
| `sae_final_logit_effect_selected_features.csv` | Whether next-token distribution shifts |
| `sae_top_logit_deltas_selected_features.csv` | Which specific tokens move |
| `sae_feature_steering_generation_summary_metrics.csv` | Whether generation changes |
| `sae_teacher_forced_kl_summary_by_feature_scale.csv` | **Most important** — full steering + KL analysis |

---

## The Shortest Rule

```
Don't reuse old feature indices.
Start with a new contrast CSV.
Then new candidates.
Then new contexts.
Then new causal checks.
Only then: steering + KL.
```

