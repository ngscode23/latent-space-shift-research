# `gemma_active/fast/sae_steering_with_kl_full_fast.py`

## Status

Active fast/cached variant of the full Gemma SAE steering runner.

## Purpose

Run the same conceptual SAE decoder-direction steering experiment as
`sae_steering_with_kl_full.py`, with less CSV overhead and more caching.

## What It Changes

Compared with the full runner:

- caches final next-token base logits;
- caches teacher-forced base forwards for baseline continuations;
- caches SAE decoder vectors;
- writes CSV checkpoints every `WRITE_EVERY_N_ROWS`;
- writes teacher-forced checkpoints every `TF_WRITE_EVERY_N_ROWS`;
- avoids duplicating the full base text in every CSV row by default.

## What It Does Not Change

The intervention logic remains:

```text
residual_stream += scale * sae.W_dec[feature]
```

Each generation is still independent. The model does not remember previous
questions or scales.

The main bottleneck is still autoregressive generation. If the run is slow,
reduce:

- number of tasks;
- number of features;
- sampled generations;
- `MAX_NEW_TOKENS`;
- teacher-forced per-token details.

## Main Inputs

Expected notebook globals:

```python
model
saes
prompts_target
```

Useful configs:

```python
WRITE_EVERY_N_ROWS
TF_WRITE_EVERY_N_ROWS
EMPTY_CACHE_EVERY_N_ROWS
INCLUDE_BASE_TEXT_FULL_IN_CSV
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION
RUN_TEACHER_FORCED_KL_AFTER_GENERATION
SAVE_PER_TOKEN_DETAILS
```

## Typical Colab Usage

```python
RUN_TAG = "gemma_sae_steering_fast"

WRITE_EVERY_N_ROWS = 20
TF_WRITE_EVERY_N_ROWS = 20
EMPTY_CACHE_EVERY_N_ROWS = 0

INCLUDE_BASE_TEXT_FULL_IN_CSV = False
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True
SAVE_PER_TOKEN_DETAILS = True

%run -i sae_steering_with_kl_full_fast.py
```

Local project path:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

## Interpretation Role

Use this when you need the same metrics as the full runner but want cleaner CSV
storage and less repeated base-computation overhead.


