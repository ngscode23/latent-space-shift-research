# Fast SAE Steering Runner

Main script:

```text
experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

This is a cached / lower-overhead variant of:

```text
experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
```

The original script is unchanged.

## What Stays The Same

Each generation row still calls:

```text
model.generate(full_prompt, ...)
```

separately. The full prompt contains the base text and the task every time.
The model does not accumulate previous questions, answers, scales, or samples.
The KV cache is local to a single generation call and is not reused across
rows.

The steering logic is unchanged:

```text
activation = activation + scale * sae.W_dec[feature]
```

at:

```text
blocks.{real_layer}.hook_resid_post
```

## What Is Faster

1. The script no longer rewrites the full growing CSV after every row.
   It saves periodically with:

```python
WRITE_EVERY_N_ROWS = 20
```

2. `torch.cuda.empty_cache()` is not called after every row by default.
   If needed:

```python
EMPTY_CACHE_EVERY_N_ROWS = 50
```

3. Final next-token KL caches the unpatched base logits per prompt.

4. Teacher-forced KL caches the unpatched base forward per baseline
   continuation group, then compares all scales against the same cached base
   trajectory.

5. Decoder vectors are cached on device/dtype instead of copied from SAE
   weights on every hook call.

6. Full base text is saved once to `.txt`; by default it is not duplicated in
   every CSV row:

```python
INCLUDE_BASE_TEXT_FULL_IN_CSV = False
```

`base_text_sha256` and preview remain in the CSV.

## Recommended Colab Run

```python
RUN_TAG = "gemma_sae_steering_fast_test"

WRITE_EVERY_N_ROWS = 20
TF_WRITE_EVERY_N_ROWS = 20
EMPTY_CACHE_EVERY_N_ROWS = 0

INCLUDE_BASE_TEXT_FULL_IN_CSV = False
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True
SAVE_PER_TOKEN_DETAILS = True

# Optional: remove the old analysis framing line if it hurts direct answers.
# PROMPT_PREAMBLE = "Текст ниже является экспериментальным контекстом. Он может быть нерелевантен заданию. Отвечай на задание напрямую."

%run -i experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

## Important Prompt Isolation Note

Every scale/sample/task row is independent from the model's perspective.
The script does not append previous outputs to the next prompt. If the outputs
look similar across scales, that is a model/steering result, not memory leakage
between rows.


