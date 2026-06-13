# Gemma Active SAE / Axis Scripts

This folder contains the current active Gemma steering pipeline.

Recommended order:

```text
01b_full_sae_evidence_candidate_patching_gemma.py
02_scale_calibration.py
sae_steering_with_kl_full.py
fast/sae_steering_with_kl_full_fast.py
x_order_orth_axis_steering_with_kl_full.py
regime_axis_grade_bridge_causal_audit.py
regime_bridge_verdict.py
```

SAE feature workflow:

```text
01b_full_sae_evidence_candidate_patching_gemma.py
  -> selects/ranks SAE feature candidates from Grade4 SAE tables
  -> writes selected_sae_order_candidates.csv,
     ranked_sae_order_candidates_full_evidence.csv,
     rough_sae_zero_ablation_logit_results.csv,
     sae_feature_top_activating_contexts.csv

02_scale_calibration.py
  -> takes chosen (real_layer, feature_index) pairs
  -> estimates native activation scale and residual-norm fractions
  -> writes paste-ready STEERING_SCALES / RECOMMENDED_SCALES_BY_FEATURE

sae_steering_with_kl_full.py or fast/sae_steering_with_kl_full_fast.py
  -> uses those calibrated features/scales for generation and KL runs

regime_axis_grade_bridge_causal_audit.py
  -> can use selected SAE directions as orthogonalization controls through
     REGIME_ORTHO_FEATURES + REGIME_ORTHO_SAE_LAYER
```

`gemma_revision_audit.py` is a helper audit script.

`regime_axis_grade_bridge_causal_audit.py` is the bank-level regime-vector
bridge test. It builds a train-only `v_regime = mean(target_bank) -
mean(control_bank)`, tests held-out separation, measures overlap with Grade
axes / SAE directions, orthogonalizes against them, and runs bidirectional
residual-stream interventions with random and label-permutation controls.

Minimal notebook input contract:

```python
prompts_target = [...]   # target texts
prompts_control = [...]  # control texts

REGIME_EXPECTED_TARGET_TEXTS = 20
REGIME_EXPECTED_CONTROL_TEXTS = 20

%run -i regime_axis_grade_bridge_causal_audit.py
```

By default the script requires `prompts_target` and `prompts_control`. Legacy
`TARGET_BASE_TEXTS` / `CONTROL_BASE_TEXTS` are ignored unless
`REGIME_REQUIRE_PROMPT_BANKS=False`, because notebook globals can be stale
between `%run -i` calls.

Every run writes:

```text
regime_bridge_input_split_<RUN_TAG>.csv
```

That file records the original text index, split, hash, length, and preview for
each text used in train/test.

Prompt modes:

```python
REGIME_BRIDGE_PROMPT_MODE = "analyze_text"   # default: tasks ask about the text
REGIME_BRIDGE_PROMPT_MODE = "context_probe"  # text conditions hidden state; task is independent
```

Use `context_probe` when testing whether the target/control context changes
the model's later response regime on an unrelated task. In that mode, `TEST_TASKS`
should be independent probes, not questions about the stimulus text.

For causal generation, prefer an explicit number or the full held-out split:

```python
REGIME_BRIDGE_N_EVAL_TEXTS_PER_SIDE = 3
REGIME_BRIDGE_N_EVAL_TEXTS_PER_SIDE = "all_test"
```

Do not set it to `len(TARGET_TEST)` before `%run`; `TARGET_TEST` is created
inside the script after config is read, so that can be undefined or stale in a
notebook session.

`regime_bridge_verdict.py` is a post-hoc analyzer for
`regime_bridge_causal_generation_*.csv`. It does not rerun the model. It checks
whether the causal generations move toward the opposite-side baseline rather
than merely moving away from the starting behavior, compares actual movement to
same-norm random controls, normalizes directness by output length, excludes
script-switch rows for directness, and can optionally add LaBSE semantic
similarity if `sentence-transformers` is installed.

See the bilingual parent README:

```text
../README_RU_EN.md
```


