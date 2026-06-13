# Regime Bridge Results Index

Canonical folder for results produced by:

```text
experiments/steering/sae_gemma_qwen/gemma_active/regime_axis_grade_bridge_causal_audit.py
```

Post-hoc verdict analyzer:

```text
experiments/steering/sae_gemma_qwen/gemma_active/regime_bridge_verdict.py
```

## Folder Layout

```text
regime_bridge_results/
  README.md
  runs/
    regime_bridge_gemma_sae_focused_l36_clean_window/
    regime_bridge_runs_l36_context_probe_clean_window_20260609/
```

All bridge CSV/NPZ/MD artifacts should live under `runs/`. Do not add duplicate
summary copies in the root `research_synthesis/` folder.

## Run 1: `gemma_sae_focused_l36_clean_window_target_3tasks`

Folder:

```text
runs/regime_bridge_gemma_sae_focused_l36_clean_window/
```

Detailed report:

```text
runs/regime_bridge_gemma_sae_focused_l36_clean_window/REGIME_BRIDGE_SYNTHESIS.md
```

Core numbers:

```text
hook = blocks.36.hook_resid_post
pool = prompt_mean
target_train = 6
control_train = 7
target_test = 3
control_test = 3
axis_tasks = 2
eval_tasks = 2
v_regime_norm = 4514.54
```

Projection:

```text
raw             AUC=1.000 gap=3705.91
sae_orth        AUC=1.000 gap=3487.27
grade_orth      AUC=1.000 gap=3634.80
grade_sae_orth  AUC=1.000 gap=3345.25
```

Controls:

```text
random same-dim p95 gap = 67.31
random same-dim max gap = 87.73
label-permutation p95 gap = 3407.84
label-permutation max gap = 3412.01
```

Reading:

```text
Strong hidden-state readout/geometry signal.
Generation-time causal steering remains inconclusive.
```

## Run 2: `l36_context_probe_clean_window`

Folder:

```text
runs/regime_bridge_runs_l36_context_probe_clean_window_20260609/
```

Detailed report:

```text
runs/regime_bridge_runs_l36_context_probe_clean_window_20260609/REGIME_BRIDGE_SYNTHESIS.md
```

Input split:

```text
target  = 6 train + 3 test = 9
control = 7 train + 3 test = 10
```

Core numbers:

```text
hook = blocks.36.hook_resid_post
pool = prompt_mean
axis_tasks = 1
eval_tasks = 1
position_mode = all_tokens
v_regime_norm = 4521.193848
causal rows = 324
```

Projection:

```text
raw             AUC=1.000 gap=3718.732
sae_orth        AUC=1.000 gap=3495.028
grade_orth      AUC=1.000 gap=3644.912
grade_sae_orth  AUC=1.000 gap=3351.831
```

Controls:

```text
random same-dim p95 gap = 130.362
random same-dim max gap = 155.138
label-permutation p95 gap = 3421.207
label-permutation max gap = 3681.045
```

Reading:

```text
Geometry/readout remains strong.
raw/sae_orth/grade_orth beat label-permutation p95 by projection gap.
grade_sae_orth remains AUC=1 but is below label-permutation p95 by gap.
Free-form causal generation is still not stronger than random causal controls.
No script-switch collapse in this run.
```

## Current Scientific Status

```text
Descriptive hidden-state shift: strong.
Content/order separation from Grade4: strong.
Bank-level L36 regime readout: strong.
Survival after Grade+SAE removal: promising but not fully closed under small-bank permutation controls.
Free-form generation steering: causal-inconclusive.
```

Do not state that `v_regime` is a stable behavioral steering handle yet. The
supported statement is narrower:

```text
A train-only L36 bank-level contrast axis separates held-out target/control
contexts and retains strong separation after selected Grade/SAE projection-out.
```

## Next Step

Stop using free-form generation as the primary behavioral endpoint. Use a
forced-choice decision-margin audit:

```text
margin = logp(A) - logp(B)
```

Required comparison:

```text
baseline margin
actual patched margin
same-norm random patched margin
label-permutation axis patched margin
```

This is the next layer needed to test:

```text
context -> hidden-state regime shift -> changed downstream decision state
```

Implementation added:

```text
experiments/steering/sae_gemma_qwen/gemma_active/regime_decision_probe_causal_audit.py
```

Output prefix:

```text
regime_decision_*
```

Primary row file:

```text
regime_decision_probe_rows_<RUN_TAG>.csv
```

Primary summary file:

```text
regime_decision_probe_summary_<RUN_TAG>.csv
```
