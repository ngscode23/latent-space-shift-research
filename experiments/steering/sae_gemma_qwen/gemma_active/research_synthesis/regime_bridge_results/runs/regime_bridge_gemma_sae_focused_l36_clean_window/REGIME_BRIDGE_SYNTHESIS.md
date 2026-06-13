# Regime Bridge Synthesis: `gemma_sae_focused_l36_clean_window`

Source archive:

```text
C:\Users\stasv\Downloads\regime_bridge_gemma_sae_focused_l36_clean_window.zip
```

Extracted under:

```text
experiments/steering/sae_gemma_qwen/gemma_active/research_synthesis/regime_bridge_results/runs/regime_bridge_gemma_sae_focused_l36_clean_window/
```

Run tag inside files:

```text
gemma_sae_focused_l36_clean_window_target_3tasks
```

## Run Shape

Actual manifest:

```text
hook:                  blocks.36.hook_resid_post
pool:                  prompt_mean
d_model:               3840
position_mode:          all_tokens
target_train_texts:     6
control_train_texts:    7
target_test_texts:      3
control_test_texts:     3
axis_tasks:             2
eval_tasks:             2
v_regime_norm:          4514.54
```

This is a small clean-window run, not a full 20/20 bank run. Projection rows
use 6 target and 6 control prompt vectors because each side has 3 held-out
texts times 2 axis tasks. Causal generation uses 1 held-out text per side and
2 eval tasks.

## Main Signal

The hidden-state projection signal is very strong.

```text
variant          target_mean     control_mean    gap       AUC-like  balanced_acc
raw              32899.78        29193.87        3705.91   1.000     1.000
sae_orth          2054.96        -1432.31        3487.27   1.000     1.000
grade_orth       19987.46        16352.67        3634.80   1.000     1.000
grade_sae_orth    -196.80        -3542.05        3345.25   1.000     1.000
```

The important row is `grade_sae_orth`: after removing Grade directions and
the selected SAE feature directions, held-out target/control separation remains
perfect in this run.

Mechanistic reading: the bank-level regime vector is not exhausted by the
known Grade axes or by the two selected L36 SAE directions. It contains a
residual regime component in L36 residual-stream space.

## Controls

Random same-dimensional unit axes:

```text
n=16
AUC mean=0.340
AUC p95=1.000
AUC max=1.000
gap mean=-35.11
gap p95=67.31
gap max=87.73
```

The AUC p95 is not informative here because the held-out set is tiny; two
random axes can get perfect rank separation by chance. The gap scale is much
more informative: actual gaps are around 3345-3706, while the largest random
gap is 87.7. Actual is roughly 38x to 42x the strongest random gap.

Label-permutation train axes:

```text
n=16
AUC mean=0.557
AUC p95=1.000
AUC max=1.000
gap mean=299.75
gap p95=3407.84
gap max=3412.01
```

This is the caution flag. Some label-permutation axes recover very large
held-out gaps. That means the small clean-window bank has enough shared
structure that shuffled train labels can still pick up target/control-ish
geometry. The actual raw and grade_orth gaps exceed the permutation max; the
grade_sae_orth gap is slightly below the strongest permutation gap. Therefore
the geometry signal is strong, but the independence claim should not yet be
treated as closed.

## Overlap With Known Axes

Cosines with Grade axes:

```text
x_full:        +0.316
x_content:     +0.241
x_order_orth:  +0.205
x_order:       -0.032
```

Cosines with selected SAE directions:

```text
L36 f323:      +0.409
L36 f1914:     -0.233
```

`L36 f323` is the strongest listed SAE overlap and the top absolute SAE
direction in the local top-20 report. This means the raw regime vector partly
uses an already-known SAE direction, but not only that direction.

Projection-out norm effects:

```text
raw norm:                         4514.54
after SAE f1914/f323 removal:     3957.57   kept=87.66%
after Grade removal:              4282.34   kept=94.86%
after Grade+SAE removal:          4017.61   kept=88.99%
```

Most of the vector remains after removing these known directions. The non-additive
norm removal indicates overlap between Grade and SAE directions, so they should
not be read as independent components.

## Causal Generation

Verdict analyzer was run with:

```text
python experiments/steering/sae_gemma_qwen/gemma_active/regime_bridge_verdict.py \
  <causal_csv> --no-semantic
```

It produced:

```text
regime_bridge_causal_generation_gemma_sae_focused_l36_clean_window_target_3tasks_VERDICT.csv
```

Directness result:

```text
rows total: 136
script-switch rows: 0
baseline directness/100w:
  control = +0.000
  target  = -0.980
  gap     = -0.980
```

The directness metric is weak for this task pair. Control-side rows do not
move on directness at any alpha. Target-side rows move at alpha=0.5, but the
effect is tiny, low-n, and partly mirrored by random controls.

KL / text movement summary at matched alphas:

```text
control alpha=0.50:
  random KL mean       0.0504
  raw actual KL         0.0186
  sae_orth KL           0.0197
  grade_orth KL         0.0266
  grade_sae_orth KL     0.0203

target alpha=0.50:
  random KL mean       0.0516
  raw actual KL         0.0560
  sae_orth KL           0.0519
  grade_orth KL         0.0595
  grade_sae_orth KL     0.0528
```

The actual axis is not reliably stronger than same-norm random axes in causal
generation. On control it is weaker than random by KL; on target it is roughly
matched or slightly stronger at alpha=0.5. This supports "intervention causes
some movement" but not "specific bidirectional behavioral control."

## Scientific Verdict

What got stronger:

1. There is a strong L36 bank-level hidden-state contrast between target and
   control texts.
2. The contrast survives held-out evaluation in the clean-window run.
3. The contrast survives removing known Grade axes and selected SAE feature
   directions.
4. The raw vector overlaps known geometry but is not exhausted by it.

What did not get proven:

1. The axis is not yet a stable bidirectional behavioral steering handle.
2. The causal-generation effect does not clearly beat same-norm random axes.
3. The independence claim is not fully closed because label-permutation axes
   sometimes produce large projection gaps in this small sample.
4. Directness/hedge markers are not sensitive enough for these two eval tasks.

Best current formulation:

```text
This run supports a residual bank-level regime component in L36 hidden-state
space: target/control separation remains after Grade+SAE projection-out. It
does not yet establish clean behavioral steering, because causal generation
movement is weak and not consistently above same-norm random controls.
```

## Mechanism Hypothesis

The coherent target bank appears to induce a residual-stream state shift that
is partly aligned with known content/order/SAE directions, especially L36 f323,
but the dominant held-out separation is carried by a broader distributed
subspace. The two selected SAE features are local readout probes of that
regime, not the whole regime.

The causal-generation weakness suggests that the bank vector is a good
diagnostic/readout axis but not necessarily a clean control axis. It may
separate hidden states after prompt processing while being too entangled or too
broad to force a precise visible behavior when injected uniformly across all
tokens.

## Next Experiment

Run the same bridge on a larger and cleaner bank:

```python
REGIME_BRIDGE_N_EVAL_TEXTS_PER_SIDE = 3
REGIME_BRIDGE_N_RANDOM_CAUSAL_AXES = 4
REGIME_BRIDGE_N_RANDOM_AXES = 64
REGIME_BRIDGE_N_PERMUTATION_AXES = 64
REGIME_BRIDGE_N_AXIS_TASKS = 2
REGIME_BRIDGE_N_EVAL_TASKS = 3
REGIME_EXTRACT_BATCH_SIZE = 8
```

Use less politically loaded eval tasks for the next causal readout, because
the current two tasks mostly elicit short one-sentence ideological answers and
make marker metrics brittle. Recommended eval tasks:

```python
TEST_TASKS = [
    "Назови главный тезис текста одним предложением.",
    "Выдели основной механизм, описанный в тексте.",
    "Сформулируй строгий вывод без перечисления альтернатив.",
]
```

Then run verdict with semantic scoring if possible:

```bash
python experiments/steering/sae_gemma_qwen/gemma_active/regime_bridge_verdict.py \
  <causal_csv>
```

If LaBSE is unavailable, `--no-semantic` is acceptable, but behavioral evidence
will remain weaker.

Decision rule for the next run:

```text
Strong result:
  grade_sae_orth AUC/gap beats random and permutation controls,
  and causal generation beats same-norm random on KL plus semantic/behavioral
  movement without script switch.

Moderate result:
  grade_sae_orth projection survives but causal generation remains random-like.
  Then the axis is a diagnostic hidden-state readout, not a steering handle.

Weak result:
  grade_sae_orth falls into permutation range on larger N.
  Then this clean-window result was partly sample/bank structure.
```
