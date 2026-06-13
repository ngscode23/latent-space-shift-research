# Regime Bridge Synthesis: `l36_context_probe_clean_window`

Source archive:
`C:\Users\stasv\Downloads\regime_bridge_runs_l36_context_probe_clean_window.zip`

Extracted under:
`experiments/steering/sae_gemma_qwen/gemma_active/research_synthesis/regime_bridge_results/runs/regime_bridge_runs_l36_context_probe_clean_window_20260609/`

## Run Shape

- Hook: `blocks.36.hook_resid_post`
- Pool: `prompt_mean`
- Prompt mode: `context_probe`
- Position mode: `all_tokens`
- Axis tasks: `1`
- Eval tasks: `1`
- Eval texts: `all_test`
- Target texts: `9` total = `6 train + 3 test`
- Control texts: `10` total = `7 train + 3 test`
- `v_regime_norm`: `4521.193848`
- Actual causal rows: `144`
- Random causal rows: `180`
- Total causal rows: `324`

The split is correct. No texts disappeared: target `6+3=9`, control `7+3=10`.

## Projection Audit

| axis variant | AUC-like | gap |
|---|---:|---:|
| raw | 1.000 | 3718.732 |
| sae_orth | 1.000 | 3495.028 |
| grade_orth | 1.000 | 3644.912 |
| grade_sae_orth | 1.000 | 3351.831 |

Projection controls:

- Random same-dim gap p95: `130.362`, max: `155.138`
- Label-permutation gap p95: `3421.207`, max: `3681.045`

Reading:

- The geometry signal is very strong against random same-dim directions.
- `raw`, `sae_orth`, and `grade_orth` exceed label-permutation p95 by gap.
- `grade_sae_orth` remains AUC=1, but its gap `3351.831` is below label-permutation p95 `3421.207`.
- Therefore: strong hidden-state readout; independence after removing both Grade and selected SAE dirs is supported but not fully closed by this small split.

## Orthogonalization

Norm kept:

- `raw`: `1.000`
- `sae_orth`: `0.876716`
- `grade_orth`: `0.948624`
- `grade_sae_orth`: `0.890000`

Removed directions:

- SAE: `L36_f1914`, `L36_f323`
- Grade: `x_content`, `x_order_orth`

The regime direction is not mostly consumed by these removals; `grade_sae_orth` still keeps about `89%` of the raw norm.

## Cosines

Notable cosines with `v_regime`:

- `x_content`: `0.241307`
- `x_full`: `0.315895`
- `x_order`: `-0.032096`
- `x_order_orth`: `0.204656`
- `L36_f1914`: `-0.233065`
- `L36_f323`: `0.408598`

Top SAE absolute cosine candidates:

- `L36_f323`: `0.408598`
- `L36_f15927`: `0.398123`
- `L36_f109`: `0.397088`
- `L36_f14074`: `0.396853`
- `L36_f56`: `-0.396374`
- `L36_f121`: `0.395794`

These are candidate alignment directions, not causal-proven features.

## Causal Generation

At `alpha=0.5`, actual next-token KL:

| variant | control KL | target KL |
|---|---:|---:|
| raw | 0.015404 | 0.062168 |
| sae_orth | 0.018380 | 0.068599 |
| grade_orth | 0.021462 | 0.071501 |
| grade_sae_orth | 0.018973 | 0.066463 |

Random causal KL quantiles at `alpha=0.5`:

- control random p50: `0.044066`, p95: `0.140361`
- target random p50: `0.052179`, p95: `0.131827`

Reading:

- Actual KL is not stronger than same-norm random controls.
- On control side, actual KL is below random median at `alpha=0.5`.
- On target side, actual KL is around random median/upper-middle but below p95.
- This does not close causal steering specificity.

Script/language:

- `script_switch_flag = 0` throughout.
- No Russian-to-English collapse in this run.

## Verdict Analyzer

Ran:

```text
regime_bridge_verdict.py regime_bridge_causal_generation_l36_context_probe_clean_window.csv --no-semantic
```

Output:

```text
regime_bridge_causal_generation_l36_context_probe_clean_window_VERDICT.csv
```

Verdict analyzer notes:

- Baseline directness/100w: control `+0.000`, target `-0.758`
- Directness rows show some positive `gap_closed` for target-side interventions.
- This is based on only `3` actual rows per side/alpha and a marker-based proxy.
- Since the single eval question already elicits direct answers, directness is not a strong behavioral endpoint here.

## Status

This run strengthens the same conclusion as the previous bridge run:

1. Hidden-state geometry/readout signal is strong.
2. Signal survives selected Grade and SAE direction removals, though `grade_sae_orth` is not above label-permutation p95 by gap in this small split.
3. Generation-time causal movement is measurable but not specific enough versus same-norm random controls.
4. The next proof step should use forced-choice decision margins, not free-form generation.

Recommended next experiment:

```text
regime_decision_probe_causal_audit.py
```

Primary metric should be:

```text
margin = logp(A) - logp(B)
```

under baseline, actual patch, same-norm random patch, and permutation-axis patch.
