# Latent Shift Research Synthesis

## Claim Ladder

1. The original attractor runs support a context-induced latent/readout regime shift.
2. The strict formal-attractor claim remains mixed unless basin, stability, return, geometry, and compression all pass.
3. The Grade 3 hidden-geometry run supports a robust causal internal latent axis in Qwen3-14B.
4. The Grade 4 run supports a separable discourse-order / rhetorical-regime component beyond sentence-shuffled content.

## Grade 3 Anchor

- Model: `Qwen/Qwen3-14B`
- Evidence status: `computed`
- Target middle projection: `0.976583`
- Middle direction cosine: `0.852397`
- Middle R2: `0.744126`
- Neutral middle +X/-X gap at alpha 0.75: `3.313378`
- Visible behavioral gate failure_code: `below_random_p95`

## Grade 4 Axis Decomposition

- Evidence status: `computed`
- Target projection on x_order_orth: `0.978944`
- Sentence-shuffle projection on x_order_orth: `0.007214`
- Neutral middle/middle x_order_orth gap at alpha 0.75: `3.726561`
- Target middle/middle x_order_orth gap at alpha 0.75: `3.698789`

Mechanistic meaning: `x_order_orth` survives removal of the content projection and remains causally steerable. This supports a separable discourse-order / rhetorical-regime component.


## Grade 4 Status

- Script exists: `True`
- Results available: `False`
- Status: `metrics_summary_available`
- Metrics summary: `C:\Users\stasv\OneDrive\Рабочий стол\agent\metrics\qwen3_14b_grade4_axis_decomposition03\summary.json`

Recommended paper wording:

```text
We observe a context-conditioned latent geometry/readout shift. In Qwen3-14B, a target-reference Vector X extracted from hidden states causally steers the internal generation trajectory under middle-layer residual-stream intervention. Grade 4 shows that this axis contains a separable discourse-order / rhetorical-regime component beyond sentence-shuffled content. We do not claim permanent weight-level change, formal basin status, or reviewer-grade visible behavioral control.
```
