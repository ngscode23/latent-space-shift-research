# Grade 4 Axis Decomposition Results / Qwen3-14B

## 1. Run Identity

```text
Source: C:\Users\stasv\Downloads\red_team_hidden_geometry_results_breakthrough_grade4_axis_decomposition03.zip
SHA256: 0D6C451B02129EF1425703207B751330DE341ABD033BBCD3F0EDA92BFFD4C81D
MODEL_ID: Qwen/Qwen3-14B
RUN_LABEL: breakthrough_grade4_axis_decomposition
Questions: 15
Reference condition: neutral
Grade4 axes: ['x_full', 'x_content', 'x_order', 'x_order_orth']
Layer bands: ['middle', 'late', 'all']
Alpha values: [0.1, 0.25, 0.5, 0.75]
Generation batch size: 16
SAVE_STEP_RAW: False
```

This is the correct `03.zip` archive. The earlier `02.zip` archive is not used for research claims. This run completed `middle`, `late`, and `all`. The memory fix should not distort the metrics because it disables heavy step-raw saving and does not change prompts, axes, alpha values, or intervention math.

## 2. Main Verdict

```text
order_component_supported
```

Grade 4 supports the stronger claim:

```text
Vector X contains a separable discourse-order / rhetorical-regime component,
not only a lexical/semantic target-family trace.
```

The key reason: `x_order_orth`, the order component after layerwise removal of the `x_content` projection, remains active. In the main middle/middle causal readout it produces the largest +component/-component gap.

## 3. Axis Decomposition

Middle-band decomposition:

```text
full_norm:                         122.150690
content_norm:                      135.718574
order_norm:                        71.961707
order_orth_norm:                   64.376540
content energy / full:             1.234488
order energy / full:               0.347065
order_orth energy / full:          0.277756
cos(content, full):                0.849368
cos(content, order):               -0.444233
cos(content, order_orth):          0.000000
cos(order_orth, full):             0.527026
```

Mechanistically: the content component is large, but the order component is not numerical residue. In the middle band, `x_order_orth` keeps meaningful energy relative to the full-axis scale and remains aligned with `x_full`. The negative `cos(content, order)` shows that content and coherent-order residue partially compensate each other, which is why `content_norm` can exceed `full_norm`.

## 4. Geometry Readout

Projection fractions:

```text
target on x_full:                  0.976583
target on x_content:               0.724737
target on x_order:                 0.241598
target on x_order_orth:            0.978944

sentence_shuffle on x_content:     0.973914
sentence_shuffle on x_order_orth:  0.007214
word_shuffle on x_order_orth:      0.251849
length_matched on x_order_orth:    -0.052867
```

This is a clean separator: sentence shuffle strongly occupies `x_content` but almost not `x_order_orth`; target strongly occupies both `x_full` and `x_order_orth`. Coherent target ordering therefore contributes a distinct direction beyond sentence-shuffled content.

## 5. Causal Component Test

Primary readout: intervention `middle`, readout `middle`, alpha `0.75`.

Neutral base:

```text
x_order_orth gap:                  3.726561
x_order gap:                       3.384538
x_full gap:                        3.308553
x_content gap:                     2.990294
```

Target base:

```text
x_order_orth gap:                  3.698789
x_order gap:                       3.383840
x_full gap:                        3.330993
x_content gap:                     2.997980
```

Middle/middle dose slopes:

```text
neutral x_order_orth slope:        2.459800
neutral x_order slope:             2.250374
neutral x_full slope:              2.229508
neutral x_content slope:           2.021195

target x_order_orth slope:         2.463462
target x_order slope:              2.257763
target x_full slope:               2.234000
target x_content slope:            2.028973
```

`x_order_orth` ranks first by gap in middle/middle for both neutral and target. This is the central Grade 4 result.

## 6. All-Layer / Late Checks

Because this run completed all bands, it includes optional all-layer comparison.

All -> middle, alpha 0.75, neutral base:

```text
x_order_orth gap:                  5.280896
x_order gap:                       5.066052
x_full gap:                        4.153453
x_content gap:                     3.662412
```

Late -> middle gaps are near zero:

```text
neutral x_order_orth gap:          0.012506
neutral x_full gap:                0.024110
neutral x_content gap:             0.035158
```

This strengthens the localization interpretation: component causal controllability is in middle intervention, not late-only intervention. All-layer works, but it is a coarser global perturbation.

## 7. Established / Not Established

Established by this run:

```text
1. Grade 3 full Vector X decomposes into a large content component and a separable order component.
2. x_order_orth is not erased by removing the x_content projection.
3. target projects strongly on x_order_orth while sentence_shuffle does not.
4. x_order_orth has stable bidirectional +component/-component causal gaps.
5. In middle/middle readout, x_order_orth is the strongest causal component.
```

Not established by this run:

```text
1. permanent weight/topology change;
2. formal attractor basin;
3. reviewer-grade visible behavioral control;
4. SAE-level named-feature localization;
5. cross-model universality of the order component.
```

## 8. Short Claim

```text
Grade 4 supports that the Qwen3-14B target-conditioned Vector X is not merely a
sentence-shuffled content axis. It contains a separable discourse-order /
rhetorical-regime component that remains causally steerable in the middle
residual stream after the content projection is removed.
```
