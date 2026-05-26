# Report Outline: Qwen3-14B latent axis package

Дата: `2026-05-25`

## Title Candidates

1. `A Target-Conditioned Causal Internal Axis in Qwen3-14B`
2. `Separating Content and Discourse-Order Components in a Target-Conditioned Latent Axis`
3. `Context-Conditioned Latent Geometry Shift with a Separable Rhetorical-Regime Component`
4. `Beyond Sentence-Shuffled Content: A Causal Order Component in Qwen3-14B Hidden States`

Рабочий русский заголовок:

```text
Причинная внутренняя ось и отделимая компонента дискурсивного порядка в
Qwen3-14B
```

## Abstract Draft

```text
Мы исследуем, вызывает ли структурированный target-контекст измеримый
context-conditioned latent/readout regime shift в Qwen3-14B. В Grade 3 мы
строим target-reference Vector X и показываем, что target condition почти
единично проектируется на эту ось в middle layers
(target_middle_projection_mean = 0.976583), тогда как random same-norm null
имеет среднее около 0.000040. Residual-stream intervention +X/-X причинно
двигает generation-time hidden trajectory: gap при alpha 0.75 достигает
3.313378 от neutral base и 3.336544 от target base.

В Grade 4 мы разлагаем Vector X на x_content, x_order и x_order_orth. Target
сильно проектируется на x_order_orth (0.978944), тогда как sentence-shuffled
content почти не проектируется на него (0.007214). x_order_orth дает самый
сильный middle/middle causal gap при alpha 0.75: 3.726561 от neutral base и
3.698789 от target base. Это поддерживает вывод, что Vector X содержит
отделимую discourse-order / rhetorical-regime компоненту beyond
sentence-shuffled content.

Мы не заявляем formal attractor basin, permanent topology/weight change,
reviewer-grade visible behavioral control или cross-model universality.
Следующий шаг - cross-model Grade 3 + Grade 4 replication.
```

## Figure / Table Plan

| Item | Purpose | Source |
|---|---|---|
| Figure 1: experimental spine | Показать broad shift -> Grade 3 -> Grade 4 | `research_synthesis/paper_draft_ru.md` |
| Table 0: fresh Level A run | С нуля собрать broad latent/readout metrics из `llm_attractor_colab_copy_paste.py` | `attractor_run_summary.csv` |
| Table 1: Grade 3 metrics | Зафиксировать causal internal Vector X | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` |
| Table 2: Grade 4 metrics | Показать x_order_orth component | `metrics/qwen3_14b_grade4_axis_decomposition03/summary.json` |
| Figure 2: +X/-X dose response | Показать causal internal steering | Grade 3 causal summaries |
| Figure 3: component causal gaps | Сравнить x_full/x_content/x_order/x_order_orth | Grade 4 summaries |
| Table 3: claim ladder | Развести supported / not supported | `research_synthesis/evidence_matrix_ru.md` |

## Results Paragraphs

### Result 1: Grade 3 identifies causal internal Vector X

```text
The Grade 3 run shows that the target-reference direction is stable in
middle-layer hidden space. The target middle projection is 0.976583, the
middle direction cosine is 0.852397, and the same-norm random baseline has a
near-zero mean of 0.000040. This rules out a trivial random-direction account.
The causal test is stronger: +X/-X residual-stream intervention moves the
generation-time hidden trajectory with alpha 0.75 gaps of 3.313378 from
neutral base and 3.336544 from target base.
```

### Result 2: Grade 4 separates order from content

```text
Grade 4 decomposes the axis into x_full, x_content, x_order and x_order_orth.
The decisive contrast is target vs sentence shuffle on x_order_orth. Target
projects at 0.978944, while sentence-shuffled content projects at 0.007214.
This means the order-orthogonal component is not carried by content-preserving
sentence shuffle.
```

### Result 3: x_order_orth is causally active

```text
The component-specific causal intervention shows that x_order_orth is not only
geometrically separable but causally active. At alpha 0.75, the middle/middle
gap is 3.726561 from neutral base and 3.698789 from target base, larger than
the corresponding x_content gaps of 2.990294 and 2.997980.
```

## Limitation Paragraphs

### Boundary 1: not visible behavioral control

```text
The current result is internal and mechanistic. Grade 3's behavioral random
p95 gate did not pass, so the result should not be described as reviewer-grade
visible behavioral control. The stronger and cleaner claim is that residual
stream intervention causally moves the internal hidden trajectory.
```

### Boundary 2: not permanent topology

```text
The experiment operates on stateless transformer calls. It measures hidden
states while the target/context is present in prompt/KV context and during
generated tokens. It does not show permanent weight-level or topology-level
change after the context is removed.
```

### Boundary 3: not cross-model yet

```text
The current Grade 3 + Grade 4 package is Qwen3-14B-specific. Cross-model
universality requires the same protocol on at least one additional model.
```

## Replication Paragraph

```text
The next default experiment is cross-model Grade 3 + Grade 4 replication.
The first candidate is mistralai/Ministral-3-14B-Instruct-2512-BF16. If that
model is unavailable or too heavy, the fallback is
allenai/OLMo-2-1124-13B-Instruct. The replication should collect the same
Grade 3 internal-axis metrics and the same Grade 4 x_order_orth decomposition
metrics, then run `python .\research_synthesis\collect_research_metrics.py`.
```

## Working Conclusion

```text
Qwen3-14B supports a target-conditioned causal internal latent axis. Grade 4
shows that this axis contains a separable discourse-order / rhetorical-regime
component beyond sentence-shuffled content. This is a strong model-specific
mechanistic result, with cross-model replication as the next major gate.
```
