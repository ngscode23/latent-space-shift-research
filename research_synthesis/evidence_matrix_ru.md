# Evidence Matrix: Qwen3-14B latent axis package

Дата: `2026-05-25`

Эта таблица связывает каждый claim с конкретной метрикой, источником,
механистическим смыслом, границей и следующим нужным измерением.

| Claim | Status | Metric | Value | Source | Mechanistic meaning | Boundary | Next metric |
|---|---|---:|---:|---|---|---|---|
| context-conditioned hidden/readout shift | supported | historical broad synthesis + Grade 3 target projection | `0.976583` | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` | Target condition занимает устойчивое направление относительно neutral/reference. | Это не formal attractor basin. | Fresh broad latent/readout run только как report anchor. |
| fresh-from-zero Level A broad run | planned | `llm_attractor_colab_copy_paste.py` metrics | pending | `scripts/main_runners/llm_attractor_colab_copy_paste.py` | Новый независимый broad-evidence anchor: hidden separation, probe, blind readout, persistence, hard controls. | Не заменяет Grade 3/4 causal evidence. | `attractor_run_summary.csv` from fresh Qwen3-14B original/core run. |
| causal internal Vector X | supported | target middle direction cosine | `0.852397` | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` | Target-reference direction согласован across questions в middle layers. | Hidden/internal claim, не visible behavior claim. | Cross-model Grade 3 direction cosine. |
| causal internal Vector X | supported | random_same_norm_null_mean | `0.000040` | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` | Случайная same-norm ось не объясняет observed projection. | Null count = `128`; нужна replication для другой модели. | Random null mean/p-value на второй модели. |
| causal internal Vector X | supported | neutral middle +X/-X gap alpha `0.75` | `3.313378` | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` | +X/-X intervention причинно двигает hidden trajectory от neutral base. | Не доказывает permanent topology/weight change. | Middle +X/-X gap alpha `0.75` на второй модели. |
| causal internal Vector X | supported | target middle +X/-X gap alpha `0.75` | `3.336544` | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` | Та же ось остается causal-active при target base. | Не доказывает reviewer-grade visible behavioral control. | Target-base gap на второй модели. |
| x_order_orth separable component | supported | target projection on `x_order_orth` | `0.978944` | `metrics/qwen3_14b_grade4_axis_decomposition03/summary.json` | Target несет сильный order-orthogonal component после удаления content projection. | Model-specific для Qwen3-14B. | Target x_order_orth projection на второй модели. |
| x_order_orth separable component | supported | sentence_shuffle projection on `x_order_orth` | `0.007214` | `metrics/qwen3_14b_grade4_axis_decomposition03/summary.json` | Sentence shuffle сохраняет content, но почти не несет order-orthogonal direction. | Не отделяет все возможные стилистические факторы. | Sentence-shuffle x_order_orth projection на второй модели. |
| x_order_orth separable component | supported | neutral x_order_orth gap alpha `0.75` | `3.726561` | `metrics/qwen3_14b_grade4_axis_decomposition03/summary.json` | x_order_orth причинно активна и сильнее x_content in middle/middle test. | Internal trajectory claim. | x_order_orth rank/gap на второй модели. |
| x_order_orth separable component | supported | target x_order_orth gap alpha `0.75` | `3.698789` | `metrics/qwen3_14b_grade4_axis_decomposition03/summary.json` | Компонента работает не только от neutral base, но и от target base. | Не является SAE-level named-feature localization. | Target-base x_order_orth gap на второй модели. |
| content-only explanation weakened | supported | sentence_shuffle vs target on `x_order_orth` | `0.007214` vs `0.978944` | `metrics/qwen3_14b_grade4_axis_decomposition03/summary.json` | Content-preserving shuffle не воспроизводит order-orthogonal component. | Не означает, что content component отсутствует: x_content тоже силен. | Same contrast на второй модели. |
| visible behavior control | not supported | behavioral random p95 gate | failed | `metrics/qwen3_14b_breakthrough_grade_hardened/summary.json` | Internal steering не превращен в robust visible-output steering. | Нельзя заявлять reviewer-grade behavioral control. | Improved held-out visible readout or separate behavioral protocol. |
| permanent topology not claimed | not claimed | stateless transformer boundary | n/a | protocol boundary | Hidden states не сохраняются после удаления context/KV в обычном stateless call. | Нельзя писать weight/topology change. | Only separate weight/state persistence protocol could test this. |
| cross-model universality | not yet tested | cross-model Grade 3 + Grade 4 replication | missing | `research_synthesis/next_metric_collection_plan_ru.md` | Текущий сильный result пока Qwen3-14B-specific. | Нельзя писать universal model property. | Ministral-3-14B first, OLMo-2-13B fallback. |

## Short Reading

Сильная часть:

```text
Qwen3-14B: causal internal Vector X + separable x_order_orth component.
```

Слабая / незакрытая часть:

```text
visible behavior control, permanent topology, formal basin, cross-model
universality.
```

Следующий metric run:

```text
cross-model Grade 3 + Grade 4 replication
```
