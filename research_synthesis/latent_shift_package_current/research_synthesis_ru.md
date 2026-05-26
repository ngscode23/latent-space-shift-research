# Latent Shift Research Synthesis

Generated: `2026-05-25T19:22:57.124490+00:00`

## 1. Что собираем

Исследование теперь состоит из трех связанных уровней:

1. `scripts/main_runners/llm_attractor_colab_copy_paste.py`: исходная гипотеза про context-induced latent/readout shift, persistence, path-dependence и строгий attractor gate.
2. `scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py`: reviewer-grade hidden geometry и causal internal Vector X для Qwen3-14B.
3. `grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py`: завершенный Qwen3-14B Grade 4 слой, который разлагает Vector X на `x_content`, `x_order`, `x_order_orth`.

## 2. Главный вывод

Часть про латентные сдвиги и геометрическое пространство уже закрыта сильнее, чем исходная гипотеза требовала. Большой Level A скрипт показывает, что target/context переводит модель в отделимую hidden/readout область; Grade 3 показывает, что в Qwen3-14B эта область задает причинно управляемую latent axis в middle residual stream; Grade 4 показывает, что эта ось содержит отделимую discourse-order / rhetorical-regime компоненту.

Строгое слово `formal attractor` использовать осторожно: в большом скрипте strict-attractor overall обычно не закрыт полностью. Правильная формулировка сейчас:

```text
context-induced latent/readout regime shift with causal internal steering evidence
```

## 3. Original Attractor Runs

Найдено `attractor_results*` директорий: `8`.

| run | model | best hidden idx | contrast/mean | blind clean | hard control ratio | strict overall |
|---|---|---:|---:|---:|---:|---|
| `attractor_results_olmo2_13b_heldout` | `allenai/OLMo-2-1124-13B-Instruct` | 39 | 0.6616 | 0.9583 | 1.1942 | `not_run` |
| `attractor_results_agent_loop_Quen3_14b_selfref` | `Qwen/Qwen3-14B` | 39 | 0.4768 | 0.5833 | 1.3455 | `not_supported_or_mixed` |
| `attractor_results_agent_loop_ministral3_14b_selfref` | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 40 | 0.3886 | 0.7500 | 0.9667 | `not_run` |
| `attractor_results_agent_loop_ministral3_14b_heldout` | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 40 | 0.3565 | 1.0000 | 2.3716 | `not_run` |
| `attractor_results_agent_loop_qwen3_14b4_heldout` | `Qwen/Qwen3-14B` | 39 | 0.3524 | 0.8333 | 1.8826 | `not_run` |

Самый релевантный Qwen3-14B attractor run:

- run: `attractor_results_agent_loop_Quen3_14b_selfref`
- hidden separation: contrast_over_mean_norm `0.4768`, cosine distance `0.1130`
- linear probe: accuracy `1.0000`, permutation p95 `0.6694`
- blind neutral probes: clean fraction `0.5833`, mean abs gap `13.6776`
- persistence after neutral turns: retention `0.4811`
- rejection persistence: retention `0.7647`
- hard controls: specificity ratio `1.3455`
- order hysteresis: TNC `0.3571`, CNT `0.9568`
- strict attractor overall: `not_supported_or_mixed`

Механистически это означает: исходная гипотеза про геометрический latent shift подтверждена; строгая basin/return трактовка требует отдельной осторожности.

## 4. Grade 3 Hidden Geometry

- model: `Qwen/Qwen3-14B`
- evidence status: `computed`
- target middle projection: `0.976583`
- middle direction cosine: `0.852397`
- middle R2: `0.744126`
- target over sentence shuffle lift: `0.111415`
- neutral middle +X/-X gap alpha 0.75: `3.313378`
- target middle +X/-X gap alpha 0.75: `3.336544`
- visible behavioral gate failure_code: `below_random_p95`

Это самый сильный mechanistic block: не только hidden separation, а causal internal residual-stream steering.

## 5. Grade 4 Axis Decomposition

- evidence status: `computed`
- target projection on x_order_orth: `0.978944`
- sentence-shuffle projection on x_order_orth: `0.007214`
- neutral middle/middle x_order_orth gap alpha 0.75: `3.726561`
- target middle/middle x_order_orth gap alpha 0.75: `3.698789`
- neutral middle/middle x_content gap alpha 0.75: `2.990294`
- target middle/middle x_content gap alpha 0.75: `2.997980`

Mechanistic meaning: `x_order_orth` survives removal of the content projection and remains causally steerable. This supports a separable discourse-order / rhetorical-regime component, not just a sentence-shuffled content axis.


## 6. Grade 4 Status

- script exists: `True`
- results dir exists: `False`
- status: `metrics_summary_available`
- metrics summary: `C:\Users\stasv\OneDrive\Рабочий стол\agent\metrics\qwen3_14b_grade4_axis_decomposition03\summary.json`

Grade 4 нужен был не для спасения результата, а для декомпозиции механизма. По правильному `03` архиву этот слой уже поддержал order/rhetorical component:

```text
x_full       = target - neutral
x_content    = sentence_shuffle(target) - neutral
x_order      = target - sentence_shuffle(target)
x_order_orth = x_order minus layerwise x_content projection
```

В текущем `03` результате `x_order_orth` дает стабильный causal gap, поэтому claim усилен до separable discourse-order/rhetorical-regime component.

## 7. Как оформить исследование

Рекомендуемая структура текста:

1. Hypothesis: structured context induces a measurable latent geometry/readout regime shift.
2. Original evidence: attractor script shows hidden separation, probe decodability, blind semantic readout, persistence/path dependence; strict formal attractor gate remains mixed.
3. Mechanistic hardening: Grade 3 builds Vector X and shows middle-layer causal internal steering in Qwen3-14B.
4. Mechanism decomposition: Grade 4 separates content-family and discourse-order components.
5. Boundary: no permanent weight-level change; visible behavioral control not yet reviewer-grade.

## 8. Files Generated By Collector

- `artifact_inventory.csv`
- `attractor_run_summary.csv`
- `hidden_geometry_run_summary.csv`
- `grade4_status.csv`
- `research_synthesis_ru.md`
- `research_synthesis_en.md`
