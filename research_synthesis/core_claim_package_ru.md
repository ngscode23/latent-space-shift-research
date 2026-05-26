# Core Claim Package: latent/readout regime shift + causal internal axis

Дата фиксации: `2026-05-25`

Этот файл нужен как рабочий центр исследования. Он не заменяет сырые CSV и не
добавляет новых метрик. Он собирает уже полученные результаты в claim ladder:
что именно поддержано, каким числом, что это значит механистически, и какие
метрики надо добирать дальше.

## 1. Главная формулировка

Мы не заявляем формальный `attractor basin` и не заявляем permanent topology
change. Рабочий claim уже сильнее и чище:

```text
Structured target context induces a measurable context-conditioned latent /
readout regime shift. In Qwen3-14B, the target-reference hidden direction
forms a causal internal residual-stream axis. Grade 4 shows that this axis
contains a separable discourse-order / rhetorical-regime component beyond
sentence-shuffled content.
```

По-русски:

```text
Структурированный target-контекст вызывает измеримый context-conditioned
латентный/readout-сдвиг. В Qwen3-14B соответствующее target-reference
направление в hidden space становится причинно управляемой внутренней осью
residual stream. Grade 4 показывает, что эта ось содержит отделимую компоненту
дискурсивного порядка / риторического режима, не сводимую к
sentence-shuffled content.
```

## 2. Что мы увидели

### Grade 3: causal internal Vector X

Источник:

```text
metrics/qwen3_14b_breakthrough_grade_hardened/summary.json
```

Ключевые числа:

| Метрика | Значение |
|---|---:|
| `target_middle_projection_mean` | `0.976583` |
| `target_middle_direction_cosine_mean` | `0.852397` |
| `target_middle_positive_projection_fraction` | `1.000000` |
| `middle_band_r2` | `0.744126` |
| `random_same_norm_null_mean` | `0.000040` |
| `random_same_norm_empirical_p` | `0.007752` |
| neutral middle +X/-X gap, alpha `0.75` | `3.313378` |
| target middle +X/-X gap, alpha `0.75` | `3.336544` |
| behavioral visible gate | `not supported` |

Что это значит:

```text
Vector X не является только описательным отличием target от neutral. Когда мы
вмешиваемся в middle residual stream и добавляем/вычитаем эту ось, внутренняя
generation-time trajectory систематически двигается в нужную сторону.
```

Механистически:

```text
В middle layers есть причинно действенное направление, связанное с
target-conditioned режимом. Это не доказывает управление видимым поведением,
но доказывает, что hidden trajectory можно причинно сдвигать вдоль этой оси.
```

### Grade 4: отделимая order/rhetorical component

Источник:

```text
metrics/qwen3_14b_grade4_axis_decomposition03/summary.json
```

Определения осей:

```text
x_full       = target - neutral
x_content    = sentence_shuffle(target) - neutral
x_order      = target - sentence_shuffle(target)
x_order_orth = x_order после layerwise удаления проекции на x_content
```

Ключевые числа:

| Метрика | Значение |
|---|---:|
| target projection on `x_order_orth` | `0.978944` |
| sentence_shuffle projection on `x_order_orth` | `0.007214` |
| word_shuffle projection on `x_order_orth` | `0.251849` |
| length-matched neutral projection on `x_order_orth` | `-0.052867` |
| `cos_content_order_orth` | `~0.000000` |
| middle `x_order_orth` energy fraction of full | `0.277756` |

Middle/middle causal gaps at alpha `0.75`:

| Base | `x_full` | `x_content` | `x_order` | `x_order_orth` |
|---|---:|---:|---:|---:|
| neutral | `3.308553` | `2.990294` | `3.384538` | `3.726561` |
| target | `3.330993` | `2.997980` | `3.383840` | `3.698789` |

Alpha slopes, middle/middle:

| Base | `x_full` | `x_content` | `x_order` | `x_order_orth` |
|---|---:|---:|---:|---:|
| neutral | `2.229508` | `2.021195` | `2.250374` | `2.459800` |
| target | `2.234000` | `2.028973` | `2.257763` | `2.463462` |

Что это значит:

```text
После удаления content-проекции компонент x_order_orth не исчезает. Наоборот,
в middle/middle causal test она дает самый сильный +component/-component gap.
Sentence shuffle почти не проектируется на x_order_orth, а target
проектируется почти единично.
```

Механистически:

```text
Vector X содержит не только lexical/semantic target-family trace. Внутри него
есть отдельная причинно активная компонента, связанная с глобальным порядком
дискурса, риторическим режимом и организацией высказывания.
```

## 3. Что усилилось и что ослабло

Усилилось:

```text
1. Hidden geometry shift: target стабильно отделяется от neutral/control.
2. Causal internal axis: middle-layer residual intervention двигает hidden trajectory.
3. Component decomposition: x_order_orth сохраняет причинную силу после удаления x_content.
4. Content-only explanation: ослаблена, потому что sentence_shuffle не несет x_order_orth.
```

Не усилилось / не заявляем:

```text
1. Permanent weight-level or topology-level change.
2. Formal attractor basin.
3. Reviewer-grade visible behavioral control.
4. Cross-model universality.
5. SAE-level named-feature localization.
```

## 4. Claim ladder для текста статьи

| Claim | Status | Evidence |
|---|---|---|
| Structured context creates hidden/readout shift | supported | broad runs + Grade 3 geometry |
| Qwen3-14B has causal internal Vector X | supported | Grade 3 +X/-X middle residual intervention |
| Vector X is not only content-family signal | supported | Grade 4 sentence-shuffle/content decomposition |
| `x_order_orth` is causally active | supported | strongest middle/middle gap at alpha `0.75` |
| Visible behavior is reliably controllable | not supported | behavioral random p95 gate fails |
| Permanent topology/weights changed | not claimed | stateless transformer boundary |
| Formal attractor basin exists | not claimed | strict basin criteria separate |

## 5. Как это писать без самоослабления

Плохая рамка:

```text
Global negative verdict that collapses all claims into one rejection.
```

Почему плохо:

```text
Она смешивает сильный узкий результат с неподдержанными более сильными
claims. Downstream evaluator потом читает весь пакет как отрицательный, хотя
метрики поддерживают causal internal axis и Grade 4 component decomposition.
```

Правильная рамка:

```text
Supported:
Qwen3-14B has a target-conditioned causal internal latent axis; Grade 4 shows
a separable discourse-order/rhetorical-regime component.

Not supported:
permanent topology change, formal basin, reviewer-grade visible behavioral
control, cross-model universality.
```

## 6. Следующий эксперимент

Сейчас не надо перезапускать Grade 4 на Qwen3-14B. Он уже сделал свою работу.
Следующий научный ход зависит от цели:

```text
Если цель — статья / отчет:
  сначала пишем Results + Methods вокруг Grade 3/Grade 4.

Если цель — усилить claim:
  делаем cross-model replication Grade 3 + Grade 4 на второй модели.

Если цель — закрыть широкую картину:
  делаем один fresh broad latent/readout run и затем collect_research_metrics.py.
```

Практический порядок:

```text
1. Зафиксировать этот core claim package.
2. Составить таблицу "какие метрики уже есть / каких не хватает".
3. Только потом запускать новые прогоны.
4. После каждого прогона запускать:
   python .\research_synthesis\collect_research_metrics.py
```
