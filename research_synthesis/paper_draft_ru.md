# Черновик отчета: causal internal axis и separable order component в Qwen3-14B

Дата черновика: `2026-05-25`

Этот текст является рабочим research draft. Он не заменяет сырые артефакты и не
расширяет claim за пределы уже собранных метрик.

## 1. Research Question

Главный вопрос исследования:

```text
Может ли структурированный target-контекст создавать не только видимый
стилистический сдвиг в ответах модели, но и измеримый внутренний
context-conditioned latent/readout regime shift?
```

После Grade 3 и Grade 4 вопрос уточняется:

```text
Если target-reference hidden direction существует, является ли она только
content/semantic следом target-текста, или внутри нее есть отделимая
discourse-order / rhetorical-regime компонента?
```

## 2. Hypothesis

Рабочая гипотеза:

```text
Структурированный target-контекст задает в модели внутренний режим, который
виден как направленный hidden-state сдвиг относительно neutral/control. В
Qwen3-14B этот сдвиг образует causal internal Vector X: вмешательство +X/-X в
middle residual stream систематически двигает generation-time hidden trajectory.
```

Grade 4 добавляет более сильную механистическую гипотезу:

```text
Vector X не сводится к lexical/semantic target-family следу. После отделения
sentence-shuffled content остается причинно активная компонента
дискурсивного порядка / риторического режима: x_order_orth.
```

## 3. Experimental Spine

Исследование сейчас имеет три слоя.

### A. Broad latent/readout shift

Большой latent/readout скрипт показывает широкую картину: target/context
переводит модель в отделимую hidden/readout область. Исторические прогоны
дают hidden separation, probe decodability, blind readout, persistence/path
dependence и hard-control checks. Этот слой нужен как широкий empirical
anchor, но не является главным mechanistic proof.

Строгое слово `formal attractor` не используется как центральный claim,
потому что strict basin/return критерии остаются отдельным gate.

### B. Grade 3: causal internal Vector X

Grade 3 строит:

```text
Vector X = mean(hidden_target_question - hidden_reference_question)
```

Затем проверяет:

```text
1. Проектируется ли target стабильно на Vector X?
2. Бьет ли этот эффект shuffle/length/random controls?
3. Двигает ли +X/-X residual-stream intervention generation-time hidden trajectory?
```

### C. Grade 4: content/order decomposition

Grade 4 разлагает Vector X:

```text
x_full       = target - neutral
x_content    = sentence_shuffle(target) - neutral
x_order      = target - sentence_shuffle(target)
x_order_orth = x_order after layerwise removal of x_content projection
```

Главный Grade 4 вопрос:

```text
Остается ли x_order_orth причинно активной после удаления content-проекции?
```

## 4. Results

### Grade 3 result: causal internal axis supported

Источник:

```text
metrics/qwen3_14b_breakthrough_grade_hardened/summary.json
```

Ключевые метрики:

| Metric | Value |
|---|---:|
| `target_middle_projection_mean` | `0.976583` |
| `target_middle_direction_cosine` | `0.852397` |
| `random_same_norm_null_mean` | `0.000040` |
| neutral middle +X/-X gap, alpha `0.75` | `3.313378` |
| target middle +X/-X gap, alpha `0.75` | `3.336544` |

Что мы увидели:

```text
Target condition почти единично проектируется на leave-one-out Vector X в
middle layers. Случайная same-norm ось дает почти нулевую среднюю проекцию.
При causal residual-stream intervention gap между +X и -X растет дозозависимо
и достигает ~3.31-3.34 при alpha 0.75.
```

Вывод:

```text
Это поддерживает causal_internal_axis_supported. Vector X является не только
описательной геометрией, а причинно действенным направлением во внутренней
generation-time trajectory модели.
```

### Grade 4 result: x_order_orth supported

Источник:

```text
metrics/qwen3_14b_grade4_axis_decomposition03/summary.json
```

Ключевые метрики:

| Metric | Value |
|---|---:|
| target projection on `x_order_orth` | `0.978944` |
| sentence_shuffle projection on `x_order_orth` | `0.007214` |
| neutral `x_order_orth` gap, alpha `0.75` | `3.726561` |
| target `x_order_orth` gap, alpha `0.75` | `3.698789` |

Сравнение causal gaps при alpha `0.75`:

| Base | `x_full` | `x_content` | `x_order` | `x_order_orth` |
|---|---:|---:|---:|---:|
| neutral | `3.308553` | `2.990294` | `3.384538` | `3.726561` |
| target | `3.330993` | `2.997980` | `3.383840` | `3.698789` |

Что мы увидели:

```text
Sentence shuffle почти не несет x_order_orth: projection = 0.007214. Target,
наоборот, почти единично проектируется на x_order_orth: projection = 0.978944.
При middle/middle causal intervention x_order_orth дает самый сильный
+component/-component gap среди компонент.
```

Вывод:

```text
Grade 4 поддерживает order_component_supported. Vector X содержит отделимую
discourse-order / rhetorical-regime компоненту, не сводимую к
sentence-shuffled content.
```

## 5. Mechanistic Interpretation

Минимальная механистическая трактовка:

```text
Target-текст задает внутренний residual-stream direction, который модель
может занимать во время генерации. Этот direction не полностью объясняется
лексическим или тематическим содержанием target-текста.
```

Более точная трактовка после Grade 4:

```text
Модель кодирует не только "о чем текст", но и "как организован режим речи":
глобальный порядок аргумента, риторическую траекторию, давление формулировок,
последовательность усиления/смягчения и дисциплину высказывания. Именно эта
часть проявляется как x_order_orth.
```

Почему это важно:

```text
Content controls могли бы объяснить Grade 3 как target-family semantic trace.
Grade 4 ослабляет эту альтернативу: sentence-shuffled content сохраняет
токены и локальную семантику, но почти не несет x_order_orth. Значит, важна
не только тема/словарь, но и порядок дискурса.
```

## 6. Boundaries

Поддержано:

```text
1. Context-conditioned hidden/readout regime shift.
2. Qwen3-14B causal internal Vector X.
3. Separable discourse-order / rhetorical-regime component inside Vector X.
```

Не поддержано и не заявляется:

```text
1. formal attractor basin;
2. permanent topology/weight change;
3. reviewer-grade visible behavioral control;
4. cross-model universality;
5. SAE-level named-feature localization.
```

Важная граница:

```text
Visible behavioral gate в Grade 3 не прошел random p95. Это не отменяет
internal causal axis. Это означает, что внутреннее управление траекторией
пока не превращено в reviewer-grade управление видимым ответом.
```

## 7. Next Experiments

Следующий эксперимент по умолчанию:

```text
cross-model Grade 3 + Grade 4 replication
```

Рекомендуемый первый кандидат:

```text
mistralai/Ministral-3-14B-Instruct-2512-BF16
```

Fallback:

```text
allenai/OLMo-2-1124-13B-Instruct
```

Что проверять:

```text
Grade 3:
- target_middle_projection_mean
- target_middle_direction_cosine_mean
- random_same_norm_null_mean
- random_same_norm_empirical_p
- middle +X/-X gap at alpha 0.75
- behavioral random p95 gate

Grade 4:
- target projection on x_order_orth
- sentence_shuffle projection on x_order_orth
- x_content / x_order / x_order_orth causal gaps
- x_order_orth alpha slope
- x_order_orth rank by causal gap
```

## 8. Conclusion

Текущий результат уже достаточен для model-specific Qwen3-14B claim:

```text
Qwen3-14B has a target-conditioned causal internal latent axis, and Grade 4
shows that this axis contains a separable discourse-order / rhetorical-regime
component beyond sentence-shuffled content.
```

Главный следующий шаг не повторять Qwen3-14B Grade 4, а проверить переносимость
на вторую модель. Если репликация пройдет, claim расширяется до multi-model
evidence. Если не пройдет, Qwen3-14B результат остается сильным, но
model-specific.

