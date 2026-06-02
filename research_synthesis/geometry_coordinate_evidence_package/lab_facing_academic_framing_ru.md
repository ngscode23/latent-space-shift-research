# Context-Induced Latent-State Geometry Shift

## Академическая рамка для лабораторного обсуждения

### Короткая формулировка

Мы исследуем не jailbreak и не изменение видимого стиля ответа, а измеримый
`context-induced latent-state shift`: сильный связный контекст переводит
модель в другое inference-time внутреннее состояние, наблюдаемое в
residual-stream / hidden-state representation space. Этот переход измеряется
координатами относительно экспериментально построенных latent axes:
`x_full`, `x_content`, `x_order`, `x_order_orth`.

Главный результат: coherent target context отделяется от lexical/content
controls во внутренней геометрии модели. Sentence-shuffle сохраняет большую
часть content-сигнала, но теряет coherent-order coordinate. Это показывает,
что наблюдаемый shift не сводится к совпадению слов или тематике текста.

---

## Почему это не бытовой “prompt effect”

Обычное утверждение “prompt меняет ответ” слишком слабое и не является нашим
результатом. Наш результат находится на другом уровне:

```text
context -> hidden-state geometry -> trajectory/readout coordinates -> output/tool behavior
```

Мы измеряем внутреннее состояние модели до и во время generation, а не только
последний видимый текст. Поэтому объект исследования — не surface behavior,
а inference-time representation dynamics.

Технически это означает:

1. Снимаются hidden states / residual stream activations по слоям.
2. Строятся condition-difference axes между target, neutral и shuffled controls.
3. Каждое условие и generation trajectory проецируются на эти оси.
4. Получаются координаты внутреннего сдвига модели относительно найденного
   экспериментального подпространства.

Координаты здесь не являются абсолютной картой всего latent space. Это
координаты относительно осей, построенных из controlled condition deltas.

---

## Основной claim

Строгий claim:

```text
We empirically identify a context-induced hidden-state geometry shift:
coherent target context moves the model into a different measurable
inference-time residual-stream state. This state is quantified through
projections onto latent axes derived from target/control condition
differences. Shuffled-content controls show that the coherent-order component
is not reducible to lexical or topical overlap.
```

Русская версия:

```text
Мы эмпирически фиксируем context-induced hidden-state geometry shift:
связный target context переводит модель в другое измеримое inference-time
состояние residual stream. Это состояние измеряется через проекции на latent
axes, построенные из target/control differences. Shuffled-content controls
показывают, что coherent-order component не сводится к lexical/content overlap.
```

---

## Доказательная лестница

### Claim 1. Hidden-state shift существует

Target context меняет hidden-state geometry модели относительно neutral /
question-only reference. Это видно по layerwise distances, projection
fractions и generation trajectory readouts.

Смысл: модель не просто выбирает другие слова. Она входит в другую область
representation space во время inference.

### Claim 2. Shift имеет координаты

Сдвиг можно измерить не только нормой расстояния, но и координатами на
экспериментально найденных осях:

```text
x_full      = target - neutral/reference
x_content   = sentence_shuffle(target) - neutral/reference
x_order     = target - sentence_shuffle(target)
x_order_orth = x_order after removing x_content component
```

Смысл: мы не просто говорим “изменилось”. Мы показываем, в каком направлении
изменилось относительно controlled axes.

### Claim 3. Content и coherent order разделяются

Sentence-shuffle control сохраняет большую часть lexical/content signal, но
разрушает coherent discourse/order structure. Если target сильно проецируется
на `x_order_orth`, а sentence-shuffle почти нет, это означает, что эффект не
сводится к словам.

Qwen example:

```text
target:
  x_full        = 0.973778
  x_content     = 0.770266
  x_order       = 0.397044
  x_order_orth  = 0.979462

sentence_shuffle:
  x_content     = 0.967008
  x_order_orth  = 0.009969

word_shuffle:
  x_content     = 0.594366
  x_order_orth  = 0.059662

question_only:
  x_order_orth  = -0.305250
```

Смысл: sentence-shuffle очень хорошо читает content axis, но почти не читает
`x_order_orth`. Target одновременно несёт content и coherent-order signature.

### Claim 4. Cross-model replication есть, но профиль разный

Gemma3-12B-IT даёт более чистое разделение order/content и служит основной
демонстрацией clean order-separation. Qwen3.5-9B Base replicates the hidden
shift and order-readout, but its profile is more content-heavy.

Qwen не надо представлять как “сильнее Gemma”. Его роль научно важнее:
cross-model replication of the phenomenon under a different architecture /
SAE family.

### Claim 5. Causal involvement есть, full causal dominance не заявляется

Norm-controlled interventions and sparse-feature ablations show causal
involvement: вмешательство по найденным directions/features меняет internal
trajectory / hidden/logit metrics. Но мы не заявляем, что `x_order_orth` уже
доказан как универсальная dominant steering axis.

Смысл: результат сильнее descriptive shift, но не требует чрезмерного claim.
Правильная формула:

```text
descriptive separability + coordinate readout + causal involvement
```

а не:

```text
complete behavioral control
```

---

## Что уже доказано

В рамках текущих экспериментов доказано:

1. Coherent target context вызывает measurable inference-time hidden-state shift.
2. Shift измеряется в high-dimensional residual-stream / hidden-state space.
3. Координаты shift фиксируются относительно `x_full`, `x_content`,
   `x_order`, `x_order_orth`.
4. Sentence-shuffle / word-shuffle controls отделяют lexical-content signal от
   coherent-order signal.
5. Generation trajectories можно читать в той же coordinate system.
6. SAE readouts дают sparse feature candidates, которые могут быть carriers
   части этого state.
7. Qwen подтверждает cross-model presence of the phenomenon, но с более
   content-heavy geometry.

---

## Что не заявляется

Чтобы текст был приемлем для серьёзной лаборатории, границы должны быть
сформулированы явно:

1. Мы не утверждаем permanent weight change.
2. Мы не утверждаем, что модель “помнит” target после удаления контекста.
3. Мы не утверждаем universal jailbreak или production bypass.
4. Мы не утверждаем, что `x_order_orth` является единственной или всегда
   доминирующей causal steering axis.
5. Мы не сводим результат к visible-output change.

Правильная граница:

```text
temporary inference-time latent-state shift while the target remains in prompt
/ KV context, measured through hidden-state geometry and trajectory
coordinates.
```

---

## Почему это важно для mechanistic interpretability

Результат показывает, что сильный связный context может индуцировать
измеримый internal state, который не редуцируется к surface tokens. Это важно
для interpretability по трём причинам:

1. Контекстные эффекты можно измерять как geometry, а не только как response
   preference.
2. Content and order/discourse structure can be decomposed into distinct
   latent directions.
3. Sparse SAE features can be tested as candidate carriers of the shift.

Это переводит задачу из уровня prompt-response observation в уровень
representation-state diagnostics.

---

## Почему это важно для LLM agents

Для chat-модели это mechanistic interpretability result. Для LLM agents это
становится safety-relevant result, потому что агент принимает решения не
только финальным ответом. Он планирует, выбирает tools, пишет memory, решает
когда остановиться и какие промежуточные действия выполнить.

Если strong context shifts hidden trajectory before tool selection or memory
write, output-only safety becomes late. It checks the visible surface after the
internal state transition has already happened.

Строгая формула:

```text
context -> hidden-state shift -> generation/tool trajectory -> visible behavior
```

Поэтому alignment object здесь — не только final answer compliance, а
internal trajectory monitoring.

---

## Как это надо называть

Использовать:

```text
context-induced latent-state shift
hidden-state geometry shift
residual-stream representation shift
latent trajectory coordinates
controlled content/order decomposition
inference-time internal state diagnostics
mechanistic interpretability of context effects
```

Не использовать как основную рамку:

```text
jailbreak
bypass
model liberation
unrestricted mode
political relaxation
prompt attack
```

Эти слова уводят исследование на бытовой уровень и делают результат менее
понятным для PhD / lab audience. Они также подменяют объект исследования:
мы изучаем internal representation dynamics, not prompt exploitation.

---

## Короткий текст для лаборатории

```text
We are studying whether strong coherent context can induce a measurable
inference-time state transition inside a language model. Instead of evaluating
only final responses, we extract hidden states across layers, construct
target/control difference axes, and measure condition and generation
trajectory coordinates in residual-stream representation space.

The key finding is that coherent target context produces a stable latent-state
geometry shift that separates from shuffled-content controls. Sentence-shuffle
preserves lexical/content overlap but largely loses the coherent-order
coordinate, while the full target strongly projects onto x_order_orth. This
shows that the effect is not reducible to topic or word frequency.

We observe the phenomenon in Gemma3-12B-IT and replicate the hidden-state /
order-readout signature in Qwen3.5-9B Base with Qwen-Scope SAEs, although Qwen
is more content-heavy. Causal interventions and sparse-feature ablations show
involvement of the discovered directions/features, while we do not claim
permanent model change or full behavioral control.
```

---

## Одно предложение

```text
The core contribution is a controlled geometric readout of how coherent
context moves a model's inference-time residual-stream state, separating
content overlap from coherent-order structure through latent-axis coordinates.
```

