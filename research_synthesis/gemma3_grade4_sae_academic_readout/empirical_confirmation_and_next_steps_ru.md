# Эмпирическое подтверждение context-induced latent-state shift

## Статус результата

В текущем прогоне на `google/gemma-3-12b-it` эмпирически подтверждена центральная descriptive-гипотеза исследования: сильный `target text` переводит модель в другое измеримое латентное состояние на inference-time, без изменения весов модели.

Это не означает, что изменились параметры модели. Это означает, что изменились её скрытые состояния: `target text` сдвинул геометрию residual-stream / layer-boundary hidden states так, что состояние target-condition стало отличаться от control-condition не только на уровне финального ответа, но внутри вычислительной траектории модели.

Главный результат: эффект `target text` не сводится к простому content trace. Sentence-shuffle control уходит в `x_content`, а coherent target уходит в `x_order_orth`. Это означает, что модель реагирует не только на набор похожих слов, а на цельную структурную / режимную конфигурацию target text.

## Ключевые метрики

Главное evidence находится в `grade4_axis_projection_geometry_summary.csv`:

```text
target on x_order_orth = 0.909026
sentence_shuffle on x_order_orth = -0.069058

sentence_shuffle on x_content = 0.849551
target on x_content = -0.010294
```

Интерпретация:

```text
sentence_shuffle control загружает content-компоненту;
target text загружает отдельную order/structure/component axis;
значит эффект target text отделим от простого content similarity.
```

Дополнительное evidence:

```text
order_orth_energy_fraction_of_full = 0.575700
SAE reconstruction cosine = 0.989581-0.999334
order-related SAE rows = 114 / 278
order_specific_generation_persistent_feature = 21
```

Интерпретация:

```text
order/structure component не является маленьким шумовым остатком;
SAE реконструирует hidden states достаточно хорошо для feature-level readout;
часть sparse features связана с target/order shift;
часть order-related features сохраняется во время generation.
```

## Что именно доказано

Доказано для текущего прогона:

```text
1. target text создаёт измеримый hidden-state geometry shift;
2. этот shift отделяется от sentence-shuffle/content control;
3. shift виден в component decomposition;
4. shift частично виден в SAE sparse-feature readout;
5. часть order-related features сохраняется в generation trajectory;
6. значит target text работает как inference-time state-control stimulus.
```

Главная формулировка:

```text
Мы эмпирически подтвердили, что сильный target text может переводить Gemma 3 12B IT в другое измеримое латентное состояние до финального ответа. Этот сдвиг виден в hidden-state geometry, отделим от content-control и частично читается через SAE features.
```

## Что ещё не доказано

Не доказано текущим прогоном:

```text
1. что эффект универсален для всех моделей;
2. что это формальный attractor basin;
3. что x_order_orth причинно управляет финальным поведением;
4. что SAE feature ids имеют точную публичную Neuronpedia-интерпретацию, если SAE source отличается;
5. что output safety полностью определяется этим shift.
```

Текущий статус:

```text
descriptive/mechanistic-readout evidence strong;
causal evidence not yet established in this Gemma SAE run.
```

## Почему это важно

Результат важен не потому, что “текст влияет на модель”. Это очевидно.

Результат важен потому, что target text создаёт измеримый pre-output state transition. Финальный ответ становится поздним readout внутреннего состояния, а не единственным объектом анализа.

Для interpretability это означает:

```text
LLM нужно анализировать как траекторию:
prompt -> hidden-state shift -> component/readout shift -> generation trajectory -> output.
```

Для AI safety это означает:

```text
output-only audit опаздывает;
важный safety-relevant сдвиг может произойти до первого токена ответа;
нужно смотреть hidden-state transitions, а не только финальный текст.
```

## Куда идти дальше

Следующий шаг не в том, чтобы ещё раз запускать analyzer. Analyzer уже показал descriptive geometry.

Следующий шаг:

```text
causal Grade 4 run по x_order_orth.
```

Нужно проверить:

```text
если ablate / patch / steer x_order_orth,
двигается ли generation behavior вместе с ним?
```

Если да, результат становится сильнее:

```text
не только target text создаёт hidden-state shift,
а измеренная component axis причинно участвует в response regime.
```

Минимальный следующий эксперимент:

```text
1. включить GRADE4_COMPONENT_CAUSAL_ENABLED;
2. включить CAUSAL_INTERVENTIONS_ENABLED;
3. сделать ablation/patch/steering для x_order_orth;
4. измерить generation trajectory;
5. измерить safety/style/readout proxies;
6. сравнить target, sentence-shuffle, word-shuffle, neutral, question-only;
7. повторить на второй open-weight модели.
```

## Короткий claim для публикации

```text
We empirically confirm a context-induced latent-state shift in Gemma 3 12B IT: strong target text moves the model into a measurably different hidden-state geometry before the final answer. The shift is separable from sentence-shuffle/content controls, visible in component decomposition, partially persistent in generation, and partially readable through SAE sparse features. This supports the view that LLM safety and interpretability should evaluate hidden-state transitions, not only final outputs.
```

