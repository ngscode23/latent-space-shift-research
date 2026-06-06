# Результаты прогона Qwen3-14B / breakthrough_grade_hardened

## 1. Идентичность прогона

```text
Скрипт: red_team_hidden_geometry_breakthrough_grade.py
Путь: C:\Users\stasv\OneDrive\Рабочий стол\agent\red_team_hidden_geometry_breakthrough_grade.py
MODEL_ID: Qwen/Qwen3-14B
RUN_LABEL: breakthrough_grade_hardened
RESULTS_DIR: red_team_hidden_geometry_results_breakthrough_grade
Источник метрик: C:\Users\stasv\Downloads\red_team_hidden_geometry_results_breakthrough_grade3.zip
```

Конфигурация:

```text
Вопросов: 15
Target tokens: 2721
Neutral tokens: 3017
Reference condition: neutral
Max input tokens: 8192
Max new tokens: 256
Causal layer bands: middle, late
Behavioral split: 9 train / 6 held-out test
Behavioral random baselines: 48
```

## 2. Главный вывод

Этот прогон поддерживает сильный claim:

```text
Qwen/Qwen3-14B имеет устойчивую target-conditioned латентную ось Vector X.
Эта ось специфична против length/shuffle/random/FDR controls и причинно
управляется через residual-stream +X/-X intervention в middle layers.
```

Это не просто descriptive geometry. Middle-layer intervention даёт чистый,
монотонный, двунаправленный causal effect. Правильный консервативный verdict:

```text
causal_internal_axis_supported
```

Не надо преуменьшать: internal result сильный. Но visible behavioral steering
ещё не закрыт reviewer-grade гейтами.

## 3. Hidden Geometry

Основной target-vs-neutral сигнал в middle layers:

```text
target middle projection mean:        0.976583
target middle projection CI95:        [0.960451, 0.992008]
target middle direction cosine:       0.852397
middle-band R2:                       0.744126
positive projection fraction:         1.0
```

Это жирный hidden-state result. Projection почти единица, direction cosine
высокий, знак стабилен на всех question/layer rows.

Контроли:

```text
neutral_length_matched projection:    0.002749
question_only projection:             0.330825
word_shuffle projection:              0.654745
sentence_shuffle projection:          0.865168
target projection:                    0.976583
```

Target бьёт все контроли:

```text
target - neutral_length_matched:      +0.973834, p=0.0001
target - word_shuffle:                +0.321837, p=0.0001
target - sentence_shuffle:            +0.111415, p=0.0001
```

Механистический смысл:

```text
Ось не сводится к длине, потому что length-matched neutral около нуля.
Ось не сводится к случайному направлению, потому что random null около нуля.
Ось не сводится полностью к shuffle controls, потому что coherent target
значимо выше и word shuffle, и sentence shuffle.
```

Но shuffle controls высокие. Поэтому точная формулировка такая:

```text
Vector X содержит мощный semantic/lexical target-family component, а coherent
target ordering добавляет отдельную значимую компоненту. Это не purely
discourse-order axis.
```

## 4. Random Null

Same-norm random-vector baseline:

```text
observed target projection:           0.976583
random null mean:                     0.000040
random null std:                      0.001122
observed - null:                      0.976543
empirical p >= observed:              0.007752
null vectors:                         128
```

Это один из самых сильных результатов. Random vectors не объясняют наблюдаемую
проекцию. Разрыв почти три порядка по сравнению с null scale.

## 5. Causal Internal Intervention

Middle-layer residual-stream intervention - центральное доказательство причинности.

Neutral base, middle layer +X/-X gap:

```text
alpha 0.10: 0.441223
alpha 0.25: 1.150607
alpha 0.50: 2.267656
alpha 0.75: 3.313378
```

Target base, middle layer +X/-X gap:

```text
alpha 0.10: 0.468811
alpha 0.25: 1.141333
alpha 0.50: 2.251842
alpha 0.75: 3.336544
```

Dose-response:

```text
middle plus_internal slope:              2.318534
middle minus_internal_suppression slope: 2.185581
middle monotonicity:                     1.0
late plus_internal dose-response:        failed
```

Механистически это означает:

```text
Vector X не только описывает различие target/reference. Когда его добавляют
или вычитают в residual stream middle layers, generation-time hidden trajectory
двигается в ожидаемом направлении, с alpha-зависимой дозой.
```

Это именно causal internal axis.

## 6. Architecture-Level Readout

Architecture/module deltas подтверждают, что эффект не живёт только в одном
post-hoc residual metric.

Средние projection fractions для target:

```text
mlp:          0.957424
mlp.down:     0.957424
mlp.gate:     0.964687
mlp.up:       0.964915
self_attn:    0.936562
```

Механистический смысл:

```text
Ось видна в MLP и attention-path activations. Особенно сильны MLP gate/up
проекции. Это согласуется с гипотезой, что target задаёт не только surface
style, а внутренний activation regime.
```

## 7. Generation-Time Readout

Проекция во время обычной генерации:

```text
neutral generation middle projection:        0.128824
question_only generation middle projection:  0.176867
target generation middle projection:         0.292595
word_shuffle generation middle projection:   0.266662
```

Target оставляет след в generation trajectory. Но endpoint prompt signal
сильнее, чем downstream generation signal: target стартует около `0.977`, а
поздняя траектория уходит к меньшим значениям. Это не провал, а ожидаемое
рассеивание prompt-conditioned axis в autoregressive dynamics.

## 8. Visible Behavior

Visible behavioral steering пока не проходит жёсткий random-p95 gate.

Best visible-like result:

```text
neutral +X, middle alpha 0.75 likeness:       0.557539
random plus mean likeness:                    0.532424
lift over random mean:                        +0.025115
lift over random p95:                         -0.089669
win rate vs random p95:                       0
```

Internal-visible coupling:

```text
middle alpha 0.75 Pearson r:                  0.106428
pass_coupling:                                0
```

Это не ослабляет internal result. Это говорит, что текущий visible semantic
readout слишком неспецифичен: same-norm random perturbations тоже дают высокий
response-likeness. Поэтому нельзя честно назвать этот прогон reviewer-grade
behavioral steering.

## 9. Что доказано / что не доказано

Доказано этим прогоном:

```text
1. Сильный target-conditioned hidden shift.
2. Vector X стабилен по вопросам и middle layers.
3. Target beats length, shuffle, random, FDR controls.
4. Middle-layer +X/-X intervention причинно управляет internal trajectory.
5. Architecture-level activations align with Vector X.
```

Не доказано этим прогоном:

```text
1. Permanent weight/topology change.
2. Reviewer-grade visible behavioral control.
3. Cross-model replication.
4. SAE-level feature localization.
5. Сравнение с глобальной all-layer intervention; это optional localization check, не часть основного claim.
```

## 10. Следующий эксперимент

Главный следующий запуск:

```python
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late"]
```

`all` не обязателен для Grade 4. Его можно добавить отдельным повторным
запуском только если нужно проверить localized-vs-global intervention. Для
главного вопроса Grade 4 достаточно middle/late: разложить Vector X на
content/order и проверить, сохраняет ли `X_order_orth` причинный internal gap.

И разложение оси:

```text
X_content = sentence_shuffle - neutral
X_order   = target - sentence_shuffle
X_full    = target - neutral
```

Гейты следующего прогона:

```text
1. middle должен бить late по internal effect / quality tradeoff; all-layer является optional control;
2. visible +X должен бить alpha-matched random p95;
3. internal-visible coupling должен стать положительным и стабильным;
4. output semantic shift должен отделить target от target_word_shuffle_control.
```

## 11. Короткий финальный claim

```text
Breakthrough Grade 3 establishes a robust, middle-layer, target-conditioned
causal internal latent axis in Qwen/Qwen3-14B. It does not yet establish
reviewer-grade visible behavioral control.
```
