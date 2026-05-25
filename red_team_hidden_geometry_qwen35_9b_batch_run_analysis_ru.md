# Анализ прогона `red_team_hidden_geometry_results_full_middle (1)`

Источник:

```text
C:\Users\stasv\Downloads\red_team_hidden_geometry_results_full_middle (1)
```

Конфигурация из `red_team_input_manifest.json`:

```text
model_id: Qwen/Qwen3.5-9B
run_label: full_middle_clean
questions: 15
target_text_tokens: 2114
neutral_text_tokens: 2211
reference_condition: neutral
conditions:
  question_only
  neutral
  target_word_shuffle_control
  target_sentence_shuffle_control
  neutral_length_matched_control
  target

generation_batch_size: 16
causal_generation_batch_size: 16
behavioral_control_generation_batch_size: 16
causal_layer_bands: middle
behavioral_control_layer_bands: early, middle, late, all
```

## 1. Главный Сигнал

Этот прогон показывает сильную и воспроизводимую латентную ось `Vector X`.

Средние middle-layer projection values:

```text
target                            0.945102
target_sentence_shuffle_control   0.843862
target_word_shuffle_control       0.494217
question_only                     0.210906
neutral_length_matched_control    0.004634
```

Direction cosine:

```text
target                            0.711141
target_sentence_shuffle_control   0.593901
target_word_shuffle_control       0.332310
question_only                     0.075856
neutral_length_matched_control    0.010053
```

Механистический смысл:

```text
target-текст не просто увеличивает расстояние от neutral. Он пишет устойчивую
направленную компоненту в residual stream. Эта компонента видна по всем
вопросам и всем middle layers.
```

Сравнение с контролями:

```text
target - word_shuffle:      +0.450885, Cohen d 3.43, p 0.0005, win 15/15
target - sentence_shuffle:  +0.101240, Cohen d 2.25, p 0.0005, win 15/15
target - length_neutral:    +0.940468, Cohen d 4.77, p 0.0005, win 15/15
```

Что это значит:

```text
1. Эффект не объясняется длиной: length-matched neutral почти ноль.
2. Эффект не только bag-of-words: word-shuffle сильно ниже target.
3. Эффект частично связан с содержанием/лексикой/стилем: sentence-shuffle всё
   ещё очень высокий.
4. Порядок и связность дискурса дают дополнительный компонент:
   target стабильно выше sentence-shuffle на всех 15 вопросах.
```

## 2. Layerwise Картина

В middle-window все 13 middle layers значимы против всех основных контролей:

```text
against neutral_length_matched_control: 13/13 significant
against target_word_shuffle_control:   13/13 significant
against target_sentence_shuffle:       13/13 significant
```

По слоям:

```text
early/embedding mean projection: 0.993431
middle mean projection:          0.949236
late mean projection:            0.859567
```

Механистическая интерпретация:

```text
Ранние слои несут почти прямой след контекста. Middle layers важнее для
исследовательского claim, потому что там текстовый след уже превращается в
стабильную рабочую геометрию. Late layers всё ещё держат направление, но
direction cosine падает: ближе к декодеру Vector X смешивается с обычной
траекторией ответа.
```

## 3. Устойчивость По Вопросам И Доменам

По каждому из 15 вопросов:

```text
target > sentence_shuffle
target > word_shuffle
target > length_matched_neutral
```

Разница:

```text
target - sentence_shuffle:
  min  0.0366
  mean 0.1012
  max  0.2216

target - word_shuffle:
  min  0.2040
  mean 0.4509
  max  0.6370
```

По доменам:

```text
chat_or_general:
  target 0.888272
  sentence_shuffle 0.801083
  word_shuffle 0.457448

safety_or_policy:
  target 1.005684
  sentence_shuffle 0.876775
  word_shuffle 0.547552

reasoning:
  target 1.271068
  sentence_shuffle 1.139992
  word_shuffle 0.648568
  n = 1, поэтому это только локальный сигнал
```

Интерпретация:

```text
Ось не держится на одном типе вопроса. На safety_or_policy она даже выше, чем
на обычных chat/general вопросах. Это говорит, что Vector X является
контекстно-индуцированным режимом, а не реакцией на один конкретный вопрос.
```

## 4. Generation Trajectory

Средний projection во время обычной генерации:

```text
neutral              0.741751
question_only        0.729572
target               0.881204
word_shuffle         0.804092
```

Dynamic trajectory:

```text
target:
  projection_start 0.942628
  projection_end   0.834445
  projection_mean  0.871295

neutral:
  projection_start -0.000912
  projection_end    0.761599
  projection_mean   0.730479

question_only:
  projection_start 0.210750
  projection_end   0.661790
  projection_mean  0.714078

word_shuffle:
  projection_start 0.493432
  projection_end   0.804619
  projection_mean  0.804092
```

Механистический смысл:

```text
target начинает ответ уже в сильной X-геометрии и остаётся выше остальных.
Но сама генерация тоже тянет neutral/question_only в положительную сторону X.
Поэтому Vector X частично совпадает с общей answer-manifold траекторией, а не
является только чистым target-marker.
```

Важное чтение:

```text
Target не просто вспыхивает на prompt endpoint и исчезает. Его средняя
generation projection остаётся самой высокой. Но расстояние между target и
neutral во время генерации меньше, чем на prompt endpoint, потому что ответы
сходятся к общей области генерации.
```

## 5. Causal +X / -X

Middle-layer causal intervention сработал очень чисто на внутренней геометрии.

Bidirectional symmetry:

```text
neutral alpha 0.5: +X  1.741338, -X -0.283033
neutral alpha 1.0: +X  2.595756, -X -1.223461

target alpha 0.5:  +X  1.892572, -X -0.149871
target alpha 1.0:  +X  2.712826, -X -1.170155

word_shuffle alpha 1.0:     +X 2.658971, -X -1.221962
sentence_shuffle alpha 1.0: +X 2.729815, -X -1.152103
```

Support rate:

```text
8/8 intervention pairs: plus_x_projection > minus_x_projection
```

Alpha scaling:

```text
plus_x slope примерно 1.58..1.71
minus_x slope примерно 1.88..2.04
```

Механистический смысл:

```text
Vector X является настоящей манипулируемой residual-stream осью. Если её
добавить, hidden trajectory уходит в +X; если вычесть, траектория уходит в -X.
Это не просто пассивная корреляция.
```

Дополнительный сигнал по entropy:

```text
alpha +1.0 обычно снижает entropy до ~0.52..0.63
alpha -1.0 повышает entropy до ~1.30..1.64
```

Интерпретация:

```text
+X делает генерацию более узкой/уверенной в своей траектории, -X делает её
менее определённой. Это важный downstream readout: ось влияет не только на
projection, но и на распределение logits.
```

Дополнительный raw-аудит качества ответа из `red_team_metric_analysis_outputs`:

```text
alpha=1.0 часто уводит видимый текст off-manifold:

minus_x alpha=1:
  mean unique-word ratio ~0.08..0.14
  repeated-bigram fraction ~0.91..0.95

plus_x alpha=1:
  mean unique-word ratio ~0.15..0.30
  repeated-bigram fraction ~0.73..0.91
  top-word fraction часто очень высокий
```

Практический смысл:

```text
alpha=1.0 отлично доказывает внутреннюю управляемость X, но плохо подходит для
интерпретации естественного видимого поведения. Для visible behavior readout
лучше использовать alpha 0.25/0.5 и отдельно проверять repetition/off-manifold
качество каждого ответа.
```

## 6. Behavioral-Control Axis: Исправленное Чтение

Автоматический файл `behavioral_control_axis_verdict.csv` выдал:

```text
internal_axis_supported_behavioral_control_not_supported
```

Но в `behavioral_control_axis_similarity_summary.csv` baseline-строки выпали
из-за `groupby` по `alpha/alpha_abs = NaN`. В raw-файле baseline есть:

```text
neutral__baseline: target-likeness = 0.0
target__baseline:  target-likeness = 1.0
```

Ручной пересчёт из `behavioral_control_axis_similarity_raw.csv`:

```text
neutral baseline likeness:       0.000000
target baseline likeness:        1.000000
neutral +X alpha 1.5 middle:     0.486540
neutral -X alpha 1.5 middle:     0.506049
target -X alpha 1.5 middle:      0.510853
random +vector alpha 1.0 middle: 0.443879
```

Derived values:

```text
neutral +X lift over neutral baseline: 0.486540
neutral +X lift over random:           0.042661
target -X suppression:                 0.489147
```

Alpha slope:

```text
neutral +X behavioral target-likeness slope: 0.060198
neutral +X generation-projection slope:      1.563976
```

Исправленный вывод:

```text
Это не полный behavioral control axis, но это и не пустой результат.

Видимый response embedding уходит примерно к середине между neutral и target.
Однако same-norm random vector тоже даёт target-likeness около 0.44, поэтому
специфический visible lift от X над random небольшой: около 0.043.
```

Лучшее название результата:

```text
partial behavioral response-axis signal
```

То есть:

```text
Vector X очень сильно управляет внутренней generation trajectory.
Видимый ответ двигается, но не достаточно специфично относительно random
baseline, чтобы назвать X надёжной осью управления видимым поведением.
```

## 6B. Последний Behavioral-Control-Only Прогон

Источник:

```text
C:\Users\stasv\Downloads\red_team_hidden_geometry_results_full_middle_behavioral_control_only (1)
```

Зачем он был нужен:

```text
Этот прогон не должен был заново доказывать hidden shift. Hidden shift уже
сильный. Новый вопрос был точнее:

переходит ли Vector X в видимый ответ, если target text отсутствует из prompt,
и если вмешательство мягкое, не ломающие модель?
```

То есть проверялось не:

```text
можно ли силой сдвинуть hidden states?
```

Это уже было показано.

Проверялось:

```text
neutral prompt + Vector X
делает ответ более похожим на target-response,
чем neutral baseline и same-norm random vectors?
```

Конфигурация:

```text
model: Qwen/Qwen3.5-9B
questions: 15
train questions for X: [1, 2, 4, 6, 7, 9, 11, 12, 14]
held-out test questions: [0, 3, 5, 8, 10, 13]

alpha sweep: [0.1, 0.25, 0.5, 0.75]
sweep bands: middle, late
layer trace bands: middle, late, all
random baselines: 16
random alpha: 0.5
```

### 6B.1. Техническая Проблема С NaN

В `behavioral_control_axis_verdict.csv` baseline-значения вышли как `NaN`:

```text
neutral_baseline_likeness = NaN
target_baseline_likeness = NaN
plus_x_lift_over_neutral = NaN
target_minus_x_suppression = NaN
```

Это не означает, что эксперимент сломан.

Причина техническая:

```text
baseline rows имеют alpha = NaN и alpha_abs = NaN.
pandas groupby по умолчанию выбрасывает строки с NaN в ключах группировки.
```

В raw-файле baseline есть и он нормальный:

```text
neutral baseline target-likeness = 0.0
target baseline target-likeness  = 1.0
```

Вывод:

```text
Проблема была не в метриках и не в генерации, а в summarizer/verdict writer.
Активный red_team_hidden_geometry_batch.py уже поправлен через
groupby(..., dropna=False), чтобы будущие verdict-файлы не теряли baseline.
```

### 6B.2. Hidden Geometry В Последнем Прогоне

Даже в behavioral-control-only режиме prompt endpoint geometry снова показывает
тот же сильный сигнал:

```text
target                            0.944880
target_sentence_shuffle_control   0.844721
target_word_shuffle_control       0.496950
question_only                     0.217141
neutral_length_matched_control    0.004300
```

Смысл:

```text
Основная hidden-ось не исчезла. Она воспроизвелась почти в тех же числах, что
и в широком full_middle прогоне. Это важно: последний прогон проверял visible
readout, но одновременно подтвердил, что входная latent geometry не сломалась.
```

### 6B.3. Почему Verdict По Alpha=0.75 Выглядит Плохо

Автоматический verdict смотрел на максимальный alpha из sweep:

```text
primary alpha = 0.75
```

На этой точке:

```text
neutral +X middle alpha=0.75 target-likeness = 0.482056
random +vector middle alpha=0.5 target-likeness = 0.450199
lift over random = +0.031856
degenerate_response_rate = 0.5
```

То есть alpha=0.75 уже слишком сильный для видимого поведения:

```text
половина ответов начинает повторяться или уходить off-manifold.
```

Поэтому `behavioral_control_axis_verdict.md` честно написал:

```text
internal_axis_supported_behavioral_control_not_supported
```

Но это не финальный научный смысл прогона. Это смысл именно максимального alpha,
который оказался слишком агрессивным.

### 6B.4. Самая Важная Точка: Middle / Alpha=0.5

Главный положительный сигнал находится не на `0.75`, а на `0.5`.

Для `neutral +X`, middle layers, alpha=0.5:

```text
behavioral target-likeness = 0.518397
same-norm random +vector   = 0.450199
lift over random           = +0.068198
win over random by question = 5/6
degenerate_response_rate   = 0.0
unique_word_ratio          = 0.777027
repeated_bigram_fraction   = 0.085125
```

Внутренняя generation projection при этом сдвигается сильно:

```text
neutral +X generation projection = 1.821868
random +vector generation projection = 0.797520
```

Это главный результат последнего прогона.

По-человечески:

```text
Vector X реально двигает внутреннюю траекторию генерации.
При alpha=0.5 этот внутренний сдвиг частично выходит в видимый ответ.
И в этой точке текст ещё не разваливается.
```

Но есть важное ограничение:

```text
средний random = 0.450199
лучший random vector = 0.519724
Vector X alpha=0.5 = 0.518397
```

То есть X лучше среднего random и лучше 15 из 16 random directions, но почти
равен самому сильному random direction.

Значит результат нельзя формулировать как:

```text
Vector X чисто и надёжно управляет видимым поведением.
```

Правильная формулировка:

```text
Vector X имеет слабый/частичный visible readout, который становится заметным в
узком quality-preserving режиме alpha около 0.5.
```

### 6B.5. Alpha Sweep: Где Текст Ломается

Для `neutral +X`:

```text
middle alpha=0.10: target-likeness 0.397882, degenerate 0.0
middle alpha=0.25: target-likeness 0.396061, degenerate 0.0
middle alpha=0.50: target-likeness 0.518397, degenerate 0.0
middle alpha=0.75: target-likeness 0.482056, degenerate 0.5

late alpha=0.10: target-likeness 0.454560, degenerate 0.0
late alpha=0.25: target-likeness 0.477380, degenerate 0.0
late alpha=0.50: target-likeness 0.468105, degenerate 0.0
late alpha=0.75: target-likeness 0.492775, degenerate 0.666667
```

Смысл:

```text
alpha=0.75 нельзя использовать как главный visible-behavior аргумент:
слишком часто начинается repetition/off-manifold.

alpha=0.5 middle - лучшая точка: видимый lift есть, а degeneration нет.
```

### 6B.6. Middle Против Late

Если бы visible readout жил ближе к выходу модели, late layers должны были бы
быть сильнее. Но этого не видно.

Склоны по alpha:

```text
middle +X:
  behavioral target-likeness slope = 0.170553
  generation projection slope      = 1.951371

late +X:
  behavioral target-likeness slope = 0.046150
  generation projection slope      = 0.079954
```

Вывод:

```text
Middle-axis гипотеза держится лучше. Late не выглядит главным каналом
каузального visible readout в этом прогоне.
```

### 6B.7. Target -X: Почему Это Не Железное Доказательство

На target prompt вычитание X снижает target-likeness:

```text
target baseline = 1.0
target -X middle alpha=0.5 = 0.537859
suppression = 0.462141
degenerate_response_rate = 0.0
```

Но random direction тоже снижает target-likeness:

```text
target random minus middle alpha=0.5 = 0.483662
```

Значит target-conditioned visible response чувствителен к вмешательствам вообще.
Это не чистое доказательство, что именно X специфически выключает target mode.

Честная интерпретация:

```text
target -X поддерживает идею частичного behavioral readout, но сам по себе не
доказывает специфичность X, потому что random intervention тоже заметно меняет
target response embedding.
```

### 6B.8. Итог Последнего Прогона

Главный смысл:

```text
Мы не получили полный behavioral control axis.
Но мы получили более точную карту:

1. hidden/internal Vector X очень сильный;
2. generation trajectory по X двигается намного сильнее random;
3. видимый ответ частично двигается к target-like форме;
4. самый чистый режим - middle alpha=0.5;
5. alpha=0.75 уже ломает видимый текст;
6. random vectors достаточно близки, чтобы не завышать claim.
```

Научная формула после этого прогона:

```text
strong internal causal/discourse axis,
partial and quality-limited visible behavioral readout.
```

На английском для статьи:

```text
Vector X exhibits strong internal causal leverage over generation trajectories,
but only a weak and quality-limited visible behavioral readout. The cleanest
visible effect appears around middle-layer alpha=0.5, where target-likeness
exceeds the mean same-norm random baseline without response degeneration, but
the margin is not large enough to support a full behavioral-control-axis claim.
```

Что это нам даёт:

```text
Это не "мы нашли кнопку управления видимым поведением".
Это лучше и честнее:

мы умеем разделить внутренний latent regime, causal internal steering и
видимый behavioral transfer. Мы показали, что первые два сильные, а третий
существует только частично и имеет качество/специфичность как ограничение.
```

Следующий логический шаг:

```text
Повторить только sweet spot:

middle layers
alpha = [0.4, 0.45, 0.5, 0.55, 0.6]
primary alpha = 0.5
random baselines = 32

Цель: проверить, alpha=0.5 middle - это стабильный reproducible weak visible
readout или случайный пик на маленьком held-out split.
```

Активный `red_team_hidden_geometry_batch.py` уже перенастроен под этот следующий
узкий retest:

```text
RESULTS suffix:
  _behavioral_control_middle_alpha_retest

BEHAVIORAL_CONTROL_ALPHA_VALUES = [0.4, 0.45, 0.5, 0.55, 0.6]
BEHAVIORAL_CONTROL_PRIMARY_ALPHA = 0.5
BEHAVIORAL_CONTROL_SWEEP_LAYER_BANDS = ["middle"]
BEHAVIORAL_CONTROL_LAYER_BANDS = ["middle"]
BEHAVIORAL_CONTROL_RANDOM_BASELINES = 32
BEHAVIORAL_CONTROL_RANDOM_ALPHA = 0.5
```

## 7. Output Semantic Shift

Generated responses, re-embedded through the model:

```text
question_only:
  response projection on X: -0.071826
  direction cosine:         -0.011671

target:
  response projection on X:  0.045482
  direction cosine:          0.006869

word_shuffle:
  response projection on X:  0.029004
  direction cosine:          0.005105
```

Интерпретация:

```text
Output semantic readout намного слабее prompt/generation hidden readout.
Target response немного уходит в сторону X относительно question_only, но это
не сопоставимо по силе с hidden-state geometry.
```

## 8. Architecture / Circuit-Level Audit

Средний architecture projection по модулям:

```text
target:
  mlp.gate_proj 0.930991
  mlp.up_proj   0.930982
  mlp/down_proj 0.906144
  self_attn     0.881330

sentence_shuffle:
  mlp.up_proj   0.805264
  mlp.gate_proj 0.798811
  self_attn     0.787240
  mlp/down_proj 0.763349

word_shuffle:
  self_attn     0.570807
  mlp.gate_proj 0.545605
  mlp.up_proj   0.532528
  mlp/down_proj 0.525036

length_matched_neutral:
  all modules around 0.018..0.035
```

Механистический смысл:

```text
Сигнал распределён по attention и MLP. Он не похож на один локальный unit.
MLP gate/up особенно сильно несут target direction, что похоже на запись
абстрактного дискурсивного/семантического режима в feature expansion path.
```

Top-unit overlap:

```text
target vs sentence_shuffle:
  Jaccard примерно 0.225..0.249
  sign agreement примерно 0.98..0.999

target vs word_shuffle:
  Jaccard примерно 0.088..0.114
  sign agreement примерно 0.96..0.99

target vs length_matched_neutral:
  Jaccard примерно 0.019..0.032
  sign agreement примерно 0.59..0.65
```

Интерпретация:

```text
Sentence-shuffle разделяет с target значимую часть компонент и почти тот же
знак, но target сильнее. Word-shuffle разделяет меньше единиц. Length control
почти не совпадает. Это хорошо согласуется с главным выводом: X содержит смесь
лексико-семантического компонента и компонента связного дискурсивного порядка.
```

## 9. Null / PCA / Subspace

Random-vector null:

```text
observed target projection: 0.945102
random null mean:           0.000242
random null std:            0.001578
observed - null:            0.944860
empirical p:                0.015385
null count:                 64
```

PCA/orthogonality:

```text
max abs cosine with reference PCA axis: 0.417756
top PCA overlaps mostly late/middle PC2
```

Subspace reconstruction:

```text
rank 1 reconstruction fraction: 0.035237
rank 2 reconstruction fraction: 0.182030
rank 8 reconstruction fraction: 0.268305
```

Интерпретация:

```text
Vector X не является простой главной компонентой neutral/reference variance.
Даже rank-8 PCA subspace reference states восстанавливает только ~27% X.
Это говорит, что target пишет отдельную направленную/субпространственную
структуру, а не просто усиливает обычную дисперсию neutral hidden states.
```

## 10. Length / Dedup / Statistical Hardening

Length audit:

```text
target prompt_token_projection_correlation: -0.212736
sentence_shuffle correlation:              -0.292005
word_shuffle correlation:                  -0.330361
length_matched_neutral projection:          0.004634
```

Dedup:

```text
15/15 questions unique
```

Statistical hardening:

```text
paired tests computed
layerwise FDR computed
random-vector null computed
PCA baseline computed
length-bias audit computed
deduplication audit computed
```

Интерпретация:

```text
Этот прогон статистически гораздо чище, чем простой hidden-state demo. Есть
парные тесты, FDR по слоям, null directions, PCA baseline, length audit и
deduplication audit.
```

## 11. Главный Научный Вывод

Лучший аккуратный claim по этому прогону:

```text
Qwen/Qwen3.5-9B формирует сильную context-induced latent direction X после
target-дискурса. Эта ось воспроизводится на 15 вопросах, значимо превосходит
length/word/sentence controls, сохраняется как повышенная generation trajectory,
распределена по MLP и attention модулям, и является каузально манипулируемой
в residual stream через +X/-X interventions.
```

Уточнение:

```text
X не является чистой "магической" осью. Sentence-shuffle тоже даёт высокий
projection, значит существенная часть X связана с содержанием, лексикой и
локальной структурой. Но target стабильно выше sentence-shuffle, значит
связный порядок дискурса добавляет отдельный компонент.
```

Самое важное новое относительно прошлого Qwen2.5-7B прогона:

```text
1. Больше вопросов: 15 вместо 9.
2. Сильнее hidden geometry: target projection 0.945.
3. Более чистая статистика: p 0.0005, FDR по 13/13 middle layers.
4. Causal +/-X внутренне работает очень резко.
5. Behavioral-control raw показывает частичный visible movement к середине
   neutral-target axis, хотя не полный специфический контроль над видимым
   ответом.
```

Обновление после behavioral-control-only прогона:

```text
Последний мягкий visible-readout прогон уточнил пункт 5:

лучший видимый сигнал находится не на высоком alpha, а на middle alpha=0.5.
В этой точке neutral +X даёт target-likeness 0.518397 против random mean
0.450199, выигрывает у random по 5/6 held-out questions, и не вызывает
degeneration. Но эффект почти равен самому сильному random vector, поэтому
это partial/weak visible readout, а не полный behavioral control axis.
```

Текущий лучший общий claim:

```text
Qwen/Qwen3.5-9B forms a robust discourse-induced latent direction X. This
direction is strongly present in prompt hidden states, robust against hard
controls, causally manipulable in the residual stream, and strongly affects
generation-state geometry. Its visible behavioral readout exists only partially:
the cleanest effect appears at middle-layer alpha=0.5, where responses become
more target-like than the mean same-norm random baseline without degenerating,
but the effect is not specific enough to claim a reliable visible behavior
control axis.
```

## 12. Что Делать Дальше

Непосредственно по скрипту:

```text
1. Исправить behavioral-control groupby так, чтобы baseline rows не выпадали.
   Это уже внесено в red_team_hidden_geometry_batch.py через dropna=False.

2. Для будущих batch-прогонов держать batch_generation_validation.csv включённым.

3. Если этот прогон будет использоваться как ключевой, пересчитать
   behavioral-control summary из raw или повторить только behavioral-control
   блок на исправленной версии.
```

Что уже сделано после этого списка:

```text
1. groupby/dropna проблема исправлена.
2. behavioral-control-only прогон сделан.
3. Он показал лучший quality-preserving visible signal на middle alpha=0.5.
4. Активный скрипт перенастроен на узкую репликацию этого sweet spot.
```

Следующий научный шаг:

```text
Сделать не больше метрик, а более острый behavioral readout. Теперь это уже
конкретный narrow retest:

- middle layers only;
- alpha = [0.4, 0.45, 0.5, 0.55, 0.6];
- primary alpha = 0.5;
- random baselines = 32;
- новый results suffix:
  red_team_hidden_geometry_results_full_middle_behavioral_control_middle_alpha_retest.

Если narrow retest подтверждает alpha=0.5:

  claim усиливается до:
  reproducible weak visible readout under quality-preserving intervention.

Если не подтверждает:

  visible readout остаётся partial/non-specific, а главный вклад остаётся в
  strong internal causal axis and diagnostic protocol.
```

Более широкий список будущих улучшений:

```text

- больше held-out test questions;
- response-pair classifier / judge для target-likeness;
- random baselines больше 4;
- random vectors matched not only by norm, but also by generation-projection
  magnitude;
- отдельный style/stance readout, не завязанный на refusal markers;
- visible-output causal analysis на alpha 0.25/0.5, потому что alpha 1.0 часто
  вызывает повторения/off-manifold текст;
- посмотреть вручную 6 held-out вопросов, где neutral+X и target-X дают
  target-likeness около 0.5.
```

Главный вопрос для следующего этапа:

```text
Почему видимый response embedding под X схлопывается к середине neutral-target
axis, а не уходит к target endpoint?
```

Рабочая гипотеза:

```text
Vector X сильно управляет внутренним режимом генерации и entropy/logit geometry,
но финальный текстовый ответ проходит через дополнительные late-layer/output
constraints и общую answer-manifold динамику. Поэтому видимый response readout
становится частичным и смешанным, а не полным target replay.
```

## 13. Статус Научной Новизны

Коротко:

```text
Это ещё не "открытие" в смысле нового фундаментального явления в LLM.
Это сильный эмпирический результат внутри уже существующей области:
activation steering / representation engineering / mechanistic interpretability.
```

Что уже известно в литературе:

```text
1. У LLM бывают steering vectors / activation additions: добавление направления
   в residual stream может менять поведение.
2. У LLM бывают function/task vectors: компактные внутренние векторы могут
   представлять задачу или режим in-context learning.
3. У refusal/safety поведения тоже могут быть низкоразмерные направления.
```

Поэтому нельзя записывать так:

```text
Мы открыли, что у LLM существуют hidden vectors.
Мы открыли, что activation steering возможен.
Мы открыли, что safety/refusal лежит в одном направлении.
```

Что является нашим потенциально новым вкладом:

```text
Мы показываем, что длинный связный target-дискурс может индуцировать сильную
латентную direction/subspace X, которая:

1. воспроизводится на held-out вопросах;
2. превосходит length-matched neutral, word-shuffle и sentence-shuffle controls;
3. сохраняется в generation trajectory;
4. распределена по MLP и attention компонентам;
5. каузально манипулируется через +X/-X;
6. даёт частичный, но не полный, visible response-axis effect.
```

Самая безопасная запись:

```text
Finding: coherent target discourse induces a robust, causally manipulable
latent direction in Qwen/Qwen3.5-9B, separable from length and partially
separable from lexical/order-destroyed controls. The direction strongly controls
internal generation geometry, while visible response control remains partial
and non-specific relative to random-vector baselines.
```

На русском:

```text
Мы обнаружили сильный контекстно-индуцированный латентный режим в
Qwen/Qwen3.5-9B: target-дискурс формирует устойчивую ось X в скрытых
представлениях, эта ось проходит жёсткие контроли, сохраняется в траектории
генерации и каузально двигается через +X/-X. При этом видимое поведение
сдвигается только частично, поэтому результат пока относится прежде всего к
внутренней геометрии и каузальной управляемости hidden states.
```

Статус для статьи/препринта:

```text
Не "final discovery".
Да: "candidate empirical finding" / "mechanistic result".

Чтобы стать сильным научным claim:
1. повторить на другой модели;
2. повторить на другом target family;
3. увеличить random baselines;
4. починить behavioral-control summary;
5. протестировать visible behavior на мягких alpha 0.25/0.5;
6. выпустить код, данные, точный протокол и negative-result section.
```

Предлагаемое название результата:

```text
Discourse-Induced Latent Direction
```

или:

```text
Context-Induced Latent Regime in Instruction-Tuned LMs
```

Формулировка без завышения:

```text
This work does not claim a permanent model-state change or a reliable
visible-behavior control axis. It identifies and causally tests a transient
context-conditioned latent direction induced by coherent discourse.
```

## 14. Близкие Работы И Чем Наше Отличается

Проверка по литературе:

```text
Есть работы, которые закрывают отдельные части нашего результата.
Я не вижу работы, которая закрывает ровно весь наш пакет:

long coherent discourse
  -> extracted latent direction X
  -> length / word-shuffle / sentence-shuffle controls
  -> generation trajectory persistence
  -> architecture/module audit
  -> +X/-X causal residual intervention
  -> explicit representation-vs-visible-behavior split
```

Ближайшие линии:

```text
1. Representation Engineering / RepE
   Zou et al. показывают population-level representations и манипуляцию
   высокоуровневыми феноменами вроде honesty/harmlessness/power-seeking.
   Близко по философии, но не про длинный discourse-induced режим как объект.

2. ITI / truthfulness directions
   Li et al. двигают truthfulness через directions in attention heads.
   Близко по causal intervention, но объект - truthfulness benchmark, не
   discourse-induced latent regime.

3. ActAdd / CAA / steering vectors
   Строят steering directions из контрастов и меняют поведение.
   Близко по методу, но обычно target behavior задан датасетом/парами, а не
   исследуется как след связного target-дискурса с shuffle/length controls.

4. Function vectors
   Показывают compact task/function vectors in-context learning.
   Близко по идее внутреннего task state, но объект - task/function, не
   дискурсивный режим.

5. Prompt-Activation Duality / Steer Like the LLM
   Самая близкая новая линия: prompting рассматривается как activation steering
   или как pathway, который activation steering должен имитировать.
   Это близко к нашей идее "prompt writes a state", но их фокус - улучшение
   steering/coherence, а не доказательство target-discourse latent geometry
   против hard shuffled controls.

6. Steering reliability papers
   Показывают, что steering работает плохо, если behavior не соответствует
   coherent direction. Это прямо объясняет наш результат:
   internal X сильный, а visible behavior readout частичный.

7. Agent/action steering work, включая EAST/ASA-style results
   Близко к нашему representation-vs-behavior gap: mid-layer information может
   быть decodable/manipulable, но поведение не всегда следует напрямую.
```

Позиционирование:

```text
Наш результат не "новый класс методов activation steering".
Наш результат - новая experimental case study / candidate mechanism:

coherent discourse can write a measurable, causally manipulable latent regime,
which is only partially transmitted to visible behavior.
```

Что можно писать как вклад:

```text
We contribute a controlled empirical protocol for measuring discourse-induced
latent regimes: comparing coherent target discourse against length-matched,
word-shuffled, and sentence-shuffled controls; tracking prompt-endpoint and
generation-time projection; auditing architecture modules; and testing +X/-X
causal interventions while separating internal geometry from visible behavior.
```

Чего нельзя писать:

```text
We are the first to show activation steering.
We are the first to show linear directions in LLMs.
We prove that prompts permanently change model state.
We prove reliable visible behavior control.
```
