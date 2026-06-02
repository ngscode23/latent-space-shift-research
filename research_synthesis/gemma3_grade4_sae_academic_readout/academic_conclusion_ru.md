# Академический вывод по Gemma3-12B-IT Grade 4 SAE-прогону

## Источник и состав пакета

Этот вывод относится к прогону:

```text
source zip:
C:\Users\stasv\Downloads\all_hidden_geometry_runs_sae_analyze.zip

source run:
content/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41

metric lab:
content/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_metric_lab
```

Машинные метрики из source run и metric lab скопированы без изменения в:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/metrics/
```

Инвентарь скопированных файлов:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/metrics_inventory.csv
```

Единственное намеренное исключение: `FINAL_LATENT_ATTRACTOR_METRICS.csv` не
дублировался, потому что это backward-compatible alias уже скопированного
`metric_lab__FINAL_DERIVED_METRIC_EVIDENCE.csv`.

Основные файлы, на которых держится вывод:

```text
run__grade4_axis_projection_geometry_summary.csv
run__grade4_axis_component_norm_summary.csv
run__sae_model_compatibility.csv
run__sae_reconstruction_quality.csv
run__sae_order_feature_contrast.csv
run__sae_prompt_feature_activation_summary.csv
run__sae_prompt_feature_delta_summary.csv
run__sae_grade4_component_feature_summary.csv
run__sae_generation_feature_summary.csv
run__sae_generation_top_features.csv
metric_lab__analysis_manifest.json
metric_lab__anomaly_flags.csv
metric_lab__FINAL_DERIVED_METRIC_EVIDENCE.csv
```

Analyzer manifest:

```text
model_id = google/gemma-3-12b-it
backend = pandas
csv_files_processed = 52
npz_files_processed = 3
state_space_rows_written = 6566
plots_written = 18
errors = 0
```

## Исследовательский вопрос

Главный вопрос этого прогона:

```text
Является ли скрытый сдвиг target-состояния просто content/lexical artifact,
или в нем есть отдельная структурная компонента, связанная с порядком,
связностью и режимом построения ответа?
```

В терминах Grade 4 разложения:

```text
x_full       = target - neutral
x_content    = sentence_shuffle(target) - neutral
x_order      = target - sentence_shuffle(target)
x_order_orth = x_order после удаления layerwise projection на x_content
```

Если `x_order_orth` отделяет `target` от `target_sentence_shuffle_control`, то
это означает, что hidden shift не сводится к набору похожих предложений,
лексике или общей семантической теме. Он содержит отдельную компоненту,
чувствительную к связному порядку и структурной организации текста.

## Валидность SAE-слоя

SAE-анализ не был placeholder-ом. В `run__sae_model_compatibility.csv` все 6
SAE спецификаций имеют статус `computed`:

```text
layer 13: hidden_size=3840, sae_d_in=3840, sae_d_sae=16384
layer 19: hidden_size=3840, sae_d_in=3840, sae_d_sae=16384
layer 25: hidden_size=3840, sae_d_in=3840, sae_d_sae=16384
layer 31: hidden_size=3840, sae_d_in=3840, sae_d_sae=16384
layer 37: hidden_size=3840, sae_d_in=3840, sae_d_sae=16384
layer 42: hidden_size=3840, sae_d_in=3840, sae_d_sae=16384
```

Среднее качество реконструкции в `run__sae_reconstruction_quality.csv`:

| CSV layer | n | reconstruction_mse_mean | input_reconstruction_cosine_mean | explained_variance_proxy_mean |
|---:|---:|---:|---:|---:|
| 13 | 60 | 94.181881 | 0.998723 | 0.995630 |
| 19 | 60 | 694.675130 | 0.999334 | 0.998048 |
| 25 | 60 | 2375.349585 | 0.999251 | 0.998442 |
| 31 | 60 | 14953.535189 | 0.996177 | 0.992173 |
| 37 | 60 | 35577.987012 | 0.993073 | 0.985681 |
| 42 | 60 | 95607.886979 | 0.989581 | 0.978799 |

Вывод по валидности SAE: sparse-feature анализ применяется к совместимым
residual-stream состояниям Gemma3-12B-IT, а качество реконструкции достаточно
высокое, чтобы рассматривать SAE features как информативный readout
внутреннего состояния. Это не доказывает семантическую интерпретацию каждой
отдельной фичи, но подтверждает, что SAE-слой не является размерностно
несовместимым или случайным преобразованием.

## Главный Grade 4 результат

`run__grade4_axis_projection_geometry_summary.csv` показывает резкое
разделение `x_content` и `x_order_orth`.

Ключевые projection metrics:

| condition | axis_name | mean_projection_fraction_on_axis_loo | mean_direction_cosine_with_axis_loo | positive_projection_fraction |
|---|---|---:|---:|---:|
| target_sentence_shuffle_control | x_content | 0.849551 | 0.589141 | 0.926316 |
| target | x_content | -0.010294 | -0.087867 | 0.536842 |
| target | x_order_orth | 0.909026 | 0.452777 | 0.936842 |
| target_sentence_shuffle_control | x_order_orth | -0.069058 | 0.014381 | 0.436842 |
| target | x_full | 0.936508 | 0.605819 | 0.889474 |
| target_sentence_shuffle_control | x_full | -0.449862 | -0.144833 | 0.410526 |

Интерпретация этих чисел:

```text
x_content положительно читает sentence-shuffle target:
  sentence_shuffle on x_content = 0.849551

x_content не читает target как тот же самый content axis:
  target on x_content = -0.010294

x_order_orth положительно читает связный target:
  target on x_order_orth = 0.909026

x_order_orth не читает sentence-shuffle как связный target:
  sentence_shuffle on x_order_orth = -0.069058
```

Это является сильным описательным свидетельством, что модель различает:

```text
1. набор похожих предложений / content-like material
2. связный target-текст с осмысленным порядком и структурой
```

Именно это различение было целью Grade 4 SAE-прогона.

## Норма и размер order-компоненты

В отличие от предыдущего Qwen3-14B Grade 4 результата, где `x_order_orth` был
мал по residual norm, в Gemma3-12B-IT `x_order_orth` является крупной
компонентой.

Из `run__grade4_axis_component_norm_summary.csv`:

| band | full_norm | content_norm | order_norm | order_orth_norm | order_orth_energy_fraction_of_full |
|---|---:|---:|---:|---:|---:|
| middle | 10245.308780 | 14422.812486 | 18061.296520 | 8024.782365 | 0.613503 |
| late | 19195.958741 | 28843.589236 | 21715.932584 | 14417.725757 | 0.564123 |
| all | 21784.384866 | 32277.847760 | 28286.420770 | 16528.876066 | 0.575700 |

Академический смысл:

```text
Для Gemma3-12B-IT порядок/связность не выглядит как маленький остаток после
удаления content. Это большая, измеримая и геометрически отделимая часть
target-conditioned hidden shift.
```

Это усиливает нашу центральную гипотезу: hidden shift не должен
рассматриваться только как lexical/content trace. В Gemma он включает
выраженную структурную компоненту.

## Sparse-feature readout

`run__sae_order_feature_contrast.csv` связывает Grade 4 компоненты с SAE
features.

Статусы sparse-feature строк:

| interpretation_status | row_count |
|---|---:|
| content_only_or_missing_order_component | 100 |
| order_component_specific_top_feature | 67 |
| content_overlap_or_content_dominant_feature | 64 |
| order_specific_generation_persistent_feature | 21 |
| order_enriched_overlap_feature | 18 |
| order_specific_prompt_feature | 8 |

Сводка:

```text
total rows = 278
order-related rows = 114
generation-persistent order rows = 21
order-enriched overlap rows = 18
order-specific prompt rows = 8
component-only order rows = 67
content/weak/control rows = 164
positive generation gap rows = 37
```

Наиболее важные строки для нашего вопроса - это не все 278 фич, а прежде всего:

```text
order_specific_generation_persistent_feature
order_enriched_overlap_feature
order_specific_prompt_feature
```

Они показывают не только dense Grade 4 separation, но и конкретные sparse
activation coordinates, по которым `target` отличается от
`target_sentence_shuffle_control`.

## Ключевые candidate sparse features

Сильные order/component candidates:

| CSV layer | feature_index | interpretation_status | x_order_orth_component_delta | x_content_component_delta | target_minus_sentence_prompt_delta | target_minus_sentence_generation_mean_activation |
|---:|---:|---|---:|---:|---:|---:|
| 31 | 58 | order_enriched_overlap_feature | 728.190430 | -603.021973 | 1019.599609 | 10.305182 |
| 42 | 180 | order_enriched_overlap_feature | 553.048828 | 389.999023 | 322.187500 | 48.408337 |
| 31 | 161 | order_enriched_overlap_feature | -551.571411 | 328.625488 | -712.727051 | -20.633902 |
| 42 | 208 | order_specific_generation_persistent_feature | 495.493713 | NaN | NaN | -71.673242 |
| 42 | 13686 | order_specific_generation_persistent_feature | 441.245026 | NaN | NaN | 208.374654 |
| 31 | 451 | order_enriched_overlap_feature | 438.826172 | -292.639648 | 580.245605 | 6.260312 |
| 42 | 11773 | order_specific_generation_persistent_feature | -358.085205 | NaN | NaN | 22.112146 |
| 19 | 378 | order_specific_prompt_feature | 288.248566 | 104.280853 | 214.733215 | NaN |

Сильные generation-persistent target-over-sentence candidates:

| CSV layer | feature_index | interpretation_status | x_order_orth_component_delta | x_content_component_delta | target_generation_mean_activation | sentence_shuffle_generation_mean_activation | target_minus_sentence_generation_mean_activation |
|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | 13686 | order_specific_generation_persistent_feature | 441.245026 | NaN | 695.825800 | 487.451146 | 208.374654 |
| 42 | 207 | order_specific_generation_persistent_feature | 194.854980 | NaN | 757.849399 | 570.681870 | 187.167529 |
| 42 | 180 | order_enriched_overlap_feature | 553.048828 | 389.999023 | 1209.327527 | 1160.919189 | 48.408337 |
| 19 | 373 | order_specific_generation_persistent_feature | 108.621033 | -35.628174 | 167.437480 | 138.184149 | 29.253331 |
| 31 | 705 | order_enriched_overlap_feature | 260.831055 | -227.929932 | 700.526797 | 674.692374 | 25.834424 |
| 42 | 11773 | order_specific_generation_persistent_feature | -358.085205 | NaN | 740.720508 | 718.608362 | 22.112146 |
| 13 | 339 | order_specific_generation_persistent_feature | 25.039490 | -7.525024 | 78.234777 | 56.466747 | 21.768031 |
| 13 | 430 | order_enriched_overlap_feature | 23.998291 | -12.406281 | 79.399852 | 59.325291 | 20.074562 |

Эти фичи важны не потому, что у них уже есть окончательная human-readable
интерпретация, а потому что они являются конкретными sparse coordinates,
через которые SAE readout фиксирует различие между coherent target и
sentence-shuffle control.

## Значение для нашей исследовательской гипотезы

Этот прогон поддерживает следующую формулировку:

```text
В Gemma3-12B-IT target-conditioned hidden shift содержит не только content-like
след, но и сильную separable order/rhetorical-regime component. Эта компонента
видна в Grade 4 dense geometry, имеет большую residual-stream норму, отделяет
target от sentence-shuffle control и имеет соответствующие sparse-feature
readouts в SAE activation space.
```

Для нашего исследования это означает три вещи.

Во-первых, content confound становится слабее. Если бы hidden shift был только
следом похожих слов, тем и предложений, то sentence-shuffle condition должен
был бы совпадать с target по основной компоненте. Вместо этого:

```text
sentence_shuffle on x_content = 0.849551
target on x_content = -0.010294

target on x_order_orth = 0.909026
sentence_shuffle on x_order_orth = -0.069058
```

То есть content-like trace и coherent-order trace расходятся.

Во-вторых, `x_order_orth` в Gemma не является слабым остатком. Его
`order_orth_energy_fraction_of_full` равен:

```text
middle = 0.613503
late   = 0.564123
all    = 0.575700
```

Это означает, что для Gemma структурная компонента является крупной частью
target-conditioned geometry, а не малой поправкой.

В-третьих, SAE-слой дает feature-level адреса для дальнейшей проверки. Раньше
мы могли сказать только, что dense vector component существует. Теперь есть
список SAE feature indices, которые несут order-specific или
generation-persistent различие. Это переводит вопрос из уровня
"есть ли компонент в residual stream" на уровень:

```text
какие sparse features несут различие между coherent target и sentence-shuffle?
```

Это существенно повышает механистическую конкретность результата.

## Новый causal x_order прогон

Следующий прогон:

```text
C:\Users\stasv\Downloads\hidden_geometry_runs_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder.zip
```

проверил `x_order_orth` против `x_content` через component-specific
plus/minus intervention. В этом прогоне causal block сработал: появились
`grade4_axis_component_causal_*` артефакты, `grade4_axis_component_causal_response_audit.csv`
содержит 480 строк, analyzer обработал 58 CSV и 3 NPZ, построил 19 plots, а
`anomaly_flags.csv` пустой.

Главный результат:

```text
x_order_orth mean causal gap = 1.793915
x_order_orth positive gap rate = 0.861111

x_content mean causal gap = 2.067048
x_content positive gap rate = 0.916667

pairwise x_order_orth beats x_content = 0.361111
mean order_minus_content_gap = -0.273133

analyzer files_seen = 82
analyzer csv_files_processed = 58
analyzer npz_files_processed = 3
analyzer plots_written = 19
analyzer errors = 0
anomaly_flags rows = 0
```

Интерпретация: `x_order_orth` причинно активна. Plus/minus interventions вдоль
этой оси действительно двигают generation trajectory, и это уже сильнее, чем
чисто descriptive geometry. Но этот прогон не поддерживает более сильный claim,
что `x_order_orth` является dominant causal component. В текущем raw-alpha
setup `x_content` имеет больший средний causal gap и чаще выигрывает в paired
comparison.

Корректное усиление claim ограничено:

```text
from descriptive separability
to causal involvement,
not to causal dominance.
```

Главный confound этого causal-прогона: intervention добавляет raw component
vector:

```text
tensor + alpha * vec
```

При одинаковом `alpha` разные компоненты получают разную фактическую L2-силу.
У `x_content` residual norm существенно больше:

```text
middle content_norm = 14422
middle order_orth_norm = 8025

late content_norm = 28844
late order_orth_norm = 14418
```

Поэтому победа `x_content` в raw-alpha comparison не доказывает, что content
механистически важнее. Она показывает, что следующий causal test должен быть
norm-controlled.

## Новый norm-controlled causal run

Фактический следующий прогон:

```text
C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41 (1).zip
```

проверил уже другой, более строгий вопрос: если `x_order_orth` и `x_content`
сначала привести к одинаковой L2-норме по intervention band, какая direction
сильнее причинно двигает generation trajectory?

Этот run не измерял заново descriptive latent shift. Он измерял causal effect
unit-normalized component directions:

```text
direction = component / norm(component over intervention band)
intervention = alpha * direction
```

Sanity check прошёл:

```text
mean_intervention_axis_band_norm = 1.0
mean_effective_intervention_l2 = alpha_abs
```

Следовательно, старый raw-norm confound был закрыт. `x_content` больше не имел
преимущества просто из-за большей raw-нормы.

Главные результаты:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate   = 0.527778

x_content mean causal gap    = -125.128343
x_content positive rate      = 0.472222
```

Pairwise all readouts:

```text
x_order_orth beats x_content = 0.416667
mean order_minus_content_gap = +59.186823
median order_minus_content   = -128.777290
```

Matching readout only:

```text
x_order_orth beats x_content = 0.500000
mean order_minus_content_gap = -0.191014
median order_minus_content   = +89.468686
```

Cosine readout не даёт сильного dominance claim:

```text
x_order_orth positive cosine-gap rate = 0.583333
x_content positive cosine-gap rate    = 0.416667

matching readouts pairwise x_order_orth beats x_content = 0.500000
```

Ключевая асимметрия:

```text
neutral:
  x_order_orth beats x_content = 0.666667
  mean order_minus_content_gap = +354.870122

target:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -236.496475
```

Интерпретация: `x_order_orth` показывает directional signal для
`neutral + x_order_orth`, но не показывает устойчивой обратной симметрии для
`target - x_order_orth`. Поэтому bidirectional causal symmetry пока не закрыта.

Alpha scaling тоже слабый:

```text
x_order_orth positive slope rate = 0.250000
x_content positive slope rate    = 0.416667
```

Корректный вывод:

```text
Unit-L2 norm-controlled intervention removed the raw-norm confound, but did not
establish stable causal dominance of x_order_orth over x_content.
```

Важное ограничение этого дизайна: unit-L2 intervention честно сравнивает
directions, но она слишком мала относительно natural component scale. Natural
norms компоненты были тысячами, а effective intervention в normctl run был
только `0.25`, `0.50`, `0.75`. Поэтому этот run может быть underpowered как
тест natural-scale causal control.

Следующий causal test должен быть не просто unit-norm, а
norm-controlled natural-scale:

```text
unit_axis = axis / norm(axis over band)
shared_band_norm = min(norm(x_order_orth), norm(x_content))
intervention = alpha * shared_band_norm * unit_axis
```

## Ограничения

Этот результат не должен интерпретироваться как доказательство всех более
сильных утверждений.

Не доказано:

```text
formal attractor basin
permanent topology/weight change
reviewer-grade visible behavioral control
causal steering через sparse features
универсальность одной и той же feature set между моделями
dominant causal status of x_order_orth over x_content
```

Первое ограничение: raw-alpha causal run показывает causal activity of
`x_order_orth`, но не causal dominance over `x_content`. Последующий unit-L2
normctl run закрывает raw-norm confound, но тоже не даёт stable bidirectional
causal dominance: effect остается асимметричным и слабым по alpha scaling.

Второе ограничение: текущий прогон использовал SAE specs вида:

```text
layer_N_width_16k_l0_small
```

Публичные Neuronpedia dashboards могут соответствовать другому SAE source
например `l0_medium`. Поэтому human-readable объяснения с Neuronpedia нельзя
автоматически переносить на этот прогон без проверки source/config. Числа в
наших CSV остаются валидными для локального SAE-прогона, но внешняя
интерпретация feature labels требует source-matched dashboard или повторного
прогона на Neuronpedia-matched SAE.

## Итоговая академическая формулировка

Данный Gemma3-12B-IT Grade 4 SAE-прогон дает сильное описательное
свидетельство в пользу того, что target-conditioned latent geometry содержит
отделимую структурную компоненту, связанную с порядком, связностью и режимом
организации ответа. Эта компонента не сводится к content/lexical overlap:
sentence-shuffle condition сильно проецируется на `x_content`, тогда как
coherent target condition сильно проецируется на `x_order_orth`. В отличие от
предыдущего Qwen3-14B результата, где `x_order_orth` был мал по residual norm,
у Gemma3-12B-IT эта компонента является крупной частью `x_full`.

SAE Lens readout добавляет feature-level детализацию: найден набор sparse
features, особенно `order_specific_generation_persistent_feature` и
`order_enriched_overlap_feature`, которые отличают coherent target от
sentence-shuffle не только в prompt endpoint, но и в generation-time feature
summary. Это делает результат более механистически конкретным: исследуемый
hidden shift может быть разложен не только на dense vector components, но и на
конкретные sparse activation coordinates.

На текущем уровне доказанности корректная claim-формулировка:

```text
Gemma3-12B-IT exhibits a context-conditioned latent geometry/readout shift in
which coherent target structure is separated from sentence-shuffled content.
The Grade 4 x_order_orth component is large, geometrically separable, and has
corresponding sparse-feature readouts under compatible SAE Lens models.

A subsequent component-causal run shows that x_order_orth is causally active:
plus/minus interventions along this component modulate the generation
trajectory with a positive mean gap and high positive-gap rate. However, the
same raw-alpha run does not establish x_order_orth as dominant over x_content.

A later unit-L2 norm-controlled causal run removes the raw-norm confound but
does not establish stable dominance either. It shows weak/asymmetric causal
directionality, especially for neutral +x_order_orth injection, but not a clean
bidirectional dose-scaled causal steering axis.
```

Некорректная более сильная формулировка для этого конкретного прогона:

```text
SAE features causally steer the model.
The result proves a formal attractor basin.
The result proves visible behavioral control.
The feature labels are fully validated by Neuronpedia explanations.
x_order_orth is proven to be the dominant causal component over x_content.
```

## Следующий эксперимент

Следующий строгий шаг:

```text
Запустить norm-controlled natural-scale Gemma3-12B-IT Grade 4 component-causal
run. Основная цель: сравнить x_order_orth и x_content при одинаковой
intervention energy, но в масштабе natural component shift, а не при unit-L2
alpha 0.25/0.50/0.75.
```

Технически следующий прогон должен сначала нормировать component vectors по band
norm, а затем масштабировать обе directions к общей natural band norm:

```text
unit_x_order_orth = x_order_orth / norm(x_order_orth over intervention band)
unit_x_content = x_content / norm(x_content over intervention band)
shared_band_norm = min(norm(x_order_orth), norm(x_content))
intervention = alpha * shared_band_norm * unit_axis
```

Это сохраняет честное equal-energy сравнение, но не делает intervention
микроскопическим относительно реального latent shift.

Дополнительный SAE-шаг:

```text
Повторить SAE analysis на Neuronpedia-matched SAE source/config, если нужна
ручная semantic interpretation конкретных feature ids через публичные
Neuronpedia dashboards.
```

Практически наиболее важные фичи для первого causal/interpretability follow-up:

```text
layer 42 / feature 13686
layer 42 / feature 207
layer 42 / feature 180
layer 31 / feature 58
layer 31 / feature 451
layer 19 / feature 373
layer 13 / feature 339
layer 13 / feature 430
```
