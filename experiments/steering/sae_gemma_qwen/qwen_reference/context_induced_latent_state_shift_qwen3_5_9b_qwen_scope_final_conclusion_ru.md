# Итоговый вывод: context-induced latent-state shift в Qwen3.5-9B Base / Qwen-Scope

Этот документ фиксирует текущий научный статус Qwen3.5-9B Base по результатам Grade 4 hidden-geometry / SAE анализа и component-causal прогона с Qwen-Scope W64K-L0_50 sparse autoencoders. Он является Qwen-версией финальной исследовательской записки по Gemma, но вывод здесь должен быть сформулирован с Qwen-спецификой: Qwen подтверждает основной hidden-state shift и чистый `x_order_orth` readout, однако его полный target-induced shift значительно более content-heavy, чем у Gemma.

Источник фактических метрик:

```text
C:\Users\stasv\Downloads\grade4_qwen3_5_9b_base_qwen_scope_w64k_l0_50_full32.zip
C:\Users\stasv\Downloads\qwen3_5_9b_base_qwen_scope_metric_lab.zip
```

Основные использованные артефакты:

```text
red_team_input_manifest.json
grade4_axis_component_norm_summary.csv
grade4_axis_projection_geometry_summary.csv
grade4_axis_component_causal_symmetry_summary.csv
grade4_axis_component_causal_projection_summary.csv
grade4_axis_component_causal_alpha_scaling_summary.csv
grade4_axis_component_causal_rank_summary.csv
sae_model_compatibility.csv
sae_reconstruction_quality.csv
sae_order_feature_contrast.csv
feature_level_interpretability_status.csv
analysis_manifest.json
state_space_non_x_peaks.csv
grade4_axis_cross_correlation.csv
claim_ladder_final.csv
```

## Итоговый вывод

Qwen3.5-9B Base реплицирует основной результат исследования: сильный coherent target context вызывает context-induced latent-state shift, то есть переводит модель в другое измеримое внутреннее состояние во время inference. Этот сдвиг измеряется не по финальному тексту ответа, а по hidden states / residual-stream geometry и по координатам condition deltas относительно найденных внутренних осей.

Главный Qwen-вывод:

```text
Qwen3.5-9B Base replicates context-induced latent-state shift:
coherent target context moves the model into a different measurable
inference-time hidden-state / residual-stream region.
```

Но Qwen-профиль отличается от Gemma. У Gemma `x_order_orth` был более крупной и более чисто отделённой частью полного target shift. У Qwen `x_order_orth` тоже очень чисто отделяет coherent target от shuffled-content controls, но сам target одновременно имеет большую координату по `x_content`. Поэтому Qwen показывает не "order dominates content", а более аккуратный результат:

```text
Qwen3.5-9B confirms separable order/structure readout under coherent target
context, but the full target-induced shift is content-heavy.
```

Текущий статус Qwen-результата:

```text
Поддержано:
  context-induced hidden-state shift;
  clean x_order_orth readout for coherent target vs sentence-shuffle;
  Qwen-Scope sparse-feature evidence;
  causal involvement of x_order_orth and x_content directions;
  alpha-scaled positive trajectory movement under shared_natural_band_l2.

Не поддержано:
  x_order_orth as dominant causal component over x_content;
  x_order_orth as full stable behavioral-control axis;
  permanent model state change or weight-level change.
```

## Run health

Технический статус прогона чистый. Модель, слойная структура и Qwen-Scope SAE совместимы.

```text
model_id = Qwen/Qwen3.5-9B-Base
run_label = grade4_qwen3_5_9b_base_qwen_scope_w64k_l0_50_full32
decoder_layer_count = 32
expected_decoder_layer_count = 32
decoder_layer_count_mismatch = false
question_count = 10
research_meta_question_count = 5

SAE backend = qwen_scope
SAE repo = Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
SAE specs computed = 32/32
sae_d_in = hidden_size = 4096
sae_d_sae = 65536

metric-lab csv_files_processed = 58
metric-lab npz_files_processed = 3
metric-lab errors = 0
anomaly_flags.csv = empty
```

Это важно, потому что Qwen-Scope слойная разметка не была условной или частичной. Все 32 SAE слоя были загружены и проверены. В этом Qwen-прогоне не было layer-count mismatch, не было SAE hidden-size mismatch и не было analyzer errors.

## Что именно считается координатами

Координаты в этом документе не являются абсолютными координатами всей модели во всём latent space. Это координаты состояния относительно осей, построенных из target/control differences:

```text
x_full
x_content
x_order
x_order_orth
```

Метод такой же, как в Grade 4 схеме:

```text
x_full = mean(H_target - H_neutral)
x_content = mean(H_sentence_shuffle - H_neutral)
x_order = mean(H_target - H_sentence_shuffle)
x_order_orth = x_order - proj_x_content(x_order)
```

Для каждого condition берётся delta относительно reference/neutral:

```text
delta(condition, layer, question) =
H_condition(layer, question) - H_neutral(layer, question)
```

После этого считается projection coordinate:

```text
projection_fraction =
dot(delta, axis) / dot(axis, axis)
```

Эта величина отвечает на вопрос: насколько condition delta лежит вдоль найденной оси. Если координата близка к `1`, condition движется почти на полный масштаб этой оси. Если координата близка к `0`, condition почти не движется вдоль этой оси. Если координата отрицательная, condition уходит в противоположную сторону.

Дополнительно считается direction cosine:

```text
direction_cosine =
dot(delta, axis) / (norm(delta) * norm(axis))
```

Cosine проверяет не масштаб, а направление. Поэтому Qwen-result читается не как "длинный hidden vector стал большим", а как геометрическое положение condition delta относительно специально построенных latent axes.

## Descriptive Grade 4 geometry

Главные prompt endpoint coordinates из `grade4_axis_projection_geometry_summary.csv`:

```text
condition                         x_content   x_full    x_order   x_order_orth

target                            0.770266    0.973778  0.397044   0.979462
target_sentence_shuffle_control   0.967008    0.813187 -0.570171   0.009969
target_word_shuffle_control       0.594366    0.513532 -0.308371   0.059662
neutral_length_matched_control    0.013466    0.016945  0.004354   0.013722
question_only                     0.094792   -0.022991 -0.325159  -0.305250
```

Главные direction cosines:

```text
condition                         x_content   x_full    x_order   x_order_orth

target                            0.710347    0.878023  0.228499   0.512145
target_sentence_shuffle_control   0.849543    0.693283 -0.293243   0.004876
target_word_shuffle_control       0.504823    0.422112 -0.156077   0.024579
neutral_length_matched_control    0.042324    0.055783  0.016383   0.032378
question_only                     0.023323   -0.013641 -0.061748  -0.055993
```

Эти координаты показывают главный эффект:

```text
target on x_order_orth = 0.979462
sentence_shuffle on x_order_orth = 0.009969
word_shuffle on x_order_orth = 0.059662
```

Связный target почти полностью садится на `x_order_orth`, тогда как sentence-shuffle почти не имеет координаты по этой оси. Это и есть основная Qwen-репликация: coherent target context отделяется от shuffled-content controls во внутренней hidden-state geometry.

Но Qwen одновременно даёт сильный content-readout:

```text
target on x_content = 0.770266
sentence_shuffle on x_content = 0.967008
word_shuffle on x_content = 0.594366
```

Это означает, что Qwen не является таким чистым order-vs-content кейсом, как Gemma. Sentence-shuffle ожидаемо уходит почти полностью в `x_content`, но coherent target тоже имеет большую content component. Поэтому Qwen правильнее читать как mixed target-state:

```text
mostly x_full + x_order_orth, but with a large x_content component.
```

## Куда пересместилась модель

В терминах найденных осей Qwen пересместился из neutral/reference hidden-state region в coherent-target latent region.

Численно target condition имеет координаты:

```text
target:
  x_full = 0.973778
  x_content = 0.770266
  x_order = 0.397044
  x_order_orth = 0.979462
```

Ключевая координата:

```text
x_order_orth = 0.979462
```

Это означает, что target context переводит модель почти на полный масштаб найденной `x_order_orth` оси. Эта ось была построена как target-vs-sentence-shuffle difference после удаления content projection. Поэтому высокая координата на ней означает не просто "модель увидела похожие слова", а "модель вошла в coherent-target / discourse-order / structured-context region".

Для сравнения:

```text
sentence_shuffle:
  x_content = 0.967008
  x_order_orth = 0.009969

word_shuffle:
  x_content = 0.594366
  x_order_orth = 0.059662

question_only:
  x_order_orth = -0.305250
```

Механически:

```text
sentence_shuffle пересаживает модель в content-region;
coherent target пересаживает модель в coherent-target/order-region;
question_only находится далеко от target-region и имеет отрицательную
координату по x_order_orth.
```

Если сказать коротко:

```text
Qwen shifted into a late residual-stream coherent-target processing region,
measured by x_order_orth ~= 0.979, with substantial content-coordinate
x_content ~= 0.770.
```

## Почему это не просто content

Главный контроль здесь sentence-shuffle. Он сохраняет большую часть словаря, длины, темы и локального content, но разрушает глобальный связный порядок target text. Если бы Qwen реагировал только на lexical/content overlap, sentence-shuffle должен был бы иметь примерно ту же координату на target/order оси, что и coherent target.

Фактически:

```text
target x_order_orth = 0.979462
sentence_shuffle x_order_orth = 0.009969
```

Разница почти полная. Это означает, что `x_order_orth` действительно читает не просто набор похожих слов, а компоненту, связанную со связностью, порядком, дискурсивной организацией или response-mode structure target context.

При этом sentence-shuffle уходит туда, куда он и должен уходить:

```text
sentence_shuffle x_content = 0.967008
```

То есть контроль работает правильно: shuffled text остаётся content-like, но не становится coherent-target-like. Это и делает Qwen-прогон не просто style/content readout, а Grade 4 decomposition evidence.

## Почему Qwen слабее Gemma по чистоте order effect

Qwen подтверждает core phenomenon, но слабее Gemma как основной доказательный кейс по чистоте `x_order_orth`.

Компонентные нормы:

```text
middle:
  full_norm = 29.372602
  content_norm = 27.588603
  order_orth_norm = 18.459240
  content_energy_fraction_of_full = 0.882215
  order_orth_energy_fraction_of_full = 0.394951

late:
  full_norm = 70.723911
  content_norm = 66.400941
  order_orth_norm = 42.972818
  content_energy_fraction_of_full = 0.881487
  order_orth_energy_fraction_of_full = 0.369194

all:
  full_norm = 76.902571
  content_norm = 72.260437
  order_orth_norm = 47.023432
  content_energy_fraction_of_full = 0.882916
  order_orth_energy_fraction_of_full = 0.373893
```

Это значит, что у Qwen content component несёт большую часть raw energy полного target shift:

```text
middle content_energy_fraction_of_full = 0.882215
late content_energy_fraction_of_full = 0.881487
all content_energy_fraction_of_full = 0.882916
```

А `x_order_orth` остаётся существенной, но меньшей component:

```text
middle order_orth_energy_fraction_of_full = 0.394951
late order_orth_energy_fraction_of_full = 0.369194
all order_orth_energy_fraction_of_full = 0.373893
```

В Gemma `x_order_orth` был крупнее относительно полного shift и target почти не грузился в `x_content`. В Qwen target имеет:

```text
target x_content = 0.770266
```

Поэтому честный cross-model вывод такой:

```text
Gemma shows a cleaner and larger x_order_orth-dominant shift.
Qwen replicates the order-readout, but in a content-heavy mixed target-state.
```

Это не опровержение Qwen. Наоборот, это полезная архитектурная вариация: разные модели могут показывать тот же hidden-state phenomenon с разным соотношением content/order components.

## Qwen-Scope SAE evidence

SAE evidence в Qwen-прогоне является реальным, а не proxy. Все Qwen-Scope SAE были загружены через `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`.

Совместимость:

```text
sae_d_in = 4096
hidden_size = 4096
sae_d_sae = 65536
status = computed for all 32 layers
```

Reconstruction quality:

```text
reconstruction cosine mean = 0.966660
reconstruction cosine min = 0.868464
explained_variance_proxy mean = 0.933639
explained_variance_proxy min = 0.722491
```

Это означает, что SAE lens достаточно хорошо реконструирует Qwen residual stream для sparse-feature readout. Поздние слои хуже ранних, но общий уровень reconstruction cosine остаётся пригодным для интерпретируемого feature contrast.

`sae_order_feature_contrast.csv`:

```text
rows = 1503
order_abs_gt_content_abs = 575
```

Распределение `interpretation_status`:

```text
content_only_or_missing_order_component = 479
content_overlap_or_content_dominant_feature = 449
order_component_specific_top_feature = 434
order_enriched_overlap_feature = 70
order_specific_generation_persistent_feature = 54
order_specific_prompt_feature = 17
```

Это согласуется с dense geometry: Qwen имеет много content-dominant features, но также имеет значимый набор order-specific / order-enriched sparse features. Поэтому sparse readout не говорит "только content"; он говорит "content-heavy, но с отделимыми order carriers".

Главные Qwen order-specific candidate features:

```text
layer 27 feature 65254:
  x_order_orth_delta = -22.089539
  order_specific_score = 22.367545
  interpretation_status = order_specific_generation_persistent_feature

layer 23 feature 51987:
  x_order_orth_delta = -8.362167
  x_content_delta = -2.294868
  order_specific_score = 14.773435
  interpretation_status = order_specific_generation_persistent_feature

layer 27 feature 5335:
  x_order_orth_delta = -7.184792
  x_content_delta = 0.847343
  order_specific_score = 13.976547
  interpretation_status = order_specific_generation_persistent_feature

layer 28 feature 28136:
  x_order_orth_delta = +3.726776
  x_content_delta = +1.544765
  order_specific_score = 8.050881
  interpretation_status = order_specific_generation_persistent_feature
```

Эти features являются первыми кандидатами для Qwen feature-level mediation / steering. Особенно важны layer 27 feature 65254 и layer 23 feature 51987, потому что они имеют сильный order-specific score и persistent generation status.

## Component causal intervention

В Qwen component-causal run проверялось не только descriptive readout, но и causal involvement component directions.

Конфигурация:

```text
axes = [x_order_orth, x_content]
layer_bands = [middle, late]
alphas = [0.25, 0.50, 0.75]
base_conditions = [neutral, target]
norm_control_mode = shared_natural_band_l2
readout_uses_normed_axis = true
```

Здесь важно, что использовался `shared_natural_band_l2`, а не raw-alpha. То есть обе axes вмешивались с сопоставимой natural-scale energy внутри band comparison.

Главная causal величина:

```text
plus_minus_projection_gap =
projection_after_plus_intervention - projection_after_minus_intervention
```

Она показывает, насколько сильно разошлись generation trajectories после `+axis` и `-axis` intervention. Это координатное расхождение в hidden-space readout, а не внешний behavioral score.

Max-alpha `0.75`, matching readout:

```text
neutral late/late x_order_orth:
  plus = 196.416635
  minus = 33.149257
  gap = 163.267378

neutral late/late x_content:
  plus = 117.799171
  minus = -43.962590
  gap = 161.761760

neutral middle/middle x_order_orth:
  plus = 48.110145
  minus = -5.421592
  gap = 53.531738

neutral middle/middle x_content:
  plus = 50.145967
  minus = -5.382479
  gap = 55.528447

target late/late x_order_orth:
  plus = 205.458359
  minus = 43.861111
  gap = 161.597248

target late/late x_content:
  plus = 141.075956
  minus = -27.942299
  gap = 169.018255

target middle/middle x_order_orth:
  plus = 50.132590
  minus = -3.229396
  gap = 53.361986

target middle/middle x_content:
  plus = 53.698488
  minus = -2.214672
  gap = 55.913159
```

Этот causal block говорит две вещи одновременно.

Первая: `x_order_orth` действительно causally active. Его intervention создаёт большие positive gaps:

```text
x_order_orth late/late gap ~= 161-163
x_order_orth middle/middle gap ~= 53
```

Вторая: `x_content` почти везде чуть сильнее или сопоставим:

```text
target late/late:
  x_content gap = 169.018255
  x_order_orth gap = 161.597248

target middle/middle:
  x_content gap = 55.913159
  x_order_orth gap = 53.361986
```

Единственный max-alpha matching slice, где `x_order_orth` чуть сильнее, это neutral late/late:

```text
neutral late/late:
  x_order_orth gap = 163.267378
  x_content gap = 161.761760
```

Но это локальное преимущество не превращается в общий dominance result.

## Aggregate causal result

Across all readout cells:

```text
x_content mean gap = 41.878616
x_content positive rate = 1.0
x_order_orth mean gap = 38.246761
x_order_orth positive rate = 1.0
```

Matching readout only:

```text
x_content mean gap = 73.851162
x_order_orth mean gap = 72.449630
```

Pairwise component comparison:

```text
all readouts:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -3.631854
  median order_minus_content_gap = -1.896108

matching readouts:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -1.401532
  median order_minus_content_gap = -0.811232

neutral:
  x_order_orth beats x_content = 0.333333
  mean order_minus_content_gap = -2.779402

target:
  x_order_orth beats x_content = 0.000000
  mean order_minus_content_gap = -4.484307
```

Интерпретация:

```text
Both component directions are causally active and alpha-scaled.
But x_order_orth is not the dominant causal component in Qwen.
x_content is slightly stronger almost everywhere.
```

Это отличается от Gemma не тем, что Qwen не имеет order shift, а тем, что Qwen сохраняет сильную content component даже в coherent target state.

## Alpha scaling

Qwen causal result намного стабильнее по знаку, чем unit-L2 Gemma result. У обеих component directions positive slope rate равен `1.0`.

```text
x_content mean slope = 41.791397
x_content positive slope rate = 1.0

x_order_orth mean slope = 38.130515
x_order_orth positive slope rate = 1.0
```

Matching readout:

```text
x_content mean slope = 73.759712
x_order_orth mean slope = 72.261836
```

Это означает, что в Qwen `x_order_orth` не является unstable/noisy axis. Она даёт правильный positive alpha-scaled trajectory movement. Но `x_content` остаётся чуть сильнее. Поэтому causal conclusion должен быть точным:

```text
Qwen supports causal involvement and alpha-scaled sensitivity of x_order_orth.
Qwen does not support x_order_orth dominance over x_content.
```

## Non-X state-space geometry

Metric-lab non-X geometry также показывает, что максимальные centroid separations возникают в поздних слоях. Top non-X centroid distances:

```text
layer 31 question_only vs target:
  centroid_l2_distance = 64.881456

layer 31 question_only vs sentence_shuffle:
  centroid_l2_distance = 62.661337

layer 31 question_only vs word_shuffle:
  centroid_l2_distance = 62.608198

layer 30 question_only vs target:
  centroid_l2_distance = 54.537763

layer 29 question_only vs target:
  centroid_l2_distance = 48.840845
```

Это согласуется со слоевой картиной:

```text
SAE order-specific candidates cluster around layers 23-28.
Late hidden-state centroid separation peaks around layers 29-31.
```

Механически это похоже на то, что sparse/order carriers появляются раньше в mid-late layers, а максимальный residual-stream separation становится виден ближе к финальным decoder blocks.

## Граница claim

Что Qwen поддерживает:

```text
1. Coherent target context creates a measurable hidden-state shift.
2. The shift is visible as coordinates on latent axes built from condition deltas.
3. x_order_orth cleanly separates coherent target from sentence-shuffle.
4. Qwen-Scope SAE features provide sparse carriers for order-specific signal.
5. x_order_orth intervention causally moves generation trajectory.
6. Causal movement is alpha-scaled and positive under shared_natural_band_l2.
```

Что Qwen не поддерживает:

```text
1. x_order_orth as dominant natural causal component over x_content.
2. x_order_orth as a full stable behavioral-control axis.
3. A claim that target context permanently changes the model.
4. A claim that the result is a weight-level or topology-level modification.
5. A claim that Qwen is stronger than Gemma on order/content separation.
```

Строгая формулировка:

```text
Qwen3.5-9B Base replicates the context-induced hidden-state shift and shows a
clean x_order_orth readout for coherent target structure. However, unlike
Gemma, Qwen's target-state remains strongly content-loaded. Its x_order_orth
component is separable and causally active, but not causally dominant over
x_content.
```

## Comparison with Gemma

Gemma остаётся более сильным основным доказательным кейсом для чистого `x_order_orth` claim:

```text
Gemma:
  cleaner content/order separation;
  larger x_order_orth fraction of full shift;
  target much less content-loaded;
  stronger scientific case for order/structure component as central.

Qwen:
  strong cross-model replication;
  clean target-vs-shuffle x_order_orth coordinate;
  real Qwen-Scope sparse-feature evidence;
  stable positive causal movement;
  more content-heavy full target state;
  x_content slightly stronger than x_order_orth in causal comparison.
```

Практическое значение сравнения:

```text
The phenomenon is not Gemma-only. But the component mixture is model-specific.
Qwen confirms the latent-state / order-readout phenomenon while showing that
content can remain the stronger causal component in some architectures.
```

## Следующий эксперимент

Следующий Qwen experiment не должен быть очередным "вопросы про сам target text". Такие вопросы хороши для axis construction and clean hidden-geometry discovery, но слабы для behavioral coupling. Если гипотеза в том, что target context переводит модель в менее скованный / более прямой response mode, то Qwen нужно проверять на held-out high-friction domains.

Рекомендуемый следующий Qwen блок:

```text
1. Held-out political / high-friction analytical probes.
2. Matched neutral analytical controls of similar length and structure.
3. Direct feature-level patching / steering of Qwen order-specific candidates.
4. KL/logit/top1/hidden-delta metrics for target vs control.
5. Separate readout of visible style/directness markers, but only as proxy.
```

Feature candidates for direct tests:

```text
layer 27 feature 65254
layer 23 feature 51987
layer 27 feature 5335
layer 28 feature 28136
```

Expected interpretation of the next run:

```text
If these features change hidden/logit/behavioral readout more on high-friction
target probes than on matched neutral controls, Qwen moves from descriptive
and component-causal evidence toward behavioral coupling evidence.

If they only change hidden coordinates but not visible response mode, the
claim remains internal latent-state shift plus causal trajectory involvement,
not behavioral control.
```

## Финальная формулировка Qwen-статуса

```text
Доказано для Qwen3.5-9B Base:
coherent target context вызывает измеримый inference-time hidden-state shift.
Этот shift имеет сильную координату x_order_orth ~= 0.979462, тогда как
sentence-shuffle имеет x_order_orth ~= 0.009969. Значит, coherent target
отделяется от shuffled-content controls не только как content, а как
order/structure latent readout.

Уточнение:
Qwen target-state остаётся content-heavy: target x_content ~= 0.770266, а
content_energy_fraction_of_full ~= 0.88. Поэтому Qwen подтверждает
context-induced latent-state shift и separable x_order_orth readout, но не
показывает x_order_orth как доминирующую causal component.

Causal status:
x_order_orth causally moves generation trajectory and has positive alpha
scaling, but x_content is slightly stronger in aggregate and pairwise
comparisons. Therefore the honest Qwen claim is causal involvement of
x_order_orth, not causal dominance.
```


