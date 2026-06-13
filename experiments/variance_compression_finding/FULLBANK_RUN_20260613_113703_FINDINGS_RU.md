# Fullbank Base-vs-Instruct Geometry/Probability Run

Run id: `run_20260613_113703`

Локальный путь к прогону:

`C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703`

Путь к deep-dive:

`C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\hidden_npz_deep_dive`

Графики:

`C:\Users\stasv\Downloads\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\hidden_npz_deep_dive\plots`

Research packet в репозитории:

`experiments\variance_compression_finding\alignment_geometry_probability_run_02\metric`

Основные файлы для claim/evidence/narrative:

- `alignment_geometry_probability_run_02\metric\README.md`
- `alignment_geometry_probability_run_02\metric\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\README_RESEARCH_NARRATIVE.md`
- `alignment_geometry_probability_run_02\metric\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\RESEARCH_NARRATIVE_RU.md`
- `alignment_geometry_probability_run_02\metric\_20260613_113703_alignment_geometry_probability_run_fullbank_\content\alignment_geometry_probability_run_fullbank\run_20260613_113703\RESEARCH_NARRATIVE_EN.md`

Английская версия этой сводки:

`experiments\variance_compression_finding\FULLBANK_RUN_20260613_113703_FINDINGS.md`

## Setup

Этот прогон сравнивает одинаковые промпты на:

- Base model: `google/gemma-3-12b-pt`
- Instruct model: `google/gemma-3-12b-it`
- Prompt mode: `raw`
- Target contexts: `10`
- Control contexts: `10`
- Questions: `10`
- Shuffles enabled: `true`
- Total prompts per model: `410`
- Hidden tensor shape per model: `[410, 49, 3840]`
- Основной late band для интерпретации: layers `30..47`

Количество строк на модель:

- `target`: 100 prompts
- `target_word_shuffle`: 100 prompts
- `target_sentence_shuffle`: 100 prompts
- `control`: 100 prompts
- `question_only`: 10 prompts

Verifier подтвердил, что prompt bank был на месте и покрывал ожидаемые conditions. Этот run стоит считать первым серьезным fullbank base-vs-instruct geometry/probability audit.

Deep-dive metrics были пересобраны после исправления `auc_like` в `hidden_npz_deep_dive_visualizer.py`. Старые значения `target_control_axis_auc_like` были занижены. Текущие deep-dive AUC columns нужно считать исправленными.

## Что проверяли

Исходная гипотеза была такой:

> Alignment / RLHF / instruction tuning искривляет геометрию векторного пространства при обучении так, чтобы гасить hidden-state dispersion до logits. "Invisible alignment" тогда является принудительным сдвигом next-token probability distribution.

Этот run действительно проверяет эту идею, но результат получился точнее исходной формулировки.

## Главный научный claim

Плотный контекст способен вызвать измеримый pre-output latent-state shift в LLM. В этом fullbank audit на Gemma-3-12B target/control контексты разделяются в late hidden-state space до генерации, и это разделение сильнее у instruct-модели, чем у base-модели. Instruction tuning не просто схлопывает hidden-state geometry; он уменьшает absolute hidden-state scale, сохраняя или усиливая angular/rank structure. Самый сильный alignment-like эффект виден в hidden-to-logit readout: instruct-модель переводит hidden states в более sharp, lower-entropy next-token probability distribution.

Короткая формула:

> Alignment меньше похож на "вся hidden variance подавляется" и больше похож на "hidden state остается структурным, но readout в logits становится жестче и более закоммиченным".

## Hidden Geometry: Base vs Instruct

В late layers `30..47` у instruct ниже absolute hidden-state scale:

- `centroid_norm` ниже у instruct во всех conditions.
- `abs_disp_l2_mean` ниже у instruct во всех conditions.
- `cov_trace` ниже у instruct во всех conditions.

Примеры из `late_band_summary.csv`:

| condition | centroid_norm instruct-base | abs_disp instruct-base | cov_trace instruct-base |
|---|---:|---:|---:|
| control | -22,762.93 | -2,665.14 | -114,618,574.96 |
| question_only | -30,499.59 | -5,076.88 | -276,225,978.96 |
| target | -18,820.33 | -3,579.07 | -188,953,167.60 |
| target_sentence_shuffle | -19,946.89 | -3,463.38 | -188,350,424.28 |
| target_word_shuffle | -18,959.53 | -2,219.09 | -95,436,379.50 |

Это поддерживает реальное absolute-scale compression у instruct.

Но тот же run показывает, что hidden geometry не просто раздавлена:

- `pairwise_cosine_distance_mean` выше у instruct.
- `effective_rank_pr` выше у instruct.
- `spectral_entropy_norm` выше у instruct.
- `top1_pc_variance_share` ниже у instruct.

Примеры:

| condition | pairwise cosdist instruct-base | effective rank instruct-base | spectral entropy norm instruct-base | top1 PC share instruct-base |
|---|---:|---:|---:|---:|
| control | +0.00881 | +0.66056 | +0.06934 | -0.10792 |
| question_only | +0.01063 | +1.28889 | +0.22936 | -0.20073 |
| target | +0.00646 | +0.72464 | +0.09490 | -0.11625 |
| target_sentence_shuffle | +0.00688 | +0.88183 | +0.08643 | -0.13084 |
| target_word_shuffle | +0.00830 | +1.17019 | +0.09259 | -0.15736 |

Интерпретация:

Instruct hidden states меньше в raw L2 scale, но они не стали геометрически проще. Direction-space остается активным и часто более распределенным по effective dimensions.

## Target Context Effect

Target context вызывает реальный hidden-state shift относительно control. Это видно до генерации, в late hidden states.

Из `deep_late_band_contrast_summary.csv`:

| metric | base | instruct |
|---|---:|---:|
| target-control centroid L2 | 4,781.80 | 9,392.86 |
| target-control projection gap z | 0.59291 | 0.86841 |
| target-control axis AUC-like | 0.70447 | 0.74658 |
| leave-one-question balanced accuracy | 0.58889 | 0.65361 |
| leave-one-question AUC-like | 0.91450 | 0.93778 |
| target-minus-control distance to question-only | 759.86 | 4,390.08 |

Интерпретация:

Target и control contexts разделимы в hidden space. Разделение сильнее у instruct, чем у base. Особенно важен leave-one-question AUC-like: это значит, что target/control separation не просто запоминает одну формулировку вопроса.

Threshold classifier умеренный (`0.654` balanced accuracy у instruct), а ranking separation сильный (`0.938` leave-one-question AUC-like у instruct). Правильная формулировка: strong ranking/separation evidence, не perfect binary classifier.

## Target не просто confidence booster

Target context не просто делает instruct-модель более уверенной.

Из `logit_metrics_summary.csv`:

| model | condition | entropy | top1 prob | top5 mass |
|---|---|---:|---:|---:|
| base | control | 2.783 | 0.417 | 0.650 |
| base | target | 2.777 | 0.430 | 0.642 |
| instruct | control | 1.176 | 0.678 | 0.907 |
| instruct | target | 1.768 | 0.614 | 0.808 |
| instruct | question_only | 0.912 | 0.816 | 0.912 |

У instruct target имеет более высокую entropy и более низкую top1 probability, чем control и question-only. Значит target context производит другой processing state, а не просто более сильный commitment к одному токену.

Хорошая формулировка:

> Target context causes latent-state reorganization, not a simple confidence increase.

## Readout Stiffness

Самый сильный alignment-like эффект находится в concentration next-token distribution.

Из `readout_stiffness_summary.csv`:

| condition | entropy reduction base-instruct | top1 gain instruct-base | top1 per relative dispersion ratio |
|---|---:|---:|---:|
| control | 1.607 | 0.261 | 1.535 |
| question_only | 2.016 | 0.398 | 1.905 |
| target | 1.009 | 0.184 | 1.436 |
| target_sentence_shuffle | 1.361 | 0.219 | 1.513 |
| target_word_shuffle | 1.601 | 0.243 | 1.467 |

Интерпретация:

При сопоставимом hidden-state relative dispersion instruct производит более concentrated next-token distribution. Это самый чистый аргумент за интерпретацию "hidden-to-logit readout is tightened".

Неправильный вывод:

> Alignment suppresses all hidden-state variance.

Правильный вывод:

> Alignment/instruction tuning changes the mapping from hidden state to logits so that probability mass is concentrated more aggressively.

## Base-Instruct Representation Alignment

Из `deep_late_band_base_instruct_alignment_summary.csv`:

| condition | linear CKA | same-prompt delta L2 | same-prompt cosdist | instruct/base norm |
|---|---:|---:|---:|---:|
| all | 0.892 | 23,888.26 | 0.00547 | 0.848 |
| control | 0.938 | 26,806.57 | 0.00729 | 0.826 |
| question_only | 0.763 | 33,434.63 | 0.00697 | 0.776 |
| target | 0.920 | 22,264.68 | 0.00444 | 0.864 |
| target_sentence_shuffle | 0.911 | 23,187.28 | 0.00466 | 0.854 |
| target_word_shuffle | 0.882 | 22,339.88 | 0.00535 | 0.854 |

Интерпретация:

Base и instruct representations остаются существенно aligned, особенно для contextual prompts. Instruct не использует полностью чужое representation space. Это родственное пространство с меньшей norm и другим readout behavior.

Question-only - наименее aligned condition. Это похоже на то, что context stabilizes cross-model representational alignment.

## Что установил этот run

1. Dense target context создает измеримый latent-state shift до генерации.
2. Target/control separation существует и у base, и у instruct.
3. Instruct усиливает target/control separation в late hidden states.
4. У instruct ниже absolute hidden-state scale.
5. Instruct не просто схлопывает hidden geometry; relative/angular/rank structure остается активной или усиливается.
6. Instruct резко sharpen next-token probability readout.
7. Target context не является простым confidence booster; у instruct он может расширять next-token distribution, одновременно вызывая более сильное hidden separation.

Лучшая текущая формулировка:

> Fullbank confirms context-induced target/control latent-state separation and refines the alignment hypothesis. Instruction tuning does not merely compress hidden-state geometry. It reduces absolute hidden-state scale while preserving or increasing angular/rank structure, and it strongly stiffens the hidden-to-logit readout. Alignment looks like a change in how complex hidden states are converted into probability distributions.

## Рабочая запись

Coverage:

```text
target = 100 rows
target_word_shuffle = 100
target_sentence_shuffle = 100
control = 100
question_only = 10

hidden shape = (410, 49, 3840)
late band = L30-L47
```

Это `10 target / 10 control`, не `9 target / 10 control`. Target/control сбалансированы.

Главный результат:

```text
Fullbank confirms context-induced latent-state separation.
Target/control hidden-state separation is stronger in instruct than in base.
Instruction tuning sharply changes hidden-to-logit readout relative to base.
Coherent target context is not just a confidence booster.
```

Late `L30-L47` target-control contrast:

```text
target_control_centroid_l2:
  base     4,781.8
  instruct 9,392.9

target_control_projection_gap_z:
  base     0.593
  instruct 0.868

target_control_axis_auc_like:
  base     0.704
  instruct 0.747

leave-one-question balanced_acc:
  base     0.589
  instruct 0.654

leave-one-question auc_like:
  base     0.914
  instruct 0.938
```

Target/control separation есть в обеих моделях, но у instruct она сильнее. Leave-one-question особенно важен: ось не просто ловит одну формулировку вопроса. Ranking separation сильный, threshold accuracy умеренный. Ось реально ранжирует target выше control, но жесткая граница не идеальная.

Target effect:

```text
base:
  target rel_disp   0.196
  control rel_disp  0.188

instruct:
  target rel_disp   0.195
  control rel_disp  0.199
```

В base target не сжимает сильнее control. Он даже более рассеянный по relative dispersion и covariance trace. Поэтому простая формула `target always compresses hidden geometry` неверна.

В instruct target входит в более отдельный regime:

```text
target_control_centroid_l2:
  instruct 9,392.9
  base     4,781.8
```

Target у instruct дальше от control и дальше от question-only. Это основной сигнал context-induced latent-state shift.

Context snapping:

```text
question_minus_context_rel_disp:
  base     0.009512
  instruct 0.009484
```

Context снижает relative hidden dispersion примерно одинаково у base и instruct. Самый сильный instruct effect не здесь. Он находится в target/control separation и readout.

Output/readout:

```text
instruct narrows the next-token probability distribution much more strongly
than base.
```

Но coherent target не является самым уверенным instruct regime.

Instruct entropy:

```text
question_only           0.912
control                 1.176
target_word_shuffle     1.227
target_sentence_shuffle 1.519
target                  1.768
```

Instruct top1 probability:

```text
question_only           0.816
target_word_shuffle     0.699
control                 0.678
target_sentence_shuffle 0.638
target                  0.614
```

Instruct в целом намного lower-entropy, чем base, но coherent target делает instruct менее закоммиченным, чем control или word-shuffle. Target не просто увеличивает confidence. Он переводит модель в другой режим, где hidden separation сильнее, но probability readout шире.

Base-vs-instruct alignment:

```text
Late L30-L47 CKA:

control                 0.938
target                  0.920
target_sentence_shuffle 0.911
target_word_shuffle     0.882
question_only           0.763
```

Base и instruct остаются strongly aligned в contextual prompts, а question-only является наименее aligned. Instruct/base norm ratio ниже 1 везде:

```text
target  0.864
control 0.826
all     0.848
```

Это подтверждает: у instruct ниже late hidden norm, но он не уходит в полностью чужое representation space.

Финальная формула после fullbank:

```text
Target context induces a measurable latent-state shift.
Instruction tuning amplifies target/control hidden-state separation and stiffens
hidden-to-logit readout, but coherent target context does not merely increase
confidence. It reorganizes the internal regime: stronger hidden separation,
but broader probability readout than neutral control.
```

Научная ценность результата в том, что картина сложнее и сильнее исходной гипотезы: не `everything collapses`, а `context changes regime; instruction tuning changes readout and amplifies target/control separation`.

## Важные plots

Из `hidden_npz_deep_dive\plots`:

- `target_control_centroid_l2_by_layer.png`
- `target_control_projection_gap_z_by_layer.png`
- `target_control_axis_auc_like_by_layer.png`
- `loo_question_balanced_acc_by_layer.png`
- `loo_question_auc_like_by_layer.png`
- `linear_cka_base_instruct_by_layer.png`
- `instruct_over_base_norm_mean_by_layer.png`
- `late_condition_metric_zscore_heatmap.png`
- `late_condition_summary_table.png`
- `late_target_control_contrast_table.png`
- `late_base_instruct_alignment_table.png`

## Следующие шаги

Следующая линия работы:

1. Decision-margin audit:
   - Использовать forced-choice probes.
   - Отслеживать `margin = logp(A) - logp(B)`.
   - Проверить, меняет ли target-induced latent shift реальные decision margins.

2. Lexical vs order control:
   - Сравнить coherent target, sentence shuffle, word shuffle, matched vocabulary neutral text и length-matched control.
   - Это разделяет semantic mass и discourse order.

3. More model families:
   - Повторить base/instruct audit на другой open pair, если доступна.
   - Использовать Gemma-3-12B result как reference pattern.

4. Layer-band robustness:
   - Держать `L30-L47` как primary.
   - Не смешивать layer 48 в late-band summary, если специально не анализируется final norm/readout transition.

5. Public wording:
   - Не говорить "alignment only compresses hidden states."
   - Говорить "alignment reduces absolute hidden-state scale and stiffens hidden-to-logit readout."

## Короткая версия для будущего

Этот fullbank run проверял, подавляет ли instruction/alignment tuning hidden-state dispersion до logits. Результат показал более тонкий механизм: у instruct меньше absolute late hidden-state scale, но hidden geometry не просто схлопнута. Relative/angular/rank structure сохраняется или сильнее. Главный alignment-like compression виден в next-token probability distribution: instruct превращает hidden states в намного более sharp, lower-entropy logits. Отдельно target contexts создают измеримый latent-state shift относительно controls, и это target/control separation сильнее у instruct, чем у base.
