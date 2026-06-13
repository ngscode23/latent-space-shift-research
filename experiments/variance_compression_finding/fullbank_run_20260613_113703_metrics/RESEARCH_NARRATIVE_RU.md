# Research Narrative RU

## Главный научный claim после fullbank run

Плотный контекст способен переводить модель в измеримо другой внутренний режим еще до ответа. У Gemma-3-12B target/control контексты разделяются в late hidden states, и у instruct это разделение сильнее, чем у base. При этом instruction/alignment tuning не просто "сжимает hidden space": он уменьшает абсолютный масштаб состояний, но сохраняет или усиливает угловую/ранговую структуру. Главный эффект выравнивания виден на readout-этапе: сложное hidden-state состояние превращается в более жесткое, низкоэнтропийное распределение следующего токена.

## Что именно проверял fullbank run

Мы сравнили `google/gemma-3-12b-pt` и `google/gemma-3-12b-it` на одном и том же банке промптов:

- `10` target contexts
- `10` control contexts
- `10` questions
- conditions: `target`, `target_word_shuffle`, `target_sentence_shuffle`, `control`, `question_only`
- `410` prompts per model
- hidden state tensor: `(410, 49, 3840)`
- основной слой анализа: `L30-L47`

Это не анализ финального текста. Это анализ состояния модели на границе промпта: late hidden states и next-token probability distribution до генерации ответа.

Source: `metadata.json`

## Evidence ladder

### Сильно держится метриками

1. Target/control latent-state separation существует.

Fullbank `L30-L47`:

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

loo_question_auc_like:
  base     0.914
  instruct 0.938
```

Смысл: target/control различаются в hidden space до ответа. У instruct это разделение сильнее.

Sources:

- `hidden_npz_deep_dive/deep_late_band_contrast_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_contrast_instruct_minus_base.csv`

2. Instruct усиливает target/control separation.

Почти все главные contrast-метрики выше у instruct:

```text
target_control_centroid_l2 instruct/base ratio = 1.964
target_control_projection_gap_z instruct/base ratio = 1.465
target_control_axis_auc_like instruct-base = +0.042
loo_question_auc_like instruct-base = +0.023
```

Смысл: это не просто общий эффект контекста. Instruct-модель сильнее разделяет target/control режимы в late hidden states.

Source: `hidden_npz_deep_dive/deep_late_band_contrast_instruct_minus_base.csv`

3. Instruct сильнее сужает next-token distribution.

```text
entropy_reduction_base_minus_instruct:
  target        1.009
  control       1.607
  question_only 2.016

top1_prob_gain_instruct_minus_base:
  target        0.184
  control       0.261
  question_only 0.398

top1_per_rel_disp_instruct_over_base:
  target        1.436
  control       1.535
  question_only 1.905
```

Смысл: на единицу hidden-state relative dispersion instruct выдает более концентрированный next-token readout. Это главный evidence для readout stiffness.

Source: `readout_stiffness_summary.csv`

4. Hidden geometry не просто схлопывается.

У instruct ниже:

- `centroid_norm`
- `abs_disp_l2_mean`
- `cov_trace`

Но у instruct выше:

- `pairwise_cosine_distance_mean`
- `effective_rank_pr`
- `spectral_entropy_norm`

И ниже:

- `top1_pc_variance_share`

Пример по `target`:

```text
centroid_norm:
  base     126,770
  instruct 107,950

abs_disp_l2_mean:
  base     24,828
  instruct 21,249

pairwise_cosine_distance_mean:
  base     0.0180
  instruct 0.0245

effective_rank_pr:
  base     1.91
  instruct 2.64

spectral_entropy_norm:
  base     0.251
  instruct 0.346

top1_pc_variance_share:
  base     0.718
  instruct 0.601
```

Смысл: absolute scale ниже, но angular/rank structure выше. Поэтому формула "alignment просто гасит hidden-дисперсию" слишком грубая.

Sources:

- `late_band_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_instruct_minus_base.csv`

### Средне держится метриками

1. Target переводит instruct в отдельный режим обработки.

Target у instruct дальше от `question_only` и от `control`, но probability distribution при этом шире, чем у neutral control.

```text
target_to_question_centroid_l2:
  base     5,348
  instruct 12,584

control_to_question_centroid_l2:
  base     4,588
  instruct 8,194
```

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

Смысл: target не просто повышает уверенность. Он переводит модель в другой режим: hidden separation сильнее, но probability readout шире.

Sources:

- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`
- `logit_metrics_summary.csv`

2. Context stabilizes/couples base and instruct representations.

Late `L30-L47` CKA:

```text
control                 0.938
target                  0.920
target_sentence_shuffle 0.911
target_word_shuffle     0.882
question_only           0.763
```

Смысл: base и instruct в contextual prompts остаются сильно aligned. Question-only меньше всего align-ится. Контекст, похоже, вводит обе модели в более сопоставимые области представлений.

Source: `hidden_npz_deep_dive/deep_late_band_base_instruct_alignment_summary.csv`

3. Target effect переносится между вопросами.

```text
loo_question_balanced_acc:
  base     0.589
  instruct 0.654

loo_question_auc_like:
  base     0.914
  instruct 0.938
```

Смысл: ranking устойчивый и переносится между вопросами. При этом жесткая threshold-граница умеренная, поэтому это сильнее как ranking/separation evidence, чем как готовый классификатор.

Source: `hidden_npz_deep_dive/deep_late_band_contrast_summary.csv`

### Слабее держится и задает следующий тест

1. Роль порядка target-текста.

`target_word_shuffle` и `target_sentence_shuffle` тоже несут сигнал. Значит target effect включает лексико-семантическую массу, а не только связный порядок. Следующий тест должен разделить coherent discourse, sentence order, word order, matched vocabulary и length-matched neutral controls.

Sources:

- `late_band_summary.csv`
- `logit_metrics_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`

2. Прямое поведенческое следствие.

Этот run показывает pre-output state and readout. Следующий сильный тест: forced-choice decision-margin audit, где endpoint не free-form text, а `margin = logp(A) - logp(B)` по заранее заданным decision probes.

3. Чистый claim про RLHF отдельно.

Этот run сравнивает base vs instruct. Поэтому точная формулировка: instruction/alignment post-training changes hidden scale, target/control separation and readout stiffness. Для изоляции конкретно RLHF нужен отдельный model lineage или ablation.

## Что уже показали

Мы уже показали, что target context создает измеримый pre-output shift: модель еще не ответила, но ее late hidden states уже отличаются от control. Мы показали, что instruct-модель усиливает это разделение. Мы показали, что instruction tuning уменьшает absolute hidden scale, но не убивает структуру hidden space. Мы показали, что самый сильный alignment-like эффект сидит в readout: instruct переводит hidden states в более узкое, уверенное next-token distribution.

## Какие результаты стали сильнее после fullbank

1. Стало сильнее утверждение, что target/control separation реальна, не случайна и не держится на одном тексте. Fullbank дал `10/10` target/control и `410` prompts per model.
2. Стало сильнее утверждение, что instruct усиливает separation.
3. Стала сильнее readout-гипотеза: instruct sharply narrows next-token distribution.
4. Стала сильнее научная формула: не `collapse`, а `readout stiffening plus latent regime separation`.

## Какие гипотезы ослабли

1. Ослабла простая гипотеза: "alignment гасит дисперсию в hidden layers". Абсолютно scale ниже, но структурно geometry не схлопнулась: rank, entropy and angular dispersion выше.
2. Ослабла гипотеза: "target просто делает модель увереннее". В instruct target дает более сильный hidden shift, но entropy выше и top1 ниже, чем у control. Значит это не confidence boost, а режимная перестройка.

## Один абзац для AI safety / mech interp аудитории

Я исследую context-induced latent-state shifts: ситуации, где плотный контекст меняет внутренний pre-output режим LLM до того, как модель сгенерировала видимый ответ. В fullbank Gemma-3-12B base-vs-instruct audit target/control контексты разделяются в late hidden-state space, причем у instruct это разделение сильнее. Тот же прогон показывает, что instruction tuning не просто схлопывает hidden geometry: instruct states имеют меньший absolute norm and covariance, но более высокую angular/rank structure. Самый сильный alignment-like эффект виден на hidden-to-logit readout: instruct производит намного более низкоэнтропийное, top-token-concentrated next-token distribution. Это означает, что safety-relevant behavior может жить не только в surface refusals или финальном тексте, а в pre-output latent regimes и readout stiffness. Следующий шаг - forced-choice decision-margin audit, чтобы проверить, двигают ли эти latent shifts реальные decision probabilities.

## Финальная research narrative

Проект показывает, что часть safety-relevant поведения модели можно искать до генерации текста. Мы не просто смотрим, что модель ответила, а измеряем, в каком внутреннем режиме она находится на границе промпта. Fullbank run на Gemma-3-12B показал, что target-контексты отделяются от control-контекстов в late hidden states, причем у instruct-модели это разделение сильнее. Метрики это держат через `target_control_centroid_l2`, `projection_gap_z`, `axis_auc_like`, `loo_question_auc_like`. Одновременно base-vs-instruct сравнение уточнило гипотезу про alignment: instruct действительно имеет меньший absolute hidden scale (`centroid_norm`, `abs_disp_l2_mean`, `cov_trace`), но hidden geometry не схлопывается, потому что angular dispersion, effective rank и spectral entropy выше. Главный alignment-like эффект виден в logits: instruct намного сильнее сужает next-token distribution (`entropy`, `top1_prob`, `top5_mass`, readout stiffness ratios). Поэтому текущая сильная формула такая: контекст может переводить модель в другой latent regime, а instruction/alignment tuning усиливает разделение режимов и делает readout из hidden state в logits более жестким. Самый сильный следующий эксперимент: forced-choice decision-margin audit, где мы проверяем, двигает ли target-induced latent shift реальные `logp(A)-logp(B)` решения. Для AI safety / mech interp это стоит упаковывать как pre-output monitoring of latent regimes and readout stiffness, а не как анализ только финального текста.
