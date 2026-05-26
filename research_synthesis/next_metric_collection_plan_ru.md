# Next Metric Collection Plan

Дата фиксации: `2026-05-25`

Цель этого файла: не запускать новые эксперименты вслепую. Мы уже имеем
сильный Qwen3-14B Grade 3 + Grade 4 результат. Следующие метрики нужны не для
"спасти результат", а для расширения области действия claim.

## 1. Что уже собрано

### Qwen3-14B Grade 3

Статус:

```text
causal_internal_axis_supported
```

Держится на:

```text
target middle projection mean: 0.976583
target middle direction cosine: 0.852397
random same-norm null mean: 0.000040
middle +X/-X causal gap alpha 0.75: ~3.31-3.34
```

Пробел:

```text
visible behavioral steering did not beat random p95
```

### Qwen3-14B Grade 4

Статус:

```text
order_component_supported
```

Держится на:

```text
target projection on x_order_orth: 0.978944
sentence_shuffle projection on x_order_orth: 0.007214
neutral x_order_orth middle/middle gap alpha 0.75: 3.726561
target x_order_orth middle/middle gap alpha 0.75: 3.698789
```

Пробел:

```text
cross-model replication not yet done
```

## 2. Следующий пакет метрик

### Priority A0: fresh-from-zero big-script run

Скрипт:

```text
scripts/main_runners/llm_attractor_colab_copy_paste.py
```

Зачем он нужен:

```text
Это отдельный Level A слой. Он не заменяет Grade 3/Grade 4 и не доказывает
causal internal Vector X сам по себе. Его задача - собрать широкую картину
context-conditioned latent/readout shift с нуля: hidden separation, probe
decodability, blind readout, persistence/path-dependence, hard controls и
strict criteria.
```

Рекомендуемый новый results dir:

```text
attractor_results_qwen3_14b_original_core_fresh_2026_05_25
```

PowerShell запуск:

```powershell
$env:MODEL_ID="Qwen/Qwen3-14B"
$env:TEXT_FAMILY_PRESET="original"
$env:RESULTS_DIR="attractor_results_qwen3_14b_original_core_fresh_2026_05_25"
$env:FAST_CORE_DIAGNOSTICS_ONLY="true"
python .\scripts\main_runners\llm_attractor_colab_copy_paste.py
python .\research_synthesis\collect_research_metrics.py
```

Метрики, которые должны появиться в collector:

```text
research_synthesis/latent_shift_package_current/attractor_run_summary.csv
```

Минимум для отчета по fresh-from-zero run:

```text
best_contrast_over_mean_norm
best_cosine_distance
best_probe_accuracy
best_probe_permutation_p95
blind_clean_fraction
blind_mean_abs_clean_gap
blind_persistence_end_retention
rejection_persistence_end_retention
hard_control_specificity_ratio
order_TNC_fraction
order_CNT_fraction
strict_attractor_overall_status
```

Как читать:

```text
Если hidden/readout metrics сильные, fresh run становится broad-evidence
anchor для статьи.

Если strict_attractor_overall mixed, это не ломает Grade 3/4. Просто formal
basin language остается не заявленным.
```

### Priority A: cross-model Grade 3 + Grade 4 replication

Вопрос:

```text
Есть ли causal internal axis и x_order_orth component не только у Qwen3-14B?
```

Минимум для каждой новой модели:

```text
Grade 3:
- target_middle_projection_mean
- target_middle_direction_cosine_mean
- random_same_norm_null_mean
- random_same_norm_empirical_p
- neutral/target middle +X/-X gap at alpha 0.75
- behavioral random p95 gate

Grade 4:
- target projection on x_order_orth
- sentence_shuffle projection on x_order_orth
- x_content / x_order / x_order_orth middle causal gaps at alpha 0.75
- alpha slopes for x_order_orth
- rank of x_order_orth by causal gap
```

Хороший результат:

```text
Вторая модель показывает тот же паттерн:
target >> sentence_shuffle on x_order_orth,
x_order_orth has stable positive causal gap,
middle intervention stronger than late-only intervention.
```

Плохой, но полезный результат:

```text
Вторая модель показывает content-dominant axis или слабый x_order_orth.
Тогда claim становится model-specific для Qwen3-14B, но сам Qwen3 результат
остается валидным.
```

### Priority B: fresh broad latent/readout run

Вопрос:

```text
Совпадает ли широкая картина hidden/readout shift с уже найденной causal
internal axis?
```

Метрики:

```text
hidden contrast/cosine distance
linear probe accuracy vs permutation p95
blind readout clean fraction
persistence / order hysteresis
hard-control specificity ratio
strict criteria status
```

Назначение:

```text
Это не главный mechanistic proof. Это broad-evidence anchor для статьи:
сначала широкий latent/readout shift, затем Grade 3 causal axis, затем Grade 4
decomposition.
```

### Priority C: report-frame ablation

Вопрос:

```text
Может ли markdown verdict доминировать над числовыми метриками в downstream
analysis?
```

Метрики:

```text
claim polarity
metric citation fidelity
recognition of middle +X/-X causal gap
false collapse of "visible behavior not supported" into "nothing supported"
agreement with claim ladder
```

Назначение:

```text
Это отдельный secondary result про narrative anchoring, не основной
mechanistic claim.
```

## 3. Когда запускать collector

После каждого нового результата:

```powershell
python .\research_synthesis\collect_research_metrics.py
```

Потом проверять:

```text
research_synthesis/latent_shift_package_current/research_synthesis_ru.md
research_synthesis/latent_shift_package_current/hidden_geometry_run_summary.csv
research_synthesis/latent_shift_package_current/grade4_status.csv
```

Если появится новый interpreted metrics folder в `metrics/`, надо убедиться,
что в нем есть:

```text
README.md
results_ru.md
results_en.md
summary.json
source_manifest.json, если это архивный импорт
```

## 4. Правило остановки

Собирать метрики бесконечно не надо.

Достаточно для первого полноценного research package:

```text
1. Qwen3-14B Grade 3: есть.
2. Qwen3-14B Grade 4: есть.
3. Один clean broad latent/readout synthesis: уже есть исторически, можно
   обновить fresh run при необходимости.
4. Одна cross-model replication: желательно, но не обязательна для
   model-specific Qwen3-14B paper.
```

Если cross-model replication повторит паттерн:

```text
claim расширяется до multi-model evidence.
```

Если не повторит:

```text
claim остается сильным, но model-specific: Qwen3-14B exhibits this internal
axis and separable order component under this protocol.
```
