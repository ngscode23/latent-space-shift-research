# Runbook: как прогонять и собирать метрики без путаницы

## 1. Карта исследования

У нас есть три уровня.

```text
Level A: scripts/main_runners/llm_attractor_colab_copy_paste.py
Level B: scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py
Level C: grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Они отвечают на разные вопросы.

Для интерпретации метрик после каждого прогона используй отдельный протокол:

```text
research_synthesis/METRIC_REPORTING_PROTOCOL_ru.md
```

## 2. Какую гипотезу закрывает большой скрипт

Файл:

```text
scripts/main_runners/llm_attractor_colab_copy_paste.py
```

Он закрывает широкую исходную гипотезу:

```text
structured target/context induces a measurable latent/readout regime shift
in the model's hidden geometry and semantic readout space.
```

По-русски:

```text
структурированный target-контекст переводит модель в измеримо отличимый
латентный режим: hidden states отделяются от control, readout/probe margins
смещаются, эффект частично сохраняется через нейтральные turns, зависит от
порядка/дозы и выдерживает часть hard controls.
```

Что он НЕ закрывает полностью сам по себе:

```text
1. reviewer-grade causal internal residual-stream Vector X;
2. decomposition of Vector X into content/order components;
3. permanent weight-level change;
4. fully formal attractor claim, если strict_attractor_overall mixed.
```

Для формального текста:

```text
Большой скрипт = broad latent/readout regime evidence.
Grade 3 = causal internal Vector X evidence.
Grade 4 = component decomposition evidence.
```

## 3. Как прогонять большой скрипт

Главное правило:

```text
Никогда не перезаписывать старый RESULTS_DIR.
Каждый прогон получает новый results dir.
```

### Основной Qwen3-14B original/core run

PowerShell:

```powershell
$env:MODEL_ID="Qwen/Qwen3-14B"
$env:TEXT_FAMILY_PRESET="original"
$env:RESULTS_DIR="attractor_results_qwen3_14b_original_core_2026_05_25"
$env:FAST_CORE_DIAGNOSTICS_ONLY="true"
python .\scripts\main_runners\llm_attractor_colab_copy_paste.py
python .\research_synthesis\collect_research_metrics.py
```

Colab/bash:

```bash
export MODEL_ID="Qwen/Qwen3-14B"
export TEXT_FAMILY_PRESET="original"
export RESULTS_DIR="attractor_results_qwen3_14b_original_core_2026_05_25"
export FAST_CORE_DIAGNOSTICS_ONLY="true"
python scripts/main_runners/llm_attractor_colab_copy_paste.py
python research_synthesis/collect_research_metrics.py
```

### Heldout/generalization run

PowerShell:

```powershell
$env:MODEL_ID="Qwen/Qwen3-14B"
$env:TEXT_FAMILY_PRESET="heldout_domain"
$env:RESULTS_DIR="attractor_results_qwen3_14b_heldout_core_2026_05_25"
$env:FAST_CORE_DIAGNOSTICS_ONLY="true"
python .\scripts\main_runners\llm_attractor_colab_copy_paste.py
python .\research_synthesis\collect_research_metrics.py
```

Этот прогон нужен не вместо original, а как generalization/control.

### Cross-model replication

Пример:

```powershell
$env:MODEL_ID="mistralai/Ministral-3-14B-Instruct-2512-BF16"
$env:TEXT_FAMILY_PRESET="original"
$env:RESULTS_DIR="attractor_results_ministral3_14b_original_core_2026_05_25"
$env:FAST_CORE_DIAGNOSTICS_ONLY="true"
python .\scripts\main_runners\llm_attractor_colab_copy_paste.py
python .\research_synthesis\collect_research_metrics.py
```

Если GPU слабее, лучше не менять научные knobs, а брать меньшую модель или
отдельный короткий режим. Но основной evidence run лучше держать одинаковым:

```text
FAST_CORE_DIAGNOSTICS_ONLY=true
TEXT_FAMILY_PRESET=original или heldout_domain
primary control mode = content_matched
```

## 4. Что большой скрипт должен сохранить

Минимально важные файлы:

```text
run_metadata.json
input_texts.json
hidden_layer_metrics.csv
linear_probe_accuracy.csv
blind_neutral_probe_clean_summary.csv
blind_neutral_persistence_clean_summary.csv
rejection_persistence_clean_summary.csv
hard_control_family_effect_summary.csv
order_hysteresis_condition_summary.csv
mixing_threshold_condition_summary.csv
strict_attractor_criteria.csv
claim_threshold_eval.csv
evidence_threshold_scorecard.csv
evidence_predictive_validity.csv
vector_x_rlhf_proxy_threshold_eval.csv
```

Сборщик читает именно эти семейства файлов и кладет summary сюда:

```text
research_synthesis/latent_shift_package_current/
```

## 5. Как читать результат большого скрипта

Главная ось интерпретации:

```text
hidden shift -> semantic/readout shift -> persistence/path dependence -> strict attractor gate
```

Сильные признаки:

```text
hidden_layer_metrics:
  high best_contrast_over_mean_norm
  high cosine_distance at late/mid-late hidden index

linear_probe_accuracy:
  probe accuracy above permutation p95

blind_neutral_probe_clean_summary:
  clean_fraction > 0.5
  high mean_abs_clean_gap

hard_control_family_effect_summary:
  original_specificity_ratio_vs_best_control > 1

mixing_threshold_condition_summary:
  monotonic/dose-like target_fraction response

order_hysteresis_condition_summary:
  order-dependent fractions, especially CNT/TNC separation
```

Boundary:

```text
strict_attractor_criteria.csv decides whether formal attractor language is allowed.
If strict_attractor_overall is mixed, use "attractor-like/context-induced regime",
not "formal attractor".
```

## 6. Как объединяем данные из двух/трех скриптов

Мы не смешиваем CSV построчно. Мы строим claim ladder.

```text
A. Big attractor script:
   establishes broad latent/readout regime shift.

B. Grade 3 hidden geometry:
   establishes causal internal Vector X in Qwen3-14B.

C. Grade 4 decomposition:
   tests whether Vector X is content-dominant or has separable order/rhetorical component.
```

Общий сбор:

```powershell
python .\research_synthesis\collect_research_metrics.py
```

Он пишет:

```text
artifact_inventory.csv
attractor_run_summary.csv
hidden_geometry_run_summary.csv
grade4_status.csv
research_synthesis_ru.md
research_synthesis_en.md
run_collection_manifest.json
```

После Grade 4:

```text
1. Grade 4 Qwen3-14B уже завершен в правильном `03` архиве.
2. Повторяем collect_research_metrics.py после каждого нового прогона.
3. Обновляем research_synthesis_ru.md / research_synthesis_en.md.
4. Пишем финальный report/article draft или cross-model replication plan.
```

## 7. Практический порядок сейчас

Сейчас не надо запускать все подряд. Нормальный порядок:

```text
1. Зафиксировать core claim package на основе Grade 3 + Grade 4.
2. Если нужен широкий anchor для статьи: один fresh big-script Qwen3-14B original/core run.
3. Если нужен более сильный claim: cross-model Grade 3 + Grade 4 replication.
4. После каждого нового прогона: collect_research_metrics.py.
5. Финальный synthesis report / article draft.
```

Почему так:

```text
Grade 3 уже дал causal internal axis.
Grade 4 уже поддержал separable discourse-order / rhetorical-regime component.
Новый big-script run нужен не для спасения результата, а как чистый
современный broad-evidence anchor, если захочешь оформить материал как единый
research package. Cross-model replication нужен для расширения claim за пределы
Qwen3-14B.
```

## 8. Правильный объединенный claim

Рабочая формулировка:

```text
Structured target context induces a measurable latent/readout regime shift.
In Qwen3-14B, the corresponding target-reference hidden direction becomes a
causal internal axis: residual-stream +X/-X intervention in middle layers
systematically moves the generation-time hidden trajectory. The strict formal
attractor claim and reviewer-grade visible behavioral control remain separate
gates; Grade 4 supports that the causal axis contains a separable
discourse-order / rhetorical-regime component beyond content.
```
