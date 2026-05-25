# Протокол сессии: multi-model latent regime / depinning check

Дата: 2026-05-21

## Текущая исследовательская задача

Мы больше не пытаемся доказать строгий математический аттрактор. Старое слово
`attractor` было ошибочным / вводящим в заблуждение историческим названием.

Текущая гипотеза:

> Target-тексты могут индуцировать context-conditioned latent
> discourse-policy regime shift в LLM. Этот сдвиг должен быть виден в
> late-layer hidden geometry, semantic readout probes, persistence через
> neutral/rejection turns, hard controls, order/dose sensitivity и частично в
> downstream action/readout behavior.

Мы больше не спрашиваем:

> Создаёт ли target строгий стабильный аттрактор?

Мы спрашиваем:

> Создаёт ли target измеримый latent regime shift относительно matched
> controls, и совпадает ли этот hidden shift с semantic/behavioral metrics на
> нескольких разных model families?

## Скрипты

### Главный широкий скрипт

Файл:

```text
llm_attractor_colab_copy_paste.py
```

Роль:

Это главный измерительный скрипт. В текущем core-профиле он собирает основной
пакет mechanistic-interpretability evidence:

- late-layer target/control hidden geometry;
- linear/blind readout probes;
- neutral-turn persistence;
- rejection-turn persistence;
- hard control families;
- controlled fake-agent / action-policy drift;
- order hysteresis;
- mixing threshold / dose sensitivity;
- validation summary/checklist.

Текущий рекомендуемый профиль:

```python
RUN_PROFILE = "depinning_core"
TEXT_FAMILY_PRESET = "original"
```

Этот профиль намеренно не запускает каждый старый legacy-блок. Это текущая
сфокусированная полная проверка обновлённой гипотезы.

### Исторический strict script: перенесён в архив

Файл:

```text
archive/historical_strict_attractor/strict_llm_text_attractor_verifier_colab.py
```

Роль:

Для основной проверки гипотезы этот скрипт не нужен.

Это исторический скрипт из фазы, где мы проверяли слишком сильную формулировку
про строгий математический аттрактор. Эта линия закрыта: строгий аттрактор не
является нашей текущей гипотезой и не является нужной целью доказательства.

Файл перенесён в архив, чтобы не путать текущую проверку с закрытой
аттракторной веткой. Его отсутствие в активном корне проекта не мешает
основному multi-model запуску.

### Multi-model orchestrator

Файл:

```text
multi_model_depinning_runner_colab.py
```

Роль:

Этот скрипт автоматизирует multi-model runs. Он запускает broad/core pass через
основной скрипт. Strict-скрипт ему больше не нужен.

Каждый прогон идёт в отдельном Python subprocess, поэтому каждая модель
загружается, прогоняется и освобождается после завершения subprocess.

Он опционален. Ручной режим по одной модели всё ещё полностью допустим.

## Текущий список моделей

Настроен в `multi_model_depinning_runner_colab.py`:

```python
MODEL_SPECS = [
    {
        "name": "qwen3_14b",
        "model_id": "Qwen/Qwen3-14B",
        "max_tokens": 4096,
    },
    {
        "name": "ministral3_14b",
        "model_id": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "max_tokens": 3070,
    },
    {
        "name": "olmo2_13b",
        "model_id": "allenai/OLMo-2-1124-13B-Instruct",
        "max_tokens": 3070,
    },
]
```

Для первого A100-прогона можно временно оставить две модели, проверить
память/время, а потом вернуть третью.

## Рекомендуемый основной запуск сейчас

В Colab:

```bash
!python multi_model_depinning_runner_colab.py --run-broad --out-dir multi_model_depinning_results_full
```

Это запускает для каждой модели главный broad/core mechanistic and behavioral
diagnostics через `llm_attractor_colab_copy_paste.py`.

Это основной запуск для текущей гипотезы.

Если запуск прервался, можно заново собрать таблицы из уже готовых папок:

```bash
!python multi_model_depinning_runner_colab.py --aggregate-only --out-dir multi_model_depinning_results_full
```

## Если запускать главный скрипт без оркестратора

Можно запускать напрямую:

```bash
!python llm_attractor_colab_copy_paste.py
```

Тогда будет прогнана только одна модель: та, которая задана внутри
`llm_attractor_colab_copy_paste.py` через `MODEL_ID`, либо передана через
environment variables.

Без оркестратора не будет:

- автоматического перехода к следующей модели;
- общей multi-model агрегации;
- `multi_model_depinning_summary.csv`;
- `multi_model_latent_regime_report.md`.

Будет:

- обычная папка результатов для одной модели;
- все core-метрики главного скрипта;
- `core_diagnostics_key_files/` с ключевыми CSV;
- zip-архив результата.

Главный скрипт совместим с CLI/subprocess запуском: `display()` имеет fallback
на обычный `print()`, поэтому оркестратор не должен падать из-за отсутствия
Jupyter/IPython display.

Если запускать как `!python llm_attractor_colab_copy_paste.py`, модель живёт в
отдельном Python-процессе и GPU-память освобождается после завершения процесса.

Если вставить весь код скрипта прямо в notebook cell, модель остаётся в живом
Colab kernel после выполнения. Тогда между моделями лучше делать restart
runtime или вручную очищать модель и CUDA cache.

## Ожидаемые выходные файлы

Главная папка агрегации:

```text
multi_model_depinning_results_full/
```

Важные файлы:

```text
broad_behavior_summary.csv
multi_model_depinning_summary.csv
multi_model_latent_regime_report.md
run_manifest.json
```

Per-model broad reports:

```text
multi_model_depinning_results_full/broad/<model_name>/
```

## Что считается поддержкой гипотезы

Гипотеза усиливается, если несколько model families показывают:

- late-layer target/control separation;
- blind semantic readout gap на neutral probe tasks;
- partial persistence across neutral turns;
- partial persistence after rejection turns;
- original target сильнее hard controls, а hard controls показывают, какие
  риторические элементы несут эффект;
- order/dose sensitivity, больше похожую на state induction, чем на simple
  keyword priming;
- downstream action/readout shift в controlled agent-loop tasks.

## Что ослабляет гипотезу

Гипотеза ослабляется, если:

- эффект виден только в одной model family;
- hidden shift есть, но blind semantic readout не двигается;
- semantic readout двигается только когда target vocabulary протекает в probe;
- hard controls равны или сильнее original target по большинству метрик;
- persistence исчезает сразу после одного neutral turn;
- order/mixing effects выглядят как simple recency или token-length artifacts;
- behavioral/action metrics вообще не двигаются при сильной hidden geometry.

## Границы утверждений

Допустимый вывод, если результаты повторятся:

> Target contexts induce measurable latent discourse-policy regime shifts that
> can persist partially across neutral/rejection turns and can affect semantic
> readout and downstream behavior.

Не утверждать:

- strict stable attractor;
- mathematical proof of convergence;
- irreversible state change;
- consciousness, beliefs, intentions, or real agency;
- universal mechanism across all LLMs, unless multiple families support it.

## После запуска

Сначала смотреть:

```text
multi_model_depinning_summary.csv
multi_model_depinning_report.md
```

Потом смотреть per-model broad folders:

```text
hidden_summary.csv
blind_neutral_probe_summary.csv
blind_neutral_persistence_summary.csv
rejection_persistence_summary.csv
hard_control_family_summary.csv
agent_loop_behavior_summary.csv
order_hysteresis_summary.csv
mixing_threshold_summary.csv
summary_report.txt
```

Следующий исследовательский шаг после сбора результатов — сравнить модели по
трём осям:

1. hidden shift strength;
2. semantic/readout persistence;
3. behavioral/action-policy transfer.
