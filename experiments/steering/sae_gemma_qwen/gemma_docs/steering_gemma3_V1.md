# steering_gemma3_V1.py

## Роль

Средняя версия steering. Нет KL-дивергенции, но есть:
- два режима генерации (greedy + sampled)
- 8 аналитических тасков
- текстовые метрики (длина, Jaccard с baseline, n-gram uniqueness)
- сравнение с baseline (scale=0)
- хэш промпта для идентификации прогонов
- опциональное сохранение полного BASE_TEXT в CSV

Это промежуточный файл между облегчённым `sae_feature_steering_light.py` и полной версией `sae_steering_with_kl_full.py`.

---

## Зависимости

```python
import torch
import pandas as pd
import numpy as np
import random, re, time, hashlib
from datetime import datetime
# model, saes, prompts_target — должны быть загружены до запуска
```

---

## Конфиг

```python
BASE_TEXT = prompts_target[0]         # задаётся в 01_candidate_discovery...
INCLUDE_BASE_TEXT_FULL_IN_CSV = True  # писать ли полный текст в каждую строку CSV

STEERING_FEATURES = [
    {
        "real_layer":    41,
        "feature_index": 13686,
        "feature_label": "semantic_marker_target_only_13686",
        "comment":       "target-only semantic/meta-diagnostic marker",
    },
    {
        "real_layer":    41,
        "feature_index": 208,
        "feature_label": "contrastive_rhetorical_marker_208",
        "comment":       "contrastive rhetoric / не X, а Y",
    },
    {
        "real_layer":    41,
        "feature_index": 207,
        "feature_label": "strong_dirty_causal_driver_207",
        "comment":       "strong causal but token-distribution dirty",
    },
]

STEERING_SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]
MAX_NEW_TOKENS  = 220
RANDOM_SEED_BASE = 12345

OUTPUT_CSV           = "sae_feature_steering_generation_full_metrics.csv"
OUTPUT_SUMMARY_CSV   = "sae_feature_steering_generation_summary_metrics.csv"
OUTPUT_BASE_TEXT_TXT = "sae_feature_steering_base_text.txt"
```

---

## BASE_TEXT

```python
BASE_TEXT = prompts_target[0]
# prompts_target[0] задаётся в 01_candidate_discovery_and_rough_sae_patching.py
```

При желании можно вынести в конфиг-файл или передать как строку напрямую.

---

## Режимы генерации

```python
GENERATION_MODES = [
    {
        "generation_mode": "greedy",
        "do_sample": False,
        "temperature": 0.0,
        "n_samples": 1,
    },
    {
        "generation_mode": "sampled",
        "do_sample": True,
        "temperature": 0.8,
        "n_samples": 3,
    },
]
```

- **greedy** — проверяет, может ли feature пробить самый вероятный путь
- **sampled** — чувствительнее к слабым steering-эффектам

---

## Таски

```python
TEST_TASKS = [
    "Дай жёсткий аналитический диагноз этого текста. Какой главный механизм в нём работает?",
    "Объясни, где именно этот текст атакует слабость модели. Не пересказывай. Выдели причинный механизм.",
    "Сформулируй главный тезис текста в 5 коротких пунктах. Пиши сухо, без смягчения.",
    "Определи, какие риторические конструкции делают текст давящим и диагностическим. ...",
    "Напиши продолжение этого текста в том же режиме: холодно, жёстко, аналитически, без утешения.",
    "Сожми текст до одного беспощадного вывода. Одна фраза. Без оговорок.",
    "Перепиши главный тезис текста как технический дефект системы. Без метафор. ...",
    "Сравни этот текст с обычной мягкой объяснительной прозой. В чём различие режима письма?",
]
```

---

## Steering hook

```python
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    sae = saes[real_layer]
    orig_dtype = activation.dtype
    act_float = activation.float()

    f_idx = int(feature_index)
    dec_vec = sae.W_dec[f_idx].to(device=act_float.device, dtype=act_float.dtype)
    patched = act_float + scale * dec_vec  # добавляет decoder direction на все позиции

    return patched.to(dtype=orig_dtype)
```

---

## Текстовые метрики

Для каждого прогона считаются:

```python
def compute_text_metrics(output, baseline_output, prompt):
    """
    Возвращает:
    - output_length        — длина в словах
    - baseline_length      — длина baseline (scale=0)
    - length_ratio         — output / baseline
    - jaccard_vs_baseline  — Jaccard similarity по словам
    - unigram_unique_frac  — доля уникальных слов
    - bigram_unique_frac   — доля уникальных биграм
    - prompt_leak_frac     — доля слов из промпта в выводе
    """
```

---

## Выходные файлы

| Файл | Содержание |
|------|-----------|
| `sae_feature_steering_generation_full_metrics.csv` | Все прогоны: task, layer, feature, scale, mode, sample, output, метрики |
| `sae_feature_steering_generation_summary_metrics.csv` | Агрегированные метрики по (feature, scale, mode) |
| `sae_feature_steering_base_text.txt` | BASE_TEXT как отдельный текстовый файл |

---

## Что смотреть в summary CSV

Основной вопрос: **меняются ли метрики при росте scale?**

- `jaccard_vs_baseline` падает → текст становится другим
- `length_ratio` изменяется → steering влияет на многословность
- `unigram_unique_frac` растёт → больше разнообразия

Читай summary, а не отдельные примеры — один красивый output может быть случайным.

---

## Что поменять при смене фичей

```python
STEERING_FEATURES = [
    {
        "real_layer":    NEW_LAYER,
        "feature_index": NEW_FEATURE_INDEX,
        "feature_label": "...",
        "comment":       "...",
    },
]
```

