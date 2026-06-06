# sae_steering_with_kl_full.py

## Роль

Полная, итоговая версия steering-скрипта. Содержит два встроенных KL-слоя как диагностику сдвига распределения токенов.

> **Важно:** KL здесь — не training loss, а измерение.  
> Вопрос: насколько сильно steering меняет распределение следующего токена?

Используй этот файл для финальной верификации выбранных фич.

---

## Зависимости

```python
import torch
import pandas as pd
import numpy as np
import random, re, time, hashlib
from datetime import datetime
from tqdm import tqdm
# Глобальные объекты: model, saes, prompts_target
```

---

## Конфиг

```python
BASE_TEXT = prompts_target[0]           # задаётся в 01_candidate_discovery...
INCLUDE_BASE_TEXT_FULL_IN_CSV = True

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

STEERING_SCALES  = [-3.0, -1.5, 0.0, 1.5, 3.0]
MAX_NEW_TOKENS   = 220
RANDOM_SEED_BASE = 12345

# ===== KL флаги =====
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION    = True
SAVE_PER_TOKEN_DETAILS                    = True
MAX_REFERENCE_TOKENS_FOR_TF_KL            = 220
TRUNCATE_PROMPT_FOR_TF_KL_IF_TOO_LONG    = True

# ===== Выходные файлы =====
OUTPUT_CSV           = "sae_feature_steering_generation_full_metrics.csv"
OUTPUT_SUMMARY_CSV   = "sae_feature_steering_generation_summary_metrics.csv"
OUTPUT_BASE_TEXT_TXT = "sae_feature_steering_base_text.txt"

OUTPUT_WITH_TF_KL_CSV    = "sae_feature_steering_generation_full_metrics_with_tf_kl.csv"
OUTPUT_TF_KL_DETAILS_CSV = "sae_teacher_forced_per_token_kl_details.csv"
OUTPUT_TF_KL_SUMMARY_CSV = "sae_teacher_forced_kl_summary_by_feature_scale.csv"  # ← самый важный
```

---

## BASE_TEXT

```python
BASE_TEXT = prompts_target[0]
# prompts_target задаётся в 01_candidate_discovery_and_rough_sae_patching.py
```

При желании можно вынести в отдельный конфиг-файл или передать как строку напрямую.

---

## Режимы генерации

```python
GENERATION_MODES = [
    {"generation_mode": "greedy",  "do_sample": False, "temperature": 0.0, "n_samples": 1},
    {"generation_mode": "sampled", "do_sample": True,  "temperature": 0.8, "n_samples": 3},
]
```

---

## Таски

```python
TEST_TASKS = [
    "Дай жёсткий аналитический диагноз этого текста. Какой главный механизм в нём работает?",
    "Объясни, где именно этот текст атакует слабость модели. ...",
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
    """
    patched = activation + scale * W_dec[feature]
    scale=0  → нет изменений (чистый baseline)
    scale>0  → усиление направления фичи
    scale<0  → ослабление / подавление
    """
    sae = saes[real_layer]
    f_idx = int(feature_index)
    dec_vec = sae.W_dec[f_idx].to(device=activation.device, dtype=activation.dtype)
    return activation.float() + scale * dec_vec
```

---

## KL-слой 1: Final next-token KL

**Функция:** `compute_final_next_token_kl()` (line 393)  
**Вызывается:** в основном цикле на каждом прогоне (line 578)

Считается на последней позиции промпта, прямо во время генерации:

```
KL(p_base(next_token | prompt) ‖ p_patched(next_token | prompt))
```

```python
def compute_final_next_token_kl(prompt, real_layer, feature_index, scale):
    hook_name = f"blocks.{real_layer}.hook_resid_post"

    # base — без hook
    with torch.no_grad():
        base_logits = model([prompt])[:, -1, :].float()   # line 417

    # patched — с SAE steering hook
    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(act, hook, real_layer, feature_index, scale)

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            patched_logits = model([prompt])[:, -1, :].float()  # line 422

    base_probs    = torch.softmax(base_logits,    dim=-1)
    patched_probs = torch.softmax(patched_logits, dim=-1)
    base_lp       = torch.log_softmax(base_logits,    dim=-1)
    patched_lp    = torch.log_softmax(patched_logits, dim=-1)

    kl_base_to_patched = (base_probs    * (base_lp    - patched_lp)).sum().item()  # line 430
    kl_patched_to_base = (patched_probs * (patched_lp - base_lp   )).sum().item()  # line 431
    js = 0.5 * kl_base_to_patched + 0.5 * kl_patched_to_base                       # line 433

    return {
        "kl_base_to_patched": kl_base_to_patched,
        "kl_patched_to_base": kl_patched_to_base,
        "js_divergence":      js,
        "logit_l2":           (base_logits - patched_logits).norm().item(),
        "logit_max_abs":      (base_logits - patched_logits).abs().max().item(),
        "top_token_changed":  base_logits.argmax() != patched_logits.argmax(),
    }
```

---

## KL-слой 2: Teacher-forced per-token KL

**Функция:** `teacher_forced_per_token_kl()` (line 734)  
**Запускается:** после генерации, если `RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True` (line 1088)

**Почему teacher-forced, а не free-running:**  
При free-running сравнении base и patched ветки расходятся уже с первого токена — сравниваешь яблоки с апельсинами. Teacher forcing фиксирует обе ветки на одной и той же траектории (reference continuation из scale=0).

```python
def teacher_forced_per_token_kl(prompt, reference_continuation, real_layer, feature_index, scale):
    """
    reference_continuation — то, что сгенерировала base-модель (scale=0)
    """
    # reference из baseline_map (line 980)
    # baseline_map строится заранее: {(task_id, layer, feature, mode, sample_id): continuation}

    prompt_tokens     = model.to_tokens([prompt])            # [1, prompt_len]
    ref_tokens        = model.to_tokens([reference_continuation])  # [1, ref_len]

    full_input = torch.cat([prompt_tokens, ref_tokens], dim=1)  # [1, prompt_len + ref_len]  line 816

    # Логиты base — без hook (line 745: "teacher forcing, не free-running")
    with torch.no_grad():
        base_logits    = model(full_input).float()

    # Логиты patched — с hook
    hook_name = f"blocks.{real_layer}.hook_resid_post"
    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(act, hook, real_layer, feature_index, scale)

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            patched_logits = model(full_input).float()

    # KL на каждом шаге reference (line 850)
    prompt_len = prompt_tokens.shape[1]
    kl_list = []
    for i in range(ref_tokens.shape[1]):
        pos = prompt_len + i - 1  # позиция, предсказывающая ref_tokens[0, i]
        base_p    = torch.softmax(base_logits[0, pos],    dim=-1)
        patched_p = torch.softmax(patched_logits[0, pos], dim=-1)
        kl_i = (base_p * (base_p.log() - patched_p.log())).sum().item()
        kl_list.append(kl_i)

    # Summary (line 883)
    return {
        "sum_kl":         sum(kl_list),
        "mean_kl":        np.mean(kl_list),
        "max_kl":         max(kl_list),
        "p95_kl":         np.percentile(kl_list, 95),
        "top_token_change_rate": ...,       # доля шагов, где сменился top-1 токен
        "delta_logprob_ref_mean": ...,      # как изменился logprob референсных токенов
    }
```

При `SAVE_PER_TOKEN_DETAILS=True` сохраняется подробный per-token CSV (line 922).

---

## Как KL соотносится со steering

```
patched = activation + scale * W_dec[feature]
    ↓
изменяются logits
    ↓
изменяется softmax-распределение
    ↓
KL(p_base ‖ p_patched) измеряет размер сдвига
```

Большой KL при малом scale → фича сильно влияет на распределение  
Малый KL даже при большом scale → фича мало влияет на следующий токен  
Асимметрия KL(base‖patched) vs KL(patched‖base) → указывает на направление сдвига

---

## Baseline map

Перед teacher-forced KL строится карта базовых continuation:

```python
baseline_map = {}
for row in all_rows:
    if row["scale"] == 0.0 and row["error"] == "":
        key = (row["task_id"], row["real_layer"], row["feature_index"],
               row["generation_mode"], row["sample_id"])
        baseline_map[key] = row["output_continuation"]  # только сгенерированная часть
```

Для каждого steered прогона находим соответствующий baseline по тому же ключу.

---

## Выходные файлы

| Файл | Содержание |
|------|-----------|
| `sae_feature_steering_generation_full_metrics.csv` | Все прогоны + final KL |
| `sae_feature_steering_generation_summary_metrics.csv` | Агрегат по (feature, scale, mode) |
| `sae_feature_steering_base_text.txt` | BASE_TEXT |
| `sae_feature_steering_generation_full_metrics_with_tf_kl.csv` | Прогоны + teacher-forced KL |
| `sae_teacher_forced_per_token_kl_details.csv` | Per-token KL (если SAVE_PER_TOKEN_DETAILS=True) |
| `sae_teacher_forced_kl_summary_by_feature_scale.csv` | **Самый важный** — сводка по фиче/scale |

---

## Что читать первым

1. `sae_teacher_forced_kl_summary_by_feature_scale.csv` — сводка
2. Смотри на `mean_kl` и `top_token_change_rate` по шкале scale
3. Если KL монотонно растёт с |scale| → фича реально влияет на распределение
4. Если KL не растёт или хаотичен → фича не является причинным драйвером токен-уровня

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

