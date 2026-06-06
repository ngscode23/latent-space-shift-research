# sae_feature_steering_v2_no_control.py

## Роль

Расширенная диагностическая версия steering без control-сравнения.  
Добавляет пять дополнительных диагностик поверх базового steering.

Используй этот файл когда:
- нужно понять, на какие конкретно токены влияет фича
- нужен позиционный профиль активации
- нужна верификация через KL на следующем токене
- нужны контрасты короткими промптами (для специфических фич)

---

## Зависимости

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # опционально
from tqdm import tqdm
# model, saes, prompts_target — должны быть загружены до запуска
```

---

## Конфиг

```python
STEERING_FEATURES = [
    {"real_layer": 41, "feature_index": 13686, "feature_label": "semantic_marker_target_only_13686"},
    {"real_layer": 41, "feature_index": 208,   "feature_label": "contrastive_rhetorical_marker_208"},
    {"real_layer": 41, "feature_index": 207,   "feature_label": "strong_dirty_causal_driver_207"},
]

STEERING_SCALES    = [-3.0, -1.5, 0.0, 1.5, 3.0]
N_SAMPLES          = 5
DO_SAMPLE          = True
TEMPERATURE        = 0.8
MAX_NEW_TOKENS     = 220
RANDOM_SEED_BASE   = 12345

# Выравнивание позиций при activation patching target → control
# "right" сохраняет хвостовую структуру промптов разной длины
CONTROL_PATCH_ALIGNMENT = "right"  # {"left", "right"}

MAX_REFERENCE_TOKENS_FOR_KL = 220

BASE_TEXT = prompts_target[0]  # задаётся в 01_candidate_discovery...
```

---

## Флаги включения диагностик

```python
RUN_GENERATION                            = True
RUN_NEXT_TOKEN_KL                         = True
RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL = False  # требует prompts_control
RUN_UNEMBED_PROJECTION                    = True
RUN_POSITIONAL_PROFILE                    = True
RUN_SHORT_PROMPT_ABLATION                 = True
```

---

## Диагностика 1: Next-token KL

**Функция:** `compute_next_token_kl_feature_steering()` (line 329)

Считает расхождение между распределением следующего токена у base и patched модели.

```python
def compute_next_token_kl_feature_steering(prompt, real_layer, feature_index, scale):
    hook_name = f"blocks.{real_layer}.hook_resid_post"

    with torch.no_grad():
        base_logits = model([prompt])[:, -1, :].float()

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(act, hook, real_layer, feature_index, scale)

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            patched_logits = model([prompt])[:, -1, :].float()

    base_probs    = torch.softmax(base_logits, dim=-1)
    patched_probs = torch.softmax(patched_logits, dim=-1)
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    pat_logprobs  = torch.log_softmax(patched_logits, dim=-1)

    kl_base_to_patched = (base_probs * (base_logprobs - pat_logprobs)).sum().item()
    kl_patched_to_base = (patched_probs * (pat_logprobs - base_logprobs)).sum().item()
    js = 0.5 * kl_base_to_patched + 0.5 * kl_patched_to_base

    return {
        "kl_base_to_patched": kl_base_to_patched,
        "kl_patched_to_base": kl_patched_to_base,
        "js_divergence":      js,
        "logit_l2":           (base_logits - patched_logits).norm().item(),
        "top_token_changed":  base_logits.argmax() != patched_logits.argmax(),
    }
```

---

## Диагностика 2: Activation patching target → control

**Функции:**
- `capture_activation_for_tokens()` (line 256) — захват активации control на нужных позициях
- `build_control_patch_hook()` (line 272) — hook, подставляющий контрольную активацию
- `compute_next_token_kl_target_to_control_patch()` (line 351) — метрика KL после патча

Требует `prompts_control`. Флаг: `RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL = True`.

Выравнивание позиций задаётся через `CONTROL_PATCH_ALIGNMENT = "right"` — при разной длине промптов right-выравнивание лучше сохраняет хвостовую структуру.

---

## Диагностика 3: Teacher-forced per-token KL

**Функция:** `teacher_forced_per_token_kl()` (line 400)

Сравнение не free-running vs free-running, а на одинаковой токенной траектории:

```python
# reference = то, что сгенерировал base (scale=0)
full_input = prompt_tokens + reference_tokens

# base логиты — без hook
base_logits = model(full_input)

# patched логиты — с hook
with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
    patched_logits = model(full_input)

# KL на каждом шаге reference
for i, ref_tok in enumerate(reference_tokens):
    kl_i = KL(base_probs[i] ‖ patched_probs[i])
```

Преимущество: не загрязняет сравнение расхождением траекторий — оба прогона проходят через одни и те же токены.

---

## Диагностика 4: Unembed-проекция W_dec[f] @ W_U

**Функция:** `run_unembed_projection()` (line 548)

Показывает, какие токены фича «хочет» поднять или опустить в логитах напрямую через unembedding matrix.

```python
def run_unembed_projection(real_layer, feature_index):
    sae = saes[real_layer]
    dec_vec = sae.W_dec[feature_index].float()  # [d_model]

    W_U = model.W_U.float()  # [d_model, vocab]
    logit_contributions = dec_vec @ W_U  # [vocab]

    top_positive = logit_contributions.topk(10)
    top_negative = (-logit_contributions).topk(10)

    # → CSV с top-10 positive и top-10 negative токенами
```

**Выход:** `sae_feature_unembed_top_tokens.csv`

Что смотреть:
- если top токены осмысленные → фича семантическая
- если это субтокены / HTML / мусор → фича «грязная», causal effect ненадёжен

---

## Диагностика 5: Позиционный профиль активации

**Функция:** `run_positional_profile()` (line 608)

```python
# encode_feature_activation (line 495) — безопасный encode с fallback
latent = sae.encode(act.float())
scores = latent[..., feature_index]  # [batch, seq]

# профиль: x = позиция токена, y = среднее activation по batch
profile = scores.mean(dim=0).cpu().numpy()
```

Сохраняет CSV и PNG с профилем активации по позициям.

**Выход:**
```
sae_feature_position_activation_profile.csv
sae_feature_position_activation_profile.png
```

---

## Диагностика 6: Short-prompt ablation для фичи 208

**Функция:** `run_short_prompt_ablation()` (line 660)

Минимальный контрастный эксперимент: 12 коротких промптов с «не X, а Y» vs 12 промптов без этого контраста.

```python
SHORT_PROMPTS_WITH_CONTRAST = [
    "Это не ошибка, а системный дефект.",
    "Проблема не в данных, а в механизме выбора.",
    ...  # 12 промптов
]

SHORT_PROMPTS_NO_CONTRAST = [
    "Это системный дефект архитектуры.",
    "Проблема находится в механизме выбора.",
    ...  # 12 промптов
]
```

Для каждого промпта считает SAE-активацию фичи, сравнивает два набора.

**Выход:**
```
sae_feature_208_short_prompt_ablation.csv
sae_feature_208_short_prompt_ablation_summary.csv
```

Используется для верификации: фича 208 должна быть значимо выше на промптах с контрастом «не X, а Y».

---

## Steering hook

```python
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    sae = saes[real_layer]
    f_idx = int(feature_index)
    dec_vec = sae.W_dec[f_idx].to(device=activation.device, dtype=activation.dtype)
    return activation.float() + scale * dec_vec
```

---

## Выходные файлы

| Файл | Содержание |
|------|-----------|
| `sae_feature_steering_generation_with_causal_metrics.csv` | Steering прогоны + KL метрики |
| `sae_feature_unembed_top_tokens.csv` | Top-10 positive/negative токены для каждой фичи |
| `sae_feature_position_activation_profile.csv` | Активация фичи по позициям токенов |
| `sae_feature_position_activation_profile.png` | Визуализация профиля |
| `sae_feature_208_short_prompt_ablation.csv` | Активации на коротких промптах |
| `sae_feature_208_short_prompt_ablation_summary.csv` | Сводка: contrast vs no-contrast |

---

## Что поменять при смене фичей

```python
STEERING_FEATURES = [
    {"real_layer": NEW_LAYER, "feature_index": NEW_FEATURE, "feature_label": "..."},
]
```

Для short-prompt ablation — обнови `SHORT_PROMPTS_WITH_CONTRAST` и `SHORT_PROMPTS_NO_CONTRAST` под семантику новой фичи.

