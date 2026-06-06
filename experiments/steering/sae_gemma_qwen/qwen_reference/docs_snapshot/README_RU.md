# SAE Feature Steering — Mechanistic Interpretability Toolkit

Набор скриптов для анализа и управления поведением языковых моделей через SAE (Sparse Autoencoders).  
Текущая цель: поиск SAE-фич, которые опосредуют «режим письма» модели (аналитический, диагностический, холодный тон), и верификация их каузального вклада через steering + KL-дивергенцию.

Разработано на базе **Gemma-3-12B-IT** + **gemma-scope-2-12b-it-res-all** (l0_small).  
Фреймворки: [TransformerLens](https://github.com/neelnanda-io/TransformerLens), [SAELens](https://github.com/jbloomAus/SAELens).

---

## Структура файлов

```
steering/
├── 01_candidate_discovery_and_rough_sae_patching.py   ← стартовый скрипт, задаёт BASE_TEXT
├── steering_gemma3_V1.py                              ← steering без KL (средняя версия)
├── sae_feature_steering_light.py                      ← steering облегчённый (без KL)
├── sae_feature_steering_v2_no_control.py              ← steering + диагностика (KL, патчинг, профили)
├── sae_steering_with_kl_full.py                       ← полная версия с двумя KL-слоями
│
└── docs/
    ├── 01_candidate_discovery.md
    ├── steering_gemma3_V1.md
    ├── sae_feature_steering_light.md
    ├── sae_feature_steering_v2_no_control.md
    └── sae_steering_with_kl_full.md
```

---

## С чего начинать

### 1. Установка зависимостей

```bash
pip install transformer_lens sae_lens pandas torch tqdm matplotlib
```

### 2. Проверить модель и SAE

Перед запуском убедись, что:
- модель есть в TransformerLens
- для неё есть SAE в SAELens
- ты знаешь правильный `release` и формат `sae_id`

### 3. Подготовить `sae_order_feature_contrast.csv`

Это входной файл с контрастными активациями. Он **не входит** в репозиторий — его нужно сгенерировать для каждой модели отдельно.  
Минимально необходимые колонки:

```
layer, feature_index, x_order_orth_component_delta, interpretation_status
```

### 4. Запустить скрипты по порядку

```
01_candidate_discovery_and_rough_sae_patching.py
→ steering_gemma3_V1.py  (опционально, базовый steering)
→ sae_feature_steering_light.py  (облегчённый steering, без KL)
→ sae_feature_steering_v2_no_control.py  (диагностика)
→ sae_steering_with_kl_full.py  (полный анализ с KL)
```

---

## BASE_TEXT — откуда берётся

`BASE_TEXT` задаётся в файле  
`01_candidate_discovery_and_rough_sae_patching.py`  
через `prompts_target[0]`.

```python
# В 01_candidate_discovery_and_rough_sae_patching.py:
prompts_target = ["Твой текст-цель здесь..."]

BASE_TEXT = prompts_target[0]  # используется всеми следующими скриптами
```

Все последующие скрипты (`steering_gemma3_V1.py`, `sae_feature_steering_light.py`, и т.д.) обращаются к `prompts_target[0]` как к глобальной переменной — они **не переопределяют** его сами.

> **По желанию** — `BASE_TEXT` можно вынести в отдельный файл конфига, `.env`, или передавать как аргумент командной строки. Логика остаётся той же: все скрипты должны получить одну и ту же строку.

---

## Полная лестница анализа

```
0. Проверить модель и SAE
1. Сгенерировать contrast CSV
2. Найти top candidate features
3. Посмотреть top activating contexts
4. Проверить target-vs-control activation contrast
5. Почистить контексты от BOS/пунктуации
6. Сделать delta-only ablation
7. Сделать all-position contribution
8. Сделать final logit effect
9. Сделать top logit deltas
10. Выбрать 2–5 фич
11. Запустить steering + final-next-token KL + teacher-forced KL
12. Читать summary CSV, а не влюбляться в один красивый output
```

---

## Описание скриптов

### `01_candidate_discovery_and_rough_sae_patching.py`

**Роль:** точка входа. Загружает модель, читает contrast CSV, загружает SAE, находит top order-specific features, запускает causal mediation analysis с patch_sae_features.

**Ключевые функции:**
- `get_feature_top_contexts()` — топ активирующих контекстов для фичи
- `inspect_top_mediators_on_texts()` — прогон всех top-медиаторов
- `patch_sae_features()` — zero-ablation фич в residual activation
- `run_mediation_experiment()` — основной цикл causal mediation

**Выходные файлы:**
```
sae_feature_top_activating_contexts.csv
causal_mediation_sae_order_features_results.csv
```

**Переменные для замены при смене модели:**
```python
MODEL_NAME = "google/gemma-3-12b-it"
CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"
TOP_K = 30
```

---

### `steering_gemma3_V1.py` — средняя версия (без KL)

**Роль:** базовый steering-прогон с несколькими тасками, двумя режимами генерации (greedy/sampled), текстовыми метриками и сравнением с baseline (scale=0).

**Поддерживает:**
- greedy + sampled generation
- 8 тасков для анализа BASE_TEXT
- сохранение полного текста + хэш промпта
- метрики: длина, Jaccard с baseline, уникальность n-грамм

**Выходные файлы:**
```
sae_feature_steering_generation_full_metrics.csv
sae_feature_steering_generation_summary_metrics.csv
sae_feature_steering_base_text.txt
```

**Что поменять:**
```python
BASE_TEXT = prompts_target[0]  # задаётся в 01_candidate...

STEERING_FEATURES = [
    {"real_layer": 41, "feature_index": 13686, ...},
]
STEERING_SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]
```

---

### `sae_feature_steering_light.py` — облегчённая версия (без KL)

**Роль:** самый простой steering, только генерация + printout. Хорош для быстрой визуальной проверки влияния фичи.

**Отличие от V1:** нет teacher-forced KL, нет метрик, нет сравнения с baseline. Только вывод текста в консоль и сохранение в CSV.

**Hook:**
```python
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale):
    dec_vec = sae.W_dec[feature_index]
    patched = activation + scale * dec_vec  # добавляет decoder direction на все позиции
    return patched
```

**Выходные файлы:**
```
sae_feature_steering_generation_test_sampled.csv
```

**Что поменять:**
```python
STEERING_FEATURES = [(41, 13686), (41, 208), (41, 207)]
STEERING_SCALES    = [-3.0, -1.5, 0.0, 1.5, 3.0]
N_SAMPLES          = 5
BASE_TEXT          = prompts_target[0]
```

---

### `sae_feature_steering_v2_no_control.py` — диагностическая версия

**Роль:** расширенный анализ без control-сравнения, но с несколькими дополнительными диагностиками.

**Что умеет:**

1. **Next-token KL** — `compute_next_token_kl_feature_steering()`  
   KL(base‖patched), KL(patched‖base), JS, logit_l2, смена top-token

2. **Activation patching target → control** (по флагу) — `capture_activation_for_tokens()`, `build_control_patch_hook()`  
   Выравнивание позиций: `CONTROL_PATCH_ALIGNMENT = "right"`

3. **Teacher-forced per-token KL** — `teacher_forced_per_token_kl()`  
   Сравнение на одной токенной траектории, не free-running

4. **Unembed-проекция W_dec[f] @ W_U** — `run_unembed_projection()`  
   Top-10 positive и top-10 negative токенов

5. **Позиционный профиль активации** — `run_positional_profile()`  
   x=позиция, y=активация → CSV + PNG

6. **Short-prompt ablation для фичи 208** — `run_short_prompt_ablation()`  
   12 промптов «не X, а Y» vs 12 без контраста

**Выходные файлы:**
```
sae_feature_steering_generation_with_causal_metrics.csv
sae_feature_unembed_top_tokens.csv
sae_feature_position_activation_profile.csv
sae_feature_position_activation_profile.png
sae_feature_208_short_prompt_ablation.csv
sae_feature_208_short_prompt_ablation_summary.csv
```

**Флаги включения/выключения:**
```python
RUN_GENERATION                              = True
RUN_NEXT_TOKEN_KL                           = True
RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL   = False  # требует prompts_control
RUN_UNEMBED_PROJECTION                      = True
RUN_POSITIONAL_PROFILE                      = True
RUN_SHORT_PROMPT_ABLATION                   = True
```

---

### `sae_steering_with_kl_full.py` — полная версия с двумя KL-слоями

**Роль:** итоговый скрипт для финальной верификации. Содержит оба KL-слоя как диагностику сдвига распределения токенов.

> KL здесь — не training loss, а измерение: насколько сильно steering меняет распределение следующего токена.

**KL-слой 1: Final next-token KL** — `compute_final_next_token_kl()` (line 393)

Считается прямо во время steering-прогона, на последней позиции промпта:
```
KL(p_base(next_token | prompt) ‖ p_patched(next_token | prompt))
```

```python
# base без hook
base_logits = model([prompt])[:, -1, :].float()

# patched — с SAE steering hook
with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
    patched_logits = model([prompt])[:, -1, :].float()

kl_base_to_patched = sum(p_base * (log_p_base - log_p_patched))
kl_patched_to_base = sum(p_patched * (log_p_patched - log_p_base))
js = 0.5 * kl_base_to_patched + 0.5 * kl_patched_to_base
```

**KL-слой 2: Teacher-forced per-token KL** — `teacher_forced_per_token_kl()` (line 734)

Запускается после генерации. Берёт reference_continuation из scale=0 (baseline), склеивает `prompt + reference_tokens`, считает logits base и patched на одинаковой траектории (teacher forcing, не free-running), считает KL на каждом шаге.

```python
# reference = то, что сгенерировала базовая модель (scale=0)
full_input = prompt_tokens + reference_tokens

base_logits    = model(full_input)          # без hook
patched_logits = model(full_input)          # с hook на каждой позиции

# KL на каждом шаге reference
for i, ref_token in enumerate(reference_tokens):
    kl_i = KL(base_probs[i] ‖ patched_probs[i])
```

Summary: sum/mean/max/p95 KL, доля смены top-token, delta logprob референсных токенов.

**Hook:**
```python
# patched = activation + scale * W_dec[feature]
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale):
    dec_vec = sae.W_dec[feature_index]
    return activation + scale * dec_vec
```

**Выходные файлы:**
```
sae_feature_steering_generation_full_metrics.csv
sae_feature_steering_generation_summary_metrics.csv
sae_feature_steering_base_text.txt
sae_feature_steering_generation_full_metrics_with_tf_kl.csv
sae_teacher_forced_per_token_kl_details.csv      (если SAVE_PER_TOKEN_DETAILS=True)
sae_teacher_forced_kl_summary_by_feature_scale.csv   ← самое важное
```

**Флаги:**
```python
RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION    = True
SAVE_PER_TOKEN_DETAILS                    = True
MAX_REFERENCE_TOKENS_FOR_TF_KL            = 220
```

---

## Что менять при переходе на другую модель

**Меняешь:**
```python
MODEL_NAME         = "new_model_name"
SAE_RELEASE        = "new_sae_release"
SAE_ID_TEMPLATE    = "layer_{layer}_..."
HOOK_POINT_TEMPLATE = "blocks.{layer}.hook_resid_post"
CONTRAST_CSV_PATH  = "путь к новому contrast CSV"
```

**Не трогаешь:**
```
feature_index
layer (номера слоёв из старого CSV)
```

> Номер фичи и слоя не переносятся между моделями. Нельзя взять feature 13686 из Gemma и применить к другой модели.

**Порядок для новой модели:**
```
1. Новый contrast CSV
2. Новые candidates
3. Новые contexts
4. Новые causal checks
5. Только потом — steering + KL
```

---

## Выходные файлы — что читать

| Файл | Что говорит |
|------|-------------|
| `sae_order_feature_contrast.csv` | Какие фичи отличаются между target и control |
| `sae_top_candidate_features.csv` | Top-K кандидаты после фильтра |
| `sae_feature_top_activating_contexts.csv` | Где они активируются в тексте |
| `sae_feature_target_vs_control_activation_contrast.csv` | Target-specific или broad/control-heavy |
| `sae_clean_selected_feature_contexts.csv` | Чистые контексты для ручной интерпретации |
| `sae_all_position_feature_contribution_selected.csv` | Есть ли вклад по всем позициям |
| `sae_final_logit_effect_selected_features.csv` | Меняется ли next-token distribution |
| `sae_top_logit_deltas_selected_features.csv` | Какие именно токены двигаются |
| `sae_feature_steering_generation_summary_metrics.csv` | Меняется ли генерация |
| `sae_teacher_forced_kl_summary_by_feature_scale.csv` | **Самая важная** — полный steering + KL анализ |

---

## Самое короткое правило

```
Старые фичи не трогаешь.
Сначала новый contrast CSV.
Потом новые candidates.
Потом новые contexts.
Потом новые causal checks.
Только потом steering + KL.
```

