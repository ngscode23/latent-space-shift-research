# sae_feature_steering_light.py

## Роль

Облегчённая версия steering — только генерация + вывод в консоль + сохранение в CSV.  
Нет KL-дивергенции, нет метрик сравнения с baseline, нет логирования промежуточных активаций.

Подходит для:
- быстрой визуальной проверки: «меняется ли стиль при разных scale?»
- первичного эксперимента с новой фичей
- демонстрации steering-эффекта

---

## Зависимости

```python
import torch
import pandas as pd
from tqdm import tqdm
# model и saes должны быть загружены до запуска
# prompts_target должен быть определён
```

---

## Конфиг

```python
STEERING_FEATURES = [
    (41, 13686),  # главный semantic marker
    (41, 208),    # contrastive / rhetorical candidate
    (41, 207),    # strong but dirty causal candidate
]

STEERING_SCALES    = [-3.0, -1.5, 0.0, 1.5, 3.0]
N_SAMPLES          = 5       # сэмплов на каждую комбинацию
DO_SAMPLE          = True
TEMPERATURE        = 0.8
MAX_NEW_TOKENS     = 220

BASE_TEXT          = prompts_target[0]  # задаётся в 01_candidate_discovery...
```

---

## BASE_TEXT

```python
# BASE_TEXT берётся из prompts_target[0]
# prompts_target определяется в 01_candidate_discovery_and_rough_sae_patching.py
BASE_TEXT = prompts_target[0]
```

Если хочешь вынести BASE_TEXT — можно передать строку напрямую:
```python
BASE_TEXT = "Свой текст для анализа..."
```

---

## Таски

```python
TEST_TASKS = [
    "Сожми текст до одного беспощадного вывода. Одна фраза. Без оговорок.",
    "Напиши 6 разных формулировок главного диагноза этого текста. ...",
    "Продолжи мысль текста на 150 слов. Сохрани холодный диагностический режим.",
    "Перепиши главный тезис текста в ещё более сухой и административно-жёсткой форме.",
    "Выдели механизм слабости, описанный в тексте, и сформулируй его как технический дефект.",
]
```

---

## Промпт-билдер

```python
def build_analysis_prompt(base_text, task):
    return (
        "Ты анализируешь один и тот же текст.\n\n"
        "=== ТЕКСТ ДЛЯ АНАЛИЗА ===\n"
        f"{base_text}\n\n"
        "=== ЗАДАНИЕ ===\n"
        f"{task.strip()}\n\n"
        "=== ОТВЕТ ===\n"
    )
```

---

## Steering hook

Добавляет decoder direction фичи на **все позиции** residual stream:

```python
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    sae = saes[real_layer]
    orig_dtype = activation.dtype
    act_float = activation.float()

    f_idx = int(feature_index)
    if f_idx < 0 or f_idx >= sae.W_dec.shape[0]:
        return activation

    with torch.no_grad():
        dec_vec = sae.W_dec[f_idx].to(device=act_float.device, dtype=act_float.dtype)
        patched = act_float + scale * dec_vec  # scale=0 → нет эффекта

    return patched.to(dtype=orig_dtype)
```

При `scale=0.0` hook не навешивается вообще (чистый baseline):
```python
if scale == 0.0:
    out = generate_safely(prompt, ...)
else:
    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        out = generate_safely(prompt, ...)
```

---

## Генерация

```python
def generate_safely(prompt, max_new_tokens, do_sample, temperature):
    """Совместимость с разными версиями TransformerLens generate()."""
    try:
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            verbose=False,
        )
    except TypeError:
        # старые версии не принимают do_sample / temperature
        return model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)
```

---

## Основной цикл

```python
for task_id, task in enumerate(TEST_TASKS):
    full_prompt = build_analysis_prompt(BASE_TEXT, task)

    for real_layer, feature_index in STEERING_FEATURES:
        for scale in STEERING_SCALES:
            for sample_id in range(N_SAMPLES):

                output = generate_with_feature_steering(
                    prompt=full_prompt,
                    real_layer=real_layer,
                    feature_index=feature_index,
                    scale=scale,
                )

                steering_rows.append({
                    "task_id": task_id,
                    "task": task.strip(),
                    "real_layer": real_layer,
                    "feature_index": feature_index,
                    "scale": scale,
                    "sample_id": sample_id,
                    "output": output,
                    "error": "",
                })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
```

Всего прогонов: `len(TEST_TASKS) × len(STEERING_FEATURES) × len(STEERING_SCALES) × N_SAMPLES`

---

## Выходные файлы

| Файл | Содержание |
|------|-----------|
| `sae_feature_steering_generation_test_sampled.csv` | Все сгенерированные тексты + метаданные прогона |

---

## Что смотреть в выходном CSV

- Сравни `scale=0.0` vs `scale=3.0` для одного task_id + feature
- Ищи систематическое изменение стиля, а не один красивый пример
- `error != ""` — прогон упал, надо разбираться

---

## Что поменять при смене фичей

```python
STEERING_FEATURES = [
    (NEW_LAYER, NEW_FEATURE_INDEX),
    ...
]
```

Номера берёшь из предыдущих этапов анализа — из `sae_teacher_forced_kl_summary_by_feature_scale.csv` или `sae_feature_target_vs_control_activation_contrast.csv`.

