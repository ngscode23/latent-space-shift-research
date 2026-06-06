# 01_candidate_discovery_and_rough_sae_patching.py

## Роль

Точка входа всего пайплайна. Загружает модель, читает contrast CSV, загружает SAE для нужных слоёв, находит top order-specific features и верифицирует их каузальный вклад через zero-ablation (patch_sae_features) + causal mediation analysis.

**Здесь определяется `BASE_TEXT`** — он берётся как `prompts_target[0]` и используется всеми последующими скриптами.

---

## Зависимости

```python
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import pandas as pd
from tqdm import tqdm
```

---

## Конфиг

```python
MODEL_NAME        = "google/gemma-3-12b-it"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"
TOP_K             = 30  # сколько самых сильных order-фич берём
```

---

## BASE_TEXT и prompts

```python
# ←←← ЗАМЕНИ НА СВОИ ПРОМПТЫ ←←←
prompts_target = ["Твой target текст здесь..."]
prompts_control = ["Твой control текст здесь..."]
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←

# BASE_TEXT для всех steering-скриптов — это:
BASE_TEXT = prompts_target[0]
```

> `BASE_TEXT` можно вынести в отдельный конфиг-файл или `.env` — все скрипты просто должны получить ту же строку. Менять логику не нужно.

---

## Загрузка модели и SAE

```python
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.bfloat16)
model.eval()

df = pd.read_csv(CONTRAST_CSV_PATH)

# Авто-определение offset (+1 в CSV → 0-based в TransformerLens)
layers_in_csv = sorted(df['layer'].unique())
if min(layers_in_csv) >= 1:
    df['real_layer'] = df['layer'] - 1
else:
    df['real_layer'] = df['layer']

target_layers = sorted(df['real_layer'].unique())

saes = {}
for real_layer in target_layers:
    sae_id = f"layer_{real_layer}_width_16k_l0_small"
    sae = SAE.from_pretrained(
        release="gemma-scope-2-12b-it-res-all",
        sae_id=sae_id,
        device=DEVICE
    )
    saes[real_layer] = sae
```

---

## Top order-mediators

```python
mediators = df[
    df['interpretation_status'].str.contains('order_specific|order_enriched', na=False)
].copy()

mediators = mediators.sort_values('x_order_orth_component_delta', ascending=False)

top_mediators = mediators.head(TOP_K)[
    ['layer', 'real_layer', 'feature_index', 'x_order_orth_component_delta', 'interpretation_status']
]
```

---

## Функции

### `get_feature_top_contexts(texts, real_layer, feature_index, top_n, context_window, batch_size)`

Показывает, где фича активируется в тексте. Это основной способ понять, «о чём» фича.

```python
contexts = get_feature_top_contexts(
    texts=prompts_target + prompts_control,
    real_layer=41,
    feature_index=13686,
    top_n=20,
    context_window=12,
    batch_size=1,
)
# → DataFrame: global_text_id, real_layer, feature_index, activation, token_position, token, context
```

Логика:
```python
hook_name = f"blocks.{real_layer}.hook_resid_post"
_, cache = model.run_with_cache(tokens, names_filter=[hook_name])
act = cache[hook_name].float()          # [batch, seq, d_model]
latent = sae.encode(act)                # [batch, seq, n_features]
scores = latent[..., feature_index]     # [batch, seq]
values, positions = torch.topk(scores[b], k=min(top_n, seq_len))
```

---

### `inspect_top_mediators_on_texts(texts, n_features, top_n_contexts)`

Прогоняет `get_feature_top_contexts` для всех top-медиаторов, сохраняет результат.

**Выход:** `sae_feature_top_activating_contexts.csv`

---

### `patch_sae_features(activation, hook, real_layer, features_to_patch, patch_value=0.0)`

Zero-ablation указанных SAE-фич в residual activation.

```python
def patch_sae_features(activation, hook, real_layer, features_to_patch, patch_value=0.0):
    sae = saes[real_layer]
    act_float = activation.to(dtype=torch.float32)

    with torch.no_grad():
        latent = sae.encode(act_float)

        for f_idx in features_to_patch:
            latent[..., int(f_idx)] = patch_value  # зануляем фичу

        decoded = sae.decode(latent)

    return decoded.to(device=..., dtype=...)
```

> **Внимание:** это «грязная» ablation — через encode → zero → decode. Для более чистой альтернативы смотри `patch_sae_feature_delta_only` в других скриптах.

---

### `run_mediation_experiment(prompts_target, prompts_control=None)`

Основной цикл causal mediation analysis.

```python
# baseline считается один раз на каждый слой
baseline_by_layer = {}
for real_layer in needed_layers:
    hook_name = f"blocks.{real_layer}.hook_resid_post"
    _, cache_base = model.run_with_cache(prompts_target, names_filter=[hook_name])
    baseline_by_layer[real_layer] = cache_base[hook_name][:, -1, :].float()

# для каждой top-фичи:
for _, row in top_mediators.iterrows():
    def patching_hook(act, hook):
        return patch_sae_features(act, hook, real_layer, [f_idx], 0.0)

    with model.hooks(fwd_hooks=[(hook_name, patching_hook)]):
        _, cache_patched = model.run_with_cache(prompts_target, names_filter=[hook_name])

    resid_base    = baseline_by_layer[real_layer]
    resid_patched = cache_patched[hook_name][:, -1, :].float()

    mediated_effect = (resid_base - resid_patched).norm(dim=-1).mean().item()
```

**Выход:** DataFrame с колонками `real_layer, feature_index, mediated_effect, ...`

---

## Выходные файлы

| Файл | Содержание |
|------|-----------|
| `sae_feature_top_activating_contexts.csv` | Где фичи активируются в тексте |
| `causal_mediation_sae_order_features_results.csv` | Каузальный вклад каждой фичи |

---

## Что менять при смене модели

```python
MODEL_NAME        = "new_model_name"         # ← СЮДА
CONTRAST_CSV_PATH = "path/to/new/csv"        # ← СЮДА
# sae_id в строке: f"layer_{real_layer}_width_16k_l0_small"  ← СЮДА
# release в SAE.from_pretrained(...)                          ← СЮДА
```

Номера `feature_index` и слоёв — не переносятся. Нужен новый contrast CSV.

