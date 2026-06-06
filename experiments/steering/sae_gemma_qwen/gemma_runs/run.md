## Общая лестница для любой новой модели

```text
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
12. Читать summary, а не влюбляться в один красивый output
```

---

# 0. Что менять для другой модели

Вынеси наверх один конфиг:

```python
MODEL_NAME = "google/gemma-3-12b-it"

SAE_RELEASE = "gemma-scope-2-12b-it-res-all"
SAE_ID_TEMPLATE = "layer_{layer}_width_16k_l0_small"

HOOK_POINT_TEMPLATE = "blocks.{layer}.hook_resid_post"

CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

LAYER_OFFSET_MODE = "auto"  # "auto", "minus_1", "none"
TOP_K = 30
```

Для новой модели обязательно проверить:

```text
1. Есть ли эта модель в TransformerLens.
2. Есть ли под неё SAE.
3. Как называется SAE release.
4. Как называется sae_id.
5. На какой hook point обучен SAE.
6. Есть ли offset слоя в CSV.
```

Самая частая ошибка:

```text
взять feature 13686 из Gemma и применить к другой модели
```

Нельзя. Номер фичи не переносится между моделями.

---

# 1. Генерация `sae_order_feature_contrast.csv`

Это самый первый этап. Он отвечает на вопрос:

```text
какие SAE-фичи отличаются между target и control?
```

Выход:

```text
sae_order_feature_contrast.csv
```

В нём должны быть минимум колонки:

```text
layer
feature_index
x_order_orth_component_delta
interpretation_status
```

Желательно ещё:

```text
target_prompt_mean_activation_delta
sentence_shuffle_prompt_mean_activation_delta
target_generation_mean_activation
sentence_shuffle_generation_mean_activation
order_specific_score
```

Важно: у меня в этой переписке есть скрипт, который **читает** этот CSV, но нет полного исходного скрипта, который его **генерирует с нуля**. Для новой модели нужен именно генератор contrast CSV. Старый `sae_order_feature_contrast.csv` от Gemma нельзя использовать для другой модели.

---

# 2. Загрузка модели, CSV и SAE

Это кусок из первого скрипта.

```python
import torch
import pandas as pd
from transformer_lens import HookedTransformer
from sae_lens import SAE

print(f"Загружаем модель: {MODEL_NAME}")
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=DEVICE,
    dtype=DTYPE,
)
model.eval()

df = pd.read_csv(CONTRAST_CSV_PATH)
print(f"Строк в contrast CSV: {len(df)}")
print(f"Колонки: {list(df.columns)}")

layers_in_csv = sorted(df["layer"].unique())
print(f"Слои в CSV: {layers_in_csv}")

if LAYER_OFFSET_MODE == "minus_1":
    df["real_layer"] = df["layer"] - 1
elif LAYER_OFFSET_MODE == "none":
    df["real_layer"] = df["layer"]
else:
    if min(layers_in_csv) >= 1:
        df["real_layer"] = df["layer"] - 1
        print("Auto offset: CSV layer - 1")
    else:
        df["real_layer"] = df["layer"]
        print("Auto offset: no correction")

target_layers = sorted(df["real_layer"].unique())
print(f"Реальные слои для SAE: {target_layers}")

saes = {}

for real_layer in target_layers:
    sae_id = SAE_ID_TEMPLATE.format(layer=real_layer)

    try:
        loaded = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
            device=DEVICE,
        )

        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        sae.eval()

        saes[int(real_layer)] = sae
        print(f"✓ SAE layer {real_layer}: {sae_id}")

    except Exception as e:
        print(f"✗ SAE layer {real_layer} failed: {repr(e)}")
```

---

# 3. Выбор top candidates из CSV

```python
mediators = df[
    df["interpretation_status"].str.contains(
        "order_specific|order_enriched|order_component",
        na=False,
    )
].copy()

mediators = mediators.sort_values(
    "x_order_orth_component_delta",
    ascending=False,
)

top_mediators = mediators.head(TOP_K)[
    [
        "layer",
        "real_layer",
        "feature_index",
        "x_order_orth_component_delta",
        "interpretation_status",
    ]
].copy()

print(f"\nTOP-{TOP_K} candidate SAE features:")
print(
    top_mediators[
        [
            "real_layer",
            "feature_index",
            "x_order_orth_component_delta",
            "interpretation_status",
        ]
    ].to_string(index=False)
)

top_mediators.to_csv("sae_top_candidate_features.csv", index=False)
```

Выход:

```text
sae_top_candidate_features.csv
```

Это ещё не доказательство. Это список подозреваемых.

---

# 4. Top activating contexts

Цель:

```text
понять, где фича активируется в тексте
```

Именно здесь видно: фича смысловая, пунктуационная, BOS, переносы строк, куски слов или реально “режим письма”.

```python
def get_feature_top_contexts_marked(
    texts,
    real_layer,
    feature_index,
    top_n=20,
    context_window=18,
    min_activation=1e-6,
    ignore_bos=True,
):
    if real_layer not in saes:
        raise ValueError(f"SAE для layer {real_layer} не загружен")

    sae = saes[real_layer]
    hook_name = HOOK_POINT_TEMPLATE.format(layer=real_layer)

    records = []

    with torch.no_grad():
        tokens = model.to_tokens(texts, prepend_bos=True)

        if hasattr(model.cfg, "n_ctx") and tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, :model.cfg.n_ctx]

        _, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name],
        )

        act = cache[hook_name].float()
        latent = sae.encode(act)

        if feature_index >= latent.shape[-1]:
            raise ValueError(
                f"feature_index={feature_index} вне диапазона. "
                f"Всего features: {latent.shape[-1]}"
            )

        scores = latent[..., feature_index]

        if ignore_bos:
            scores[:, 0] = -1e9

        flat_scores = scores.flatten()
        k = min(top_n * 10, flat_scores.numel())

        values, flat_positions = torch.topk(flat_scores, k=k)

        batch_size, seq_len = scores.shape

        for value, flat_pos in zip(values.tolist(), flat_positions.tolist()):
            if value <= min_activation:
                continue

            b = int(flat_pos // seq_len)
            pos = int(flat_pos % seq_len)

            str_tokens = model.to_str_tokens(tokens[b])

            left = max(0, pos - context_window)
            right = min(len(str_tokens), pos + context_window + 1)

            left_context = "".join(str_tokens[left:pos])
            active_token = str_tokens[pos]
            right_context = "".join(str_tokens[pos + 1:right])

            context = left_context + " >>>" + active_token + "<<< " + right_context

            records.append({
                "text_id": b,
                "real_layer": int(real_layer),
                "feature_index": int(feature_index),
                "activation": float(value),
                "token_position": int(pos),
                "token": active_token,
                "context": context,
            })

            if len(records) >= top_n:
                break

    del cache, act, latent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(records)
```

Запуск:

```python
texts_for_interp = prompts_target + prompts_control

all_contexts = []

for _, row in top_mediators.head(15).iterrows():
    real_layer = int(row["real_layer"])
    feature_index = int(row["feature_index"])

    print(f"\n=== layer {real_layer}, feature {feature_index} ===")

    ctx = get_feature_top_contexts_marked(
        texts=texts_for_interp,
        real_layer=real_layer,
        feature_index=feature_index,
        top_n=15,
        context_window=18,
    )

    print(ctx[["activation", "token", "context"]].to_string(index=False))
    all_contexts.append(ctx)

feature_contexts = pd.concat(all_contexts, ignore_index=True)
feature_contexts.to_csv("sae_feature_top_activating_contexts_marked.csv", index=False)
```

---

# 5. Target-vs-control activation contrast

Цель:

```text
выкинуть фичи, которые одинаково активны и на target, и на control
```

```python
def feature_activation_stats(
    texts,
    label,
    candidate_features,
    threshold=1e-6,
    ignore_bos=True,
):
    rows = []

    candidate_features = candidate_features.copy()
    candidate_features["real_layer"] = candidate_features["real_layer"].astype(int)
    candidate_features["feature_index"] = candidate_features["feature_index"].astype(int)

    for real_layer, group in candidate_features.groupby("real_layer"):
        if real_layer not in saes:
            continue

        sae = saes[real_layer]
        hook_name = HOOK_POINT_TEMPLATE.format(layer=real_layer)

        with torch.no_grad():
            tokens = model.to_tokens(texts, prepend_bos=True)

            if hasattr(model.cfg, "n_ctx") and tokens.shape[1] > model.cfg.n_ctx:
                tokens = tokens[:, :model.cfg.n_ctx]

            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )

            act = cache[hook_name].float()
            latent = sae.encode(act)

            mask = torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device)

            if ignore_bos:
                mask[:, 0] = False

            for _, row in group.iterrows():
                f_idx = int(row["feature_index"])

                if f_idx < 0 or f_idx >= latent.shape[-1]:
                    continue

                scores = latent[..., f_idx]
                vals = scores[mask]

                active_vals = vals[vals > threshold]

                rows.append({
                    "set": label,
                    "real_layer": real_layer,
                    "feature_index": f_idx,
                    "mean_activation": vals.mean().item(),
                    "max_activation": vals.max().item(),
                    "sum_activation_per_text": vals.sum().item() / len(texts),
                    "active_fraction": (vals > threshold).float().mean().item(),
                    "nonzero_mean_activation": active_vals.mean().item() if active_vals.numel() > 0 else 0.0,
                    "nonzero_count": int(active_vals.numel()),
                    "x_order_orth_delta": float(row.get("x_order_orth_component_delta", 0.0)),
                    "status": row.get("interpretation_status", ""),
                })

        del cache, act, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)
```

Запуск:

```python
candidate_features = top_mediators.head(30).copy()

target_stats = feature_activation_stats(
    texts=prompts_target,
    label="target",
    candidate_features=candidate_features,
)

control_stats = feature_activation_stats(
    texts=prompts_control,
    label="control",
    candidate_features=candidate_features,
)

contrast = target_stats.merge(
    control_stats,
    on=["real_layer", "feature_index"],
    suffixes=("_target", "_control"),
)

contrast["sum_delta"] = (
    contrast["sum_activation_per_text_target"]
    - contrast["sum_activation_per_text_control"]
)

contrast["target_control_sum_ratio"] = (
    contrast["sum_activation_per_text_target"] + 1e-9
) / (
    contrast["sum_activation_per_text_control"] + 1e-9
)

contrast = contrast.sort_values("sum_delta", ascending=False)

contrast.to_csv("sae_feature_target_vs_control_activation_contrast.csv", index=False)

print(
    contrast[
        [
            "real_layer",
            "feature_index",
            "sum_activation_per_text_target",
            "sum_activation_per_text_control",
            "sum_delta",
            "target_control_sum_ratio",
            "status_target",
        ]
    ].head(30).to_string(index=False)
)
```

---

# 6. Выбор selected features

После contrast выбираешь руками 2–5 фич.

Автоматический первичный фильтр:

```python
selected_features_df = contrast[
    (contrast["sum_delta"] > 0) &
    (contrast["sum_activation_per_text_target"] > 0) &
    (contrast["target_control_sum_ratio"] > 1.5)
].copy()

selected_features_df = selected_features_df.sort_values("sum_delta", ascending=False).head(10)

print(selected_features_df[
    [
        "real_layer",
        "feature_index",
        "sum_activation_per_text_target",
        "sum_activation_per_text_control",
        "sum_delta",
        "target_control_sum_ratio",
    ]
].to_string(index=False))

selected_features_df.to_csv("sae_feature_target_vs_control_best.csv", index=False)
```

Потом руками делаешь список:

```python
SELECTED_FEATURES = [
    (41, 13686),
    (41, 208),
    (41, 207),
]
```

Для новой модели здесь будут другие номера.

---

# 7. Clean contexts

Цель:

```text
выкинуть BOS, точки, запятые, переносы, pad, мусор
```

```python
BAD_TOKENS_EXACT = {
    "<bos>", "<pad>", "\n", "\n\n", "\n\n\n\n",
    ".", ",", "-", "—", "–", ":", ";", "!", "?",
    "(", ")", "[", "]", "{", "}", '"', "'", "«", "»",
}

def is_bad_token(tok):
    s = str(tok)

    if s in BAD_TOKENS_EXACT:
        return True

    if s.strip() == "":
        return True

    if "<pad>" in s or "<bos>" in s:
        return True

    if all(ch in ".,:;!?-—–()[]{}\"'«» \n\t" for ch in s):
        return True

    return False
```

Используешь тот же `get_feature_top_contexts_marked`, но добавляешь:

```python
if is_bad_token(active_token):
    continue
```

Выход:

```text
sae_clean_selected_feature_contexts.csv
```

---

# 8. Delta-only ablation

Это важная замена старого SAE decode-patch.

Старое было грязным:

```text
activation → SAE encode → zero feature → SAE decode
```

Чище:

```text
activation - latent_f * W_dec_f
```

```python
def patch_sae_feature_delta_only(activation, hook, real_layer, feature_index, scale=1.0):
    sae = saes[real_layer]

    orig_dtype = activation.dtype
    act_float = activation.float()

    with torch.no_grad():
        latent = sae.encode(act_float)

        f_idx = int(feature_index)
        if f_idx < 0 or f_idx >= latent.shape[-1]:
            return activation

        feature_activation = latent[..., f_idx:f_idx + 1]

        dec_vec = sae.W_dec[f_idx].to(
            device=act_float.device,
            dtype=act_float.dtype,
        )

        delta = feature_activation * dec_vec
        patched = act_float - scale * delta

    return patched.to(dtype=orig_dtype)
```

---

# 9. All-position contribution

Цель:

```text
не смотреть только last token
```

```python
def feature_position_contribution_stats(texts, label, selected_features, min_activation=1e-6):
    rows = []

    for real_layer, feature_index in selected_features:
        if real_layer not in saes:
            continue

        sae = saes[real_layer]
        hook_name = HOOK_POINT_TEMPLATE.format(layer=real_layer)

        with torch.no_grad():
            tokens = model.to_tokens(texts, prepend_bos=True)

            if hasattr(model.cfg, "n_ctx") and tokens.shape[1] > model.cfg.n_ctx:
                tokens = tokens[:, :model.cfg.n_ctx]

            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )

            act = cache[hook_name].float()
            latent = sae.encode(act)

            f_idx = int(feature_index)

            if f_idx < 0 or f_idx >= latent.shape[-1]:
                continue

            feature_scores = latent[..., f_idx]
            dec_vec = sae.W_dec[f_idx].to(device=feature_scores.device, dtype=torch.float32)
            dec_norm = dec_vec.norm().item()

            contribution_norm = feature_scores * dec_norm

            vals = contribution_norm.flatten()
            scores = feature_scores.flatten()

            active_vals = vals[scores > min_activation]
            active_scores = scores[scores > min_activation]

            rows.append({
                "set": label,
                "real_layer": real_layer,
                "feature_index": feature_index,
                "sum_contribution": vals.sum().item(),
                "mean_contribution": vals.mean().item(),
                "max_contribution": vals.max().item(),
                "active_sum_contribution": active_vals.sum().item() if active_vals.numel() > 0 else 0.0,
                "active_mean_contribution": active_vals.mean().item() if active_vals.numel() > 0 else 0.0,
                "active_max_contribution": active_vals.max().item() if active_vals.numel() > 0 else 0.0,
                "mean_activation": scores.mean().item(),
                "max_activation": scores.max().item(),
                "active_token_count": int(active_scores.numel()),
                "active_fraction": (scores > min_activation).float().mean().item(),
                "decoder_norm": dec_norm,
            })

        del cache, act, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)
```

Запуск:

```python
target_pos = feature_position_contribution_stats(
    prompts_target,
    "target",
    SELECTED_FEATURES,
)

control_pos = feature_position_contribution_stats(
    prompts_control,
    "control",
    SELECTED_FEATURES,
)

position_compare = target_pos.merge(
    control_pos,
    on=["real_layer", "feature_index"],
    suffixes=("_target", "_control"),
)

position_compare["active_sum_delta"] = (
    position_compare["active_sum_contribution_target"]
    - position_compare["active_sum_contribution_control"]
)

position_compare["active_sum_ratio"] = (
    position_compare["active_sum_contribution_target"] + 1e-9
) / (
    position_compare["active_sum_contribution_control"] + 1e-9
)

position_compare.to_csv("sae_all_position_feature_contribution_selected.csv", index=False)
```

---

# 10. Final logit effect

Цель:

```text
проверить, меняет ли фича распределение следующего токена
```

```python
def run_final_logit_effect(texts, label, selected_features):
    rows = []

    for real_layer, feature_index in selected_features:
        if real_layer not in saes:
            continue

        hook_name = HOOK_POINT_TEMPLATE.format(layer=real_layer)

        def patching_hook(act, hook):
            return patch_sae_feature_delta_only(
                activation=act,
                hook=hook,
                real_layer=real_layer,
                feature_index=feature_index,
                scale=1.0,
            )

        for text_id, text in enumerate(texts):
            with torch.no_grad():
                base_logits = model([text])[:, -1, :].float()
                base_logprobs = torch.log_softmax(base_logits, dim=-1)
                base_probs = torch.softmax(base_logits, dim=-1)

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_name, patching_hook)]):
                    patched_logits = model([text])[:, -1, :].float()

                patched_logprobs = torch.log_softmax(patched_logits, dim=-1)

            logit_l2 = (base_logits - patched_logits).norm(dim=-1).item()
            logit_max_abs = (base_logits - patched_logits).abs().max(dim=-1).values.item()

            kl_base_to_patched = (
                base_probs * (base_logprobs - patched_logprobs)
            ).sum(dim=-1).item()

            top_base_id = int(base_logits.argmax(dim=-1).item())
            top_patched_id = int(patched_logits.argmax(dim=-1).item())

            rows.append({
                "set": label,
                "text_id": text_id,
                "real_layer": real_layer,
                "feature_index": feature_index,
                "final_logit_l2": logit_l2,
                "final_logit_max_abs": logit_max_abs,
                "kl_base_to_patched": kl_base_to_patched,
                "top_base_token": model.to_string([top_base_id]),
                "top_patched_token": model.to_string([top_patched_id]),
                "top_token_changed": top_base_id != top_patched_id,
            })

    return pd.DataFrame(rows)
```

---

# 11. Top logit deltas

Цель:

```text
понять, какие именно токены двигаются
```

Выход:

```text
sae_top_logit_deltas_selected_features.csv
```

Критерий:

```text
если двигаются смысловые токены — интересно
если двигаются субтокены/мусор/HTML — грязная фича
```

---

# 12. Steering + KL

На этом этапе уже не надо искать фичи. Надо проверять выбранные.

Используешь готовый файл:

```text
sae_steering_with_kl_full.py
```

Он делает:

```text
1. base text + tasks
2. scale steering
3. greedy + sampled generation
4. текстовые метрики
5. final-next-token KL
6. teacher-forced per-token KL
7. summary CSV
```

Внутри меняешь только:

```python
STEERING_FEATURES = [
    {
        "real_layer": NEW_LAYER,
        "feature_index": NEW_FEATURE,
        "feature_label": "...",
        "comment": "...",
    },
]
```

И модель/SAE должны уже быть загружены до запуска файла.

---

# Минимальный порядок файлов/ячеек

```text
00_config_and_prompts.py
    MODEL_NAME
    SAE_RELEASE
    SAE_ID_TEMPLATE
    prompts_target
    prompts_control

01_generate_contrast_csv.py
    создаёт sae_order_feature_contrast.csv
    для новой модели нужен заново

02_load_model_sae_candidates.py
    грузит model
    читает contrast CSV
    грузит SAE
    создаёт top_mediators

03_feature_contexts.py
    top activating contexts
    clean contexts

04_activation_contrast.py
    target-vs-control activation contrast
    выбираешь SELECTED_FEATURES

05_causal_checks.py
    delta-only ablation
    all-position contribution
    final logit effect
    top logit deltas

06_steering_with_kl.py
    steering
    final-next-token KL
    teacher-forced per-token KL
```

---

# Какой файл читать после каждого этапа

```text
sae_order_feature_contrast.csv
→ какие фичи подозрительны

sae_top_candidate_features.csv
→ top-K кандидаты после фильтра

sae_feature_top_activating_contexts_marked.csv
→ где они активируются

sae_feature_target_vs_control_activation_contrast.csv
→ target-specific или broad/control-heavy

sae_clean_selected_feature_contexts.csv
→ нормальная ручная интерпретация

sae_all_position_feature_contribution_selected.csv
→ есть ли вклад по всем позициям

sae_final_logit_effect_selected_features.csv
→ меняется ли next-token distribution

sae_top_logit_deltas_selected_features.csv
→ какие токены двигаются

sae_feature_steering_generation_summary_metrics.csv
→ меняется ли генерация

sae_feature_steering_generation_full_metrics_with_tf_kl.csv
→ полный steering + KL анализ

sae_teacher_forced_kl_summary_by_feature_scale.csv
→ самая важная summary по KL
```

---

# Что переносится на другую модель

Переносится:

```text
логика эксперимента
метрики
структура файлов
target/control дизайн
teacher-forced KL
steering validation
```

Не переносится:

```text
feature_index
layer
SAE release
hook point
порог scale
интерпретация фичи
```

---

# Самое короткое правило

Для новой модели:

```text
старые фичи не трогаешь
сначала новый contrast CSV
потом новые candidates
потом новые contexts
потом новые causal checks
только потом steering + KL
```

Иначе это будет не mechanistic interpretability, а спиритический сеанс с `feature_index`.
