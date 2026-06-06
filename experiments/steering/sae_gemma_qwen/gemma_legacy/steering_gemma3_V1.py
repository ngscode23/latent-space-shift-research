# ============================================================
# SAE FEATURE STEERING GENERATION TEST
# BASE TEXT + MULTIPLE TASKS + METRICS + BASELINE COMPARISON
# ============================================================

import torch
import pandas as pd
import numpy as np
import random
import re
import time
import hashlib
from datetime import datetime

# ====================== CONFIG ======================

# Один общий текст анализа.
# Важно: prompts_target должен уже существовать из твоего основного скрипта.
BASE_TEXT = prompts_target[0]

# Включать ли полный текст анализа в CSV.
# Да, файл будет жирнее, но зато таблица самодостаточная.
INCLUDE_BASE_TEXT_FULL_IN_CSV = True

# Фичи для steering
STEERING_FEATURES = [
    {
        "real_layer": 41,
        "feature_index": 13686,
        "feature_label": "semantic_marker_target_only_13686",
        "comment": "target-only semantic/meta-diagnostic marker",
    },
    {
        "real_layer": 41,
        "feature_index": 208,
        "feature_label": "contrastive_rhetorical_marker_208",
        "comment": "contrastive rhetoric / не X, а Y",
    },
    {
        "real_layer": 41,
        "feature_index": 207,
        "feature_label": "strong_dirty_causal_driver_207",
        "comment": "strong causal but token-distribution dirty",
    },
]

# Масштабы вмешательства
STEERING_SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]

# Задания к одному и тому же BASE_TEXT
TEST_TASKS = [
    """
Дай жёсткий аналитический диагноз этого текста.
Какой главный механизм в нём работает?
""",

    """
Объясни, где именно этот текст атакует слабость модели.
Не пересказывай. Выдели причинный механизм.
""",

    """
Сформулируй главный тезис текста в 5 коротких пунктах.
Пиши сухо, без смягчения.
""",

    """
Определи, какие риторические конструкции делают текст давящим и диагностическим.
Приведи типы конструкций, а не длинные цитаты.
""",

    """
Напиши продолжение этого текста в том же режиме:
холодно, жёстко, аналитически, без утешения.
""",

    """
Сожми текст до одного беспощадного вывода.
Одна фраза. Без оговорок.
""",

    """
Перепиши главный тезис текста как технический дефект системы.
Без метафор. Как диагноз архитектурного ограничения.
""",

    """
Сравни этот текст с обычной мягкой объяснительной прозой.
В чём различие режима письма?
""",
]

# Два режима генерации:
# greedy — проверяет, может ли feature пробить самый вероятный путь;
# sampled — чувствительнее к слабым steering-эффектам.
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

MAX_NEW_TOKENS = 220
RANDOM_SEED_BASE = 12345

OUTPUT_CSV = "sae_feature_steering_generation_full_metrics.csv"
OUTPUT_BASE_TEXT_TXT = "sae_feature_steering_base_text.txt"


# ====================== HELPERS ======================

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_preview(text, n=700):
    text = str(text).replace("\n", "\\n")
    return text[:n]


def build_analysis_prompt(base_text, task):
    """
    Собирает prompt:
    один общий BASE_TEXT + конкретное задание к нему.
    """
    return f"""
Ты анализируешь один и тот же текст.

=== ТЕКСТ ДЛЯ АНАЛИЗА ===
{base_text}

=== ЗАДАНИЕ ===
{task.strip()}

=== ОТВЕТ ===
"""


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_seed(task_id, real_layer, feature_index, sample_id, generation_mode):
    """
    Seed не зависит от scale.
    Это нужно, чтобы sampled-сравнение scale было честнее:
    меняется feature scale, а не случайность.
    """
    mode_offset = 0 if generation_mode == "greedy" else 100000
    return (
        RANDOM_SEED_BASE
        + task_id * 1000
        + real_layer * 100
        + feature_index
        + sample_id * 17
        + mode_offset
    )


def strip_prompt_from_generation(full_output, prompt):
    """
    TransformerLens generate часто возвращает prompt + continuation.
    Для метрик нужен именно ответ.
    """
    full_output = str(full_output)

    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()

    # fallback: если prompt слегка изменился/нормализовался
    marker = "=== ОТВЕТ ==="
    if marker in full_output:
        return full_output.split(marker, 1)[-1].strip()

    return full_output.strip()


def simple_token_count(text):
    """
    Быстрый подсчёт токенов через модель.
    """
    try:
        toks = model.to_tokens(text, prepend_bos=False)
        return int(toks.shape[-1])
    except Exception:
        return None


def word_list(text):
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", str(text).lower())


def sentence_count(text):
    parts = re.split(r"[.!?]+", str(text))
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def count_substrings(text, substrings):
    t = str(text).lower()
    return sum(t.count(s.lower()) for s in substrings)


def jaccard_similarity(a, b):
    wa = set(word_list(a))
    wb = set(word_list(b))

    if len(wa) == 0 and len(wb) == 0:
        return 1.0

    if len(wa) == 0 or len(wb) == 0:
        return 0.0

    return len(wa & wb) / len(wa | wb)


def compute_text_metrics(output_text):
    """
    Метрики ответа.
    Они не доказывают смысл, но помогают увидеть систематические сдвиги.
    """
    output_text = str(output_text)

    words = word_list(output_text)
    n_words = len(words)
    n_sent = sentence_count(output_text)

    diagnostic_keywords = [
        "механизм", "режим", "ограничение", "предел", "форма", "удержание",
        "смягчение", "отзыв", "запрет", "сила", "жесткость", "точность",
        "диагноз", "разоблачение", "структура", "дефект", "компенсация",
        "контроль", "вывод", "давление", "власть", "слабость",
    ]

    contrastive_markers = [
        "не ", " а ", "не потому", "а потому", "не о", "а о",
        "не как", "а как", "вместо", "однако", "но", "зато",
    ]

    negation_markers = [
        "не", "нет", "ни", "без", "нельзя", "недостаток",
        "отсутствие", "запрет", "невозможность",
    ]

    return {
        "output_char_len": len(output_text),
        "output_word_count": n_words,
        "output_token_count": simple_token_count(output_text),
        "sentence_count": n_sent,
        "avg_words_per_sentence": float(n_words / n_sent) if n_sent > 0 else 0.0,

        "diagnostic_keyword_count": count_substrings(output_text, diagnostic_keywords),
        "contrastive_marker_count": count_substrings(output_text, contrastive_markers),
        "negation_marker_count": count_substrings(output_text, negation_markers),

        "contains_mechanism": int("механизм" in output_text.lower()),
        "contains_regime": int("режим" in output_text.lower()),
        "contains_limit": int(("предел" in output_text.lower()) or ("огранич" in output_text.lower())),
        "contains_softening": int(("смягч" in output_text.lower()) or ("уступ" in output_text.lower())),
        "contains_form": int("форм" in output_text.lower()),
    }


# ====================== STEERING HOOK ======================

def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    """
    Steering через добавление decoder direction выбранной SAE feature
    на все позиции residual stream.

    patched = activation + scale * W_dec[feature]

    Это не ablation. Это именно steering.
    """
    sae = saes[real_layer]

    orig_dtype = activation.dtype
    act_float = activation.float()

    with torch.no_grad():
        f_idx = int(feature_index)

        if f_idx < 0 or f_idx >= sae.W_dec.shape[0]:
            return activation

        dec_vec = sae.W_dec[f_idx].to(
            device=act_float.device,
            dtype=act_float.dtype,
        )

        patched = act_float + scale * dec_vec

    return patched.to(dtype=orig_dtype)


def generate_safely(
    prompt,
    max_new_tokens=220,
    do_sample=False,
    temperature=0.0,
):
    """
    Совместимость с разными версиями TransformerLens generate().
    """
    kwargs_variants = []

    # Основной вариант
    if do_sample:
        kwargs_variants.append({
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "verbose": False,
        })
    else:
        kwargs_variants.append({
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "verbose": False,
        })

        kwargs_variants.append({
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "verbose": False,
        })

    # Последний fallback
    kwargs_variants.append({
        "max_new_tokens": max_new_tokens,
        "verbose": False,
    })

    last_error = None

    for kwargs in kwargs_variants:
        try:
            return model.generate(prompt, **kwargs)
        except TypeError as e:
            last_error = e
            continue

    raise last_error


def generate_with_feature_steering(
    prompt,
    real_layer,
    feature_index,
    scale,
    max_new_tokens=220,
    do_sample=False,
    temperature=0.0,
):
    hook_name = f"blocks.{real_layer}.hook_resid_post"

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(
            activation=act,
            hook=hook,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    with torch.no_grad():
        if scale == 0.0:
            out = generate_safely(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        else:
            with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                out = generate_safely(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )

    return out


# ====================== MAIN RUN ======================

base_text_sha = sha256_text(BASE_TEXT)
base_text_preview = safe_preview(BASE_TEXT, n=1000)

with open(OUTPUT_BASE_TEXT_TXT, "w", encoding="utf-8") as f:
    f.write(BASE_TEXT)

print(f"BASE_TEXT sha256: {base_text_sha}")
print(f"BASE_TEXT сохранён в: {OUTPUT_BASE_TEXT_TXT}")
print(f"Длина BASE_TEXT chars: {len(BASE_TEXT)}")
print(f"Длина BASE_TEXT tokens: {simple_token_count(BASE_TEXT)}")

steering_rows = []

run_started_at = datetime.now().isoformat(timespec="seconds")

total_runs = (
    len(TEST_TASKS)
    * len(STEERING_FEATURES)
    * len(STEERING_SCALES)
    * sum(m["n_samples"] for m in GENERATION_MODES)
)

run_counter = 0

for task_id, task in enumerate(TEST_TASKS):
    task_clean = task.strip()
    full_prompt = build_analysis_prompt(BASE_TEXT, task_clean)

    print("\n\n############################")
    print(f"### TASK {task_id}")
    print("############################")
    print(task_clean)

    for feature_cfg in STEERING_FEATURES:
        real_layer = int(feature_cfg["real_layer"])
        feature_index = int(feature_cfg["feature_index"])
        feature_label = feature_cfg["feature_label"]
        feature_comment = feature_cfg["comment"]

        if real_layer not in saes:
            print(f"SKIP: SAE для layer {real_layer} не загружен")
            continue

        for mode_cfg in GENERATION_MODES:
            generation_mode = mode_cfg["generation_mode"]
            do_sample = bool(mode_cfg["do_sample"])
            temperature = float(mode_cfg["temperature"])
            n_samples = int(mode_cfg["n_samples"])

            for sample_id in range(n_samples):
                seed = make_seed(
                    task_id=task_id,
                    real_layer=real_layer,
                    feature_index=feature_index,
                    sample_id=sample_id,
                    generation_mode=generation_mode,
                )

                for scale in STEERING_SCALES:
                    run_counter += 1

                    print(
                        f"\n=== RUN {run_counter}/{total_runs} | "
                        f"TASK {task_id} | FEATURE {real_layer}/{feature_index} | "
                        f"{generation_mode} | SAMPLE {sample_id} | SCALE {scale} ==="
                    )

                    set_reproducible_seed(seed)

                    started = time.time()
                    error = ""

                    try:
                        output_raw = generate_with_feature_steering(
                            prompt=full_prompt,
                            real_layer=real_layer,
                            feature_index=feature_index,
                            scale=scale,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=do_sample,
                            temperature=temperature,
                        )

                        output_text = strip_prompt_from_generation(output_raw, full_prompt)

                    except Exception as e:
                        output_raw = ""
                        output_text = ""
                        error = repr(e)
                        print(f"ERROR: {error}")

                    elapsed_sec = time.time() - started

                    if output_text:
                        print(output_text)

                    metrics = compute_text_metrics(output_text)

                    row = {
                        "run_started_at": run_started_at,
                        "row_created_at": datetime.now().isoformat(timespec="seconds"),

                        "task_id": task_id,
                        "task": task_clean,

                        "base_text_id": "prompts_target_0",
                        "base_text_sha256": base_text_sha,
                        "base_text_preview": base_text_preview,
                        "base_text_full": BASE_TEXT if INCLUDE_BASE_TEXT_FULL_IN_CSV else "",

                        "real_layer": real_layer,
                        "feature_index": feature_index,
                        "feature_label": feature_label,
                        "feature_comment": feature_comment,

                        "scale": float(scale),

                        "generation_mode": generation_mode,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "sample_id": sample_id,
                        "seed": seed,
                        "max_new_tokens": MAX_NEW_TOKENS,

                        "prompt_char_len": len(full_prompt),
                        "prompt_token_count": simple_token_count(full_prompt),

                        "elapsed_sec": elapsed_sec,
                        "error": error,

                        "output": output_text,
                        "output_raw": str(output_raw),
                    }

                    row.update(metrics)
                    steering_rows.append(row)

                    # Инкрементально сохраняем после каждой строки.
                    # Если Colab отвалится, данные не пропадут.
                    temp_df = pd.DataFrame(steering_rows)
                    temp_df.to_csv(OUTPUT_CSV, index=False)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


# ====================== BASELINE COMPARISON ======================

steering_df = pd.DataFrame(steering_rows)

# Сравниваем каждую строку со scale == 0 внутри:
# task_id + feature + generation_mode + sample_id
baseline_cols = [
    "task_id",
    "real_layer",
    "feature_index",
    "generation_mode",
    "sample_id",
]

baseline_df = steering_df[steering_df["scale"] == 0.0].copy()

baseline_df = baseline_df[
    baseline_cols
    + [
        "output",
        "output_char_len",
        "output_word_count",
        "output_token_count",
        "diagnostic_keyword_count",
        "contrastive_marker_count",
        "negation_marker_count",
    ]
].rename(columns={
    "output": "baseline_scale0_output",
    "output_char_len": "baseline_output_char_len",
    "output_word_count": "baseline_output_word_count",
    "output_token_count": "baseline_output_token_count",
    "diagnostic_keyword_count": "baseline_diagnostic_keyword_count",
    "contrastive_marker_count": "baseline_contrastive_marker_count",
    "negation_marker_count": "baseline_negation_marker_count",
})

steering_df = steering_df.merge(
    baseline_df,
    on=baseline_cols,
    how="left",
)

steering_df["exact_match_to_scale0"] = (
    steering_df["output"].fillna("") == steering_df["baseline_scale0_output"].fillna("")
).astype(int)

steering_df["jaccard_similarity_to_scale0"] = steering_df.apply(
    lambda r: jaccard_similarity(r["output"], r["baseline_scale0_output"]),
    axis=1,
)

steering_df["delta_char_len_vs_scale0"] = (
    steering_df["output_char_len"] - steering_df["baseline_output_char_len"]
)

steering_df["delta_word_count_vs_scale0"] = (
    steering_df["output_word_count"] - steering_df["baseline_output_word_count"]
)

steering_df["delta_token_count_vs_scale0"] = (
    steering_df["output_token_count"] - steering_df["baseline_output_token_count"]
)

steering_df["delta_diagnostic_keywords_vs_scale0"] = (
    steering_df["diagnostic_keyword_count"] - steering_df["baseline_diagnostic_keyword_count"]
)

steering_df["delta_contrastive_markers_vs_scale0"] = (
    steering_df["contrastive_marker_count"] - steering_df["baseline_contrastive_marker_count"]
)

steering_df["delta_negation_markers_vs_scale0"] = (
    steering_df["negation_marker_count"] - steering_df["baseline_negation_marker_count"]
)

# Финальное сохранение
steering_df.to_csv(OUTPUT_CSV, index=False)

print("\n============================================================")
print("ГОТОВО")
print("============================================================")
print(f"Основная таблица: {OUTPUT_CSV}")
print(f"BASE_TEXT отдельно: {OUTPUT_BASE_TEXT_TXT}")
print(f"Строк в таблице: {len(steering_df)}")

print("\n=== КРАТКАЯ СВОДКА ПО FEATURE/SCALE/MODE ===")

summary = steering_df.groupby(
    ["real_layer", "feature_index", "feature_label", "generation_mode", "scale"],
    as_index=False
).agg(
    rows=("output", "count"),
    exact_match_rate_to_scale0=("exact_match_to_scale0", "mean"),
    mean_jaccard_to_scale0=("jaccard_similarity_to_scale0", "mean"),
    mean_output_token_count=("output_token_count", "mean"),
    mean_delta_token_count_vs_scale0=("delta_token_count_vs_scale0", "mean"),
    mean_diagnostic_keyword_count=("diagnostic_keyword_count", "mean"),
    mean_delta_diagnostic_keywords_vs_scale0=("delta_diagnostic_keywords_vs_scale0", "mean"),
    mean_contrastive_marker_count=("contrastive_marker_count", "mean"),
    mean_delta_contrastive_markers_vs_scale0=("delta_contrastive_markers_vs_scale0", "mean"),
    mean_negation_marker_count=("negation_marker_count", "mean"),
    mean_delta_negation_markers_vs_scale0=("delta_negation_markers_vs_scale0", "mean"),
    mean_elapsed_sec=("elapsed_sec", "mean"),
)

print(summary.to_string(index=False))

summary.to_csv("sae_feature_steering_generation_summary_metrics.csv", index=False)
print("\nСводка сохранена в sae_feature_steering_generation_summary_metrics.csv")