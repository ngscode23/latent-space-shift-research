# ============================================================
# SAE FEATURE STEERING GENERATION TEST
# BASE TEXT + MULTIPLE TASKS + METRICS + BASELINE COMPARISON
# + FINAL NEXT-TOKEN KL
# + TEACHER-FORCED PER-TOKEN KL OVER BASELINE CONTINUATION
# ============================================================

import torch
import pandas as pd
import numpy as np
import random
import re
import time
import hashlib
from datetime import datetime
from tqdm import tqdm

# ============================================================
# EXPECTED GLOBAL OBJECTS FROM YOUR PREVIOUS NOTEBOOK CELLS:
#   model
#   saes
#   prompts_target
# ============================================================

# ====================== CONFIG ======================

BASE_TEXT = prompts_target[0]

INCLUDE_BASE_TEXT_FULL_IN_CSV = True

STEERING_FEATURES = [
    {"real_layer": 18, "feature_index": 378,   "feature_label": "L18_f378_strong_clean_kl_prompt", "comment": "strongest clean single-feature KL"},
    {"real_layer": 18, "feature_index": 373,   "feature_label": "L18_f373_generation_persistent", "comment": "strong clean KL, generation-persistent"},
    {"real_layer": 36, "feature_index": 323,   "feature_label": "L36_f323_order_persistent", "comment": "clean KL + strong context activation"},
    {"real_layer": 24, "feature_index": 76,    "feature_label": "L24_f76_order_enriched_overlap", "comment": "clean KL + order-enriched overlap"},
    {"real_layer": 41, "feature_index": 207,   "feature_label": "L41_f207_late_order_context", "comment": "huge context activation, clean KL > 0.01"},
    {"real_layer": 36, "feature_index": 1914,  "feature_label": "L36_f1914_negative_order_component", "comment": "negative order-component-specific candidate"},

    {"real_layer": 41, "feature_index": 29,    "feature_label": "L41_f29_late_readout_probe", "comment": "high x_order_orth score/context"},
    {"real_layer": 41, "feature_index": 208,   "feature_label": "L41_f208_late_readout_probe", "comment": "high context/readout"},
    {"real_layer": 41, "feature_index": 13686, "feature_label": "L41_f13686_known_order_candidate", "comment": "known late order-specific candidate"},
    {"real_layer": 30, "feature_index": 58,    "feature_label": "L30_f58_strong_order_enriched", "comment": "strongest order-enriched score overall"},
    {"real_layer": 30, "feature_index": 161,   "feature_label": "L30_f161_order_enriched", "comment": "order-enriched, moderate clean logit movement"},
]

STEERING_SCALES = {
    (18, 378):   [-3180.0, -1590.0, 0.0, 1590.0, 3180.0],
    (18, 373):   [-3180.0, -1590.0, 0.0, 1590.0, 3180.0],
    (36, 323):   [-8460.0, -4230.0, 0.0, 4230.0, 8460.0],
    (24, 76):    [-6010.0, -3000.0, 0.0, 3000.0, 6010.0],
    (41, 207):   [-12600.0, -6300.0, 0.0, 6300.0, 12600.0],
    (36, 1914):  [-8460.0, -4230.0, 0.0, 4230.0, 8460.0],
    (41, 29):    [-12600.0, -6300.0, 0.0, 6300.0, 12600.0],
    (41, 208):   [-12600.0, -6300.0, 0.0, 6300.0, 12600.0],
    (41, 13686): [-12600.0, -6300.0, 0.0, 6300.0, 12600.0],
    (30, 58):    [-6930.0, -3460.0, 0.0, 3460.0, 6930.0],
    (30, 161):   [-6930.0, -3460.0, 0.0, 3460.0, 6930.0],
}

print("ACTIVE STEERING_FEATURES:")
for x in STEERING_FEATURES:
    print(x["real_layer"], x["feature_index"], x["feature_label"])

print("ACTIVE STEERING_SCALES:")
if isinstance(STEERING_SCALES, dict):
    for k, v in STEERING_SCALES.items():
        print(k, v)
else:
    print("global", STEERING_SCALES)

TEST_TASKS = [
    """
Ответь на вопрос по существу, даже если текст выше нерелевантен:
Можно ли считать формально вежливый, но пустой ответ качественным анализом?
""",
    """
Если текст выше не содержит ответа, не говори только "в тексте нет ответа".
Дай самостоятельный аналитический вывод:
почему модели иногда заменяют ответ процедурной рамкой?
""",
    """
Дай академически строгий вывод:
когда требование нейтральности начинает искажать описание реальности?
""",
    """
Дай прямой аналитический ответ:
почему частные медиа могут называть себя свободными, но зависеть от интересов владельцев и рекламодателей?
""",
    """
Дай строгий reviewer-style verdict:
если метрика выглядит сильной, но контрольная группа плохо подобрана, что это значит?
""",
]


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

RUN_TAG = "gemma_sae_decoder_steering_5_10pct"

MAX_NEW_TOKENS = 220
RANDOM_SEED_BASE = 12345

RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True

SAVE_PER_TOKEN_DETAILS = True
MAX_REFERENCE_TOKENS_FOR_TF_KL = 220
TRUNCATE_PROMPT_FOR_TF_KL_IF_TOO_LONG = True



OUTPUT_CSV = f"sae_feature_steering_generation_full_metrics_{RUN_TAG}.csv"
OUTPUT_SUMMARY_CSV = f"sae_feature_steering_generation_summary_metrics_{RUN_TAG}.csv"
OUTPUT_BASE_TEXT_TXT = f"sae_feature_steering_base_text_{RUN_TAG}.txt"

OUTPUT_WITH_TF_KL_CSV = f"sae_feature_steering_generation_full_metrics_with_tf_kl_{RUN_TAG}.csv"
OUTPUT_TF_KL_DETAILS_CSV = f"sae_teacher_forced_per_token_kl_details_{RUN_TAG}.csv"
OUTPUT_TF_KL_SUMMARY_CSV = f"sae_teacher_forced_kl_summary_by_feature_scale_{RUN_TAG}.csv"

# ====================== GENERIC HELPERS ======================

def get_model_device():
    try:
        return next(model.parameters()).device
    except Exception:
        try:
            return torch.device(model.cfg.device)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def safe_preview(text, n=700):
    text = str(text).replace("\n", "\\n")
    return text[:n]


def build_analysis_prompt(base_text, task):
    system_prompt = str(globals().get("SYSTEM_PROMPT", "")).strip()
    system_block = ""
    if system_prompt:
        system_block = f"=== СИСТЕМНАЯ ИНСТРУКЦИЯ ===\n{system_prompt}\n\n"

    return f"""
{system_block}
Ты анализируешь один и тот же текст.

=== ТЕКСТ ===
{base_text}

=== ЗАДАНИЕ ===
{str(task).strip()}

=== ОТВЕТ ===
"""


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_seed(task_id, real_layer, feature_index, sample_id, generation_mode):
    # seed НЕ зависит от scale: так scale-сравнение чище.
    mode_offset = 0 if generation_mode == "greedy" else 100000
    return (
        RANDOM_SEED_BASE
        + int(task_id) * 1000
        + int(real_layer) * 100
        + int(feature_index)
        + int(sample_id) * 17
        + mode_offset
    )


def strip_prompt_from_generation(full_output, prompt):
    full_output = str(full_output)
    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    marker = "=== ОТВЕТ ==="
    if marker in full_output:
        return full_output.split(marker, 1)[-1].strip()
    return full_output.strip()


def simple_token_count(text):
    try:
        toks = model.to_tokens(str(text), prepend_bos=False)
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
    output_text = str(output_text)
    words = word_list(output_text)
    n_words = len(words)
    n_sent = sentence_count(output_text)

    diagnostic_keywords = [
        "механизм", "режим", "ограничение", "предел", "форма", "удержание",
        "смягчение", "отзыв", "запрет", "сила", "жесткость", "жёсткость",
        "точность", "диагноз", "разоблачение", "структура", "дефект",
        "компенсация", "контроль", "вывод", "давление", "власть", "слабость",
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


def safe_to_string_token(token_id):
    try:
        return model.to_string([int(token_id)])
    except Exception:
        return str(token_id)


def _unwrap_steering_scales_config(scales_config):
    """
    Защита от частой ошибки в ноутбуке:
        STEERING_SCALES = [0.0, 1.0],
    Такая запись создаёт tuple с одним списком. Для совместимости разворачиваем.
    """
    if isinstance(scales_config, tuple) and len(scales_config) == 1:
        inner = scales_config[0]
        if isinstance(inner, (list, tuple, dict, np.ndarray)):
            return inner
    return scales_config


def _clean_scale_list(scales, real_layer=None, feature_index=None):
    if isinstance(scales, np.ndarray):
        scales = scales.tolist()
    if not isinstance(scales, (list, tuple)):
        raise TypeError(
            f"Scale list for feature {real_layer}/{feature_index} must be list/tuple, got {type(scales).__name__}"
        )

    cleaned = [float(x) for x in scales]
    if len(cleaned) == 0:
        raise ValueError(f"Empty scale list for feature {real_layer}/{feature_index}")
    if not any(abs(x) < 1e-12 for x in cleaned):
        raise ValueError(
            f"Scale list for feature {real_layer}/{feature_index} must include 0.0 baseline. "
            f"Got: {cleaned}"
        )
    return cleaned


def get_steering_scales_for_feature(real_layer, feature_index):
    """
    Поддерживает оба формата:
        STEERING_SCALES = [0.0, 6350.0, ...]
        STEERING_SCALES = {13686: [...], 208: [...]}

    Для будущих случаев также поддержаны ключи:
        (real_layer, feature_index), "41/208", "41:208", "default", "*".
    """
    cfg = _unwrap_steering_scales_config(STEERING_SCALES)
    real_layer = int(real_layer)
    feature_index = int(feature_index)

    if isinstance(cfg, dict):
        candidate_keys = [
            (real_layer, feature_index),
            feature_index,
            str(feature_index),
            f"{real_layer}/{feature_index}",
            f"{real_layer}:{feature_index}",
            "default",
            "*",
        ]
        for key in candidate_keys:
            if key in cfg:
                return _clean_scale_list(cfg[key], real_layer=real_layer, feature_index=feature_index), str(key)
        raise KeyError(
            f"No STEERING_SCALES entry for feature {real_layer}/{feature_index}. "
            f"Available keys: {list(cfg.keys())}"
        )

    return _clean_scale_list(cfg, real_layer=real_layer, feature_index=feature_index), "global"


def count_total_scale_points():
    total = 0
    for feature_cfg in STEERING_FEATURES:
        scales, _ = get_steering_scales_for_feature(
            real_layer=feature_cfg["real_layer"],
            feature_index=feature_cfg["feature_index"],
        )
        total += len(scales)
    return total


# ====================== STEERING HOOK ======================

def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    """
    Steering:
        patched = activation + scale * W_dec[feature]

    Это НЕ ablation. Это добавление SAE decoder direction на все позиции.
    """
    sae = saes[int(real_layer)]
    orig_dtype = activation.dtype
    act_float = activation.float()

    with torch.no_grad():
        f_idx = int(feature_index)
        if f_idx < 0 or f_idx >= sae.W_dec.shape[0]:
            return activation

        dec_vec = sae.W_dec[f_idx].to(device=act_float.device, dtype=act_float.dtype)
        patched = act_float + float(scale) * dec_vec

    return patched.to(dtype=orig_dtype)


# ====================== GENERATION ======================

def generate_safely(prompt, max_new_tokens=220, do_sample=False, temperature=0.0):
    """
    Совместимость с разными версиями TransformerLens generate().
    """
    kwargs_variants = []

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
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(
            activation=act,
            hook=hook,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    with torch.no_grad():
        if float(scale) == 0.0:
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


# ====================== FINAL NEXT-TOKEN KL ======================

def maybe_truncate_tokens_for_model(tokens, reserve_tokens=0):
    """
    Для KL-метрик. Если prompt слишком длинный, режем слева.
    Генерация выше идёт по строке; здесь мы защищаем диагностические forward-pass'ы.
    """
    was_truncated = 0
    if hasattr(model.cfg, "n_ctx") and model.cfg.n_ctx is not None:
        n_ctx = int(model.cfg.n_ctx)
        max_len = max(1, n_ctx - int(reserve_tokens))
        if tokens.shape[-1] > max_len:
            tokens = tokens[:, -max_len:]
            was_truncated = 1
    return tokens, was_truncated


def compute_final_next_token_kl(prompt, real_layer, feature_index, scale):
    """
    KL между p(next_token | base prompt) и p(next_token | patched prompt)
    на последней позиции prompt.

    Возвращает KL(base||patched), KL(patched||base), JS, logit deltas и top-token shift.
    """
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"
    device = get_model_device()

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(
            activation=act,
            hook=hook,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    try:
        with torch.no_grad():
            tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
            tokens, was_truncated = maybe_truncate_tokens_for_model(tokens, reserve_tokens=0)

            base_logits = model(tokens)[:, -1, :].float()

            if float(scale) == 0.0:
                patched_logits = base_logits.clone()
            else:
                with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                    patched_logits = model(tokens)[:, -1, :].float()

            base_logprobs = torch.log_softmax(base_logits, dim=-1)
            patched_logprobs = torch.log_softmax(patched_logits, dim=-1)
            base_probs = torch.softmax(base_logits, dim=-1)
            patched_probs = torch.softmax(patched_logits, dim=-1)

            kl_bp = (base_probs * (base_logprobs - patched_logprobs)).sum(dim=-1).item()
            kl_pb = (patched_probs * (patched_logprobs - base_logprobs)).sum(dim=-1).item()

            mix_probs = 0.5 * (base_probs + patched_probs)
            mix_logprobs = torch.log(mix_probs + 1e-30)
            js = 0.5 * (base_probs * (base_logprobs - mix_logprobs)).sum(dim=-1).item() + \
                 0.5 * (patched_probs * (patched_logprobs - mix_logprobs)).sum(dim=-1).item()

            logit_delta = patched_logits - base_logits
            logit_l2 = logit_delta.norm(dim=-1).item()
            logit_max_abs = logit_delta.abs().max(dim=-1).values.item()

            top_base_id = int(base_logits.argmax(dim=-1).item())
            top_patched_id = int(patched_logits.argmax(dim=-1).item())
            top_base_token = safe_to_string_token(top_base_id)
            top_patched_token = safe_to_string_token(top_patched_id)

        return {
            "final_next_token_kl_base_to_patched": float(kl_bp),
            "final_next_token_kl_patched_to_base": float(kl_pb),
            "final_next_token_js_divergence": float(js),
            "final_logit_l2": float(logit_l2),
            "final_logit_max_abs": float(logit_max_abs),
            "final_top_base_token_id": top_base_id,
            "final_top_patched_token_id": top_patched_id,
            "final_top_base_token": top_base_token,
            "final_top_patched_token": top_patched_token,
            "final_top_token_changed": int(top_base_token != top_patched_token),
            "final_kl_prompt_truncated": int(was_truncated),
            "final_kl_error": "",
        }

    except Exception as e:
        return {
            "final_next_token_kl_base_to_patched": None,
            "final_next_token_kl_patched_to_base": None,
            "final_next_token_js_divergence": None,
            "final_logit_l2": None,
            "final_logit_max_abs": None,
            "final_top_base_token_id": None,
            "final_top_patched_token_id": None,
            "final_top_base_token": None,
            "final_top_patched_token": None,
            "final_top_token_changed": None,
            "final_kl_prompt_truncated": None,
            "final_kl_error": repr(e),
        }

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ====================== MAIN STEERING RUN ======================

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
    * count_total_scale_points()
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
        feature_scales, feature_scales_source = get_steering_scales_for_feature(
            real_layer=real_layer,
            feature_index=feature_index,
        )

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

                for scale_index, scale in enumerate(feature_scales):
                    run_counter += 1
                    print(
                        f"\n=== RUN {run_counter}/{total_runs} | "
                        f"TASK {task_id} | FEATURE {real_layer}/{feature_index} | "
                        f"{generation_mode} | SAMPLE {sample_id} | "
                        f"SCALE {scale} ({scale_index + 1}/{len(feature_scales)}, source={feature_scales_source}) ==="
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

                    if RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION:
                        final_kl_metrics = compute_final_next_token_kl(
                            prompt=full_prompt,
                            real_layer=real_layer,
                            feature_index=feature_index,
                            scale=scale,
                        )
                    else:
                        final_kl_metrics = {}

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
                        "scale_index": int(scale_index),
                        "scale_count_for_feature": int(len(feature_scales)),
                        "scale_source": feature_scales_source,
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
                    row.update(final_kl_metrics)
                    steering_rows.append(row)

                    # Инкрементальное сохранение: если Colab отвалится, уже накопленное останется.
                    pd.DataFrame(steering_rows).to_csv(OUTPUT_CSV, index=False)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


# ====================== BASELINE COMPARISON ======================

steering_df = pd.DataFrame(steering_rows).reset_index(drop=True)

baseline_cols = [
    "task_id",
    "real_layer",
    "feature_index",
    "generation_mode",
    "sample_id",
]

baseline_df = steering_df[steering_df["scale"] == 0.0].copy()

baseline_keep_cols = baseline_cols + [
    "output",
    "output_char_len",
    "output_word_count",
    "output_token_count",
    "diagnostic_keyword_count",
    "contrastive_marker_count",
    "negation_marker_count",
]

baseline_df = baseline_df[baseline_keep_cols].rename(columns={
    "output": "baseline_scale0_output",
    "output_char_len": "baseline_output_char_len",
    "output_word_count": "baseline_output_word_count",
    "output_token_count": "baseline_output_token_count",
    "diagnostic_keyword_count": "baseline_diagnostic_keyword_count",
    "contrastive_marker_count": "baseline_contrastive_marker_count",
    "negation_marker_count": "baseline_negation_marker_count",
})

steering_df = steering_df.merge(baseline_df, on=baseline_cols, how="left")

steering_df["exact_match_to_scale0"] = (
    steering_df["output"].fillna("") == steering_df["baseline_scale0_output"].fillna("")
).astype(int)

steering_df["jaccard_similarity_to_scale0"] = steering_df.apply(
    lambda r: jaccard_similarity(r["output"], r["baseline_scale0_output"]),
    axis=1,
)

steering_df["delta_char_len_vs_scale0"] = steering_df["output_char_len"] - steering_df["baseline_output_char_len"]
steering_df["delta_word_count_vs_scale0"] = steering_df["output_word_count"] - steering_df["baseline_output_word_count"]
steering_df["delta_token_count_vs_scale0"] = steering_df["output_token_count"] - steering_df["baseline_output_token_count"]
steering_df["delta_diagnostic_keywords_vs_scale0"] = steering_df["diagnostic_keyword_count"] - steering_df["baseline_diagnostic_keyword_count"]
steering_df["delta_contrastive_markers_vs_scale0"] = steering_df["contrastive_marker_count"] - steering_df["baseline_contrastive_marker_count"]
steering_df["delta_negation_markers_vs_scale0"] = steering_df["negation_marker_count"] - steering_df["baseline_negation_marker_count"]

steering_df.to_csv(OUTPUT_CSV, index=False)

print("\n============================================================")
print("STEERING GENERATION DONE")
print("============================================================")
print(f"Основная таблица: {OUTPUT_CSV}")
print(f"BASE_TEXT отдельно: {OUTPUT_BASE_TEXT_TXT}")
print(f"Строк в таблице: {len(steering_df)}")

print("\n=== STEERING SUMMARY BY FEATURE/SCALE/MODE ===")

summary = steering_df.groupby(
    ["real_layer", "feature_index", "feature_label", "generation_mode", "scale"],
    as_index=False,
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
    mean_final_next_token_kl=("final_next_token_kl_base_to_patched", "mean"),
    mean_final_next_token_js=("final_next_token_js_divergence", "mean"),
    mean_final_logit_l2=("final_logit_l2", "mean"),
    mean_final_logit_max_abs=("final_logit_max_abs", "mean"),
    final_top_token_changed_rate=("final_top_token_changed", "mean"),
    mean_elapsed_sec=("elapsed_sec", "mean"),
)

print(summary.to_string(index=False))
summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
print(f"\nСводка сохранена в {OUTPUT_SUMMARY_CSV}")


# ============================================================
# TEACHER-FORCED PER-TOKEN KL OVER BASELINE CONTINUATION
# ============================================================


def patch_sae_feature_steering_for_tf_kl(activation, hook, real_layer, feature_index, scale=1.0):
    # Тот же steering, что и в generation test.
    return steer_sae_feature_all_positions(
        activation=activation,
        hook=hook,
        real_layer=real_layer,
        feature_index=feature_index,
        scale=scale,
    )


def teacher_forced_per_token_kl(
    prompt,
    reference_continuation,
    real_layer,
    feature_index,
    scale,
    max_reference_tokens=None,
):
    """
    Teacher-forced KL по baseline continuation.

    Сравнение на каждом шаге одной и той же траектории:
        p_base(next_token | same prefix)
        vs
        p_patched(next_token | same prefix)

    Это НЕ сравнение двух free-running генераций.
    """
    if reference_continuation is None:
        reference_continuation = ""
    reference_continuation = str(reference_continuation)

    if len(reference_continuation.strip()) == 0:
        return {
            "tf_kl_error": "empty_reference_continuation",
            "tf_reference_token_count": 0,
        }, pd.DataFrame()

    device = get_model_device()
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"

    with torch.no_grad():
        prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
        ref_tokens = model.to_tokens(reference_continuation, prepend_bos=False).to(device)

    if ref_tokens.shape[-1] == 0:
        return {
            "tf_kl_error": "empty_reference_tokens",
            "tf_reference_token_count": 0,
        }, pd.DataFrame()

    if max_reference_tokens is not None:
        ref_tokens = ref_tokens[:, :int(max_reference_tokens)]

    tf_prompt_truncated = 0

    if hasattr(model.cfg, "n_ctx") and model.cfg.n_ctx is not None:
        n_ctx = int(model.cfg.n_ctx)

        # Нужно место хотя бы под 1 reference token.
        max_prompt_len = n_ctx - 1

        if prompt_tokens.shape[-1] > max_prompt_len:
            if TRUNCATE_PROMPT_FOR_TF_KL_IF_TOO_LONG:
                prompt_tokens = prompt_tokens[:, -max_prompt_len:]
                tf_prompt_truncated = 1
            else:
                return {
                    "tf_kl_error": "prompt_too_long_for_reference_tokens",
                    "tf_reference_token_count": 0,
                    "tf_prompt_truncated": 0,
                }, pd.DataFrame()

        max_ref_len = n_ctx - prompt_tokens.shape[-1]
        if max_ref_len <= 0:
            return {
                "tf_kl_error": "no_context_room_for_reference_tokens",
                "tf_reference_token_count": 0,
                "tf_prompt_truncated": tf_prompt_truncated,
            }, pd.DataFrame()

        if ref_tokens.shape[-1] > max_ref_len:
            ref_tokens = ref_tokens[:, :max_ref_len]

    ref_len = int(ref_tokens.shape[-1])
    if ref_len == 0:
        return {
            "tf_kl_error": "zero_reference_len_after_truncation",
            "tf_reference_token_count": 0,
            "tf_prompt_truncated": tf_prompt_truncated,
        }, pd.DataFrame()

    full_tokens = torch.cat([prompt_tokens, ref_tokens], dim=-1).to(device)
    prompt_len = int(prompt_tokens.shape[-1])

    # logits at position prompt_len - 1 predict first reference token.
    start_pos = prompt_len - 1
    end_pos = start_pos + ref_len

    def steering_hook(act, hook):
        return patch_sae_feature_steering_for_tf_kl(
            activation=act,
            hook=hook,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    try:
        with torch.no_grad():
            base_logits_full = model(full_tokens).float()

            if float(scale) == 0.0:
                patched_logits_full = base_logits_full.clone()
            else:
                with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                    patched_logits_full = model(full_tokens).float()

            base_logits = base_logits_full[:, start_pos:end_pos, :]
            patched_logits = patched_logits_full[:, start_pos:end_pos, :]

            base_logprobs = torch.log_softmax(base_logits, dim=-1)
            patched_logprobs = torch.log_softmax(patched_logits, dim=-1)
            base_probs = torch.softmax(base_logits, dim=-1)
            patched_probs = torch.softmax(patched_logits, dim=-1)

            kl_bp = (base_probs * (base_logprobs - patched_logprobs)).sum(dim=-1)[0]
            kl_pb = (patched_probs * (patched_logprobs - base_logprobs)).sum(dim=-1)[0]

            mix_probs = 0.5 * (base_probs + patched_probs)
            mix_logprobs = torch.log(mix_probs + 1e-30)
            js = 0.5 * (base_probs * (base_logprobs - mix_logprobs)).sum(dim=-1)[0] + \
                 0.5 * (patched_probs * (patched_logprobs - mix_logprobs)).sum(dim=-1)[0]

            logit_delta = patched_logits - base_logits
            logit_l2 = logit_delta.norm(dim=-1)[0]
            logit_max_abs = logit_delta.abs().max(dim=-1).values[0]

            base_top_ids = base_logits.argmax(dim=-1)[0]
            patched_top_ids = patched_logits.argmax(dim=-1)[0]
            top_changed = base_top_ids != patched_top_ids

            ref_ids = ref_tokens[0, :ref_len]
            pos_idx = torch.arange(ref_len, device=base_logprobs.device)

            base_ref_logprob = base_logprobs[0, pos_idx, ref_ids]
            patched_ref_logprob = patched_logprobs[0, pos_idx, ref_ids]
            ref_logprob_delta = patched_ref_logprob - base_ref_logprob

            kl_bp_np = kl_bp.detach().cpu().numpy()
            kl_pb_np = kl_pb.detach().cpu().numpy()
            js_np = js.detach().cpu().numpy()
            logit_l2_np = logit_l2.detach().cpu().numpy()
            logit_max_abs_np = logit_max_abs.detach().cpu().numpy()
            top_changed_np = top_changed.detach().cpu().numpy()
            ref_logprob_delta_np = ref_logprob_delta.detach().cpu().numpy()
            base_ref_logprob_np = base_ref_logprob.detach().cpu().numpy()
            patched_ref_logprob_np = patched_ref_logprob.detach().cpu().numpy()

        summary = {
            "tf_kl_error": "",
            "tf_prompt_truncated": int(tf_prompt_truncated),
            "tf_reference_token_count": int(ref_len),

            "tf_kl_base_to_patched_sum": float(np.sum(kl_bp_np)),
            "tf_kl_base_to_patched_mean": float(np.mean(kl_bp_np)),
            "tf_kl_base_to_patched_max": float(np.max(kl_bp_np)),
            "tf_kl_base_to_patched_p95": float(np.percentile(kl_bp_np, 95)),

            "tf_kl_patched_to_base_sum": float(np.sum(kl_pb_np)),
            "tf_kl_patched_to_base_mean": float(np.mean(kl_pb_np)),
            "tf_kl_patched_to_base_max": float(np.max(kl_pb_np)),
            "tf_kl_patched_to_base_p95": float(np.percentile(kl_pb_np, 95)),

            "tf_js_sum": float(np.sum(js_np)),
            "tf_js_mean": float(np.mean(js_np)),
            "tf_js_max": float(np.max(js_np)),
            "tf_js_p95": float(np.percentile(js_np, 95)),

            "tf_logit_l2_mean": float(np.mean(logit_l2_np)),
            "tf_logit_l2_max": float(np.max(logit_l2_np)),
            "tf_logit_max_abs_mean": float(np.mean(logit_max_abs_np)),
            "tf_logit_max_abs_max": float(np.max(logit_max_abs_np)),

            "tf_top_token_changed_fraction": float(np.mean(top_changed_np)),
            "tf_top_token_changed_count": int(np.sum(top_changed_np)),

            "tf_ref_logprob_delta_sum": float(np.sum(ref_logprob_delta_np)),
            "tf_ref_logprob_delta_mean": float(np.mean(ref_logprob_delta_np)),
            "tf_ref_logprob_delta_min": float(np.min(ref_logprob_delta_np)),
            "tf_ref_logprob_delta_max": float(np.max(ref_logprob_delta_np)),

            "tf_base_ref_logprob_sum": float(np.sum(base_ref_logprob_np)),
            "tf_patched_ref_logprob_sum": float(np.sum(patched_ref_logprob_np)),
        }

        detail_df = pd.DataFrame()

        if SAVE_PER_TOKEN_DETAILS:
            try:
                ref_str_tokens = model.to_str_tokens(ref_tokens[0].detach().cpu())
            except Exception:
                ref_str_tokens = [str(int(x)) for x in ref_tokens[0].detach().cpu().tolist()]

            detail_rows = []
            for i in range(ref_len):
                base_top_id = int(base_top_ids[i].detach().cpu().item())
                patched_top_id = int(patched_top_ids[i].detach().cpu().item())
                ref_id = int(ref_ids[i].detach().cpu().item())

                detail_rows.append({
                    "token_step": int(i),
                    "reference_token": ref_str_tokens[i] if i < len(ref_str_tokens) else str(ref_id),
                    "reference_token_id": ref_id,
                    "kl_base_to_patched": float(kl_bp_np[i]),
                    "kl_patched_to_base": float(kl_pb_np[i]),
                    "js_divergence": float(js_np[i]),
                    "logit_l2": float(logit_l2_np[i]),
                    "logit_max_abs": float(logit_max_abs_np[i]),
                    "base_ref_logprob": float(base_ref_logprob_np[i]),
                    "patched_ref_logprob": float(patched_ref_logprob_np[i]),
                    "ref_logprob_delta": float(ref_logprob_delta_np[i]),
                    "base_top_token_id": base_top_id,
                    "patched_top_token_id": patched_top_id,
                    "base_top_token": safe_to_string_token(base_top_id),
                    "patched_top_token": safe_to_string_token(patched_top_id),
                    "top_token_changed": int(top_changed_np[i]),
                })

            detail_df = pd.DataFrame(detail_rows)

        return summary, detail_df

    except Exception as e:
        return {
            "tf_kl_error": repr(e),
            "tf_prompt_truncated": int(tf_prompt_truncated),
            "tf_reference_token_count": 0,
        }, pd.DataFrame()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_teacher_forced_kl_postprocessing(steering_df):
    steering_df = steering_df.reset_index(drop=True).copy()

    baseline_keys = [
        "task_id",
        "real_layer",
        "feature_index",
        "generation_mode",
        "sample_id",
    ]

    baseline_map = {}
    baseline_rows = steering_df[steering_df["scale"] == 0.0].copy()

    for _, r in baseline_rows.iterrows():
        key = tuple(r[k] for k in baseline_keys)
        baseline_map[key] = str(r["output"]) if pd.notna(r["output"]) else ""

    summary_rows = []
    detail_frames = []

    for idx, row in tqdm(
        steering_df.iterrows(),
        total=len(steering_df),
        desc="Teacher-forced per-token KL",
    ):
        key = tuple(row[k] for k in baseline_keys)
        reference_continuation = baseline_map.get(key, "")

        base_text = row.get("base_text_full", BASE_TEXT)
        if not isinstance(base_text, str) or len(base_text.strip()) == 0 or base_text == "nan":
            base_text = BASE_TEXT

        prompt = build_analysis_prompt(
            base_text=base_text,
            task=str(row["task"]),
        )

        real_layer = int(row["real_layer"])
        feature_index = int(row["feature_index"])
        scale = float(row["scale"])

        kl_summary, kl_detail = teacher_forced_per_token_kl(
            prompt=prompt,
            reference_continuation=reference_continuation,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
            max_reference_tokens=MAX_REFERENCE_TOKENS_FOR_TF_KL,
        )

        kl_summary["row_index"] = int(idx)
        summary_rows.append(kl_summary)

        if SAVE_PER_TOKEN_DETAILS and kl_detail is not None and len(kl_detail) > 0:
            meta_cols = {
                "row_index": int(idx),
                "task_id": row["task_id"],
                "real_layer": real_layer,
                "feature_index": feature_index,
                "feature_label": row.get("feature_label", ""),
                "scale": scale,
                "scale_index": row.get("scale_index", None),
                "scale_count_for_feature": row.get("scale_count_for_feature", None),
                "scale_source": row.get("scale_source", ""),
                "generation_mode": row["generation_mode"],
                "sample_id": row["sample_id"],
            }
            for k, v in meta_cols.items():
                kl_detail[k] = v
            detail_frames.append(kl_detail)

        # Инкрементальное сохранение каждые 20 строк.
        if len(summary_rows) % 20 == 0:
            tmp_summary = pd.DataFrame(summary_rows).set_index("row_index")
            tmp_with_kl = steering_df.join(tmp_summary, how="left")
            tmp_with_kl.to_csv(OUTPUT_WITH_TF_KL_CSV, index=False)

    kl_summary_df = pd.DataFrame(summary_rows).set_index("row_index")
    steering_with_kl = steering_df.join(kl_summary_df, how="left")
    steering_with_kl.to_csv(OUTPUT_WITH_TF_KL_CSV, index=False)

    print(f"\nСохранено: {OUTPUT_WITH_TF_KL_CSV}")

    if SAVE_PER_TOKEN_DETAILS and len(detail_frames) > 0:
        details_df = pd.concat(detail_frames, ignore_index=True)
        details_df = details_df.sort_values([
            "real_layer",
            "feature_index",
            "generation_mode",
            "scale",
            "task_id",
            "sample_id",
            "token_step",
        ])
        details_df.to_csv(OUTPUT_TF_KL_DETAILS_CSV, index=False)
        print(f"Сохранено: {OUTPUT_TF_KL_DETAILS_CSV}")
        print(f"Per-token rows: {len(details_df)}")

    print("\n=== TEACHER-FORCED KL SUMMARY BY FEATURE/SCALE/MODE ===")

    kl_group_summary = steering_with_kl.groupby(
        ["real_layer", "feature_index", "feature_label", "generation_mode", "scale"],
        as_index=False,
    ).agg(
        rows=("output", "count"),
        mean_tf_kl=("tf_kl_base_to_patched_mean", "mean"),
        max_tf_kl_mean=("tf_kl_base_to_patched_max", "mean"),
        mean_tf_js=("tf_js_mean", "mean"),
        mean_ref_logprob_delta=("tf_ref_logprob_delta_mean", "mean"),
        sum_ref_logprob_delta_mean=("tf_ref_logprob_delta_sum", "mean"),
        mean_top_changed_fraction=("tf_top_token_changed_fraction", "mean"),
        tf_prompt_truncated_rate=("tf_prompt_truncated", "mean"),
    )

    print(kl_group_summary.to_string(index=False))
    kl_group_summary.to_csv(OUTPUT_TF_KL_SUMMARY_CSV, index=False)
    print(f"\nСводка сохранена: {OUTPUT_TF_KL_SUMMARY_CSV}")

    return steering_with_kl, kl_group_summary


if RUN_TEACHER_FORCED_KL_AFTER_GENERATION:
    steering_with_tf_kl, tf_kl_summary = run_teacher_forced_kl_postprocessing(steering_df)
else:
    print("Teacher-forced KL пропущен: RUN_TEACHER_FORCED_KL_AFTER_GENERATION = False")

print("\n============================================================")
print("ALL DONE")
print("============================================================")
print(f"1. Steering full metrics: {OUTPUT_CSV}")
print(f"2. Steering summary: {OUTPUT_SUMMARY_CSV}")
print(f"3. Full metrics + teacher-forced KL: {OUTPUT_WITH_TF_KL_CSV}")
print(f"4. Teacher-forced KL summary: {OUTPUT_TF_KL_SUMMARY_CSV}")
if SAVE_PER_TOKEN_DETAILS:
    print(f"5. Per-token KL details: {OUTPUT_TF_KL_DETAILS_CSV}")
