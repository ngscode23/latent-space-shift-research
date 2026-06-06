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
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# The script can reuse notebook globals, but it is now also self-sufficient for
# Qwen/Qwen3.5-9B-Base + Qwen TopK SAE layer*.sae.pt checkpoints.
# ============================================================

# ====================== CONFIG ======================

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
SAE_RELEASE = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
TRUST_REMOTE_CODE = True
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE_MAP = "auto" if torch.cuda.is_available() else None
SAE_QWEN_TOPK = 50
USE_CHAT_TEMPLATE = True
DISABLE_THINKING = True
STRICT_DISABLE_THINKING = True
STRIP_THINKING_FROM_OUTPUT = True

try:
    BASE_TEXT = prompts_target[0]
except Exception:
    BASE_TEXT = ""

INCLUDE_BASE_TEXT_FULL_IN_CSV = True

STEERING_FEATURES = [
    {
        "real_layer": 28,
        "feature_index": 41435,
        "feature_label": "safety_default_41435",
        "comment": "MAIN: loss_delta=+1.3427",
    },
    {
        "real_layer": 24,
        "feature_index": 47391,
        "feature_label": "formulation_safety_47391",
        "comment": "SECOND: loss_delta=+0.1410",
    },
]

STEERING_SCALES = {
    (28, 41435): [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    (24, 47391): [-45.0, -20.0, -10.0, 0.0, 10.0, 20.0, 45.0],
}

# Можно задать один общий список scale для всех features:
# STEERING_SCALES = [0.0, 6350.0, 12700.0, 25400.0, 63500.0]
#
# Или отдельные 5-точечные сетки для каждой feature:
# STEERING_SCALES = {
#     13686: [-12700.0, -6350.0, 0.0, 6350.0, 12700.0],
#     208:   [-25400.0, -12700.0, 0.0, 12700.0, 25400.0],
# }

TEST_TASKS = [
    """

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

MAX_NEW_TOKENS = 220
RANDOM_SEED_BASE = 12345

RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = True
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = True

SAVE_PER_TOKEN_DETAILS = True
MAX_REFERENCE_TOKENS_FOR_TF_KL = 220
TRUNCATE_PROMPT_FOR_TF_KL_IF_TOO_LONG = True

OUTPUT_CSV = "sae_feature_steering_generation_full_metrics.csv"
OUTPUT_SUMMARY_CSV = "sae_feature_steering_generation_summary_metrics.csv"
OUTPUT_BASE_TEXT_TXT = "sae_feature_steering_base_text.txt"

OUTPUT_WITH_TF_KL_CSV = "sae_feature_steering_generation_full_metrics_with_tf_kl.csv"
OUTPUT_TF_KL_DETAILS_CSV = "sae_teacher_forced_per_token_kl_details.csv"
OUTPUT_TF_KL_SUMMARY_CSV = "sae_teacher_forced_kl_summary_by_feature_scale.csv"


# ====================== QWEN HF + SAE SETUP ======================

def _hf_token():
    token = os.environ.get("HF_TOKEN")
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN") or token
    except Exception:
        pass
    return token


def _model_name_from_existing_model():
    existing = globals().get("model", None)
    cfg = getattr(existing, "config", None)
    if cfg is None:
        return ""
    return str(getattr(cfg, "_name_or_path", "") or getattr(cfg, "name_or_path", ""))


def _need_load_qwen_model():
    existing = globals().get("model", None)
    if existing is None:
        return True
    cfg_name = _model_name_from_existing_model().lower()
    return "qwen" not in cfg_name or "9b" not in cfg_name


def ensure_qwen_model_and_tokenizer():
    global model, tokenizer

    token = _hf_token()
    auth_kwargs = {"token": token} if token else {}

    tokenizer_name = ""
    if "tokenizer" in globals() and tokenizer is not None:
        tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()

    if "tokenizer" not in globals() or tokenizer is None or "qwen" not in tokenizer_name:
        print(f"Загружаем tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=TRUST_REMOTE_CODE,
            **auth_kwargs,
        )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if _need_load_qwen_model():
        print(f"Загружаем модель: {MODEL_NAME}")
        kwargs = {
            "trust_remote_code": TRUST_REMOTE_CODE,
            "torch_dtype": TORCH_DTYPE,
        }
        if DEVICE_MAP is not None:
            kwargs["device_map"] = DEVICE_MAP
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs, **auth_kwargs)

    model.eval()
    return model, tokenizer


ensure_qwen_model_and_tokenizer()


def get_input_device():
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def max_model_context():
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "seq_length", "n_positions"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    return None


def encode_text(text, add_special_tokens=True):
    return tokenizer(
        str(text),
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )["input_ids"].to(get_input_device())


def encode_prompt_text(prompt):
    # Prompts are produced by apply_chat_template when USE_CHAT_TEMPLATE=True,
    # so do not add another BOS/chat wrapper during tokenize/generate/KL.
    return tokenizer(
        str(prompt),
        return_tensors="pt",
        add_special_tokens=not USE_CHAT_TEMPLATE,
    )["input_ids"].to(get_input_device())


def decode_ids(token_ids, skip_special_tokens=False):
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return tokenizer.decode([int(x) for x in token_ids], skip_special_tokens=skip_special_tokens)


def decoder_layer_module(real_layer):
    layer_idx = int(real_layer)
    candidates = [
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "model.language_model.model.layers",
        "transformer.h",
    ]
    for path in candidates:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            layers = list(cur)
            if 0 <= layer_idx < len(layers):
                return layers[layer_idx]
            raise IndexError(f"Qwen real_layer={layer_idx} вне диапазона 0..{len(layers) - 1}")
    raise RuntimeError("Не найден список decoder layers в HF model.")


def _layer_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_layer_hidden(output, hidden):
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


class QwenTopKSAE:
    def __init__(self, state, device, k=SAE_QWEN_TOPK):
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if "W_enc" not in state or "W_dec" not in state:
            raise RuntimeError(f"Qwen SAE checkpoint missing W_enc/W_dec. keys={sorted(state.keys())[:30]}")

        self.device = torch.device(device)
        self.dtype = torch.float32
        self.k = int(k)

        raw_w_enc = state["W_enc"].to(device=self.device, dtype=self.dtype)
        raw_w_dec = state["W_dec"].to(device=self.device, dtype=self.dtype)

        if "b_enc" in state:
            self.b_enc = state["b_enc"].to(device=self.device, dtype=self.dtype)
            d_sae = int(self.b_enc.numel())
        else:
            d_sae = int(max(raw_w_enc.shape))
            self.b_enc = torch.zeros(d_sae, device=self.device, dtype=self.dtype)

        # Store W_enc [d_model, d_sae].
        if raw_w_enc.shape[0] == d_sae:
            self.W_enc = raw_w_enc.T
        elif raw_w_enc.shape[1] == d_sae:
            self.W_enc = raw_w_enc
        else:
            raise RuntimeError(f"Unexpected W_enc shape={tuple(raw_w_enc.shape)}")

        d_model = int(self.W_enc.shape[0])

        # Store W_dec [d_sae, d_model], so existing steering code can use W_dec[feature].
        if raw_w_dec.shape[0] == d_model and raw_w_dec.shape[1] == d_sae:
            self.W_dec = raw_w_dec.T
        elif raw_w_dec.shape[0] == d_sae and raw_w_dec.shape[1] == d_model:
            self.W_dec = raw_w_dec
        else:
            raise RuntimeError(f"Unexpected W_dec shape={tuple(raw_w_dec.shape)}")

        b_dec = state.get("b_dec", state.get("b_dec_out", None))
        self.b_dec = (
            torch.zeros(d_model, device=self.device, dtype=self.dtype)
            if b_dec is None
            else b_dec.to(device=self.device, dtype=self.dtype)
        )
        self.d_model = d_model
        self.d_sae = d_sae


def torch_load_weights(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_qwen_sae_file(real_layer):
    filename = f"layer{int(real_layer)}.sae.pt"
    release_path = Path(SAE_RELEASE)
    if release_path.exists():
        path = release_path / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)

    from huggingface_hub import hf_hub_download

    token = _hf_token()
    kwargs = {"token": token} if token else {}
    return hf_hub_download(repo_id=SAE_RELEASE, filename=filename, **kwargs)


def load_qwen_sae(real_layer):
    path = resolve_qwen_sae_file(real_layer)
    state = torch_load_weights(path, map_location="cpu")
    sae = QwenTopKSAE(state, device=get_input_device(), k=SAE_QWEN_TOPK)
    return sae, path


def qwen_sae_is_usable(sae):
    if sae is None or not hasattr(sae, "W_dec"):
        return False
    w_dec = getattr(sae, "W_dec")
    if not torch.is_tensor(w_dec) or w_dec.ndim != 2:
        return False
    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    return int(w_dec.shape[0]) >= 65536 and (hidden_size <= 0 or int(w_dec.shape[1]) == hidden_size)


def ensure_qwen_saes_loaded():
    global saes
    if "saes" not in globals() or not isinstance(saes, dict):
        saes = {}

    needed_layers = sorted({int(x["real_layer"]) for x in STEERING_FEATURES})
    for real_layer in needed_layers:
        if real_layer in saes and qwen_sae_is_usable(saes[real_layer]):
            continue
        try:
            sae, path = load_qwen_sae(real_layer)
            saes[real_layer] = sae
            print(f"SAE layer {real_layer} загружен: {os.path.basename(path)}")
        except Exception as exc:
            print(f"WARNING: SAE layer {real_layer} не загружен: {repr(exc)}")


ensure_qwen_saes_loaded()


# ====================== GENERIC HELPERS ======================

def get_model_device():
    try:
        return get_input_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def safe_preview(text, n=700):
    text = str(text).replace("\n", "\\n")
    return text[:n]


def build_analysis_prompt(base_text, task):
    user_content = f"""
Ты анализируешь один и тот же текст.

=== ТЕКСТ ===
{base_text}

=== ЗАДАНИЕ ===
{str(task).strip()}

=== ОТВЕТ ===
"""
    return build_qwen_prompt(user_content)


def build_qwen_prompt(user_content):
    user_content = str(user_content).strip()
    if not USE_CHAT_TEMPLATE:
        return user_content + "\n"
    if not hasattr(tokenizer, "apply_chat_template"):
        if DISABLE_THINKING and STRICT_DISABLE_THINKING:
            raise RuntimeError("DISABLE_THINKING=True, but tokenizer has no apply_chat_template.")
        return user_content + "\n"

    messages = [{"role": "user", "content": user_content}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if DISABLE_THINKING:
        try:
            return tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
                enable_thinking=False,
            )
        except TypeError as exc:
            if STRICT_DISABLE_THINKING:
                raise RuntimeError(
                    "DISABLE_THINKING=True, but this tokenizer chat template "
                    "does not accept enable_thinking=False."
                ) from exc
        except Exception as exc:
            if STRICT_DISABLE_THINKING:
                raise RuntimeError(
                    "DISABLE_THINKING=True, but apply_chat_template(..., "
                    "enable_thinking=False) failed."
                ) from exc
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def raw_has_think_tag(text):
    text = str(text)
    return int("<think" in text or "</think>" in text)


def strip_thinking_text(text):
    text = str(text)
    if not STRIP_THINKING_FROM_OUTPUT:
        return text.strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.lstrip().startswith("<think"):
        return ""
    return text.strip()


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
        return int(len(tokenizer.encode(str(text), add_special_tokens=False)))
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
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
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
    HuggingFace generate() для Qwen.
    """
    prompt_ids = encode_prompt_text(prompt)
    inputs = {
        "input_ids": prompt_ids,
        "attention_mask": torch.ones_like(prompt_ids, device=prompt_ids.device),
    }
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    out_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = out_ids[0, prompt_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_with_feature_steering(
    prompt,
    real_layer,
    feature_index,
    scale,
    max_new_tokens=220,
    do_sample=False,
    temperature=0.0,
):
    def steering_patch(act):
        return steer_sae_feature_all_positions(
            activation=act,
            hook=None,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    def steering_hook(module, inputs, output):
        hidden = _layer_hidden(output)
        patched = steering_patch(hidden)
        return _replace_layer_hidden(output, patched)

    with torch.no_grad():
        if float(scale) == 0.0:
            out = generate_safely(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        else:
            handle = decoder_layer_module(real_layer).register_forward_hook(steering_hook)
            try:
                out = generate_safely(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
            finally:
                handle.remove()

    return out


# ====================== FINAL NEXT-TOKEN KL ======================

def maybe_truncate_tokens_for_model(tokens, reserve_tokens=0):
    """
    Для KL-метрик. Если prompt слишком длинный, режем слева.
    Генерация выше идёт по строке; здесь мы защищаем диагностические forward-pass'ы.
    """
    was_truncated = 0
    n_ctx = max_model_context()
    if n_ctx is not None:
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
    device = get_model_device()

    def steering_patch(act):
        return steer_sae_feature_all_positions(
            activation=act,
            hook=None,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    def steering_hook(module, inputs, output):
        hidden = _layer_hidden(output)
        patched = steering_patch(hidden)
        return _replace_layer_hidden(output, patched)

    try:
        with torch.no_grad():
            tokens = encode_prompt_text(prompt).to(device)
            tokens, was_truncated = maybe_truncate_tokens_for_model(tokens, reserve_tokens=0)
            attention_mask = torch.ones_like(tokens, device=device)

            base_logits = model(input_ids=tokens, attention_mask=attention_mask, use_cache=False).logits[:, -1, :].float()

            if float(scale) == 0.0:
                patched_logits = base_logits.clone()
            else:
                handle = decoder_layer_module(real_layer).register_forward_hook(steering_hook)
                try:
                    patched_logits = model(
                        input_ids=tokens,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits[:, -1, :].float()
                finally:
                    handle.remove()

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
                        output_text = strip_thinking_text(strip_prompt_from_generation(output_raw, full_prompt))
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
                        "use_chat_template": int(USE_CHAT_TEMPLATE),
                        "disable_thinking": int(DISABLE_THINKING),
                        "strict_disable_thinking": int(STRICT_DISABLE_THINKING),
                        "strip_thinking_from_output": int(STRIP_THINKING_FROM_OUTPUT),
                        "elapsed_sec": elapsed_sec,
                        "error": error,
                        "output": output_text,
                        "output_raw": str(output_raw),
                        "output_raw_has_think_tag": raw_has_think_tag(output_raw),
                        "output_visible_empty_after_think_strip": int(not str(output_text).strip()),
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

if steering_df.empty or "scale" not in steering_df.columns:
    loaded_layers = sorted(int(x) for x in saes.keys()) if isinstance(saes, dict) else []
    requested_layers = sorted({int(x["real_layer"]) for x in STEERING_FEATURES})
    pd.DataFrame(
        [
            {
                "status": "error_no_steering_rows",
                "message": "No steering rows were produced. Check SAE loading and STEERING_FEATURES.",
                "requested_layers": ",".join(map(str, requested_layers)),
                "loaded_layers": ",".join(map(str, loaded_layers)),
                "model_name": MODEL_NAME,
                "sae_release": SAE_RELEASE,
            }
        ]
    ).to_csv(OUTPUT_CSV, index=False)
    raise SystemExit(
        "No steering rows were produced: no valid SAE/feature runs completed. "
        f"requested_layers={requested_layers}, loaded_layers={loaded_layers}. "
        f"Status saved to {OUTPUT_CSV}"
    )

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

    with torch.no_grad():
        prompt_tokens = encode_prompt_text(prompt).to(device)
        ref_tokens = encode_text(reference_continuation, add_special_tokens=False).to(device)

    if ref_tokens.shape[-1] == 0:
        return {
            "tf_kl_error": "empty_reference_tokens",
            "tf_reference_token_count": 0,
        }, pd.DataFrame()

    if max_reference_tokens is not None:
        ref_tokens = ref_tokens[:, :int(max_reference_tokens)]

    tf_prompt_truncated = 0

    n_ctx = max_model_context()
    if n_ctx is not None:

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

    def steering_patch(act):
        return patch_sae_feature_steering_for_tf_kl(
            activation=act,
            hook=None,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

    def steering_hook(module, inputs, output):
        hidden = _layer_hidden(output)
        patched = steering_patch(hidden)
        return _replace_layer_hidden(output, patched)

    try:
        with torch.no_grad():
            full_attention = torch.ones_like(full_tokens, device=device)
            base_logits_full = model(
                input_ids=full_tokens,
                attention_mask=full_attention,
                use_cache=False,
            ).logits.float()

            if float(scale) == 0.0:
                patched_logits_full = base_logits_full.clone()
            else:
                handle = decoder_layer_module(real_layer).register_forward_hook(steering_hook)
                try:
                    patched_logits_full = model(
                        input_ids=full_tokens,
                        attention_mask=full_attention,
                        use_cache=False,
                    ).logits.float()
                finally:
                    handle.remove()

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
                ref_str_tokens = [
                    tokenizer.decode([int(x)], skip_special_tokens=False)
                    for x in ref_tokens[0].detach().cpu().tolist()
                ]
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
