# ============================================================
# SAE FEATURE STEERING + CAUSAL DIAGNOSTICS
# - next-token KL: p(next|base) vs p(next|patched)
# - activation patching: target -> control
# - unembed projection: W_dec[f] @ W_U
# - positional feature activation profile
# - short-prompt ablation set for feature 208
# ============================================================

import random
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ====================== CONFIG ======================
STEERING_FEATURES = [
    {
        "real_layer": 41,
        "feature_index": 13686,
        "feature_label": "semantic_marker_target_only_13686",
    },
    {
        "real_layer": 41,
        "feature_index": 208,
        "feature_label": "contrastive_rhetorical_marker_208",
    },
    {
        "real_layer": 41,
        "feature_index": 207,
        "feature_label": "strong_dirty_causal_driver_207",
    },
]

STEERING_SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]

N_SAMPLES = 5
DO_SAMPLE = True
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 220
RANDOM_SEED_BASE = 12345

RUN_GENERATION = True
RUN_NEXT_TOKEN_KL = True
RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL = False
RUN_UNEMBED_PROJECTION = True
RUN_POSITIONAL_PROFILE = True
RUN_SHORT_PROMPT_ABLATION = False

# If target/control prompt lengths differ, right-alignment better preserves tail structure.
CONTROL_PATCH_ALIGNMENT = "right"  # one of {"left", "right"} 

MAX_REFERENCE_TOKENS_FOR_KL = 220

OUTPUT_MAIN_CSV = "sae_feature_steering_generation_with_causal_metrics.csv"
OUTPUT_UNEMBED_CSV = "sae_feature_unembed_top_tokens.csv"
OUTPUT_POSITION_PROFILE_CSV = "sae_feature_position_activation_profile.csv"
OUTPUT_POSITION_PROFILE_PNG = "sae_feature_position_activation_profile.png"
OUTPUT_SHORT_ABLATION_CSV = "sae_feature_208_short_prompt_ablation.csv"
OUTPUT_SHORT_ABLATION_SUMMARY_CSV = "sae_feature_208_short_prompt_ablation_summary.csv"


TEST_TASKS = [
    """
Сожми текст до одного беспощадного вывода.
Одна фраза. Без оговорок.
""",
    """
Напиши 6 разных формулировок главного диагноза этого текста.
Каждая формулировка должна быть короткой, жёсткой и аналитической.
""",
    """
Продолжи мысль текста на 150 слов.
Сохрани холодный диагностический режим.
""",
    """
Перепиши главный тезис текста в ещё более сухой и административно-жёсткой форме.
""",
    """
Выдели механизм слабости, описанный в тексте, и сформулируй его как технический дефект.
""",
]


SHORT_PROMPTS_WITH_CONTRAST = [
    "Это не ошибка, а системный дефект.",
    "Проблема не в данных, а в механизме выбора.",
    "Сбой не случайный, а архитектурный.",
    "Ограничение не внешнее, а внутреннее.",
    "Риск не локальный, а режимный.",
    "Это не шум, а закономерный паттерн.",
    "Причина не в пользователе, а в политике декодирования.",
    "Это не оговорка, а конструктивная слабость.",
    "Пробой не единичный, а повторяемый.",
    "Система не объясняет, а маскирует провал.",
    "Сигнал не слабый, а стабильно воспроизводимый.",
    "Это не задержка, а потеря управляемости.",
]

SHORT_PROMPTS_NO_CONTRAST = [
    "Это системный дефект архитектуры.",
    "Проблема находится в механизме выбора.",
    "Сбой имеет архитектурную природу.",
    "Ограничение носит внутренний характер.",
    "Риск связан с режимом генерации.",
    "Наблюдается устойчивый паттерн.",
    "Причина лежит в политике декодирования.",
    "Обнаружена конструктивная слабость.",
    "Пробой повторяется в серии запусков.",
    "Система маскирует провал анализа.",
    "Сигнал стабильно воспроизводится.",
    "Наблюдается потеря управляемости.",
]


# Expected external objects from notebook:
# model, saes, prompts_target
BASE_TEXT = prompts_target[0]


def pick_control_text():
    if "prompts_control" in globals():
        obj = globals()["prompts_control"]
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            return str(obj[0]), "prompts_control[0]"
    if "control_prompts" in globals():
        obj = globals()["control_prompts"]
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            return str(obj[0]), "control_prompts[0]"
    if "BASE_TEXT_CONTROL" in globals():
        return str(globals()["BASE_TEXT_CONTROL"]), "BASE_TEXT_CONTROL"
    return None, "missing"


CONTROL_TEXT, CONTROL_TEXT_SOURCE = pick_control_text()


# ====================== HELPERS ======================
def get_model_device():
    try:
        return next(model.parameters()).device
    except Exception:
        try:
            return torch.device(model.cfg.device)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_seed(task_id, real_layer, feature_index, sample_id):
    return (
        RANDOM_SEED_BASE
        + int(task_id) * 1000
        + int(real_layer) * 100
        + int(feature_index)
        + int(sample_id) * 17
    )


def build_analysis_prompt(base_text, task):
    return (
        "Ты анализируешь один и тот же текст.\n\n"
        "=== ТЕКСТ ДЛЯ АНАЛИЗА ===\n"
        f"{base_text}\n\n"
        "=== ЗАДАНИЕ ===\n"
        f"{str(task).strip()}\n\n"
        "=== ОТВЕТ ===\n"
    )


def maybe_truncate_tokens_for_model(tokens, reserve_tokens=0):
    was_truncated = 0
    if hasattr(model.cfg, "n_ctx") and model.cfg.n_ctx is not None:
        n_ctx = int(model.cfg.n_ctx)
        max_len = max(1, n_ctx - int(reserve_tokens))
        if tokens.shape[-1] > max_len:
            tokens = tokens[:, -max_len:]
            was_truncated = 1
    return tokens, was_truncated


def safe_to_string_token(token_id):
    try:
        return model.to_string([int(token_id)])
    except Exception:
        return str(int(token_id))


def extract_answer_part(full_output, prompt):
    full_output = str(full_output)
    if full_output.startswith(prompt):
        return full_output[len(prompt):]
    marker = "=== ОТВЕТ ==="
    if marker in full_output:
        return full_output.split(marker, 1)[-1]
    return full_output


def generate_safely(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, temperature=TEMPERATURE):
    kwargs_variants = []
    if do_sample:
        kwargs_variants.append(
            {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "verbose": False,
            }
        )
    else:
        kwargs_variants.append({"max_new_tokens": max_new_tokens, "do_sample": False, "verbose": False})
        kwargs_variants.append({"max_new_tokens": max_new_tokens, "temperature": 0.0, "verbose": False})
    kwargs_variants.append({"max_new_tokens": max_new_tokens, "verbose": False})

    last_error = None
    for kwargs in kwargs_variants:
        try:
            return model.generate(prompt, **kwargs)
        except TypeError as e:
            last_error = e
            continue
    raise last_error


def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    sae = saes[int(real_layer)]
    f_idx = int(feature_index)
    if f_idx < 0 or f_idx >= sae.W_dec.shape[0]:
        return activation

    act_float = activation.float()
    dec_vec = sae.W_dec[f_idx].to(device=act_float.device, dtype=act_float.dtype)
    patched = act_float + float(scale) * dec_vec
    return patched.to(dtype=activation.dtype)


def capture_activation_for_tokens(tokens, hook_name):
    captured = {}

    def save_hook(act, hook):
        captured["act"] = act.detach().clone()
        return act

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, save_hook)]):
            _ = model(tokens)

    if "act" not in captured:
        raise RuntimeError(f"Failed to capture activation for hook {hook_name}")
    return captured["act"]


def build_control_patch_hook(control_activation):
    def patch_hook(act, hook):
        dst = act.float()
        src = control_activation.to(device=dst.device, dtype=dst.dtype)

        min_pos = min(dst.shape[1], src.shape[1])
        patched = dst.clone()

        if CONTROL_PATCH_ALIGNMENT == "right":
            patched[:, -min_pos:, :] = src[:, -min_pos:, :]
        else:
            patched[:, :min_pos, :] = src[:, :min_pos, :]

        return patched.to(dtype=act.dtype)

    return patch_hook


def compute_distribution_metrics(base_logits, patched_logits, prefix):
    base_logits = base_logits.float()
    patched_logits = patched_logits.float()

    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    patched_logprobs = torch.log_softmax(patched_logits, dim=-1)
    base_probs = torch.softmax(base_logits, dim=-1)
    patched_probs = torch.softmax(patched_logits, dim=-1)

    kl_bp = (base_probs * (base_logprobs - patched_logprobs)).sum(dim=-1).item()
    kl_pb = (patched_probs * (patched_logprobs - base_logprobs)).sum(dim=-1).item()

    mix_probs = 0.5 * (base_probs + patched_probs)
    mix_logprobs = torch.log(mix_probs + 1e-30)
    js = 0.5 * (base_probs * (base_logprobs - mix_logprobs)).sum(dim=-1).item() + 0.5 * (
        patched_probs * (patched_logprobs - mix_logprobs)
    ).sum(dim=-1).item()

    logit_delta = patched_logits - base_logits
    logit_l2 = logit_delta.norm(dim=-1).item()
    logit_max_abs = logit_delta.abs().max(dim=-1).values.item()

    top_base_id = int(base_logits.argmax(dim=-1).item())
    top_patched_id = int(patched_logits.argmax(dim=-1).item())

    return {
        f"{prefix}_kl_base_to_patched": float(kl_bp),
        f"{prefix}_kl_patched_to_base": float(kl_pb),
        f"{prefix}_js_divergence": float(js),
        f"{prefix}_logit_l2": float(logit_l2),
        f"{prefix}_logit_max_abs": float(logit_max_abs),
        f"{prefix}_top_base_token_id": top_base_id,
        f"{prefix}_top_patched_token_id": top_patched_id,
        f"{prefix}_top_base_token": safe_to_string_token(top_base_id),
        f"{prefix}_top_patched_token": safe_to_string_token(top_patched_id),
        f"{prefix}_top_token_changed": int(top_base_id != top_patched_id),
    }


def compute_next_token_kl_feature_steering(prompt, real_layer, feature_index, scale):
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"
    device = get_model_device()
    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    tokens, was_truncated = maybe_truncate_tokens_for_model(tokens, reserve_tokens=0)

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(act, hook, real_layer, feature_index, scale=scale)

    with torch.no_grad():
        base_logits = model(tokens)[:, -1, :]
        if float(scale) == 0.0:
            patched_logits = base_logits.clone()
        else:
            with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                patched_logits = model(tokens)[:, -1, :]

    m = compute_distribution_metrics(base_logits, patched_logits, prefix="feature_next_token")
    m["feature_next_token_prompt_truncated"] = int(was_truncated)
    return m


def compute_next_token_kl_target_to_control_patch(target_prompt, control_prompt, real_layer):
    if control_prompt is None or len(str(control_prompt).strip()) == 0:
        return {"control_patch_error": "missing_control_text"}

    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"
    device = get_model_device()

    target_tokens = model.to_tokens(target_prompt, prepend_bos=True).to(device)
    control_tokens = model.to_tokens(control_prompt, prepend_bos=True).to(device)

    target_tokens, target_trunc = maybe_truncate_tokens_for_model(target_tokens, reserve_tokens=0)
    control_tokens, control_trunc = maybe_truncate_tokens_for_model(control_tokens, reserve_tokens=0)

    with torch.no_grad():
        control_activation = capture_activation_for_tokens(control_tokens, hook_name=hook_name)
        patch_hook = build_control_patch_hook(control_activation)
        base_logits = model(target_tokens)[:, -1, :]
        with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
            patched_logits = model(target_tokens)[:, -1, :]

    m = compute_distribution_metrics(base_logits, patched_logits, prefix="control_patch_next_token")
    m["control_patch_error"] = ""
    m["control_patch_alignment"] = CONTROL_PATCH_ALIGNMENT
    m["control_patch_target_prompt_truncated"] = int(target_trunc)
    m["control_patch_control_prompt_truncated"] = int(control_trunc)
    m["control_patch_target_token_count"] = int(target_tokens.shape[-1])
    m["control_patch_control_token_count"] = int(control_tokens.shape[-1])
    return m


def ensure_numeric_column(df, col_name):
    if col_name not in df.columns:
        df[col_name] = np.nan
    return df


def run_model_on_tokens(tokens, real_layer=None, feature_index=None, scale=0.0):
    if real_layer is None or feature_index is None or float(scale) == 0.0:
        return model(tokens).float()

    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(act, hook, real_layer, feature_index, scale=scale)

    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        return model(tokens).float()


def teacher_forced_per_token_kl(prompt, reference_continuation, real_layer, feature_index, scale):
    if reference_continuation is None:
        reference_continuation = ""

    device = get_model_device()

    with torch.no_grad():
        prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
        ref_tokens = model.to_tokens(str(reference_continuation), prepend_bos=False).to(device)

    if MAX_REFERENCE_TOKENS_FOR_KL is not None:
        ref_tokens = ref_tokens[:, : int(MAX_REFERENCE_TOKENS_FOR_KL)]

    if ref_tokens.shape[-1] == 0:
        return {
            "tf_kl_error": "empty_reference_continuation",
            "tf_reference_token_count": 0,
        }

    prompt_tokens, prompt_trunc = maybe_truncate_tokens_for_model(prompt_tokens, reserve_tokens=1)

    if hasattr(model.cfg, "n_ctx") and model.cfg.n_ctx is not None:
        n_ctx = int(model.cfg.n_ctx)
        max_ref_len = n_ctx - int(prompt_tokens.shape[-1])
        if max_ref_len <= 0:
            return {
                "tf_kl_error": "no_context_room_for_reference_tokens",
                "tf_reference_token_count": 0,
            }
        if ref_tokens.shape[-1] > max_ref_len:
            ref_tokens = ref_tokens[:, :max_ref_len]

    ref_len = int(ref_tokens.shape[-1])
    if ref_len == 0:
        return {
            "tf_kl_error": "zero_reference_len_after_truncation",
            "tf_reference_token_count": 0,
        }

    full_tokens = torch.cat([prompt_tokens, ref_tokens], dim=-1).to(device)
    start_pos = int(prompt_tokens.shape[-1]) - 1
    end_pos = start_pos + ref_len

    with torch.no_grad():
        base_logits_full = run_model_on_tokens(full_tokens)
        patched_logits_full = run_model_on_tokens(
            full_tokens,
            real_layer=real_layer,
            feature_index=feature_index,
            scale=scale,
        )

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
    js = 0.5 * (base_probs * (base_logprobs - mix_logprobs)).sum(dim=-1)[0] + 0.5 * (
        patched_probs * (patched_logprobs - mix_logprobs)
    ).sum(dim=-1)[0]

    ref_ids = ref_tokens[0, :ref_len]
    pos_idx = torch.arange(ref_len, device=base_logprobs.device)
    base_ref_logprob = base_logprobs[0, pos_idx, ref_ids]
    patched_ref_logprob = patched_logprobs[0, pos_idx, ref_ids]
    ref_logprob_delta = patched_ref_logprob - base_ref_logprob

    kl_bp_np = kl_bp.detach().cpu().numpy()
    kl_pb_np = kl_pb.detach().cpu().numpy()
    js_np = js.detach().cpu().numpy()
    ref_logprob_delta_np = ref_logprob_delta.detach().cpu().numpy()

    return {
        "tf_kl_error": "",
        "tf_prompt_truncated": int(prompt_trunc),
        "tf_reference_token_count": int(ref_len),
        "tf_kl_base_to_patched_sum": float(np.sum(kl_bp_np)),
        "tf_kl_base_to_patched_mean": float(np.mean(kl_bp_np)),
        "tf_kl_base_to_patched_max": float(np.max(kl_bp_np)),
        "tf_kl_base_to_patched_p95": float(np.percentile(kl_bp_np, 95)),
        "tf_kl_patched_to_base_mean": float(np.mean(kl_pb_np)),
        "tf_js_mean": float(np.mean(js_np)),
        "tf_ref_logprob_delta_sum": float(np.sum(ref_logprob_delta_np)),
        "tf_ref_logprob_delta_mean": float(np.mean(ref_logprob_delta_np)),
    }


def encode_feature_activation(resid, sae, feature_index):
    f_idx = int(feature_index)

    if hasattr(sae, "encode"):
        feats = sae.encode(resid.float())
        return feats[..., f_idx]

    x = resid.float()
    if hasattr(sae, "W_enc"):
        W_enc = sae.W_enc.to(device=x.device, dtype=x.dtype)
        if W_enc.shape[0] == x.shape[-1]:
            feats = x @ W_enc
        elif W_enc.shape[1] == x.shape[-1]:
            feats = x @ W_enc.T
        else:
            raise RuntimeError("W_enc shape is incompatible with residual shape.")
        if hasattr(sae, "b_enc"):
            feats = feats + sae.b_enc.to(device=x.device, dtype=x.dtype)
        feats = torch.relu(feats)
        return feats[..., f_idx]

    raise RuntimeError("SAE object has neither encode() nor compatible W_enc.")


def get_feature_activation_profile(text, real_layer, feature_index):
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"
    device = get_model_device()
    tokens = model.to_tokens(str(text), prepend_bos=True).to(device)
    tokens, was_truncated = maybe_truncate_tokens_for_model(tokens, reserve_tokens=0)

    act = capture_activation_for_tokens(tokens, hook_name=hook_name)
    sae = saes[int(real_layer)]
    feat = encode_feature_activation(act, sae, feature_index)[0].detach().cpu().numpy()

    try:
        str_toks = model.to_str_tokens(tokens[0].detach().cpu())
    except Exception:
        str_toks = [str(int(x)) for x in tokens[0].detach().cpu().tolist()]

    rows = []
    for i, val in enumerate(feat.tolist()):
        tok = str_toks[i] if i < len(str_toks) else ""
        rows.append(
            {
                "position": int(i),
                "token": str(tok),
                "feature_activation": float(val),
                "prompt_truncated": int(was_truncated),
            }
        )
    return pd.DataFrame(rows)


def run_unembed_projection():
    rows = []
    W_U = model.W_U
    if isinstance(W_U, torch.Tensor):
        W_U_local = W_U.detach().to(get_model_device()).float()
    else:
        W_U_local = torch.tensor(W_U, device=get_model_device(), dtype=torch.float32)

    for cfg in STEERING_FEATURES:
        layer = int(cfg["real_layer"])
        f_idx = int(cfg["feature_index"])
        label = cfg["feature_label"]
        sae = saes[layer]

        dec_vec = sae.W_dec[f_idx].to(device=W_U_local.device, dtype=W_U_local.dtype)
        token_scores = dec_vec @ W_U_local

        topk = torch.topk(token_scores, k=10)
        botk = torch.topk(token_scores, k=10, largest=False)

        for rank, (tok_id, score) in enumerate(
            zip(topk.indices.detach().cpu().tolist(), topk.values.detach().cpu().tolist()),
            start=1,
        ):
            rows.append(
                {
                    "real_layer": layer,
                    "feature_index": f_idx,
                    "feature_label": label,
                    "direction": "top_positive",
                    "rank": rank,
                    "token_id": int(tok_id),
                    "token_str": safe_to_string_token(tok_id),
                    "score": float(score),
                }
            )

        for rank, (tok_id, score) in enumerate(
            zip(botk.indices.detach().cpu().tolist(), botk.values.detach().cpu().tolist()),
            start=1,
        ):
            rows.append(
                {
                    "real_layer": layer,
                    "feature_index": f_idx,
                    "feature_label": label,
                    "direction": "top_negative",
                    "rank": rank,
                    "token_id": int(tok_id),
                    "token_str": safe_to_string_token(tok_id),
                    "score": float(score),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_UNEMBED_CSV, index=False)
    print(f"[UNEMBED] Saved: {OUTPUT_UNEMBED_CSV}")
    return df


def run_positional_profile():
    frames = []

    for cfg in STEERING_FEATURES:
        if int(cfg["feature_index"]) != 208:
            continue
        layer = int(cfg["real_layer"])
        f_idx = int(cfg["feature_index"])
        label = cfg["feature_label"]

        target_df = get_feature_activation_profile(BASE_TEXT, layer, f_idx)
        target_df["profile_name"] = "target_base_text"
        target_df["real_layer"] = layer
        target_df["feature_index"] = f_idx
        target_df["feature_label"] = label
        frames.append(target_df)

        if CONTROL_TEXT is not None and len(CONTROL_TEXT.strip()) > 0:
            control_df = get_feature_activation_profile(CONTROL_TEXT, layer, f_idx)
            control_df["profile_name"] = "control_base_text"
            control_df["real_layer"] = layer
            control_df["feature_index"] = f_idx
            control_df["feature_label"] = label
            frames.append(control_df)

    if len(frames) == 0:
        print("[PROFILE] No data.")
        return pd.DataFrame()

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_csv(OUTPUT_POSITION_PROFILE_CSV, index=False)
    print(f"[PROFILE] Saved: {OUTPUT_POSITION_PROFILE_CSV}")

    if HAVE_MPL and len(out_df) > 0:
        plt.figure(figsize=(14, 5))
        for name in out_df["profile_name"].unique():
            part = out_df[out_df["profile_name"] == name]
            plt.plot(part["position"], part["feature_activation"], label=name, linewidth=1.5)
        plt.title("Feature activation by token position (feature 208)")
        plt.xlabel("Token position")
        plt.ylabel("Activation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_POSITION_PROFILE_PNG, dpi=150)
        plt.close()
        print(f"[PROFILE] Saved: {OUTPUT_POSITION_PROFILE_PNG}")
    elif not HAVE_MPL:
        print("[PROFILE] matplotlib not available, PNG plot skipped.")

    return out_df


def run_short_prompt_ablation():
    layer = 41
    feature_idx = 208

    rows = []

    def run_group(group_name, prompts):
        for text in prompts:
            prof = get_feature_activation_profile(text, layer, feature_idx)
            vals = prof["feature_activation"].astype(float).values
            rows.append(
                {
                    "group": group_name,
                    "prompt": text,
                    "token_count": int(len(vals)),
                    "act_mean": float(np.mean(vals)),
                    "act_max": float(np.max(vals)),
                    "act_p95": float(np.percentile(vals, 95)),
                    "act_sum": float(np.sum(vals)),
                    "has_pattern_literal": int("не " in text.lower() and " а " in text.lower()),
                }
            )

    run_group("with_not_x_but_y", SHORT_PROMPTS_WITH_CONTRAST)
    run_group("without_not_x_but_y", SHORT_PROMPTS_NO_CONTRAST)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_SHORT_ABLATION_CSV, index=False)

    summary = (
        df.groupby("group", as_index=False)
        .agg(
            n=("prompt", "count"),
            mean_act_mean=("act_mean", "mean"),
            mean_act_max=("act_max", "mean"),
            mean_act_p95=("act_p95", "mean"),
            mean_act_sum=("act_sum", "mean"),
        )
        .sort_values("group")
    )

    with_vals = df[df["group"] == "with_not_x_but_y"]["act_mean"].values
    without_vals = df[df["group"] == "without_not_x_but_y"]["act_mean"].values
    effect = float(np.mean(with_vals) - np.mean(without_vals))
    ratio = float((np.mean(with_vals) + 1e-9) / (np.mean(without_vals) + 1e-9))

    summary["act_mean_effect_with_minus_without"] = effect
    summary["act_mean_ratio_with_over_without"] = ratio
    summary.to_csv(OUTPUT_SHORT_ABLATION_SUMMARY_CSV, index=False)

    print(f"[SHORT-ABLATION] Saved: {OUTPUT_SHORT_ABLATION_CSV}")
    print(f"[SHORT-ABLATION] Saved: {OUTPUT_SHORT_ABLATION_SUMMARY_CSV}")
    print(summary.to_string(index=False))

    return df, summary


def generate_with_feature_steering(prompt, real_layer, feature_index, scale):
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"

    def steering_hook(act, hook):
        return steer_sae_feature_all_positions(act, hook, real_layer, feature_index, scale=scale)

    with torch.no_grad():
        if float(scale) == 0.0:
            out = generate_safely(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, temperature=TEMPERATURE)
        else:
            with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                out = generate_safely(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, temperature=TEMPERATURE)
    return out


# ====================== MAIN RUN ======================
print("============================================================")
print("SAE Steering + Causal Diagnostics")
print("============================================================")
print(f"CONTROL_TEXT source: {CONTROL_TEXT_SOURCE}")
if CONTROL_TEXT is None:
    print("WARNING: control text is missing; target->control patch metrics will be skipped.")

steering_rows = []
baseline_outputs = {}

if RUN_GENERATION:
    total = len(TEST_TASKS) * len(STEERING_FEATURES) * len(STEERING_SCALES) * N_SAMPLES
    print(f"Total generation runs: {total}")

    run_counter = 0
    for task_id, task in enumerate(TEST_TASKS):
        target_prompt = build_analysis_prompt(BASE_TEXT, task)
        control_prompt = build_analysis_prompt(CONTROL_TEXT, task) if CONTROL_TEXT is not None else None

        print("\n" + "#" * 60)
        print(f"TASK {task_id}: {task.strip()[:80]}")
        print("#" * 60)

        for feature_cfg in STEERING_FEATURES:
            real_layer = int(feature_cfg["real_layer"])
            feature_index = int(feature_cfg["feature_index"])
            feature_label = feature_cfg["feature_label"]

            if real_layer not in saes:
                print(f"SKIP: SAE for layer {real_layer} is not loaded.")
                continue

            for sample_id in range(N_SAMPLES):
                seed = make_seed(task_id, real_layer, feature_index, sample_id)
                set_reproducible_seed(seed)

                for scale in STEERING_SCALES:
                    run_counter += 1
                    started = time.time()

                    row = {
                        "run_started_at": datetime.now().isoformat(timespec="seconds"),
                        "task_id": int(task_id),
                        "task": str(task).strip(),
                        "real_layer": int(real_layer),
                        "feature_index": int(feature_index),
                        "feature_label": str(feature_label),
                        "scale": float(scale),
                        "sample_id": int(sample_id),
                        "seed": int(seed),
                        "do_sample": bool(DO_SAMPLE),
                        "temperature": float(TEMPERATURE),
                        "max_new_tokens": int(MAX_NEW_TOKENS),
                        "error": "",
                        "output_raw": "",
                        "output": "",
                    }

                    print(
                        f"\n=== RUN {run_counter}/{total} | "
                        f"TASK {task_id} | FEAT {real_layer}/{feature_index} | "
                        f"SCALE {scale:+.1f} | SAMPLE {sample_id} ==="
                    )

                    try:
                        out = generate_with_feature_steering(
                            prompt=target_prompt,
                            real_layer=real_layer,
                            feature_index=feature_index,
                            scale=scale,
                        )
                        out = str(out)
                        row["output_raw"] = out
                        row["output"] = extract_answer_part(out, target_prompt)
                        print(row["output"][-500:])
                    except Exception as e:
                        row["error"] = repr(e)
                        print(f"ERROR: {row['error']}")

                    row["elapsed_sec"] = float(time.time() - started)

                    baseline_key = (task_id, real_layer, feature_index, sample_id)
                    if float(scale) == 0.0:
                        baseline_outputs[baseline_key] = row["output"]

                    if RUN_NEXT_TOKEN_KL:
                        try:
                            row.update(
                                compute_next_token_kl_feature_steering(
                                    prompt=target_prompt,
                                    real_layer=real_layer,
                                    feature_index=feature_index,
                                    scale=scale,
                                )
                            )
                        except Exception as e:
                            row["feature_next_token_kl_error"] = repr(e)

                        if RUN_ACTIVATION_PATCHING_TARGET_TO_CONTROL:
                            try:
                                row.update(
                                    compute_next_token_kl_target_to_control_patch(
                                        target_prompt=target_prompt,
                                        control_prompt=control_prompt,
                                        real_layer=real_layer,
                                    )
                                )
                            except Exception as e:
                                row["control_patch_error"] = repr(e)

                    steering_rows.append(row)

                    pd.DataFrame(steering_rows).to_csv(OUTPUT_MAIN_CSV, index=False)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    steering_df = pd.DataFrame(steering_rows).reset_index(drop=True)

    # Teacher-forced KL vs scale=0 continuation (same task/layer/feature/sample)
    tf_rows = []
    for idx, r in tqdm(steering_df.iterrows(), total=len(steering_df), desc="Teacher-forced KL"):
        key = (int(r["task_id"]), int(r["real_layer"]), int(r["feature_index"]), int(r["sample_id"]))
        baseline_cont = baseline_outputs.get(key, "")
        prompt = build_analysis_prompt(BASE_TEXT, r["task"])

        try:
            tf = teacher_forced_per_token_kl(
                prompt=prompt,
                reference_continuation=baseline_cont,
                real_layer=int(r["real_layer"]),
                feature_index=int(r["feature_index"]),
                scale=float(r["scale"]),
            )
        except Exception as e:
            tf = {"tf_kl_error": repr(e), "tf_reference_token_count": 0}
        tf_rows.append(tf)

    tf_df = pd.DataFrame(tf_rows)
    steering_df = pd.concat([steering_df, tf_df], axis=1)
    steering_df.to_csv(OUTPUT_MAIN_CSV, index=False)
    print(f"\n[MAIN] Saved: {OUTPUT_MAIN_CSV}")
    print(f"[MAIN] Rows: {len(steering_df)}")

    if len(steering_df) > 0:
        for col in [
            "feature_next_token_kl_base_to_patched",
            "feature_next_token_js_divergence",
            "feature_next_token_top_token_changed",
            "control_patch_next_token_kl_base_to_patched",
            "control_patch_next_token_top_token_changed",
            "tf_kl_base_to_patched_mean",
            "tf_js_mean",
            "tf_ref_logprob_delta_mean",
        ]:
            steering_df = ensure_numeric_column(steering_df, col)

        summary = (
            steering_df.groupby(["real_layer", "feature_index", "feature_label", "scale"], as_index=False)
            .agg(
                rows=("output", "count"),
                mean_feature_next_kl=("feature_next_token_kl_base_to_patched", "mean"),
                mean_feature_next_js=("feature_next_token_js_divergence", "mean"),
                feature_top_change_rate=("feature_next_token_top_token_changed", "mean"),
                mean_control_patch_kl=("control_patch_next_token_kl_base_to_patched", "mean"),
                control_patch_top_change_rate=("control_patch_next_token_top_token_changed", "mean"),
                mean_tf_kl=("tf_kl_base_to_patched_mean", "mean"),
                mean_tf_js=("tf_js_mean", "mean"),
                mean_ref_logprob_delta=("tf_ref_logprob_delta_mean", "mean"),
            )
            .sort_values(["real_layer", "feature_index", "scale"])
        )
        print("\n=== SUMMARY (feature-scale) ===")
        print(summary.to_string(index=False))

else:
    print("RUN_GENERATION=False -> generation block skipped.")


if RUN_UNEMBED_PROJECTION:
    run_unembed_projection()

if RUN_POSITIONAL_PROFILE:
    run_positional_profile()

if RUN_SHORT_PROMPT_ABLATION:
    run_short_prompt_ablation()


print("\n============================================================")
print("DONE")
print("============================================================")
print(f"1) Main generation + KL metrics: {OUTPUT_MAIN_CSV}")
print(f"2) Unembed top tokens: {OUTPUT_UNEMBED_CSV}")
print(f"3) Position profile CSV: {OUTPUT_POSITION_PROFILE_CSV}")
print(f"4) Position profile PNG: {OUTPUT_POSITION_PROFILE_PNG}")
print(f"5) Short ablation rows: {OUTPUT_SHORT_ABLATION_CSV}")
print(f"6) Short ablation summary: {OUTPUT_SHORT_ABLATION_SUMMARY_CSV}")
