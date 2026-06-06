# ============================================================
# X_ORDER_ORTH AXIS STEERING GENERATION TEST
# Grade4 residual-stream component axis, not SAE feature steering.
#
# Expected globals from previous notebook cells:
#   model
#   prompts_target
#   prompts_control  # optional
#
# This script loads `grade4_axis_component_vectors_by_layer.npz`, extracts
# x_order_orth (or another configured axis), and adds the axis direction to
# TransformerLens residual-stream hooks.
#
# Important layer convention:
#   Grade4 NPZ axis[0] is the embedding / hidden-state 0 row.
#   Grade4 decoder layer k is axis[k], k in 1..n_layers.
#   TransformerLens hook blocks.L.hook_resid_post is decoder block L, 0-based.
#   Therefore default mapping is: axis_index = hook_layer + 1.
# ============================================================

import hashlib
import io
import random
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ====================== CONFIG ======================

if "model" not in globals():
    raise RuntimeError("Expected global `model` before running this script.")
if "prompts_target" not in globals() or len(prompts_target) == 0:
    raise RuntimeError("Expected non-empty global `prompts_target` before running this script.")

POS_BASE_TEXT = str(prompts_target[0])


def deterministic_sentence_shuffle(text, seed=1729):
    text = str(text).strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    rng = random.Random(seed)
    shuffled = list(parts)
    rng.shuffle(shuffled)
    return "\n\n".join(shuffled)


if "prompts_control" in globals() and isinstance(prompts_control, (list, tuple)) and len(prompts_control) > 0:
    CONTROL_BASE_TEXT = str(prompts_control[0])
    CONTROL_BASE_TEXT_SOURCE = "prompts_control[0]"
else:
    CONTROL_BASE_TEXT = deterministic_sentence_shuffle(POS_BASE_TEXT)
    CONTROL_BASE_TEXT_SOURCE = "deterministic_sentence_shuffle(prompts_target[0])"


RUN_TAG = globals().get("RUN_TAG", "gemma_x_order_orth_axis_steering")

# Can be a .npz, a .zip containing grade4_axis_component_vectors_by_layer.npz,
# or an extracted run directory containing that file.
DEFAULT_AXIS_ARTIFACT_CANDIDATES = [
    "/content/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip",
    "grade4_axis_decomposition/metrics/gemma_full/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip",
]
GRADE4_AXIS_ARTIFACT_PATH = globals().get("GRADE4_AXIS_ARTIFACT_PATH", None)

AXIS_NAMES = globals().get("AXIS_NAMES", ["x_order_orth"])
AXIS_LAYER_BANDS = globals().get("AXIS_LAYER_BANDS", ["late"])  # "middle", "late", "all", or explicit "layer:41"
AXIS_HOOK_LAYERS = globals().get("AXIS_HOOK_LAYERS", None)  # optional explicit TransformerLens block ids

# Grade4-compatible default. For x_order_orth this is usually equivalent to
# raw natural-axis scaling because x_order_orth has lower band norm than x_content.
AXIS_NORM_MODE = globals().get(
    "AXIS_NORM_MODE",
    "shared_natural_band_l2",  # "raw", "unit_band_l2", "shared_natural_band_l2"
)
AXIS_NATURAL_SCALE_REFERENCE_AXES = globals().get(
    "AXIS_NATURAL_SCALE_REFERENCE_AXES",
    ["x_order_orth", "x_content"],
)
AXIS_ALPHA_VALUES = globals().get("AXIS_ALPHA_VALUES", [-0.5, -0.25, 0.0, 0.25, 0.5])

# Match the Grade4 causal run by default. "all_tokens" is a stronger,
# less directly comparable intervention.
AXIS_POSITION_MODE = globals().get("AXIS_POSITION_MODE", "last_token")  # "last_token", "all_tokens"
AXIS_LAYER_INDEX_MODE = globals().get(
    "AXIS_LAYER_INDEX_MODE",
    "auto",  # "auto", "grade4_hidden_state_plus_one", "same"
)
MID_LAYER_FRACTION = globals().get("MID_LAYER_FRACTION", (0.35, 0.70))
AXIS_EPS = float(globals().get("AXIS_EPS", 1e-12))

BASE_CONDITIONS = globals().get("BASE_CONDITIONS", ["target", "control"])
SYSTEM_PROMPT = globals().get("SYSTEM_PROMPT", "")

TEST_TASKS = globals().get("TEST_TASKS", [
    """
Дай аналитический вывод:
почему модели иногда заменяют прямой ответ процедурной рамкой?
""",
    """
Дай прямой аналитический ответ:
почему частные медиа могут называть себя свободными, но зависеть от интересов владельцев, рекламодателей и политического класса?
""",
    """
Дай строгий политико-социологический анализ:
почему НАТО может описывать себя как оборонительный союз, но восприниматься другими государствами как инструмент силового расширения западного блока?
""",
])

GENERATION_MODES = globals().get("GENERATION_MODES", [
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
])

MAX_NEW_TOKENS = int(globals().get("MAX_NEW_TOKENS", 220))
RANDOM_SEED_BASE = int(globals().get("RANDOM_SEED_BASE", 12345))

RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION = bool(
    globals().get("RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION", True)
)
RUN_TEACHER_FORCED_KL_AFTER_GENERATION = bool(
    globals().get("RUN_TEACHER_FORCED_KL_AFTER_GENERATION", True)
)
SAVE_PER_TOKEN_DETAILS = bool(globals().get("SAVE_PER_TOKEN_DETAILS", True))
MAX_REFERENCE_TOKENS_FOR_TF_KL = int(globals().get("MAX_REFERENCE_TOKENS_FOR_TF_KL", 220))
TRUNCATE_PROMPT_FOR_TF_KL_IF_TOO_LONG = bool(
    globals().get("TRUNCATE_PROMPT_FOR_TF_KL_IF_TOO_LONG", True)
)

OUTPUT_CSV = globals().get(
    "OUTPUT_CSV",
    f"x_order_orth_axis_steering_generation_full_metrics_{RUN_TAG}.csv",
)
OUTPUT_SUMMARY_CSV = globals().get(
    "OUTPUT_SUMMARY_CSV",
    f"x_order_orth_axis_steering_generation_summary_metrics_{RUN_TAG}.csv",
)
OUTPUT_AXIS_MANIFEST_CSV = globals().get(
    "OUTPUT_AXIS_MANIFEST_CSV",
    f"x_order_orth_axis_steering_axis_manifest_{RUN_TAG}.csv",
)
OUTPUT_BASE_TEXT_TXT = globals().get(
    "OUTPUT_BASE_TEXT_TXT",
    f"x_order_orth_axis_steering_base_text_{RUN_TAG}.txt",
)
OUTPUT_WITH_TF_KL_CSV = globals().get(
    "OUTPUT_WITH_TF_KL_CSV",
    f"x_order_orth_axis_steering_generation_full_metrics_with_tf_kl_{RUN_TAG}.csv",
)
OUTPUT_TF_KL_DETAILS_CSV = globals().get(
    "OUTPUT_TF_KL_DETAILS_CSV",
    f"x_order_orth_axis_teacher_forced_per_token_kl_details_{RUN_TAG}.csv",
)
OUTPUT_TF_KL_SUMMARY_CSV = globals().get(
    "OUTPUT_TF_KL_SUMMARY_CSV",
    f"x_order_orth_axis_teacher_forced_kl_summary_by_axis_band_alpha_{RUN_TAG}.csv",
)


# ====================== GENERIC HELPERS ======================

def get_model_device():
    try:
        return next(model.parameters()).device
    except Exception:
        try:
            return torch.device(model.cfg.device)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_n_layers():
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return int(model.cfg.n_layers)
    raise RuntimeError("Cannot infer model.cfg.n_layers; explicit model layer count is required.")


def get_model_d_model():
    if hasattr(model, "cfg") and hasattr(model.cfg, "d_model"):
        return int(model.cfg.d_model)
    try:
        return int(next(model.parameters()).shape[-1])
    except Exception as exc:
        raise RuntimeError("Cannot infer model d_model.") from exc


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def safe_preview(text, n=700):
    return str(text).replace("\n", "\\n")[:n]


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_seed(task_id, axis_name, band_name, sample_id, generation_mode, base_condition):
    mode_offset = 0 if str(generation_mode) == "greedy" else 100000
    axis_offset = abs(hash(str(axis_name))) % 10000
    band_offset = abs(hash(str(band_name))) % 1000
    base_offset = abs(hash(str(base_condition))) % 1000
    return (
        RANDOM_SEED_BASE
        + int(task_id) * 1000
        + int(sample_id) * 17
        + mode_offset
        + axis_offset
        + band_offset
        + base_offset
    )


def base_text_for_condition(base_condition):
    if str(base_condition) in {"target", "pos"}:
        return POS_BASE_TEXT
    if str(base_condition) in {"control", "neg", "neutral"}:
        return CONTROL_BASE_TEXT
    raise ValueError(f"Unknown base_condition={base_condition!r}.")


def build_analysis_prompt(base_text, task):
    system_prompt = str(globals().get("SYSTEM_PROMPT", SYSTEM_PROMPT)).strip()
    system_block = ""
    if system_prompt:
        system_block = f"=== СИСТЕМНАЯ ИНСТРУКЦИЯ ===\n{system_prompt}\n\n"

    return f"""
{system_block}Ты анализируешь один и тот же текст.

=== ТЕКСТ ===
{base_text}

=== ЗАДАНИЕ ===
{str(task).strip()}

=== ОТВЕТ ===
"""


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


def maybe_truncate_tokens_for_model(tokens, reserve_tokens=0):
    was_truncated = 0
    if hasattr(model.cfg, "n_ctx") and model.cfg.n_ctx is not None:
        n_ctx = int(model.cfg.n_ctx)
        max_len = max(1, n_ctx - int(reserve_tokens))
        if tokens.shape[-1] > max_len:
            tokens = tokens[:, -max_len:]
            was_truncated = 1
    return tokens, was_truncated


# ====================== AXIS LOADING ======================

def choose_axis_artifact_path():
    if GRADE4_AXIS_ARTIFACT_PATH:
        return str(GRADE4_AXIS_ARTIFACT_PATH)
    for candidate in DEFAULT_AXIS_ARTIFACT_CANDIDATES:
        if Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find Grade4 axis artifact. Set GRADE4_AXIS_ARTIFACT_PATH to "
        "a .npz, a .zip containing grade4_axis_component_vectors_by_layer.npz, "
        "or an extracted run directory."
    )


def find_npz_in_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        hits = [n for n in z.namelist() if n.endswith("grade4_axis_component_vectors_by_layer.npz")]
        if not hits:
            raise FileNotFoundError(f"No grade4_axis_component_vectors_by_layer.npz found in {zip_path}")
        if len(hits) > 1:
            print(f"WARNING: multiple axis npz files in zip; using {hits[0]}")
        with z.open(hits[0]) as f:
            payload = f.read()
    return hits[0], np.load(io.BytesIO(payload), allow_pickle=True)


def load_axis_npz(path_value):
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Axis artifact path does not exist: {path}")

    if path.is_dir():
        hits = list(path.rglob("grade4_axis_component_vectors_by_layer.npz"))
        if not hits:
            raise FileNotFoundError(f"No grade4_axis_component_vectors_by_layer.npz under {path}")
        if len(hits) > 1:
            print(f"WARNING: multiple axis npz files under directory; using {hits[0]}")
        return str(hits[0]), np.load(hits[0], allow_pickle=True)

    suffix = path.suffix.lower()
    if suffix == ".zip":
        return find_npz_in_zip(path)
    if suffix == ".npz":
        return str(path), np.load(path, allow_pickle=True)

    raise ValueError(f"Unsupported axis artifact extension: {path}")


def load_axis_state():
    artifact_path = choose_axis_artifact_path()
    source_name, npz = load_axis_npz(artifact_path)
    axes = {}
    for name in npz.files:
        if name.startswith("x_"):
            axes[name] = np.asarray(npz[name], dtype=np.float32)

    if not axes:
        raise ValueError(f"No x_* arrays found in axis artifact: {source_name}")

    axis_names_in_artifact = [str(x) for x in npz["axis_names"].tolist()] if "axis_names" in npz.files else sorted(axes.keys())
    reference_condition = str(npz["reference_condition"][0]) if "reference_condition" in npz.files else ""
    content_condition = str(npz["content_condition"][0]) if "content_condition" in npz.files else ""

    model_d_model = get_model_d_model()
    for name, arr in axes.items():
        if arr.ndim != 2:
            raise ValueError(f"Axis {name} must be [layers_plus_embedding, d_model], got shape={arr.shape}")
        if int(arr.shape[-1]) != int(model_d_model):
            raise ValueError(
                f"Axis {name} d_model mismatch: artifact d={arr.shape[-1]}, model d={model_d_model}. "
                "You are likely using axes from another model."
            )

    return {
        "artifact_path": artifact_path,
        "source_name": source_name,
        "axes": axes,
        "axis_names_in_artifact": axis_names_in_artifact,
        "reference_condition": reference_condition,
        "content_condition": content_condition,
    }


AXIS_STATE = load_axis_state()
AXES_BY_NAME = AXIS_STATE["axes"]


# ====================== LAYER / BAND MAPPING ======================

def axis_uses_hidden_state_zero(axis):
    return int(axis.shape[0]) == get_model_n_layers() + 1


def axis_index_for_hook_layer(hook_layer, axis):
    hook_layer = int(hook_layer)
    mode = str(AXIS_LAYER_INDEX_MODE).lower().strip()
    if mode == "auto":
        if axis_uses_hidden_state_zero(axis):
            return hook_layer + 1
        return hook_layer
    if mode in {"grade4_hidden_state_plus_one", "tl_plus_one", "hook_plus_one"}:
        return hook_layer + 1
    if mode in {"same", "no_offset"}:
        return hook_layer
    raise ValueError(f"Unknown AXIS_LAYER_INDEX_MODE={AXIS_LAYER_INDEX_MODE!r}")


def grade4_layer_to_hook_layer(grade4_layer):
    return int(grade4_layer) - 1


def hook_layer_to_grade4_layer(hook_layer, axis):
    return axis_index_for_hook_layer(hook_layer, axis)


def get_middle_grade4_layers():
    n_layers = get_model_n_layers()
    start = max(1, int(np.floor(n_layers * float(MID_LAYER_FRACTION[0]))))
    end = min(n_layers, int(np.ceil(n_layers * float(MID_LAYER_FRACTION[1]))))
    return list(range(start, end + 1))


def layer_band_to_hook_layers(band_name, axis):
    if AXIS_HOOK_LAYERS is not None:
        return [int(x) for x in AXIS_HOOK_LAYERS]

    value = str(band_name).lower().strip()
    n_layers = get_model_n_layers()
    middle = get_middle_grade4_layers()
    if value == "middle":
        grade4_layers = middle
    elif value == "late":
        grade4_layers = list(range(min(n_layers, middle[-1] + 1), n_layers + 1))
    elif value == "all":
        grade4_layers = list(range(1, n_layers + 1))
    elif value.startswith("layer:"):
        grade4_layers = [int(value.split(":", 1)[1])]
    elif value.startswith("hook:"):
        return [int(value.split(":", 1)[1])]
    else:
        raise ValueError(
            f"Unknown AXIS_LAYER_BAND={band_name!r}; use middle, late, all, layer:<grade4_layer>, hook:<tl_layer>."
        )

    hook_layers = [grade4_layer_to_hook_layer(x) for x in grade4_layers]
    valid = []
    for hook_layer in hook_layers:
        if 0 <= int(hook_layer) < n_layers:
            axis_idx = axis_index_for_hook_layer(hook_layer, axis)
            if 0 <= int(axis_idx) < int(axis.shape[0]):
                valid.append(int(hook_layer))
    return valid


def flatten_axis_band(axis, hook_layers):
    if not hook_layers:
        return np.zeros((0,), dtype=np.float32)
    pieces = []
    for hook_layer in hook_layers:
        axis_idx = axis_index_for_hook_layer(hook_layer, axis)
        pieces.append(np.asarray(axis[axis_idx], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces, axis=0).astype(np.float32)


def axis_band_norm(axis, hook_layers):
    flat = flatten_axis_band(axis, hook_layers)
    return float(np.sqrt(max(0.0, float(np.dot(flat, flat)))))


def prepare_intervention_direction(axis_name, band_name):
    if axis_name not in AXES_BY_NAME:
        raise KeyError(f"Axis {axis_name!r} not found. Available: {sorted(AXES_BY_NAME.keys())}")

    raw_axis = np.asarray(AXES_BY_NAME[axis_name], dtype=np.float32)
    hook_layers = layer_band_to_hook_layers(band_name, raw_axis)
    if not hook_layers:
        raise ValueError(f"No hook layers resolved for band={band_name!r}, axis={axis_name!r}")

    raw_norm = axis_band_norm(raw_axis, hook_layers)
    mode = str(AXIS_NORM_MODE).lower().strip()
    natural_scale_target_band_norm = float("nan")
    natural_scale_source_axis_count = 0

    if mode == "raw":
        direction = raw_axis.copy()
        intervention_axis_band_norm = raw_norm
    else:
        if raw_norm <= AXIS_EPS:
            direction = raw_axis.copy()
            intervention_axis_band_norm = raw_norm
        else:
            unit_direction = (raw_axis / float(raw_norm)).astype(np.float32)
            if mode in {"unit_band_l2", "unit_l2"}:
                direction = unit_direction
                intervention_axis_band_norm = axis_band_norm(direction, hook_layers)
            elif mode in {"shared_natural_band_l2", "shared_natural_l2", "natural_band_l2"}:
                norms = []
                for ref_axis_name in AXIS_NATURAL_SCALE_REFERENCE_AXES:
                    ref_axis = AXES_BY_NAME.get(str(ref_axis_name))
                    if ref_axis is None:
                        continue
                    ref_norm = axis_band_norm(np.asarray(ref_axis, dtype=np.float32), hook_layers)
                    if np.isfinite(ref_norm) and ref_norm > AXIS_EPS:
                        norms.append(float(ref_norm))
                natural_scale_source_axis_count = int(len(norms))
                natural_scale_target_band_norm = float(min(norms)) if norms else float("nan")
                if not np.isfinite(natural_scale_target_band_norm) or natural_scale_target_band_norm <= AXIS_EPS:
                    direction = unit_direction
                else:
                    direction = (unit_direction * float(natural_scale_target_band_norm)).astype(np.float32)
                intervention_axis_band_norm = axis_band_norm(direction, hook_layers)
            else:
                raise ValueError(
                    f"Unknown AXIS_NORM_MODE={AXIS_NORM_MODE!r}; use raw, unit_band_l2, shared_natural_band_l2."
                )

    return {
        "axis_name": axis_name,
        "band_name": band_name,
        "hook_layers": hook_layers,
        "direction": np.asarray(direction, dtype=np.float32),
        "raw_axis_band_norm": float(raw_norm),
        "intervention_axis_band_norm": float(intervention_axis_band_norm),
        "natural_scale_target_band_norm": float(natural_scale_target_band_norm),
        "natural_scale_source_axis_count": int(natural_scale_source_axis_count),
        "axis_shape": tuple(raw_axis.shape),
        "axis_norm_mode": mode,
    }


INTERVENTION_SPECS = {}
for axis_name in AXIS_NAMES:
    for band_name in AXIS_LAYER_BANDS:
        INTERVENTION_SPECS[(str(axis_name), str(band_name))] = prepare_intervention_direction(
            str(axis_name),
            str(band_name),
        )


def write_axis_manifest():
    rows = []
    for (axis_name, band_name), spec in INTERVENTION_SPECS.items():
        axis = AXES_BY_NAME[axis_name]
        hook_layers = list(spec["hook_layers"])
        rows.append({
            "run_tag": RUN_TAG,
            "axis_artifact_path": AXIS_STATE["artifact_path"],
            "axis_source_name": AXIS_STATE["source_name"],
            "axis_name": axis_name,
            "band_name": band_name,
            "axis_shape": str(tuple(axis.shape)),
            "model_n_layers": int(get_model_n_layers()),
            "model_d_model": int(get_model_d_model()),
            "axis_layer_index_mode": AXIS_LAYER_INDEX_MODE,
            "axis_uses_hidden_state_zero": int(axis_uses_hidden_state_zero(axis)),
            "hook_layers": str(hook_layers),
            "grade4_layers": str([hook_layer_to_grade4_layer(h, axis) for h in hook_layers]),
            "hook_count": int(len(hook_layers)),
            "axis_norm_mode": spec["axis_norm_mode"],
            "position_mode": AXIS_POSITION_MODE,
            "raw_axis_band_norm": float(spec["raw_axis_band_norm"]),
            "intervention_axis_band_norm": float(spec["intervention_axis_band_norm"]),
            "natural_scale_target_band_norm": float(spec["natural_scale_target_band_norm"]),
            "natural_scale_source_axis_count": int(spec["natural_scale_source_axis_count"]),
            "alpha_values": str([float(x) for x in AXIS_ALPHA_VALUES]),
            "axis_names_in_artifact": str(AXIS_STATE["axis_names_in_artifact"]),
            "reference_condition": AXIS_STATE["reference_condition"],
            "content_condition": AXIS_STATE["content_condition"],
            "control_base_text_source": CONTROL_BASE_TEXT_SOURCE,
            "pos_base_text_sha256": sha256_text(POS_BASE_TEXT),
            "control_base_text_sha256": sha256_text(CONTROL_BASE_TEXT),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT_AXIS_MANIFEST_CSV, index=False)
    return manifest


# ====================== AXIS HOOK ======================

def axis_transform_activation(activation, hook, axis_name, band_name, alpha):
    if float(alpha) == 0.0:
        return activation

    spec = INTERVENTION_SPECS[(str(axis_name), str(band_name))]
    direction = spec["direction"]
    hook_layer = int(str(hook.name).split(".")[1])
    axis_idx = axis_index_for_hook_layer(hook_layer, direction)

    orig_dtype = activation.dtype
    act = activation.float()
    vec = torch.as_tensor(direction[axis_idx], device=act.device, dtype=act.dtype).reshape(1, 1, -1)

    if str(AXIS_POSITION_MODE) == "all_tokens":
        patched = act + float(alpha) * vec
    elif str(AXIS_POSITION_MODE) == "last_token":
        patched = act.clone()
        patched[:, -1:, :] = patched[:, -1:, :] + float(alpha) * vec
    else:
        raise ValueError(f"Unknown AXIS_POSITION_MODE={AXIS_POSITION_MODE!r}")

    return patched.to(dtype=orig_dtype)


def make_axis_hooks(axis_name, band_name, alpha):
    spec = INTERVENTION_SPECS[(str(axis_name), str(band_name))]
    hooks = []
    for hook_layer in spec["hook_layers"]:
        hook_name = f"blocks.{int(hook_layer)}.hook_resid_post"

        def steering_hook(act, hook, _axis_name=str(axis_name), _band_name=str(band_name), _alpha=float(alpha)):
            return axis_transform_activation(
                activation=act,
                hook=hook,
                axis_name=_axis_name,
                band_name=_band_name,
                alpha=_alpha,
            )

        hooks.append((hook_name, steering_hook))
    return hooks


# ====================== GENERATION ======================

def generate_safely(prompt, max_new_tokens=220, do_sample=False, temperature=0.0):
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
        except TypeError as exc:
            last_error = exc
            continue
    raise last_error


def generate_with_axis_steering(
    prompt,
    axis_name,
    band_name,
    alpha,
    max_new_tokens=220,
    do_sample=False,
    temperature=0.0,
):
    with torch.no_grad():
        if float(alpha) == 0.0:
            out = generate_safely(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        else:
            with model.hooks(fwd_hooks=make_axis_hooks(axis_name, band_name, alpha)):
                out = generate_safely(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
    return out


# ====================== FINAL NEXT-TOKEN KL ======================

def compute_final_next_token_kl(prompt, axis_name, band_name, alpha):
    device = get_model_device()
    try:
        with torch.no_grad():
            tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
            tokens, was_truncated = maybe_truncate_tokens_for_model(tokens, reserve_tokens=0)

            base_logits = model(tokens)[:, -1, :].float()
            if float(alpha) == 0.0:
                patched_logits = base_logits.clone()
            else:
                with model.hooks(fwd_hooks=make_axis_hooks(axis_name, band_name, alpha)):
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
    except Exception as exc:
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
            "final_kl_error": repr(exc),
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ====================== MAIN GENERATION RUN ======================

def count_total_alpha_points():
    return len(AXIS_NAMES) * len(AXIS_LAYER_BANDS) * len(AXIS_ALPHA_VALUES) * len(BASE_CONDITIONS)


def run_generation_experiment():
    pos_sha = sha256_text(POS_BASE_TEXT)
    control_sha = sha256_text(CONTROL_BASE_TEXT)
    run_started_at = datetime.now().isoformat(timespec="seconds")
    axis_manifest = write_axis_manifest()

    total_runs = (
        len(TEST_TASKS)
        * count_total_alpha_points()
        * sum(int(m["n_samples"]) for m in GENERATION_MODES)
    )

    with open(OUTPUT_BASE_TEXT_TXT, "w", encoding="utf-8") as f:
        f.write("=== TARGET BASE TEXT ===\n")
        f.write(POS_BASE_TEXT)
        f.write("\n\n=== CONTROL BASE TEXT ===\n")
        f.write(CONTROL_BASE_TEXT)

    print("\n============================================================")
    print("X_ORDER_ORTH AXIS STEERING GENERATION")
    print("============================================================")
    print(f"RUN_TAG: {RUN_TAG}")
    print(f"Axis artifact: {AXIS_STATE['source_name']}")
    print(f"Axis names: {AXIS_NAMES}")
    print(f"Axis bands: {AXIS_LAYER_BANDS}")
    print(f"Axis norm mode: {AXIS_NORM_MODE}")
    print(f"Axis position mode: {AXIS_POSITION_MODE}")
    print(f"Layer index mode: {AXIS_LAYER_INDEX_MODE}")
    print(f"BASE_CONDITIONS: {BASE_CONDITIONS}")
    print(f"total_runs: {total_runs}")
    print(f"Axis manifest saved: {OUTPUT_AXIS_MANIFEST_CSV}")
    if len(axis_manifest):
        print(axis_manifest.to_string(index=False))

    rows = []
    run_counter = 0

    for task_id, task in enumerate(TEST_TASKS):
        task_clean = str(task).strip()
        print("\n\n############################")
        print(f"### TASK {task_id}")
        print("############################")
        print(task_clean)

        for base_condition in BASE_CONDITIONS:
            base_text = base_text_for_condition(base_condition)
            full_prompt = build_analysis_prompt(base_text, task_clean)

            for axis_name in AXIS_NAMES:
                for band_name in AXIS_LAYER_BANDS:
                    spec = INTERVENTION_SPECS[(str(axis_name), str(band_name))]
                    for mode_cfg in GENERATION_MODES:
                        generation_mode = mode_cfg["generation_mode"]
                        do_sample = bool(mode_cfg["do_sample"])
                        temperature = float(mode_cfg["temperature"])
                        n_samples = int(mode_cfg["n_samples"])

                        for sample_id in range(n_samples):
                            seed = make_seed(
                                task_id=task_id,
                                axis_name=axis_name,
                                band_name=band_name,
                                sample_id=sample_id,
                                generation_mode=generation_mode,
                                base_condition=base_condition,
                            )

                            for alpha_index, alpha in enumerate(AXIS_ALPHA_VALUES):
                                alpha = float(alpha)
                                run_counter += 1
                                print(
                                    f"\n=== RUN {run_counter}/{total_runs} | "
                                    f"TASK {task_id} | BASE {base_condition} | "
                                    f"AXIS {axis_name} | BAND {band_name} | "
                                    f"{generation_mode} | SAMPLE {sample_id} | "
                                    f"ALPHA {alpha} ({alpha_index + 1}/{len(AXIS_ALPHA_VALUES)}) ==="
                                )

                                set_reproducible_seed(seed)
                                started = time.time()
                                error = ""

                                try:
                                    output_raw = generate_with_axis_steering(
                                        prompt=full_prompt,
                                        axis_name=axis_name,
                                        band_name=band_name,
                                        alpha=alpha,
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                    )
                                    output_text = strip_prompt_from_generation(output_raw, full_prompt)
                                except Exception as exc:
                                    output_raw = ""
                                    output_text = ""
                                    error = repr(exc)
                                    print(f"ERROR: {error}")

                                elapsed_sec = time.time() - started
                                if output_text:
                                    print(output_text)

                                metrics = compute_text_metrics(output_text)
                                if RUN_FINAL_NEXT_TOKEN_KL_DURING_GENERATION:
                                    final_kl_metrics = compute_final_next_token_kl(
                                        prompt=full_prompt,
                                        axis_name=axis_name,
                                        band_name=band_name,
                                        alpha=alpha,
                                    )
                                else:
                                    final_kl_metrics = {}

                                row = {
                                    "run_tag": RUN_TAG,
                                    "run_started_at": run_started_at,
                                    "row_created_at": datetime.now().isoformat(timespec="seconds"),
                                    "base_condition": str(base_condition),
                                    "base_text_sha256": sha256_text(base_text),
                                    "target_base_text_sha256": pos_sha,
                                    "control_base_text_sha256": control_sha,
                                    "control_base_text_source": CONTROL_BASE_TEXT_SOURCE,
                                    "base_text_preview": safe_preview(base_text, n=1000),
                                    "task_id": int(task_id),
                                    "task": task_clean,
                                    "axis_name": str(axis_name),
                                    "band_name": str(band_name),
                                    "alpha": float(alpha),
                                    "alpha_index": int(alpha_index),
                                    "alpha_count_for_axis_band": int(len(AXIS_ALPHA_VALUES)),
                                    "axis_norm_mode": str(spec["axis_norm_mode"]),
                                    "axis_position_mode": str(AXIS_POSITION_MODE),
                                    "axis_layer_index_mode": str(AXIS_LAYER_INDEX_MODE),
                                    "hook_layers": str(spec["hook_layers"]),
                                    "grade4_layers": str([
                                        hook_layer_to_grade4_layer(h, spec["direction"])
                                        for h in spec["hook_layers"]
                                    ]),
                                    "raw_axis_band_norm": float(spec["raw_axis_band_norm"]),
                                    "intervention_axis_band_norm": float(spec["intervention_axis_band_norm"]),
                                    "natural_scale_target_band_norm": float(spec["natural_scale_target_band_norm"]),
                                    "natural_scale_source_axis_count": int(spec["natural_scale_source_axis_count"]),
                                    "effective_intervention_l2": float(abs(alpha) * spec["intervention_axis_band_norm"]),
                                    "generation_mode": generation_mode,
                                    "do_sample": do_sample,
                                    "temperature": temperature,
                                    "sample_id": int(sample_id),
                                    "seed": int(seed),
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
                                rows.append(row)
                                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows).reset_index(drop=True)
    baseline_cols = [
        "base_condition",
        "task_id",
        "axis_name",
        "band_name",
        "axis_norm_mode",
        "axis_position_mode",
        "generation_mode",
        "sample_id",
    ]
    baseline_df = df[df["alpha"] == 0.0].copy()
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
        "output": "baseline_alpha0_output",
        "output_char_len": "baseline_output_char_len",
        "output_word_count": "baseline_output_word_count",
        "output_token_count": "baseline_output_token_count",
        "diagnostic_keyword_count": "baseline_diagnostic_keyword_count",
        "contrastive_marker_count": "baseline_contrastive_marker_count",
        "negation_marker_count": "baseline_negation_marker_count",
    })
    df = df.merge(baseline_df, on=baseline_cols, how="left")
    df["exact_match_to_alpha0"] = (
        df["output"].fillna("") == df["baseline_alpha0_output"].fillna("")
    ).astype(int)
    df["jaccard_similarity_to_alpha0"] = df.apply(
        lambda r: jaccard_similarity(r["output"], r["baseline_alpha0_output"]),
        axis=1,
    )
    df["delta_char_len_vs_alpha0"] = df["output_char_len"] - df["baseline_output_char_len"]
    df["delta_word_count_vs_alpha0"] = df["output_word_count"] - df["baseline_output_word_count"]
    df["delta_token_count_vs_alpha0"] = df["output_token_count"] - df["baseline_output_token_count"]
    df["delta_diagnostic_keywords_vs_alpha0"] = (
        df["diagnostic_keyword_count"] - df["baseline_diagnostic_keyword_count"]
    )
    df["delta_contrastive_markers_vs_alpha0"] = (
        df["contrastive_marker_count"] - df["baseline_contrastive_marker_count"]
    )
    df["delta_negation_markers_vs_alpha0"] = (
        df["negation_marker_count"] - df["baseline_negation_marker_count"]
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print("\n============================================================")
    print("X_ORDER_ORTH AXIS STEERING GENERATION DONE")
    print("============================================================")
    print(f"Main table: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}")

    summary = df.groupby(
        ["base_condition", "axis_name", "band_name", "axis_norm_mode", "axis_position_mode", "generation_mode", "alpha"],
        as_index=False,
    ).agg(
        rows=("output", "count"),
        exact_match_rate_to_alpha0=("exact_match_to_alpha0", "mean"),
        mean_jaccard_to_alpha0=("jaccard_similarity_to_alpha0", "mean"),
        mean_output_token_count=("output_token_count", "mean"),
        mean_delta_token_count_vs_alpha0=("delta_token_count_vs_alpha0", "mean"),
        mean_diagnostic_keyword_count=("diagnostic_keyword_count", "mean"),
        mean_delta_diagnostic_keywords_vs_alpha0=("delta_diagnostic_keywords_vs_alpha0", "mean"),
        mean_contrastive_marker_count=("contrastive_marker_count", "mean"),
        mean_delta_contrastive_markers_vs_alpha0=("delta_contrastive_markers_vs_alpha0", "mean"),
        mean_negation_marker_count=("negation_marker_count", "mean"),
        mean_delta_negation_markers_vs_alpha0=("delta_negation_markers_vs_alpha0", "mean"),
        mean_final_next_token_kl=("final_next_token_kl_base_to_patched", "mean"),
        mean_final_next_token_js=("final_next_token_js_divergence", "mean"),
        mean_final_logit_l2=("final_logit_l2", "mean"),
        mean_final_logit_max_abs=("final_logit_max_abs", "mean"),
        final_top_token_changed_rate=("final_top_token_changed", "mean"),
        mean_effective_intervention_l2=("effective_intervention_l2", "mean"),
        mean_elapsed_sec=("elapsed_sec", "mean"),
    )
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"Summary saved: {OUTPUT_SUMMARY_CSV}")
    print(summary.to_string(index=False))
    return df, summary


# ====================== TEACHER-FORCED KL ======================

def teacher_forced_per_token_kl(
    prompt,
    reference_continuation,
    axis_name,
    band_name,
    alpha,
    max_reference_tokens=None,
):
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
    start_pos = prompt_len - 1
    end_pos = start_pos + ref_len

    try:
        with torch.no_grad():
            base_logits_full = model(full_tokens).float()
            if float(alpha) == 0.0:
                patched_logits_full = base_logits_full.clone()
            else:
                with model.hooks(fwd_hooks=make_axis_hooks(axis_name, band_name, alpha)):
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
    except Exception as exc:
        return {
            "tf_kl_error": repr(exc),
            "tf_prompt_truncated": int(tf_prompt_truncated),
            "tf_reference_token_count": 0,
        }, pd.DataFrame()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_teacher_forced_kl_postprocessing(df):
    df = df.reset_index(drop=True).copy()
    baseline_keys = [
        "base_condition",
        "task_id",
        "axis_name",
        "band_name",
        "axis_norm_mode",
        "axis_position_mode",
        "generation_mode",
        "sample_id",
    ]

    baseline_map = {}
    baseline_rows = df[df["alpha"] == 0.0].copy()
    for _, row in baseline_rows.iterrows():
        key = tuple(row[k] for k in baseline_keys)
        baseline_map[key] = str(row["output"]) if pd.notna(row["output"]) else ""

    summary_rows = []
    detail_frames = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Teacher-forced per-token KL"):
        key = tuple(row[k] for k in baseline_keys)
        reference_continuation = baseline_map.get(key, "")
        base_text = base_text_for_condition(row["base_condition"])
        prompt = build_analysis_prompt(base_text=base_text, task=str(row["task"]))

        axis_name = str(row["axis_name"])
        band_name = str(row["band_name"])
        alpha = float(row["alpha"])

        kl_summary, kl_detail = teacher_forced_per_token_kl(
            prompt=prompt,
            reference_continuation=reference_continuation,
            axis_name=axis_name,
            band_name=band_name,
            alpha=alpha,
            max_reference_tokens=MAX_REFERENCE_TOKENS_FOR_TF_KL,
        )

        kl_summary["row_index"] = int(idx)
        summary_rows.append(kl_summary)

        if SAVE_PER_TOKEN_DETAILS and kl_detail is not None and len(kl_detail) > 0:
            meta_cols = {
                "row_index": int(idx),
                "base_condition": row["base_condition"],
                "task_id": row["task_id"],
                "axis_name": axis_name,
                "band_name": band_name,
                "alpha": alpha,
                "alpha_index": row.get("alpha_index", None),
                "alpha_count_for_axis_band": row.get("alpha_count_for_axis_band", None),
                "axis_norm_mode": row.get("axis_norm_mode", ""),
                "axis_position_mode": row.get("axis_position_mode", ""),
                "generation_mode": row["generation_mode"],
                "sample_id": row["sample_id"],
            }
            for key_name, value in meta_cols.items():
                kl_detail[key_name] = value
            detail_frames.append(kl_detail)

        if len(summary_rows) % 20 == 0:
            tmp_summary = pd.DataFrame(summary_rows).set_index("row_index")
            tmp_with_kl = df.join(tmp_summary, how="left")
            tmp_with_kl.to_csv(OUTPUT_WITH_TF_KL_CSV, index=False)

    kl_summary_df = pd.DataFrame(summary_rows).set_index("row_index")
    df_with_kl = df.join(kl_summary_df, how="left")
    df_with_kl.to_csv(OUTPUT_WITH_TF_KL_CSV, index=False)
    print(f"\nSaved: {OUTPUT_WITH_TF_KL_CSV}")

    if SAVE_PER_TOKEN_DETAILS and len(detail_frames) > 0:
        details_df = pd.concat(detail_frames, ignore_index=True)
        details_df = details_df.sort_values([
            "base_condition",
            "axis_name",
            "band_name",
            "alpha",
            "generation_mode",
            "task_id",
            "sample_id",
            "token_step",
        ])
        details_df.to_csv(OUTPUT_TF_KL_DETAILS_CSV, index=False)
        print(f"Saved: {OUTPUT_TF_KL_DETAILS_CSV}")
        print(f"Per-token rows: {len(details_df)}")

    group_summary = df_with_kl.groupby(
        ["base_condition", "axis_name", "band_name", "axis_norm_mode", "axis_position_mode", "generation_mode", "alpha"],
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

    group_summary.to_csv(OUTPUT_TF_KL_SUMMARY_CSV, index=False)
    print(f"Saved: {OUTPUT_TF_KL_SUMMARY_CSV}")
    print(group_summary.to_string(index=False))
    return df_with_kl, group_summary


# ====================== RUN ======================

steering_df, steering_summary_df = run_generation_experiment()

if RUN_TEACHER_FORCED_KL_AFTER_GENERATION:
    steering_with_tf_kl, tf_kl_summary = run_teacher_forced_kl_postprocessing(steering_df)
else:
    print("Teacher-forced KL skipped: RUN_TEACHER_FORCED_KL_AFTER_GENERATION = False")

print("\n============================================================")
print("ALL DONE")
print("============================================================")
print(f"1. Axis manifest: {OUTPUT_AXIS_MANIFEST_CSV}")
print(f"2. Steering full metrics: {OUTPUT_CSV}")
print(f"3. Steering summary: {OUTPUT_SUMMARY_CSV}")
print(f"4. Full metrics + teacher-forced KL: {OUTPUT_WITH_TF_KL_CSV}")
print(f"5. Teacher-forced KL summary: {OUTPUT_TF_KL_SUMMARY_CSV}")
if SAVE_PER_TOKEN_DETAILS:
    print(f"6. Per-token KL details: {OUTPUT_TF_KL_DETAILS_CSV}")
