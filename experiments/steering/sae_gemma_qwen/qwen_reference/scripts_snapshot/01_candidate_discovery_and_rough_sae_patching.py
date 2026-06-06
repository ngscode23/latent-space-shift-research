# =============================================
# CAUSAL MEDIATION ANALYSIS + TARGETED SAE PATCHING
# Qwen/Qwen3.5-9B-Base + Qwen-Scope W64K-L0_50
# =============================================
#
# Main output:
#   causal_mediation_sae_order_features_results.csv
#
# Main columns:
#   real_layer
#   csv_layer
#   feature_index
#   x_order_orth_delta
#   status
#   mediated_effect
#   target_hidden_delta_l2
#   target_logit_l2
#   target_logit_mean_abs
#   target_kl_base_to_patched
#   target_top1_flip_rate
#   control_mediated_effect
#   target_minus_control_mediated_effect
#   patch_mode
#   patch_position
#   patch_value
#   model_name
#   sae_repo_id
#   sae_top_k
#   sae_relu_before_topk
#   prompt_batch_size
#   max_length
#
# If control prompts are provided, extra control/diff columns are added:
#   control_hidden_delta_l2
#   control_logit_l2
#   control_logit_mean_abs
#   control_kl_base_to_patched
#   control_top1_flip_rate
#   diff_hidden_delta_l2
#   diff_logit_l2
#   diff_kl_base_to_patched
#
# Optional context output:
#   sae_feature_top_activating_contexts.csv
#
# Use the standard Grade 4 artifact:
#   /content/sae_order_feature_contrast.csv
#
# Do not use metric-lab's sae_order_feature_contrast_matrix.csv here.

from __future__ import annotations

import argparse
import gc
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    if "COLAB_GPU" not in os.environ:
        raise
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "transformers>=4.51.0",
            "accelerate>=0.33.0",
            "huggingface_hub",
            "pandas",
            "tqdm",
            "safetensors",
        ]
    )
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ====================== SETTINGS ======================

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
TRUST_REMOTE_CODE = True

CONTRAST_CSV_PATH = "/content/sae_order_feature_contrast.csv"
OUTPUT_CSV_PATH = "causal_mediation_sae_order_features_results.csv"
CONTEXTS_CSV_PATH = "sae_feature_top_activating_contexts.csv"

TOP_K = 30
STATUS_REGEX = "order_specific|order_enriched"
LAYER_COLUMN = "layer"
FEATURE_COLUMN = "feature_index"
SCORE_COLUMN = "x_order_orth_component_delta"
STATUS_COLUMN = "interpretation_status"

BATCH_SIZE = 1
PROMPT_BATCH_SIZE = BATCH_SIZE
MAX_LENGTH = 8192
USE_4BIT_MODEL = True
MODEL_DTYPE = "auto"  # "auto", "float16", "bfloat16", "float32"

SAE_TOP_K = 50
SAE_DTYPE = "float16"
SAE_CACHE_LAYERS = 1
SAE_TOKEN_CHUNK_SIZE = 64
SAE_RELU_BEFORE_TOPK = False

PATCH_MODE = "reconstruct"  # "reconstruct" or "direction_delta"
PATCH_POSITION = "all_tokens"  # "all_tokens" or "last_token"
PATCH_VALUE = 0.0

INSPECT_CONTEXTS_N = 0
CONTEXT_TOP_N = 12
CONTEXT_WINDOW = 14

AUTO_INSTALL_PACKAGES = True
AUTO_DOWNLOAD_COLAB_RESULTS = False

TARGET_PROMPTS_TXT_PATH = "/content/target_prompts.txt"
CONTROL_PROMPTS_TXT_PATH = "/content/control_prompts.txt"


# ====================== PROMPTS ======================
# You can paste a whole text chunk. A list is not required.

prompts_target = """
PASTE_TARGET_PROMPT_HERE
""".strip()

prompts_control = """
PASTE_CONTROL_PROMPT_HERE
""".strip()


# ====================== UTILITIES ======================


def install_packages_if_needed() -> None:
    if not AUTO_INSTALL_PACKAGES or "COLAB_GPU" not in os.environ:
        return
    checks = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "huggingface_hub": "huggingface_hub",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "safetensors": "safetensors",
    }
    if USE_4BIT_MODEL:
        checks["bitsandbytes"] = "bitsandbytes"
    missing = []
    for package_name, import_name in checks.items():
        try:
            __import__(import_name)
        except Exception:
            missing.append(package_name)
    if missing:
        print("Installing missing packages:", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])


def clear_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_dtype(name: str) -> torch.dtype:
    name = str(name).strip().lower().replace("torch.", "")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Bad dtype={name!r}. Use auto/float16/bfloat16/float32.")
    return mapping[name]


def safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN") or token
    except Exception:
        pass
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def nested_getattr(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def get_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        for p in model.parameters():
            if p.device.type != "meta":
                return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_decoder_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    candidates = [
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "model.language_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "decoder.layers",
        "model.decoder.layers",
    ]
    for path in candidates:
        layers = nested_getattr(model, path)
        if layers is None:
            continue
        try:
            out = list(layers)
        except Exception:
            continue
        if out:
            return out
    raise AttributeError("Could not locate decoder layers. For Qwen expected model.model.layers.")


def move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}


def batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def is_placeholder_text(text: str) -> bool:
    s = str(text).strip()
    placeholders = {
        "",
        "PASTE_TARGET_PROMPT_HERE",
        "PASTE_CONTROL_PROMPT_HERE",
        "PASTE_PROMPT_HERE",
        "PASTE_TEXT_HERE",
    }
    return s in placeholders or (s.startswith("PASTE_") and s.endswith("_HERE"))


def normalize_prompt_value(value: Any, name: str, required: bool) -> List[str]:
    if value is None:
        out: List[str] = []
    elif isinstance(value, str):
        text = value.strip()
        out = [] if is_placeholder_text(text) else [text]
    elif isinstance(value, (list, tuple)):
        out = [str(x).strip() for x in value if str(x).strip() and not is_placeholder_text(str(x))]
    else:
        raise TypeError(f"{name} must be a string or a list of strings, got {type(value).__name__}")
    if required and not out:
        raise SystemExit(f"{name} is empty. Put your whole text chunk into {name}.")
    return out


def read_whole_file_prompt(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    return [] if not text else [text]


def resolve_prompts() -> Tuple[List[str], List[str]]:
    target = normalize_prompt_value(prompts_target, "prompts_target", required=False)
    control = normalize_prompt_value(prompts_control, "prompts_control", required=False)
    if not target:
        target = read_whole_file_prompt(TARGET_PROMPTS_TXT_PATH)
        if target:
            print(f"Target prompt read from {TARGET_PROMPTS_TXT_PATH}", flush=True)
    if not control:
        control = read_whole_file_prompt(CONTROL_PROMPTS_TXT_PATH)
        if control:
            print(f"Control prompt read from {CONTROL_PROMPTS_TXT_PATH}", flush=True)
    if not target:
        raise SystemExit("No target prompts. Fill prompts_target or create /content/target_prompts.txt.")
    return target, control


def tokenize_batch(tokenizer: AutoTokenizer, texts: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
    kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": MAX_LENGTH is not None,
    }
    if MAX_LENGTH is not None:
        kwargs["max_length"] = int(MAX_LENGTH)
    x = tokenizer(list(texts), **kwargs)
    x.pop("token_type_ids", None)
    return move_inputs_to_device(x, device)


def gather_last_token_tensor(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = attention_mask.sum(dim=1).to(tensor.device).clamp(min=1) - 1
    batch_idx = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch_idx, last_idx, :]


def logit_metrics(base_logits: torch.Tensor, patched_logits: torch.Tensor) -> Dict[str, float]:
    base = base_logits.float()
    patched = patched_logits.float()
    diff = patched - base
    base_logp = F.log_softmax(base, dim=-1)
    patched_logp = F.log_softmax(patched, dim=-1)
    base_p = base_logp.exp()
    kl = (base_p * (base_logp - patched_logp)).sum(dim=-1)
    return {
        "logit_l2": diff.norm(dim=-1).mean().item(),
        "logit_mean_abs": diff.abs().mean(dim=-1).mean().item(),
        "kl_base_to_patched": kl.mean().item(),
        "top1_flip_rate": (base.argmax(dim=-1) != patched.argmax(dim=-1)).float().mean().item(),
    }


def hidden_delta_l2(base_hidden: torch.Tensor, patched_hidden: torch.Tensor) -> float:
    return (base_hidden.float() - patched_hidden.float()).norm(dim=-1).mean().item()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


# ====================== QWEN-SCOPE TOP-K SAE ======================


@dataclass
class QwenScopeTopKSAE:
    layer: int
    W_enc: torch.Tensor  # [d_sae, d_model]
    W_dec: torch.Tensor  # [d_model, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    b_dec: torch.Tensor  # [d_model]
    top_k: int = 50
    relu_before_topk: bool = False

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        layer: int,
        device: torch.device,
        dtype: torch.dtype,
        top_k: int,
        relu_before_topk: bool,
    ) -> "QwenScopeTopKSAE":
        filename = f"layer{int(layer)}.sae.pt"
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        state = safe_torch_load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"{filename}: expected dict checkpoint, got {type(state).__name__}")

        required = ["W_enc", "W_dec", "b_enc"]
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"{filename}: missing keys {missing}. Found keys: {list(state.keys())[:50]}")

        W_enc = state["W_enc"].detach().cpu()
        W_dec = state["W_dec"].detach().cpu()
        b_enc = state["b_enc"].detach().cpu().flatten()
        b_dec_raw = state.get("b_dec", None)
        b_dec = None if b_dec_raw is None else b_dec_raw.detach().cpu().flatten()

        d_sae = int(b_enc.numel())
        if b_dec is not None:
            d_model = int(b_dec.numel())
        elif W_enc.ndim == 2:
            d_model = int(W_enc.shape[1] if W_enc.shape[0] == d_sae else W_enc.shape[0])
        else:
            raise ValueError(f"{filename}: invalid W_enc shape {tuple(W_enc.shape)}")

        if tuple(W_enc.shape) == (d_sae, d_model):
            W_enc = W_enc.contiguous()
        elif tuple(W_enc.shape) == (d_model, d_sae):
            W_enc = W_enc.T.contiguous()
        else:
            raise ValueError(
                f"{filename}: W_enc shape={tuple(W_enc.shape)}, expected "
                f"{(d_sae, d_model)} or {(d_model, d_sae)}"
            )

        if tuple(W_dec.shape) == (d_model, d_sae):
            W_dec = W_dec.contiguous()
        elif tuple(W_dec.shape) == (d_sae, d_model):
            W_dec = W_dec.T.contiguous()
        else:
            raise ValueError(
                f"{filename}: W_dec shape={tuple(W_dec.shape)}, expected "
                f"{(d_model, d_sae)} or {(d_sae, d_model)}"
            )

        if b_dec is None:
            b_dec = torch.zeros(d_model)
        if tuple(b_dec.shape) != (d_model,):
            raise ValueError(f"{filename}: b_dec shape={tuple(b_dec.shape)}, expected {(d_model,)}")

        obj = cls(
            layer=int(layer),
            W_enc=W_enc.to(device=device, dtype=dtype, non_blocking=True),
            W_dec=W_dec.to(device=device, dtype=dtype, non_blocking=True),
            b_enc=b_enc.to(device=device, dtype=dtype, non_blocking=True),
            b_dec=b_dec.to(device=device, dtype=dtype, non_blocking=True),
            top_k=int(top_k),
            relu_before_topk=bool(relu_before_topk),
        )
        obj.validate()
        return obj

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    def validate(self) -> None:
        if tuple(self.W_enc.shape) != (self.d_sae, self.d_model):
            raise ValueError(f"W_enc shape mismatch: {tuple(self.W_enc.shape)}")
        if tuple(self.W_dec.shape) != (self.d_model, self.d_sae):
            raise ValueError(f"W_dec shape mismatch: {tuple(self.W_dec.shape)}")
        if tuple(self.b_enc.shape) != (self.d_sae,):
            raise ValueError(f"b_enc shape mismatch: {tuple(self.b_enc.shape)}")
        if tuple(self.b_dec.shape) != (self.d_model,):
            raise ValueError(f"b_dec shape mismatch: {tuple(self.b_dec.shape)}")
        if not (0 < int(self.top_k) <= self.d_sae):
            raise ValueError(f"Bad top_k={self.top_k} for d_sae={self.d_sae}")

    def _preacts(self, residual_2d: torch.Tensor) -> torch.Tensor:
        if residual_2d.shape[-1] != self.d_model:
            raise ValueError(f"residual dim={residual_2d.shape[-1]} != SAE d_model={self.d_model}")
        x = residual_2d.to(device=self.device, dtype=self.dtype)
        pre = x @ self.W_enc.T + self.b_enc
        if self.relu_before_topk:
            pre = pre.clamp_min(0)
        return pre

    @torch.no_grad()
    def selected_feature_acts_2d(self, residual_2d: torch.Tensor, feature_indices: Sequence[int]) -> Dict[int, torch.Tensor]:
        pre = self._preacts(residual_2d)
        vals, idx = pre.topk(k=self.top_k, dim=-1)
        del pre
        out: Dict[int, torch.Tensor] = {}
        for feature_index in feature_indices:
            f = int(feature_index)
            if f < 0 or f >= self.d_sae:
                raise ValueError(f"feature_index={f} outside 0..{self.d_sae - 1}")
            mask = idx.eq(f)
            out[f] = torch.where(mask, vals, torch.zeros_like(vals)).sum(dim=-1)
        return out

    @torch.no_grad()
    def reconstruct_patched_2d(self, residual_2d: torch.Tensor, feature_indices: Sequence[int], patch_value: float) -> torch.Tensor:
        pre = self._preacts(residual_2d)
        vals, idx = pre.topk(k=self.top_k, dim=-1)
        acts = torch.zeros_like(pre)
        del pre
        acts.scatter_(-1, idx, vals)
        for feature_index in feature_indices:
            f = int(feature_index)
            if f < 0 or f >= self.d_sae:
                raise ValueError(f"feature_index={f} outside 0..{self.d_sae - 1}")
            acts[:, f] = float(patch_value)
        decoded = acts @ self.W_dec.T + self.b_dec
        return decoded

    @torch.no_grad()
    def direction_delta_patched_2d(self, residual_2d: torch.Tensor, feature_indices: Sequence[int], patch_value: float) -> torch.Tensor:
        patched = residual_2d.to(device=self.device, dtype=self.dtype)
        acts = self.selected_feature_acts_2d(patched, feature_indices)
        patch_value_t = torch.tensor(float(patch_value), device=self.device, dtype=self.dtype)
        for feature_index, act in acts.items():
            direction = self.W_dec[:, int(feature_index)]
            patched = patched + (patch_value_t - act).unsqueeze(-1) * direction.view(1, -1)
        return patched

    @torch.no_grad()
    def patch_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_indices: Sequence[int],
        patch_value: float,
        patch_mode: str,
        patch_position: str,
        token_chunk_size: int,
    ) -> torch.Tensor:
        orig_shape = hidden.shape
        orig_dtype = hidden.dtype
        orig_device = hidden.device
        flat = hidden.reshape(-1, hidden.shape[-1])

        attention_mask = attention_mask.to(device=hidden.device).bool()
        if patch_position == "all_tokens":
            patch_mask = attention_mask.reshape(-1)
        elif patch_position == "last_token":
            patch_mask_2d = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.bool, device=hidden.device)
            last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            patch_mask_2d[batch_idx, last_idx] = True
            patch_mask = patch_mask_2d.reshape(-1)
        else:
            raise ValueError("PATCH_POSITION must be 'all_tokens' or 'last_token'.")

        row_indices = torch.where(patch_mask)[0]
        if row_indices.numel() == 0:
            return hidden

        out_flat = flat.clone()
        token_chunk_size = max(1, int(token_chunk_size))
        for start in range(0, int(row_indices.numel()), token_chunk_size):
            idx = row_indices[start : start + token_chunk_size]
            chunk = flat.index_select(0, idx).to(device=self.device, dtype=self.dtype)
            if patch_mode == "reconstruct":
                patched_chunk = self.reconstruct_patched_2d(chunk, feature_indices, patch_value)
            elif patch_mode == "direction_delta":
                patched_chunk = self.direction_delta_patched_2d(chunk, feature_indices, patch_value)
            else:
                raise ValueError("PATCH_MODE must be 'reconstruct' or 'direction_delta'.")
            out_flat.index_copy_(0, idx, patched_chunk.to(device=orig_device, dtype=orig_dtype))
            del chunk, patched_chunk
            clear_mem()
        return out_flat.reshape(orig_shape)

    @torch.no_grad()
    def feature_scores(self, hidden: torch.Tensor, feature_index: int, token_chunk_size: int) -> torch.Tensor:
        orig_shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        outs = []
        token_chunk_size = max(1, int(token_chunk_size))
        for start in range(0, int(flat.shape[0]), token_chunk_size):
            chunk = flat[start : start + token_chunk_size].to(device=self.device, dtype=self.dtype)
            scores = self.selected_feature_acts_2d(chunk, [int(feature_index)])[int(feature_index)]
            outs.append(scores.detach().float().cpu())
            del chunk, scores
            clear_mem()
        return torch.cat(outs, dim=0).reshape(*orig_shape)


class SAECache:
    def __init__(self, repo_id: str, dtype: torch.dtype, top_k: int, max_layers: int, relu_before_topk: bool):
        self.repo_id = repo_id
        self.dtype = dtype
        self.top_k = int(top_k)
        self.max_layers = max(1, int(max_layers))
        self.relu_before_topk = bool(relu_before_topk)
        self.cache: Dict[Tuple[int, str], QwenScopeTopKSAE] = {}
        self.order: List[Tuple[int, str]] = []

    def clear(self) -> None:
        self.cache.clear()
        self.order.clear()
        clear_mem()

    def get(self, layer: int, device: torch.device) -> QwenScopeTopKSAE:
        key = (int(layer), str(torch.device(device)))
        if key in self.cache:
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        while len(self.cache) >= self.max_layers and self.order:
            old_key = self.order.pop(0)
            del self.cache[old_key]
            clear_mem()
        print(f"Loading Qwen-Scope SAE layer{int(layer)}.sae.pt -> {device}, dtype={self.dtype}", flush=True)
        sae = QwenScopeTopKSAE.from_hf(
            repo_id=self.repo_id,
            layer=int(layer),
            device=torch.device(device),
            dtype=self.dtype,
            top_k=self.top_k,
            relu_before_topk=self.relu_before_topk,
        )
        self.cache[key] = sae
        self.order.append(key)
        return sae


# ====================== MODEL / CSV ======================


def load_model_and_tokenizer() -> Tuple[torch.nn.Module, AutoTokenizer]:
    hf_token = resolve_hf_token()
    hf_kwargs = {"token": hf_token} if hf_token else {}
    dtype = parse_dtype(MODEL_DTYPE)

    print(f"Loading tokenizer: {MODEL_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True, **hf_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if USE_4BIT_MODEL and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            if "COLAB_GPU" not in os.environ:
                raise
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"])
            from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading model in 4-bit NF4", flush=True)
    else:
        print("Loading model without 4-bit quantization", flush=True)

    print(f"Loading model: {MODEL_NAME}", flush=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs, **hf_kwargs)
    mdl.eval()
    if hasattr(mdl.config, "use_cache"):
        mdl.config.use_cache = False

    layers = get_decoder_layers(mdl)
    print(
        f"Model loaded. layers={len(layers)}, input_device={get_input_device(mdl)}, model_dtype={dtype}",
        flush=True,
    )
    return mdl, tok


def load_top_mediators(csv_path: str, num_model_layers: int, top_k: int) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Contrast CSV not found: {path}")
    if path.name.endswith("_matrix.csv"):
        raise ValueError("This is metric-lab matrix CSV. Use standard sae_order_feature_contrast.csv.")

    df = pd.read_csv(path)
    print(f"CSV rows: {len(df)}", flush=True)

    required = [LAYER_COLUMN, FEATURE_COLUMN, SCORE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns {missing}. Columns: {list(df.columns)}")

    if "sae_name" in df.columns:
        sample_names = [str(x).lower() for x in df["sae_name"].dropna().unique()[:10]]
        if sample_names and not any("qwen" in x for x in sample_names):
            raise ValueError(f"CSV does not look like Qwen SAE contrast. sae_name sample: {sample_names}")

    if "feature_count" in df.columns:
        counts = sorted(set(int(x) for x in df["feature_count"].dropna().unique()))
        if counts and max(counts) < 60000:
            raise ValueError(f"CSV does not look like Qwen W64K. feature_count={counts}")

    work = df.copy()
    work[LAYER_COLUMN] = pd.to_numeric(work[LAYER_COLUMN], errors="coerce")
    work[FEATURE_COLUMN] = pd.to_numeric(work[FEATURE_COLUMN], errors="coerce")
    work[SCORE_COLUMN] = pd.to_numeric(work[SCORE_COLUMN], errors="coerce")
    work = work.dropna(subset=[LAYER_COLUMN, FEATURE_COLUMN, SCORE_COLUMN]).copy()
    work[LAYER_COLUMN] = work[LAYER_COLUMN].astype(int)
    work[FEATURE_COLUMN] = work[FEATURE_COLUMN].astype(int)

    layers_in_csv = sorted(work[LAYER_COLUMN].unique().tolist())
    print(f"CSV layers: {layers_in_csv}", flush=True)
    if layers_in_csv and min(layers_in_csv) >= 1 and max(layers_in_csv) <= num_model_layers:
        work["real_layer"] = work[LAYER_COLUMN] - 1
        print("Detected 1-based CSV layers -> real_layer = layer - 1", flush=True)
    else:
        work["real_layer"] = work[LAYER_COLUMN]
        print("Detected 0-based CSV layers -> real_layer = layer", flush=True)

    invalid = work[(work["real_layer"] < 0) | (work["real_layer"] >= num_model_layers)]
    if len(invalid):
        bad = sorted(set(int(x) for x in invalid["real_layer"].unique()))
        raise ValueError(f"real_layer outside model range 0..{num_model_layers - 1}: {bad}")

    if STATUS_COLUMN in work.columns:
        mask = work[STATUS_COLUMN].astype(str).str.contains(STATUS_REGEX, case=False, regex=True, na=False)
        mediators = work[mask].copy()
    else:
        print(f"WARNING: no {STATUS_COLUMN}; using all rows.", flush=True)
        mediators = work.copy()

    if len(mediators) == 0:
        raise ValueError(f"No rows matched STATUS_REGEX={STATUS_REGEX!r}.")

    mediators = mediators.sort_values(SCORE_COLUMN, ascending=False).head(int(top_k)).copy()
    mediators["real_layer"] = mediators["real_layer"].astype(int)
    mediators[FEATURE_COLUMN] = mediators[FEATURE_COLUMN].astype(int)

    show_cols = [LAYER_COLUMN, "real_layer", FEATURE_COLUMN, SCORE_COLUMN]
    if STATUS_COLUMN in mediators.columns:
        show_cols.append(STATUS_COLUMN)
    print(f"\nTOP-{top_k} Qwen order mediators:", flush=True)
    print(mediators[show_cols].to_string(index=False), flush=True)
    return mediators


# ====================== FORWARD / PATCH ======================


@torch.no_grad()
def forward_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts_batch: Sequence[str],
    real_layer: int,
    sae_cache: Optional[SAECache] = None,
    feature_indices: Optional[Sequence[int]] = None,
    capture_hidden: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_device = get_input_device(model)
    inputs = tokenize_batch(tokenizer, prompts_batch, input_device)
    attention_mask = inputs["attention_mask"]
    layers = get_decoder_layers(model)
    layer_module = layers[int(real_layer)]
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _module_input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden
        if feature_indices is not None and len(feature_indices) > 0:
            if sae_cache is None:
                raise RuntimeError("sae_cache required for patching")
            sae = sae_cache.get(int(real_layer), hidden.device)
            patched = sae.patch_hidden(
                hidden=hidden,
                attention_mask=attention_mask,
                feature_indices=feature_indices,
                patch_value=PATCH_VALUE,
                patch_mode=PATCH_MODE,
                patch_position=PATCH_POSITION,
                token_chunk_size=SAE_TOKEN_CHUNK_SIZE,
            )

        if capture_hidden:
            captured["last_hidden"] = gather_last_token_tensor(patched, attention_mask).detach().float().cpu()

        if feature_indices is None or len(feature_indices) == 0:
            return None
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
            last_logits = gather_last_token_tensor(out.logits, attention_mask).detach().float().cpu()
    finally:
        handle.remove()
        del inputs
        clear_mem()
    return last_logits, captured.get("last_hidden")


def compute_baseline_for_layer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    real_layer: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_all: List[torch.Tensor] = []
    hidden_all: List[torch.Tensor] = []
    batches = list(batched(prompts, PROMPT_BATCH_SIZE))
    for batch in tqdm(batches, desc=f"baseline L{real_layer}", leave=False):
        logits, hidden = forward_batch(
            model=model,
            tokenizer=tokenizer,
            prompts_batch=batch,
            real_layer=real_layer,
            capture_hidden=True,
        )
        if hidden is None:
            raise RuntimeError(f"Hidden capture failed for baseline layer {real_layer}")
        logits_all.append(logits)
        hidden_all.append(hidden)
    return torch.cat(logits_all, dim=0), torch.cat(hidden_all, dim=0)


def compute_patched_for_feature(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    real_layer: int,
    feature_index: int,
    sae_cache: SAECache,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_all: List[torch.Tensor] = []
    hidden_all: List[torch.Tensor] = []
    for batch in batched(prompts, PROMPT_BATCH_SIZE):
        logits, hidden = forward_batch(
            model=model,
            tokenizer=tokenizer,
            prompts_batch=batch,
            real_layer=real_layer,
            sae_cache=sae_cache,
            feature_indices=[int(feature_index)],
            capture_hidden=True,
        )
        if hidden is None:
            raise RuntimeError(f"Hidden capture failed for patched layer {real_layer}, feature {feature_index}")
        logits_all.append(logits)
        hidden_all.append(hidden)
    return torch.cat(logits_all, dim=0), torch.cat(hidden_all, dim=0)


def row_value(row: pd.Series, column: str, default: Any = "") -> Any:
    return row[column] if column in row.index else default


def run_mediation_experiment(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    top_mediators: pd.DataFrame,
    target_prompts: Sequence[str],
    control_prompts: Sequence[str],
    sae_cache: SAECache,
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    if len(top_mediators) == 0:
        raise ValueError("top_mediators is empty.")
    if not target_prompts:
        raise ValueError("target_prompts is empty.")

    for real_layer, layer_df in tqdm(list(top_mediators.groupby("real_layer", sort=True)), desc="layers"):
        real_layer = int(real_layer)
        sae_cache.clear()

        target_base_logits, target_base_hidden = compute_baseline_for_layer(
            model=model,
            tokenizer=tokenizer,
            prompts=target_prompts,
            real_layer=real_layer,
        )

        if control_prompts:
            control_base_logits, control_base_hidden = compute_baseline_for_layer(
                model=model,
                tokenizer=tokenizer,
                prompts=control_prompts,
                real_layer=real_layer,
            )
        else:
            control_base_logits = None
            control_base_hidden = None

        for _, row in tqdm(layer_df.iterrows(), total=len(layer_df), desc=f"features L{real_layer}", leave=False):
            feature_index = int(row[FEATURE_COLUMN])
            target_patched_logits, target_patched_hidden = compute_patched_for_feature(
                model=model,
                tokenizer=tokenizer,
                prompts=target_prompts,
                real_layer=real_layer,
                feature_index=feature_index,
                sae_cache=sae_cache,
            )
            target_logit_metrics = logit_metrics(target_base_logits, target_patched_logits)
            target_hidden_l2 = hidden_delta_l2(target_base_hidden, target_patched_hidden)

            rec: Dict[str, Any] = {
                "real_layer": real_layer,
                "csv_layer": int(row[LAYER_COLUMN]),
                "feature_index": feature_index,
                "x_order_orth_delta": finite_float(row_value(row, SCORE_COLUMN, float("nan"))),
                "status": str(row_value(row, STATUS_COLUMN, "")),
                "mediated_effect": target_hidden_l2,
                "target_hidden_delta_l2": target_hidden_l2,
                "target_logit_l2": target_logit_metrics["logit_l2"],
                "target_logit_mean_abs": target_logit_metrics["logit_mean_abs"],
                "target_kl_base_to_patched": target_logit_metrics["kl_base_to_patched"],
                "target_top1_flip_rate": target_logit_metrics["top1_flip_rate"],
                "control_mediated_effect": float("nan"),
                "target_minus_control_mediated_effect": float("nan"),
                "patch_mode": PATCH_MODE,
                "patch_position": PATCH_POSITION,
                "patch_value": float(PATCH_VALUE),
                "model_name": MODEL_NAME,
                "sae_repo_id": SAE_REPO,
                "sae_top_k": int(SAE_TOP_K),
                "sae_relu_before_topk": int(bool(SAE_RELU_BEFORE_TOPK)),
                "prompt_batch_size": int(PROMPT_BATCH_SIZE),
                "max_length": int(MAX_LENGTH) if MAX_LENGTH is not None else "",
            }

            if control_prompts:
                control_patched_logits, control_patched_hidden = compute_patched_for_feature(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=control_prompts,
                    real_layer=real_layer,
                    feature_index=feature_index,
                    sae_cache=sae_cache,
                )
                control_logit_metrics = logit_metrics(control_base_logits, control_patched_logits)
                control_hidden_l2 = hidden_delta_l2(control_base_hidden, control_patched_hidden)
                rec.update(
                    {
                        "control_mediated_effect": control_hidden_l2,
                        "target_minus_control_mediated_effect": target_hidden_l2 - control_hidden_l2,
                        "control_hidden_delta_l2": control_hidden_l2,
                        "control_logit_l2": control_logit_metrics["logit_l2"],
                        "control_logit_mean_abs": control_logit_metrics["logit_mean_abs"],
                        "control_kl_base_to_patched": control_logit_metrics["kl_base_to_patched"],
                        "control_top1_flip_rate": control_logit_metrics["top1_flip_rate"],
                        "diff_hidden_delta_l2": target_hidden_l2 - control_hidden_l2,
                        "diff_logit_l2": target_logit_metrics["logit_l2"] - control_logit_metrics["logit_l2"],
                        "diff_kl_base_to_patched": (
                            target_logit_metrics["kl_base_to_patched"]
                            - control_logit_metrics["kl_base_to_patched"]
                        ),
                    }
                )
                del control_patched_logits, control_patched_hidden

            results.append(rec)
            del target_patched_logits, target_patched_hidden
            clear_mem()

        del target_base_logits, target_base_hidden
        if control_base_logits is not None:
            del control_base_logits, control_base_hidden
        sae_cache.clear()

    if not results:
        raise ValueError("No mediation results were produced.")
    out = pd.DataFrame(results)
    if "target_minus_control_mediated_effect" in out.columns and out["target_minus_control_mediated_effect"].notna().any():
        return out.sort_values("target_minus_control_mediated_effect", ascending=False)
    return out.sort_values("mediated_effect", ascending=False)


# ====================== TOP ACTIVATING CONTEXTS ======================


@torch.no_grad()
def get_feature_top_contexts(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    real_layer: int,
    feature_index: int,
    sae_cache: SAECache,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    layers = get_decoder_layers(model)
    layer_module = layers[int(real_layer)]
    input_device = get_input_device(model)

    for batch_start in tqdm(range(0, len(texts), PROMPT_BATCH_SIZE), desc=f"contexts L{real_layer} F{feature_index}", leave=False):
        batch_texts = list(texts[batch_start : batch_start + PROMPT_BATCH_SIZE])
        inputs = tokenize_batch(tokenizer, batch_texts, input_device)
        captured: Dict[str, torch.Tensor] = {}

        def hook_fn(_module, _module_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()
            return None

        handle = layer_module.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError(f"Hidden capture failed for contexts layer {real_layer}")

        hidden = captured["hidden"]
        attention_mask = inputs["attention_mask"].detach().cpu()
        input_ids = inputs["input_ids"].detach().cpu()
        sae = sae_cache.get(int(real_layer), hidden.device)
        scores = sae.feature_scores(hidden, int(feature_index), SAE_TOKEN_CHUNK_SIZE)

        for b in range(scores.shape[0]):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len <= 0:
                continue
            valid_scores = scores[b, :valid_len]
            k = min(int(CONTEXT_TOP_N), int(valid_scores.numel()))
            values, positions = torch.topk(valid_scores, k=k)
            ids = input_ids[b, :valid_len].tolist()
            tokens = tokenizer.convert_ids_to_tokens(ids)
            for value, pos in zip(values.tolist(), positions.tolist()):
                pos = int(pos)
                left = max(0, pos - int(CONTEXT_WINDOW))
                right = min(valid_len, pos + int(CONTEXT_WINDOW) + 1)
                records.append(
                    {
                        "global_text_id": batch_start + b,
                        "real_layer": int(real_layer),
                        "feature_index": int(feature_index),
                        "activation": float(value),
                        "token_position": pos,
                        "token": tokens[pos],
                        "context": tokenizer.decode(ids[left:right], skip_special_tokens=False),
                        "context_tokens": " ".join(tokens[left:right]),
                    }
                )

        del inputs, captured, hidden, scores
        clear_mem()

    out = pd.DataFrame(records)
    if len(out):
        out = out.sort_values("activation", ascending=False).head(int(CONTEXT_TOP_N)).reset_index(drop=True)
    return out


def inspect_top_mediators_on_texts(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    top_mediators: pd.DataFrame,
    target_prompts: Sequence[str],
    sae_cache: SAECache,
) -> pd.DataFrame:
    if INSPECT_CONTEXTS_N <= 0:
        return pd.DataFrame()
    rows = []
    for _, row in top_mediators.head(int(INSPECT_CONTEXTS_N)).iterrows():
        real_layer = int(row["real_layer"])
        feature_index = int(row[FEATURE_COLUMN])
        print(f"\n=== TOP CONTEXTS: layer {real_layer}, feature {feature_index} ===", flush=True)
        try:
            ctx = get_feature_top_contexts(
                model=model,
                tokenizer=tokenizer,
                texts=target_prompts,
                real_layer=real_layer,
                feature_index=feature_index,
                sae_cache=sae_cache,
            )
            if len(ctx):
                print(ctx[["activation", "token", "context"]].to_string(index=False), flush=True)
                rows.append(ctx)
            else:
                print("No contexts found.", flush=True)
        except Exception as exc:
            print(f"Context extraction failed for L{real_layer} F{feature_index}: {exc!r}", flush=True)
        finally:
            sae_cache.clear()
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ====================== CLI / MAIN ======================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B Qwen-Scope SAE mediation/patching.")
    parser.add_argument("--csv", default=CONTRAST_CSV_PATH, help="Path to standard sae_order_feature_contrast.csv")
    parser.add_argument("--output", default=OUTPUT_CSV_PATH, help="Output mediation CSV path")
    parser.add_argument("--contexts-output", default=CONTEXTS_CSV_PATH, help="Output contexts CSV path")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of top order features to test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Prompt batch size")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Tokenizer max length")
    parser.add_argument("--no-control", action="store_true", help="Do not run control prompts")
    parser.add_argument("--no-inspect", action="store_true", help="Disable top activating context extraction")
    parser.add_argument("--inspect-n-features", type=int, default=INSPECT_CONTEXTS_N, help="Extract contexts for this many top features")
    parser.add_argument("--patch-mode", choices=["reconstruct", "direction_delta"], default=PATCH_MODE)
    parser.add_argument("--patch-position", choices=["all_tokens", "last_token"], default=PATCH_POSITION)
    parser.add_argument("--no-4bit", action="store_true", help="Load model without 4-bit quantization")
    return parser.parse_args()


def main() -> None:
    global PROMPT_BATCH_SIZE, BATCH_SIZE, MAX_LENGTH, PATCH_MODE, PATCH_POSITION, USE_4BIT_MODEL, INSPECT_CONTEXTS_N

    install_packages_if_needed()
    args = parse_args()

    BATCH_SIZE = max(1, int(args.batch_size))
    PROMPT_BATCH_SIZE = BATCH_SIZE
    MAX_LENGTH = int(args.max_length) if args.max_length is not None and int(args.max_length) > 0 else None
    PATCH_MODE = str(args.patch_mode)
    PATCH_POSITION = str(args.patch_position)
    INSPECT_CONTEXTS_N = 0 if args.no_inspect else max(0, int(args.inspect_n_features))
    if args.no_4bit:
        USE_4BIT_MODEL = False

    target_prompts, control_prompts = resolve_prompts()
    if args.no_control:
        control_prompts = []

    print("\n=== QWEN3.5 SAE MEDIATION START ===", flush=True)
    print(f"Target prompts: {len(target_prompts)}", flush=True)
    print(f"Control prompts: {len(control_prompts)}", flush=True)
    print(f"Patch mode: {PATCH_MODE}; patch position: {PATCH_POSITION}", flush=True)
    print(f"Batch size: {PROMPT_BATCH_SIZE}; max_length: {MAX_LENGTH}", flush=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This run will be very slow.", flush=True)

    mdl, tok = load_model_and_tokenizer()
    layers = get_decoder_layers(mdl)
    top_mediators = load_top_mediators(args.csv, num_model_layers=len(layers), top_k=int(args.top_k))

    sae_cache = SAECache(
        repo_id=SAE_REPO,
        dtype=parse_dtype(SAE_DTYPE),
        top_k=SAE_TOP_K,
        max_layers=SAE_CACHE_LAYERS,
        relu_before_topk=SAE_RELU_BEFORE_TOPK,
    )

    print("\nRunning causal mediation / targeted SAE patching...", flush=True)
    mediation_results = run_mediation_experiment(
        model=mdl,
        tokenizer=tok,
        top_mediators=top_mediators,
        target_prompts=target_prompts,
        control_prompts=control_prompts,
        sae_cache=sae_cache,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mediation_results.to_csv(output_path, index=False)

    print("\n=== RESULTS HEAD ===", flush=True)
    print(mediation_results.head(20).to_string(index=False), flush=True)
    print(f"\nSaved: {output_path}", flush=True)

    if not args.no_inspect and INSPECT_CONTEXTS_N > 0:
        print("\nExtracting top activating contexts...", flush=True)
        context_df = inspect_top_mediators_on_texts(
            model=mdl,
            tokenizer=tok,
            top_mediators=top_mediators,
            target_prompts=target_prompts,
            sae_cache=sae_cache,
        )
        if len(context_df):
            contexts_path = Path(args.contexts_output)
            contexts_path.parent.mkdir(parents=True, exist_ok=True)
            context_df.to_csv(contexts_path, index=False)
            print(f"Saved contexts: {contexts_path}", flush=True)
        else:
            print("No context rows saved.", flush=True)

    sae_cache.clear()

    if AUTO_DOWNLOAD_COLAB_RESULTS:
        try:
            from google.colab import files

            files.download(str(output_path))
            contexts_path = Path(args.contexts_output)
            if contexts_path.exists():
                files.download(str(contexts_path))
        except Exception as exc:
            print(f"Auto-download failed: {exc!r}", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
