from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DECODER_LAYER_CANDIDATES = [
    "model.layers",
    "model.model.layers",
    "language_model.model.layers",
    "model.language_model.model.layers",
    "language_model.layers",
    "model.language_model.layers",
    "text_model.layers",
    "model.text_model.layers",
    "decoder.layers",
    "model.decoder.layers",
    "transformer.layers",
    "transformer.h",
    "gpt_neox.layers",
]


DEFAULT_ARCHITECTURE_MODULES = [
    "self_attn",
    "mlp",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


@dataclass
class DecoderLayerProbe:
    source: str
    count: int
    expected_count: Optional[int]
    count_mismatch: bool
    status: str


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return name.strip("_") or "unnamed"


def nested_getattr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def expected_decoder_layer_count_from_config(config: Any) -> Optional[int]:
    candidates = [
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(config, "llm_config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ["num_hidden_layers", "n_layer", "n_layers", "num_layers"]:
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)
    return None


def find_decoder_layers(model: Any) -> Tuple[List[Any], DecoderLayerProbe]:
    for path in DECODER_LAYER_CANDIDATES:
        layers = nested_getattr(model, path)
        if layers is None:
            continue
        try:
            layer_list = list(layers)
        except Exception:
            continue
        if not layer_list:
            continue
        expected = expected_decoder_layer_count_from_config(getattr(model, "config", None))
        mismatch = expected is not None and len(layer_list) != expected
        status = "count_mismatch" if mismatch else "ok"
        return layer_list, DecoderLayerProbe(
            source=path,
            count=len(layer_list),
            expected_count=expected,
            count_mismatch=mismatch,
            status=status,
        )

    expected = expected_decoder_layer_count_from_config(getattr(model, "config", None))
    return [], DecoderLayerProbe(
        source="",
        count=0,
        expected_count=expected,
        count_mismatch=False,
        status="not_found",
    )


def get_layer_module(layer: Any, module_name: str) -> Any:
    if module_name == "self_attn":
        return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if module_name == "mlp":
        return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    if module_name.startswith("mlp."):
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is None:
            return None
        return nested_getattr(mlp, module_name.split(".", 1)[1])
    return nested_getattr(layer, module_name)


def module_probe_rows(layers: Sequence[Any], module_names: Sequence[str]) -> List[Dict[str, Any]]:
    if not layers:
        return [
            {
                "layer_index": "",
                "module": module_name,
                "found": 0,
                "module_class": "",
                "status": "not_run_no_decoder_layers",
            }
            for module_name in module_names
        ]

    rows: List[Dict[str, Any]] = []
    layer = layers[0]
    for module_name in module_names:
        module = get_layer_module(layer, module_name)
        rows.append(
            {
                "layer_index": 1,
                "module": module_name,
                "found": int(module is not None),
                "module_class": module.__class__.__name__ if module is not None else "",
                "status": "ok" if module is not None else "not_found",
            }
        )
    return rows


def resolve_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN") or token
    except Exception:
        pass
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def resolve_dtype(name: str):
    import torch

    lowered = str(name).lower().strip()
    if lowered == "auto":
        return "auto"
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


def build_prompt(
    tokenizer: Any,
    content: str,
    use_chat_template: bool = True,
    disable_thinking: bool = False,
) -> str:
    content = str(content or "").strip()
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if disable_thinking:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
            except Exception:
                pass
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass
    return content + "\n"


def token_count(tokenizer: Any, text: str) -> int:
    return int(len(tokenizer(str(text), add_special_tokens=False).input_ids))


def build_condition_prompts(
    tokenizer: Any,
    target_text: str,
    neutral_text: str,
    questions: Sequence[str],
    max_input_tokens: int,
    overhead_tokens: int,
    disable_thinking: bool,
) -> List[Dict[str, Any]]:
    conditions = {
        "question_only": "",
        "neutral": neutral_text or "",
        "target": target_text or "",
    }
    rows: List[Dict[str, Any]] = []
    for q_idx, question in enumerate(questions):
        question = str(question or "").strip()
        for condition, prefix in conditions.items():
            if condition != "question_only" and not prefix:
                continue
            user_content = "\n\n".join(part for part in [prefix.strip(), question] if part).strip()
            prompt = build_prompt(tokenizer, user_content, disable_thinking=disable_thinking)
            prompt_tokens = token_count(tokenizer, prompt)
            prefix_tokens = token_count(tokenizer, prefix) if prefix else 0
            question_tokens = token_count(tokenizer, question)
            estimated_total = prefix_tokens + question_tokens + int(overhead_tokens)
            overflow = estimated_total > int(max_input_tokens)
            question_may_be_truncated = prefix_tokens + int(overhead_tokens) >= int(max_input_tokens)
            rows.append(
                {
                    "question_index": q_idx,
                    "condition": condition,
                    "prompt_tokens": prompt_tokens,
                    "prefix_tokens": prefix_tokens,
                    "question_tokens": question_tokens,
                    "estimated_total_tokens": estimated_total,
                    "max_input_tokens": int(max_input_tokens),
                    "estimated_overflow_tokens": estimated_total - int(max_input_tokens),
                    "question_may_be_truncated": int(question_may_be_truncated),
                    "pass_prompt_budget": int(not overflow and not question_may_be_truncated),
                }
            )
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)

