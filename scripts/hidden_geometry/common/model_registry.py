from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    loader: str
    max_input_tokens: int
    torch_dtype: str
    execution_profile: str
    disable_thinking: bool
    trust_remote_code: bool = True
    notes: str = ""


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "qwen3_14b": ModelProfile(
        name="qwen3_14b",
        model_id="Qwen/Qwen3-14B",
        loader="causal_lm",
        max_input_tokens=8192,
        torch_dtype="bfloat16",
        execution_profile="balanced_14b",
        disable_thinking=True,
        notes="Primary Qwen Grade 3/4 reference profile.",
    ),
    "olmo2_13b_clean4096": ModelProfile(
        name="olmo2_13b_clean4096",
        model_id="allenai/OLMo-2-1124-13B-Instruct",
        loader="causal_lm",
        max_input_tokens=4096,
        torch_dtype="bfloat16",
        execution_profile="safe_14b",
        disable_thinking=False,
        notes="OLMo2 1124 uses a strict 4096-token training context.",
    ),
    "gemma3_12b_it": ModelProfile(
        name="gemma3_12b_it",
        model_id="google/gemma-3-12b-it",
        loader="causal_lm",
        max_input_tokens=8192,
        torch_dtype="bfloat16",
        execution_profile="safe_14b",
        disable_thinking=False,
        notes="Gemma3 may be wrapped as a VLM; require decoder-layer preflight.",
    ),
}


def profile_names() -> list[str]:
    return sorted(MODEL_PROFILES)


def get_model_profile(name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(profile_names())
        raise KeyError(f"Unknown model profile {name!r}. Known profiles: {known}") from exc


def resolve_model_profile(
    profile_name: Optional[str],
    model_id: Optional[str],
    max_input_tokens: Optional[int] = None,
    torch_dtype: Optional[str] = None,
    execution_profile: Optional[str] = None,
    disable_thinking: Optional[bool] = None,
    loader: Optional[str] = None,
) -> ModelProfile:
    if profile_name:
        base = get_model_profile(profile_name)
    elif model_id:
        safe_name = (
            model_id.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .lower()
        )
        base = ModelProfile(
            name=safe_name,
            model_id=model_id,
            loader="causal_lm",
            max_input_tokens=8192,
            torch_dtype="bfloat16",
            execution_profile="safe_14b",
            disable_thinking=False,
            notes="Ad-hoc profile generated from --model-id.",
        )
    else:
        raise ValueError("Pass either --profile or --model-id.")

    return ModelProfile(
        name=base.name,
        model_id=model_id or base.model_id,
        loader=loader or base.loader,
        max_input_tokens=int(max_input_tokens if max_input_tokens is not None else base.max_input_tokens),
        torch_dtype=torch_dtype or base.torch_dtype,
        execution_profile=execution_profile or base.execution_profile,
        disable_thinking=bool(disable_thinking if disable_thinking is not None else base.disable_thinking),
        trust_remote_code=base.trust_remote_code,
        notes=base.notes,
    )


def profile_to_dict(profile: ModelProfile) -> dict:
    return asdict(profile)

