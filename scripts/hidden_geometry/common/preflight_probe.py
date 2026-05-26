from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.model_compat import (  # type: ignore
        DEFAULT_ARCHITECTURE_MODULES,
        build_condition_prompts,
        build_prompt,
        dataclass_to_dict,
        find_decoder_layers,
        get_layer_module,
        module_probe_rows,
        resolve_dtype,
        resolve_hf_token,
        safe_name,
        token_count,
        write_csv,
        write_json,
    )
    from common.model_registry import (  # type: ignore
        profile_names,
        profile_to_dict,
        resolve_model_profile,
    )
else:
    from .model_compat import (
        DEFAULT_ARCHITECTURE_MODULES,
        build_condition_prompts,
        build_prompt,
        dataclass_to_dict,
        find_decoder_layers,
        get_layer_module,
        module_probe_rows,
        resolve_dtype,
        resolve_hf_token,
        safe_name,
        token_count,
        write_csv,
        write_json,
    )
    from .model_registry import profile_names, profile_to_dict, resolve_model_profile


def read_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def read_questions(path: Optional[str], fallback_question: str) -> List[str]:
    if not path:
        return [fallback_question.strip()]
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return [fallback_question.strip()]
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            questions = [str(item).strip() for item in obj if str(item).strip()]
            return questions or [fallback_question.strip()]
    except Exception:
        pass
    questions = [line.strip() for line in text.splitlines() if line.strip()]
    return questions or [fallback_question.strip()]


def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def tensor_shape_from_output(output: Any) -> str:
    if isinstance(output, tuple) and output:
        output = output[0]
    if isinstance(output, list) and output:
        output = output[0]
    if torch.is_tensor(output):
        return "x".join(str(int(dim)) for dim in output.shape)
    return ""


def run_hidden_state_smoke(model, tokenizer, prompt: str, max_input_tokens: int) -> Dict[str, Any]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    device = input_device(model)
    enc = {key: value.to(device) for key, value in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hidden_states = getattr(out, "hidden_states", None)
    if not hidden_states:
        return {
            "hidden_state_smoke_status": "failed_no_hidden_states",
            "hidden_state_count": 0,
            "hidden_size": "",
            "smoke_prompt_tokens": int(enc["input_ids"].shape[1]),
        }
    last = hidden_states[-1]
    hidden_size = int(last.shape[-1]) if torch.is_tensor(last) and last.ndim >= 1 else ""
    return {
        "hidden_state_smoke_status": "ok",
        "hidden_state_count": int(len(hidden_states)),
        "hidden_size": hidden_size,
        "smoke_prompt_tokens": int(enc["input_ids"].shape[1]),
    }


def run_hook_smoke(
    model,
    tokenizer,
    layers,
    module_names: List[str],
    prompt: str,
    max_input_tokens: int,
) -> List[Dict[str, Any]]:
    rows = module_probe_rows(layers, module_names)
    if not layers:
        return rows

    captured: Dict[str, str] = {}
    handles = []
    first_layer = layers[0]

    def make_hook(module_name: str):
        def hook(_module, _inputs, output):
            captured[module_name] = tensor_shape_from_output(output)
        return hook

    for module_name in module_names:
        module = get_layer_module(first_layer, module_name)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(module_name)))

    try:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        device = input_device(model)
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            _ = model(**enc, output_hidden_states=True, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    for row in rows:
        module_name = str(row["module"])
        row["hook_fired"] = int(module_name in captured)
        row["hook_output_shape"] = captured.get(module_name, "")
        if row["found"] and not row["hook_fired"]:
            row["status"] = "found_but_hook_not_fired"
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Hidden-geometry multi-model preflight probe.")
    parser.add_argument("--profile", choices=profile_names(), help="Known model profile.")
    parser.add_argument("--model-id", help="Ad-hoc Hugging Face model id.")
    parser.add_argument("--results-dir", help="Output directory for preflight artifacts.")
    parser.add_argument("--target-file", help="UTF-8 file containing TARGET_TEXT for prompt budget probe.")
    parser.add_argument("--neutral-file", help="UTF-8 file containing NEUTRAL_TEXT for prompt budget probe.")
    parser.add_argument("--questions-file", help="JSON list or newline-separated questions.")
    parser.add_argument("--question", default="Compatibility smoke question.", help="Fallback question.")
    parser.add_argument("--max-input-tokens", type=int, help="Override profile max input tokens.")
    parser.add_argument("--torch-dtype", help="Override profile torch dtype.")
    parser.add_argument("--execution-profile", help="Record-only execution profile override.")
    parser.add_argument("--loader", default=None, help="Model loader name. Default comes from profile.")
    parser.add_argument("--disable-thinking", action="store_true", help="Force enable_thinking=False in chat template when supported.")
    parser.add_argument("--no-disable-thinking", action="store_true", help="Force normal chat template.")
    parser.add_argument("--load-model", action="store_true", help="Load full model and run layer/hook/hidden-state smoke tests.")
    parser.add_argument("--device-map", default="auto", help="Device map for full model load.")
    parser.add_argument("--smoke-text", default="Compatibility smoke test.", help="Short prompt for model smoke tests.")
    parser.add_argument("--prompt-overhead-token-budget", type=int, default=128)
    args = parser.parse_args()

    disable_thinking_override: Optional[bool] = None
    if args.disable_thinking:
        disable_thinking_override = True
    if args.no_disable_thinking:
        disable_thinking_override = False

    profile = resolve_model_profile(
        profile_name=args.profile,
        model_id=args.model_id,
        max_input_tokens=args.max_input_tokens,
        torch_dtype=args.torch_dtype,
        execution_profile=args.execution_profile,
        disable_thinking=disable_thinking_override,
        loader=args.loader,
    )

    results_dir = Path(args.results_dir or f"hidden_geometry_preflight_results/{safe_name(profile.name)}")
    results_dir.mkdir(parents=True, exist_ok=True)

    token = resolve_hf_token()
    hf_kwargs = {"token": token} if token else {}
    status_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "artifact_type": "protocol_reference",
        "profile": profile_to_dict(profile),
        "load_model": bool(args.load_model),
        "target_file": args.target_file or "",
        "neutral_file": args.neutral_file or "",
        "questions_file": args.questions_file or "",
        "hf_token_present": bool(token),
    }

    try:
        config = AutoConfig.from_pretrained(
            profile.model_id,
            trust_remote_code=profile.trust_remote_code,
            **hf_kwargs,
        )
        manifest["config_class"] = config.__class__.__name__
        manifest["model_type"] = getattr(config, "model_type", "")
        manifest["config_num_hidden_layers"] = getattr(config, "num_hidden_layers", None)
        text_config = getattr(config, "text_config", None)
        manifest["text_config_num_hidden_layers"] = getattr(text_config, "num_hidden_layers", None) if text_config else None
        status_rows.append({"check": "config_load", "status": "pass", "detail": ""})
    except Exception as exc:
        manifest["config_load_error"] = repr(exc)
        status_rows.append({"check": "config_load", "status": "fail", "detail": repr(exc)})
        write_json(results_dir / "model_compatibility_manifest.json", manifest)
        write_csv(results_dir / "preflight_status.csv", status_rows)
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            profile.model_id,
            trust_remote_code=profile.trust_remote_code,
            **hf_kwargs,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        manifest["tokenizer_class"] = tokenizer.__class__.__name__
        manifest["tokenizer_has_chat_template"] = bool(getattr(tokenizer, "chat_template", None))
        manifest["tokenizer_model_max_length"] = getattr(tokenizer, "model_max_length", None)
        status_rows.append({"check": "tokenizer_load", "status": "pass", "detail": ""})
    except Exception as exc:
        manifest["tokenizer_load_error"] = repr(exc)
        status_rows.append({"check": "tokenizer_load", "status": "fail", "detail": repr(exc)})
        write_json(results_dir / "model_compatibility_manifest.json", manifest)
        write_csv(results_dir / "preflight_status.csv", status_rows)
        return 2

    target_text = read_text_file(args.target_file)
    neutral_text = read_text_file(args.neutral_file)
    questions = read_questions(args.questions_file, args.question)
    prompt_rows = build_condition_prompts(
        tokenizer=tokenizer,
        target_text=target_text,
        neutral_text=neutral_text,
        questions=questions,
        max_input_tokens=profile.max_input_tokens,
        overhead_tokens=args.prompt_overhead_token_budget,
        disable_thinking=profile.disable_thinking,
    )
    write_csv(results_dir / "prompt_budget_probe.csv", prompt_rows)
    prompt_budget_pass = bool(prompt_rows) and all(int(row["pass_prompt_budget"]) == 1 for row in prompt_rows)
    prompt_budget_status = "pass" if prompt_budget_pass else "fail"
    if not target_text or not neutral_text:
        prompt_budget_status = "not_representative_missing_target_or_neutral_file"
    status_rows.append(
        {
            "check": "prompt_budget",
            "status": prompt_budget_status,
            "detail": f"rows={len(prompt_rows)}",
        }
    )
    manifest["target_text_tokens"] = token_count(tokenizer, target_text) if target_text else 0
    manifest["neutral_text_tokens"] = token_count(tokenizer, neutral_text) if neutral_text else 0
    manifest["question_count"] = len(questions)

    decoder_rows: List[Dict[str, Any]] = []
    module_rows: List[Dict[str, Any]] = []
    module_hook_status = "not_run"
    hidden_smoke: Dict[str, Any] = {
        "hidden_state_smoke_status": "not_run_model_not_loaded",
        "hidden_state_count": 0,
        "hidden_size": "",
        "smoke_prompt_tokens": 0,
    }

    if args.load_model:
        try:
            model_kwargs = {
                "trust_remote_code": profile.trust_remote_code,
                "device_map": args.device_map,
                "torch_dtype": resolve_dtype(profile.torch_dtype),
            }
            model = AutoModelForCausalLM.from_pretrained(profile.model_id, **model_kwargs, **hf_kwargs)
            model.eval()
            manifest["model_class"] = model.__class__.__name__
            status_rows.append({"check": "model_load", "status": "pass", "detail": ""})
        except Exception as exc:
            manifest["model_load_error"] = repr(exc)
            manifest["model_load_traceback"] = traceback.format_exc()
            status_rows.append({"check": "model_load", "status": "fail", "detail": repr(exc)})
            write_json(results_dir / "model_compatibility_manifest.json", manifest)
            write_csv(results_dir / "preflight_status.csv", status_rows)
            return 2

        layers, decoder_probe = find_decoder_layers(model)
        decoder_rows.append(dataclass_to_dict(decoder_probe))
        write_csv(results_dir / "decoder_layer_probe.csv", decoder_rows)
        status_rows.append(
            {
                "check": "decoder_layers",
                "status": "pass" if decoder_probe.status == "ok" else "fail",
                "detail": json.dumps(dataclass_to_dict(decoder_probe), ensure_ascii=False),
            }
        )

        if decoder_probe.status == "ok":
            smoke_prompt = build_prompt(
                tokenizer,
                args.smoke_text,
                disable_thinking=profile.disable_thinking,
            )
            try:
                hidden_smoke = run_hidden_state_smoke(
                    model,
                    tokenizer,
                    smoke_prompt,
                    min(profile.max_input_tokens, 2048),
                )
                status_rows.append(
                    {
                        "check": "hidden_state_smoke",
                        "status": "pass" if hidden_smoke["hidden_state_smoke_status"] == "ok" else "fail",
                        "detail": json.dumps(hidden_smoke, ensure_ascii=False),
                    }
                )
            except Exception as exc:
                hidden_smoke = {
                    "hidden_state_smoke_status": "failed_exception",
                    "hidden_state_count": 0,
                    "hidden_size": "",
                    "smoke_prompt_tokens": 0,
                    "error": repr(exc),
                }
                status_rows.append({"check": "hidden_state_smoke", "status": "fail", "detail": repr(exc)})

            try:
                module_rows = run_hook_smoke(
                    model,
                    tokenizer,
                    layers,
                    DEFAULT_ARCHITECTURE_MODULES,
                    smoke_prompt,
                    min(profile.max_input_tokens, 2048),
                )
                module_pass = all(int(row.get("found", 0)) == 1 and int(row.get("hook_fired", 0)) == 1 for row in module_rows)
                module_hook_status = "pass" if module_pass else "partial"
                status_rows.append(
                    {
                        "check": "module_hook_smoke",
                        "status": module_hook_status,
                        "detail": f"modules={len(module_rows)}",
                    }
                )
            except Exception as exc:
                module_hook_status = "fail"
                module_rows = [
                    {
                        "layer_index": 1,
                        "module": "",
                        "found": 0,
                        "module_class": "",
                        "hook_fired": 0,
                        "hook_output_shape": "",
                        "status": f"failed_exception: {exc!r}",
                    }
                ]
                status_rows.append({"check": "module_hook_smoke", "status": "fail", "detail": repr(exc)})
        else:
            module_rows = module_probe_rows(layers, DEFAULT_ARCHITECTURE_MODULES)
            module_hook_status = "not_run_decoder_layers_failed"
    else:
        decoder_rows.append(
            {
                "source": "",
                "count": 0,
                "expected_count": manifest.get("text_config_num_hidden_layers") or manifest.get("config_num_hidden_layers"),
                "count_mismatch": False,
                "status": "not_run_model_not_loaded",
            }
        )
        module_rows = module_probe_rows([], DEFAULT_ARCHITECTURE_MODULES)
        module_hook_status = "not_run_model_not_loaded"
        write_csv(results_dir / "decoder_layer_probe.csv", decoder_rows)
        status_rows.append({"check": "model_load", "status": "skipped", "detail": "pass --load-model for layer/hook checks"})

    write_csv(results_dir / "module_hook_probe.csv", module_rows)
    manifest["decoder_layer_probe"] = decoder_rows[0] if decoder_rows else {}
    manifest["hidden_state_smoke"] = hidden_smoke

    hard_fail_checks = {"config_load", "tokenizer_load", "model_load", "decoder_layers", "hidden_state_smoke"}
    failures = [row for row in status_rows if row["check"] in hard_fail_checks and row["status"] == "fail"]
    if not args.load_model:
        final_status = "config_tokenizer_only"
    elif failures:
        final_status = "fail"
    elif prompt_budget_status == "fail":
        final_status = "fail_prompt_budget"
    elif module_hook_status not in {"pass", "not_run_decoder_layers_failed"}:
        if prompt_budget_status.startswith("not_representative"):
            final_status = "pass_core_model_compat_partial_module_coverage_prompt_budget_not_representative"
        else:
            final_status = "pass_core_model_compat_partial_module_coverage"
    elif prompt_budget_status.startswith("not_representative"):
        final_status = "pass_model_compat_prompt_budget_not_representative"
    else:
        final_status = "pass"
    status_rows.append({"check": "final_preflight_status", "status": final_status, "detail": ""})
    manifest["final_preflight_status"] = final_status

    write_json(results_dir / "model_compatibility_manifest.json", manifest)
    write_csv(results_dir / "preflight_status.csv", status_rows)
    return 0 if final_status.startswith("pass") or final_status == "config_tokenizer_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
