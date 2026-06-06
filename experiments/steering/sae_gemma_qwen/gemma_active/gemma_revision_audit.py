# ============================================================
# GEMMA REVISION / RUNTIME AUDIT
#
# Purpose:
#   Check whether a local/Colab run is using the expected public Gemma repo
#   revision and record tokenizer/runtime details that can change outputs even
#   when weights are unchanged.
#
# Expected optional globals when run with `%run -i`:
#   model
#   prompts_target
#
# Usage:
#   HF_MODEL_ID = "google/gemma-3-12b-it"
#   EXPECTED_HF_SHA = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
#   %run -i steering/gemma_revision_audit.py
# ============================================================

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path


HF_MODEL_ID = globals().get("HF_MODEL_ID", "google/gemma-3-12b-it")
EXPECTED_HF_SHA = globals().get("EXPECTED_HF_SHA", "96b6f1eccf38110c56df3a15bffe176da04bfd80")
OUTPUT_JSON = globals().get("OUTPUT_JSON", f"gemma_revision_audit_{HF_MODEL_ID.replace('/', '_')}.json")


def sha256_text(value):
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def safe_str(value, max_len=2000):
    if value is None:
        return None
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + f"...<truncated {len(text) - max_len} chars>"
    return text


def package_version(name):
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        return None


def get_hf_model_info(model_id):
    out = {
        "model_id": model_id,
        "ok": False,
        "error": "",
    }
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, files_metadata=True)
        out.update({
            "ok": True,
            "sha": getattr(info, "sha", None),
            "last_modified": str(getattr(info, "last_modified", None)),
            "private": getattr(info, "private", None),
            "gated": getattr(info, "gated", None),
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
        })
        siblings = []
        for s in getattr(info, "siblings", []) or []:
            item = {
                "rfilename": getattr(s, "rfilename", None),
                "size": getattr(s, "size", None),
                "blob_id": getattr(s, "blob_id", None),
            }
            lfs = getattr(s, "lfs", None)
            if lfs is not None:
                item["lfs_oid"] = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
                item["lfs_size"] = getattr(lfs, "size", None)
            siblings.append(item)
        out["siblings"] = siblings
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def get_local_model_info():
    out = {
        "model_global_present": "model" in globals(),
    }
    if "model" not in globals():
        return out

    m = globals()["model"]
    cfg = getattr(m, "cfg", None)
    tok = getattr(m, "tokenizer", None)
    out.update({
        "cfg_model_name": safe_str(getattr(cfg, "model_name", None)),
        "cfg_tokenizer_name": safe_str(getattr(cfg, "tokenizer_name", None)),
        "cfg_dtype": safe_str(getattr(cfg, "dtype", None)),
        "cfg_device": safe_str(getattr(cfg, "device", None)),
        "cfg_n_layers": getattr(cfg, "n_layers", None),
        "cfg_d_model": getattr(cfg, "d_model", None),
        "cfg_n_ctx": getattr(cfg, "n_ctx", None),
        "tokenizer_name_or_path": safe_str(getattr(tok, "name_or_path", None)),
        "tokenizer_chat_template_present": getattr(tok, "chat_template", None) is not None,
        "tokenizer_chat_template_sha256": sha256_text(getattr(tok, "chat_template", None)),
        "tokenizer_chat_template_preview": safe_str(getattr(tok, "chat_template", None), max_len=1200),
    })
    try:
        out["parameter_device"] = str(next(m.parameters()).device)
        out["parameter_dtype"] = str(next(m.parameters()).dtype)
    except Exception as exc:
        out["parameter_error"] = repr(exc)
    try:
        hf_model = getattr(m, "hf_model", None)
        gen_cfg = getattr(hf_model, "generation_config", None)
        out["hf_generation_config"] = safe_str(gen_cfg.to_dict() if hasattr(gen_cfg, "to_dict") else gen_cfg)
    except Exception as exc:
        out["hf_generation_config_error"] = repr(exc)
    return out


def get_prompt_info():
    out = {
        "prompts_target_present": "prompts_target" in globals(),
    }
    if "prompts_target" in globals() and len(globals()["prompts_target"]) > 0:
        text = str(globals()["prompts_target"][0])
        out.update({
            "prompts_target_0_sha256": sha256_text(text),
            "prompts_target_0_char_len": len(text),
            "prompts_target_0_preview": safe_str(text, max_len=1000),
        })
    return out


def get_cache_info(model_id):
    out = {"cache_scan_ok": False, "snapshots": [], "error": ""}
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if getattr(repo, "repo_id", None) == model_id:
                for rev in getattr(repo, "revisions", []) or []:
                    out["snapshots"].append({
                        "commit_hash": getattr(rev, "commit_hash", None),
                        "snapshot_path": str(getattr(rev, "snapshot_path", "")),
                        "size_on_disk": getattr(rev, "size_on_disk", None),
                        "last_modified": str(getattr(rev, "last_modified", None)),
                    })
        out["cache_scan_ok"] = True
    except Exception as exc:
        out["error"] = repr(exc)
    return out


audit = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "hf_model_id": HF_MODEL_ID,
    "expected_hf_sha": EXPECTED_HF_SHA,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "transformer_lens": package_version("transformer-lens"),
        "huggingface_hub": package_version("huggingface_hub"),
        "sae_lens": package_version("sae-lens"),
    },
    "hf_remote": get_hf_model_info(HF_MODEL_ID),
    "hf_cache": get_cache_info(HF_MODEL_ID),
    "local_model": get_local_model_info(),
    "prompt": get_prompt_info(),
}

remote_sha = audit["hf_remote"].get("sha")
audit["expected_sha_matches_remote"] = bool(remote_sha == EXPECTED_HF_SHA) if remote_sha else None

Path(OUTPUT_JSON).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

print("=" * 78)
print("GEMMA REVISION / RUNTIME AUDIT")
print("=" * 78)
print("HF_MODEL_ID:", HF_MODEL_ID)
print("Remote SHA:", remote_sha)
print("Expected SHA:", EXPECTED_HF_SHA)
print("Expected SHA matches remote:", audit["expected_sha_matches_remote"])
print("Remote last_modified:", audit["hf_remote"].get("last_modified"))
print("Local cfg model:", audit["local_model"].get("cfg_model_name"))
print("Local tokenizer:", audit["local_model"].get("tokenizer_name_or_path"))
print("Local n_layers/d_model/n_ctx:", audit["local_model"].get("cfg_n_layers"), audit["local_model"].get("cfg_d_model"), audit["local_model"].get("cfg_n_ctx"))
print("Tokenizer chat_template sha256:", audit["local_model"].get("tokenizer_chat_template_sha256"))
print("Cached snapshots:")
for snapshot in audit["hf_cache"].get("snapshots", []):
    print(" ", snapshot.get("commit_hash"), snapshot.get("snapshot_path"))
print("Saved:", OUTPUT_JSON)

