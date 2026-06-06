# ============================================================
# GEMMA SAE ORDER-FEATURE EVIDENCE + ROUGH PATCHING V2
#
# This script is a safer/full replacement for exploratory SAE candidate
# discovery. It does NOT modify the original 01 script.
#
# Main idea:
#   1. Read the full Gemma Grade4 SAE tables, preferably directly from the
#      raw-run zip:
#        grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip
#   2. Use sae_order_feature_contrast.csv as the primary ranking table.
#   3. Add evidence from reconstruction quality, component summaries,
#      prompt deltas, generation summaries, generation top features, and
#      top changed features.
#   4. Rank order candidates by order_specific_score / abs x_order_orth,
#      preserving the sign. Negative x_order_orth features are valid.
#   5. Optionally run rough SAE zero-ablation and top activating contexts.
#
# Expected Colab usage:
#
#   SAE_TABLE_ZIP_PATH = "/content/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale.zip"
#   PATCH_PROMPTS = prompts_target
#   RUN_MEDIATION_PATCHING = True
#   RUN_TOP_CONTEXT_INSPECTION = True
#   %run -i steering/01b_full_sae_evidence_candidate_patching_gemma.py
#
# If the CSVs are extracted instead:
#
#   SAE_TABLE_DIR = "/content/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale"
#   PATCH_PROMPTS = prompts_target
#   %run -i steering/01b_full_sae_evidence_candidate_patching_gemma.py
# ============================================================

import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ====================== CONFIG ======================

MODEL_NAME = globals().get("MODEL_NAME", "google/gemma-3-12b-it")
SAE_RELEASE = globals().get("SAE_RELEASE", "gemma-scope-2-12b-it-res-all")
SAE_ID_TEMPLATE = globals().get("SAE_ID_TEMPLATE", "layer_{real_layer}_width_16k_l0_small")
DEVICE = globals().get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = globals().get("MODEL_DTYPE", torch.bfloat16)
USE_EXISTING_MODEL_IF_AVAILABLE = bool(globals().get("USE_EXISTING_MODEL_IF_AVAILABLE", True))
PREPEND_BOS = bool(globals().get("PREPEND_BOS", True))

RUN_TAG = globals().get("RUN_TAG", "gemma_sae_order_feature_patching_v2")
OUTPUT_DIR = Path(globals().get("OUTPUT_DIR", f"sae_order_feature_patching_outputs_{RUN_TAG}"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best source: raw Grade4 run zip, not metric-lab zip.
DEFAULT_RUN_BASENAME = "grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl_natscale"
SAE_TABLE_ZIP_PATH = globals().get(
    "SAE_TABLE_ZIP_PATH",
    f"/content/{DEFAULT_RUN_BASENAME}.zip",
)
SAE_TABLE_DIR = globals().get(
    "SAE_TABLE_DIR",
    f"/content/hidden_geometry_runs/{DEFAULT_RUN_BASENAME}",
)

# Optional direct override for the primary table.
CONTRAST_CSV_PATH = globals().get("CONTRAST_CSV_PATH", None)

SAE_TABLE_FILENAMES = [
    "sae_order_feature_contrast.csv",
    "sae_reconstruction_quality.csv",
    "sae_grade4_component_feature_summary.csv",
    "sae_prompt_feature_delta_summary.csv",
    "sae_generation_feature_summary.csv",
    "sae_generation_top_features.csv",
    "sae_top_changed_features.csv",
    "sae_model_compatibility.csv",
    "sae_prompt_feature_activation_summary.csv",
]

TOP_K_CANDIDATES = int(globals().get("TOP_K_CANDIDATES", 30))


def parse_optional_limit(value):
    """None/all/unlimited/0 means no limit."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "all",
        "unlimited",
        "no_limit",
    }:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


# Default is intentionally unlimited, matching the old script's behavior:
# rank all candidates globally and take TOP_K_CANDIDATES without per-layer caps.
MAX_FEATURES_PER_LAYER = parse_optional_limit(globals().get("MAX_FEATURES_PER_LAYER", None))

ORDER_STATUS_REGEX = globals().get(
    "ORDER_STATUS_REGEX",
    "order_specific|order_enriched|order_component",
)
EXCLUDE_STATUS_REGEX = globals().get(
    "EXCLUDE_STATUS_REGEX",
    "content_only_or_missing_order_component",
)

RUN_MEDIATION_PATCHING = bool(globals().get("RUN_MEDIATION_PATCHING", True))
RUN_TOP_CONTEXT_INSPECTION = bool(globals().get("RUN_TOP_CONTEXT_INSPECTION", True))
SKIP_MODEL_LOAD = bool(globals().get("SKIP_MODEL_LOAD", False))

# Target-only by default. Control prompts are optional and must be explicitly
# added to CONTEXT_TEXTS by the caller if desired.
PATCH_PROMPTS = globals().get("PATCH_PROMPTS", globals().get("prompts_target", None))
CONTEXT_TEXTS = globals().get("CONTEXT_TEXTS", PATCH_PROMPTS)

PATCH_BATCH_SIZE = int(globals().get("PATCH_BATCH_SIZE", 1))
PATCH_MODE = globals().get("PATCH_MODE", "zero")  # "zero", "scale", "set"
PATCH_SCALE = float(globals().get("PATCH_SCALE", 0.0))
PATCH_VALUE = float(globals().get("PATCH_VALUE", 0.0))
PATCH_POSITION_MODE = globals().get("PATCH_POSITION_MODE", "all_tokens")  # "all_tokens", "last_token"
CONTEXT_BATCH_SIZE = int(globals().get("CONTEXT_BATCH_SIZE", 1))

# Number of different feature-ablation variants to pack into one model forward
# per layer. This uses more VRAM but much better utilizes large GPUs.
PATCH_FEATURE_BATCH_SIZE = int(globals().get("PATCH_FEATURE_BATCH_SIZE", 8))

# Correct default for feature intervention:
#   delta  = activation + (decode(latent_patched) - decode(latent_original))
#   replace = decode(latent_patched)
# The old rough script used replace, which measures SAE reconstruction error
# plus feature removal. "delta" is the clean single-feature intervention.
PATCH_RESIDUAL_UPDATE_MODE = globals().get("PATCH_RESIDUAL_UPDATE_MODE", "delta")

# Default is all selected candidates. Set e.g. N_CONTEXT_FEATURES=8 for a quick
# run; leave unset/None/all to inspect every selected feature.
N_CONTEXT_FEATURES = parse_optional_limit(globals().get("N_CONTEXT_FEATURES", None))
TOP_N_CONTEXTS = int(globals().get("TOP_N_CONTEXTS", 12))
CONTEXT_WINDOW = int(globals().get("CONTEXT_WINDOW", 14))

# Context inspection token filter. This only affects the human-readable
# "top activating contexts" table; feature patching still applies normally.
FILTER_CONTEXT_TOKENS = bool(globals().get("FILTER_CONTEXT_TOKENS", True))
CONTEXT_TOKEN_BLACKLIST = globals().get(
    "CONTEXT_TOKEN_BLACKLIST",
    [
        "",
        " ",
        "\t",
        "\n",
        "\n\n",
        "\n\n\n",
        "\r",
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        "؟",
        "…",
        "-",
        "--",
        "---",
        "‑",
        "–",
        "—",
        "_",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "\"",
        "'",
        "`",
        "``",
        "''",
        "«",
        "»",
        "“",
        "”",
        "‘",
        "’",
        "/",
        "\\",
        "|",
    ],
)
CONTEXT_TOKEN_BLACKLIST_SET = {str(x) for x in CONTEXT_TOKEN_BLACKLIST}
CONTEXT_TOKEN_BLACKLIST_STRIPPED_SET = {str(x).strip().lower() for x in CONTEXT_TOKEN_BLACKLIST}
CONTEXT_TOKEN_PUNCT_RE = re.compile(r"^[\s\.,;:!\?؟…\-\u2010-\u2015_()\[\]{}<>\"'`«»“”‘’/\\|]+$")

MAX_PROMPT_TOKENS = globals().get("MAX_PROMPT_TOKENS", None)
if MAX_PROMPT_TOKENS is not None:
    MAX_PROMPT_TOKENS = int(MAX_PROMPT_TOKENS)

SAVE_SELECTED_CANDIDATES_CSV = OUTPUT_DIR / "selected_sae_order_candidates.csv"
SAVE_RANKED_CANDIDATES_CSV = OUTPUT_DIR / "ranked_sae_order_candidates_full_evidence.csv"
SAVE_TABLE_MANIFEST_CSV = OUTPUT_DIR / "sae_table_manifest.csv"
SAVE_RECON_SUMMARY_CSV = OUTPUT_DIR / "sae_layer_reconstruction_quality_summary.csv"
SAVE_PATCHING_RESULTS_CSV = OUTPUT_DIR / "rough_sae_zero_ablation_logit_results.csv"
SAVE_CONTEXTS_CSV = OUTPUT_DIR / "sae_feature_top_activating_contexts.csv"
SAVE_SUMMARY_MD = OUTPUT_DIR / "summary.md"


# ====================== TABLE LOADING ======================

def _path_exists(value):
    return value is not None and Path(str(value)).exists()


def _find_zip_member(zip_path, filename):
    with zipfile.ZipFile(zip_path) as zf:
        matches = [
            name for name in zf.namelist()
            if name.endswith("/" + filename) or name == filename
        ]
        if not matches:
            return None
        # Prefer the hidden_geometry_runs raw folder if there are duplicates.
        matches = sorted(
            matches,
            key=lambda x: (
                0 if "/hidden_geometry_runs/" in x else 1,
                len(x),
                x,
            ),
        )
        return matches[0]


def read_csv_from_zip(zip_path, filename):
    member = _find_zip_member(zip_path, filename)
    if member is None:
        return None, None
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f), f"{zip_path}::{member}"


def read_sae_table(filename, required=False):
    candidate_paths = []

    if filename == "sae_order_feature_contrast.csv" and CONTRAST_CSV_PATH:
        candidate_paths.append(Path(str(CONTRAST_CSV_PATH)))

    if SAE_TABLE_DIR:
        candidate_paths.append(Path(str(SAE_TABLE_DIR)) / filename)

    candidate_paths.extend([
        Path("/content") / filename,
        Path(filename),
        Path("content") / "hidden_geometry_runs" / DEFAULT_RUN_BASENAME / filename,
    ])

    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path), str(path)

    if _path_exists(SAE_TABLE_ZIP_PATH):
        df, source = read_csv_from_zip(str(SAE_TABLE_ZIP_PATH), filename)
        if df is not None:
            return df, source

    if required:
        raise FileNotFoundError(
            f"Required SAE table not found: {filename}. "
            f"Set SAE_TABLE_DIR or SAE_TABLE_ZIP_PATH."
        )
    return pd.DataFrame(), ""


def load_all_sae_tables():
    tables = {}
    manifest_rows = []

    for filename in SAE_TABLE_FILENAMES:
        required = filename == "sae_order_feature_contrast.csv"
        df, source = read_sae_table(filename, required=required)
        tables[filename] = df
        manifest_rows.append({
            "filename": filename,
            "loaded": int(len(df) > 0),
            "rows": int(len(df)),
            "columns": int(len(df.columns)) if len(df.columns) else 0,
            "source": source,
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SAVE_TABLE_MANIFEST_CSV, index=False)
    print("\nLoaded SAE tables:")
    print(manifest.to_string(index=False))
    return tables, manifest


# ====================== EVIDENCE MERGE ======================

def feature_keys_for(df):
    keys = []
    if "sae_spec_index" in df.columns:
        keys.append("sae_spec_index")
    keys.extend(["layer", "feature_index"])
    return keys


def layer_keys_for(df):
    keys = []
    if "sae_spec_index" in df.columns:
        keys.append("sae_spec_index")
    keys.append("layer")
    return keys


def make_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flatten_pivot_columns(df, prefix, id_cols=None):
    id_cols = set(id_cols or [])
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) not in ("", "nan", "None")]
            if len(parts) == 1 and parts[0] in id_cols:
                flat_cols.append(parts[0])
            elif len(parts) >= 2:
                flat_cols.append(f"{prefix}{parts[0]}__{parts[1]}")
            elif len(parts) == 1:
                flat_cols.append(parts[0])
            else:
                flat_cols.append("")
        else:
            flat_cols.append(str(col))
    df.columns = flat_cols
    return df


def merge_component_summary(candidates, component_df):
    if len(component_df) == 0:
        return candidates
    component_df = component_df.copy()
    component_df = component_df[component_df.get("status", "computed").fillna("computed") == "computed"]
    if len(component_df) == 0:
        return candidates

    keys = feature_keys_for(component_df)
    value_cols = [
        "component_feature_delta",
        "abs_component_feature_delta",
        "rank_by_abs_component_delta",
    ]
    component_df = make_numeric(component_df, value_cols)
    pivot = component_df.pivot_table(
        index=keys,
        columns="component_name",
        values=value_cols,
        aggfunc="first",
    ).reset_index()
    pivot = flatten_pivot_columns(pivot, "grade4_", id_cols=keys)
    return candidates.merge(pivot, on=keys, how="left")


def merge_condition_feature_table(candidates, table_df, prefix, value_cols):
    if len(table_df) == 0:
        return candidates
    table_df = table_df.copy()
    table_df = table_df[table_df.get("status", "computed").fillna("computed") == "computed"]
    if len(table_df) == 0 or "condition" not in table_df.columns:
        return candidates

    keys = feature_keys_for(table_df)
    table_df = make_numeric(table_df, value_cols)
    usable_cols = [col for col in value_cols if col in table_df.columns]
    if not usable_cols:
        return candidates

    pivot = table_df.pivot_table(
        index=keys,
        columns="condition",
        values=usable_cols,
        aggfunc="first",
    ).reset_index()
    pivot = flatten_pivot_columns(pivot, prefix, id_cols=keys)
    return candidates.merge(pivot, on=keys, how="left")


def merge_reconstruction_quality(candidates, recon_df):
    if len(recon_df) == 0:
        return candidates, pd.DataFrame()
    recon_df = recon_df.copy()
    recon_df = recon_df[recon_df.get("status", "computed").fillna("computed") == "computed"]
    if len(recon_df) == 0:
        return candidates, pd.DataFrame()

    keys = layer_keys_for(recon_df)
    recon_df = make_numeric(
        recon_df,
        [
            "input_reconstruction_cosine",
            "explained_variance_proxy",
            "reconstruction_mse",
            "reconstruction_l2_norm",
        ],
    )
    summary = recon_df.groupby(keys, as_index=False).agg(
        recon_rows=("input_reconstruction_cosine", "count"),
        recon_cosine_mean=("input_reconstruction_cosine", "mean"),
        recon_cosine_min=("input_reconstruction_cosine", "min"),
        explained_variance_proxy_mean=("explained_variance_proxy", "mean"),
        explained_variance_proxy_min=("explained_variance_proxy", "min"),
        reconstruction_mse_mean=("reconstruction_mse", "mean"),
    )
    summary.to_csv(SAVE_RECON_SUMMARY_CSV, index=False)
    return candidates.merge(summary, on=keys, how="left"), summary


def add_real_layer(df):
    df = df.copy()
    layers = pd.to_numeric(df["layer"], errors="coerce").dropna()
    if len(layers) == 0:
        raise ValueError("No numeric layer column in sae_order_feature_contrast.csv")
    if int(layers.min()) >= 1:
        df["real_layer"] = pd.to_numeric(df["layer"], errors="coerce").astype(int) - 1
        print("Detected 1-based layer ids in CSV. Added real_layer = layer - 1.")
    else:
        df["real_layer"] = pd.to_numeric(df["layer"], errors="coerce").astype(int)
        print("Detected 0-based layer ids in CSV. real_layer = layer.")
    return df


def build_candidate_table(tables):
    contrast = tables["sae_order_feature_contrast.csv"].copy()
    contrast = contrast[contrast.get("status", "computed").fillna("computed") == "computed"].copy()

    numeric_cols = [
        "layer",
        "feature_index",
        "feature_count",
        "x_content_component_delta",
        "x_order_orth_component_delta",
        "x_content_component_rank",
        "x_order_orth_component_rank",
        "target_prompt_mean_activation_delta",
        "sentence_shuffle_prompt_mean_activation_delta",
        "target_prompt_activation_rate_delta",
        "sentence_shuffle_prompt_activation_rate_delta",
        "target_generation_mean_activation",
        "sentence_shuffle_generation_mean_activation",
        "target_generation_activation_rate",
        "sentence_shuffle_generation_activation_rate",
        "target_generation_late_minus_early_activation",
        "sentence_shuffle_generation_late_minus_early_activation",
        "target_generation_generation_mean_activation_delta",
        "sentence_shuffle_generation_generation_mean_activation_delta",
        "abs_x_content_component_delta",
        "abs_x_order_orth_component_delta",
        "order_minus_content_abs_component_delta",
        "order_over_content_abs_ratio",
        "target_minus_sentence_shuffle_prompt_delta",
        "target_minus_sentence_shuffle_generation_mean_activation",
        "order_specific_score",
    ]
    contrast = make_numeric(contrast, numeric_cols)
    contrast = add_real_layer(contrast)

    contrast["interpretation_status"] = contrast["interpretation_status"].fillna("")
    contrast["candidate_is_order_like"] = contrast["interpretation_status"].str.contains(
        ORDER_STATUS_REGEX,
        case=False,
        regex=True,
        na=False,
    ).astype(int)
    contrast["candidate_is_excluded"] = contrast["interpretation_status"].str.contains(
        EXCLUDE_STATUS_REGEX,
        case=False,
        regex=True,
        na=False,
    ).astype(int)

    # Preserve sign, but rank by magnitude/evidence score. This fixes the old
    # positive-only x_order_orth sorting problem.
    if "abs_x_order_orth_component_delta" not in contrast.columns:
        contrast["abs_x_order_orth_component_delta"] = contrast["x_order_orth_component_delta"].abs()
    else:
        contrast["abs_x_order_orth_component_delta"] = contrast["abs_x_order_orth_component_delta"].fillna(
            contrast["x_order_orth_component_delta"].abs()
        )

    if "abs_x_content_component_delta" not in contrast.columns:
        contrast["abs_x_content_component_delta"] = contrast["x_content_component_delta"].abs()
    else:
        contrast["abs_x_content_component_delta"] = contrast["abs_x_content_component_delta"].fillna(
            contrast["x_content_component_delta"].abs()
        )

    contrast["order_specific_score_filled"] = contrast["order_specific_score"].fillna(
        contrast["abs_x_order_orth_component_delta"]
    )
    content_abs = contrast["abs_x_content_component_delta"]
    order_abs = contrast["abs_x_order_orth_component_delta"]
    valid_content_abs = content_abs.notna() & (content_abs.abs() > 1e-9)
    contrast["has_x_content_component_delta"] = content_abs.notna().astype(int)
    contrast["order_abs_over_content_abs_safe"] = np.where(
        valid_content_abs,
        order_abs / content_abs,
        np.nan,
    )
    # For sorting only: a missing content component means "no measured content
    # competitor", so pure order-component features should not be penalized.
    contrast["order_abs_over_content_abs_sort"] = pd.Series(
        contrast["order_abs_over_content_abs_safe"]
    ).fillna(np.inf)
    contrast["selection_score"] = contrast["order_specific_score_filled"]

    candidates = contrast[
        (contrast["candidate_is_order_like"] == 1)
        & (contrast["candidate_is_excluded"] == 0)
        & contrast["abs_x_order_orth_component_delta"].notna()
    ].copy()

    if len(candidates) == 0:
        print("WARNING: no order-like candidates found by status regex; falling back to abs x_order_orth ranking.")
        candidates = contrast[
            (contrast["candidate_is_excluded"] == 0)
            & contrast["abs_x_order_orth_component_delta"].notna()
        ].copy()

    candidates = merge_component_summary(
        candidates,
        tables["sae_grade4_component_feature_summary.csv"],
    )
    candidates = merge_condition_feature_table(
        candidates,
        tables["sae_prompt_feature_delta_summary.csv"],
        prefix="prompt_",
        value_cols=[
            "mean_activation_delta",
            "abs_mean_activation_delta",
            "activation_rate_delta",
            "condition_mean_activation",
            "reference_mean_activation",
            "rank_by_abs_delta",
        ],
    )
    candidates = merge_condition_feature_table(
        candidates,
        tables["sae_generation_feature_summary.csv"],
        prefix="generation_",
        value_cols=[
            "mean_activation",
            "max_activation",
            "activation_rate",
            "early_mean_activation",
            "late_mean_activation",
            "late_minus_early_activation",
            "generation_mean_activation_delta",
            "rank_by_generation_activity",
        ],
    )
    candidates = merge_condition_feature_table(
        candidates,
        tables["sae_generation_top_features.csv"],
        prefix="generation_top_",
        value_cols=[
            "mean_activation",
            "max_activation",
            "activation_rate",
            "generation_mean_activation_delta",
            "abs_generation_mean_activation_delta",
            "rank_by_generation_activity",
            "rank_by_abs_generation_delta",
        ],
    )
    candidates = merge_condition_feature_table(
        candidates,
        tables["sae_top_changed_features.csv"],
        prefix="top_changed_",
        value_cols=[
            "mean_activation_delta",
            "abs_mean_activation_delta",
            "activation_rate_delta",
            "condition_mean_activation",
            "reference_mean_activation",
            "rank_by_abs_delta",
        ],
    )
    candidates, recon_summary = merge_reconstruction_quality(
        candidates,
        tables["sae_reconstruction_quality.csv"],
    )

    sort_cols = [
        "selection_score",
        "abs_x_order_orth_component_delta",
        "order_abs_over_content_abs_sort",
    ]
    candidates = candidates.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)

    selected = candidates.copy()
    if MAX_FEATURES_PER_LAYER is not None:
        selected = selected.groupby("real_layer", group_keys=False).head(MAX_FEATURES_PER_LAYER)
    selected = selected.head(TOP_K_CANDIDATES).reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)

    candidates.to_csv(SAVE_RANKED_CANDIDATES_CSV, index=False)
    selected.to_csv(SAVE_SELECTED_CANDIDATES_CSV, index=False)

    print(f"\nRanked candidates saved: {SAVE_RANKED_CANDIDATES_CSV}")
    print(f"Selected candidates saved: {SAVE_SELECTED_CANDIDATES_CSV}")
    print("\nSelected candidates:")
    show_cols = [
        "selected_rank",
        "layer",
        "real_layer",
        "feature_index",
        "x_order_orth_component_delta",
        "x_content_component_delta",
        "order_specific_score",
        "order_abs_over_content_abs_safe",
        "interpretation_status",
        "recon_cosine_mean",
    ]
    show_cols = [c for c in show_cols if c in selected.columns]
    print(selected[show_cols].to_string(index=False))
    return candidates, selected, recon_summary


# ====================== MODEL / SAE LOADING ======================

def get_model_device(model_obj):
    try:
        return next(model_obj.parameters()).device
    except Exception:
        return torch.device(DEVICE)


def get_model_n_layers(model_obj):
    try:
        return int(model_obj.cfg.n_layers)
    except Exception:
        pass
    try:
        return int(len(model_obj.blocks))
    except Exception:
        return None


def validate_selected_layers_against_model(model_obj, selected):
    n_layers = get_model_n_layers(model_obj)
    real_layers = sorted(set(int(x) for x in selected["real_layer"].tolist()))
    csv_layers = sorted(set(int(x) for x in selected["layer"].tolist()))

    if n_layers is None:
        print(
            "WARNING: could not read model layer count; "
            "using CSV layer -> real_layer mapping without model-range validation."
        )
        return

    invalid = [x for x in real_layers if x < 0 or x >= n_layers]
    print(
        f"Layer mapping validation: CSV layers {csv_layers} -> real_layers {real_layers}; "
        f"model n_layers={n_layers}, valid real_layer range=0..{n_layers - 1}."
    )
    if invalid:
        raise ValueError(
            f"Invalid real_layer ids for model with n_layers={n_layers}: {invalid}. "
            "The expected mapping for 1-based CSV layers is real_layer = layer - 1."
        )


def load_model():
    if SKIP_MODEL_LOAD:
        return None
    if USE_EXISTING_MODEL_IF_AVAILABLE and "model" in globals():
        existing = globals()["model"]
        if hasattr(existing, "run_with_cache") and hasattr(existing, "to_tokens"):
            print("Using existing global TransformerLens model.")
            existing.eval()
            return existing

    print(f"Loading TransformerLens model: {MODEL_NAME}")
    from transformer_lens import HookedTransformer

    loaded_model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=MODEL_DTYPE,
    )
    loaded_model.eval()
    return loaded_model


def load_sae(real_layer):
    from sae_lens import SAE

    sae_id = SAE_ID_TEMPLATE.format(real_layer=int(real_layer))
    loaded = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=DEVICE,
    )
    if isinstance(loaded, tuple):
        sae = loaded[0]
    else:
        sae = loaded
    print(f"Loaded SAE: real_layer={real_layer}, sae_id={sae_id}")
    return sae


def load_needed_saes(selected):
    saes = {}
    for real_layer in sorted(set(int(x) for x in selected["real_layer"].tolist())):
        try:
            saes[real_layer] = load_sae(real_layer)
        except Exception as exc:
            print(f"WARNING: failed to load SAE layer {real_layer}: {repr(exc)}")
    return saes


# ====================== PATCHING METRICS ======================

def ensure_prompt_list(prompts):
    if prompts is None:
        return None
    if isinstance(prompts, str):
        return [prompts]
    return [str(x) for x in list(prompts)]


def tokenize_prompts(model_obj, prompts):
    tokens = model_obj.to_tokens(prompts, prepend_bos=PREPEND_BOS).to(get_model_device(model_obj))
    if MAX_PROMPT_TOKENS is not None and tokens.shape[-1] > MAX_PROMPT_TOKENS:
        tokens = tokens[:, -MAX_PROMPT_TOKENS:]
    elif hasattr(model_obj.cfg, "n_ctx") and model_obj.cfg.n_ctx is not None:
        n_ctx = int(model_obj.cfg.n_ctx)
        if tokens.shape[-1] > n_ctx:
            tokens = tokens[:, -n_ctx:]
    return tokens


def patch_sae_feature(activation, sae, feature_index):
    orig_dtype = activation.dtype
    orig_device = activation.device
    act_float = activation.to(dtype=torch.float32)

    with torch.no_grad():
        latent_original = sae.encode(act_float)
        latent = latent_original.clone()
        feature_index = int(feature_index)
        if feature_index < 0 or feature_index >= latent.shape[-1]:
            return activation

        if PATCH_POSITION_MODE == "last_token":
            target_slice = latent[:, -1:, feature_index]
        else:
            target_slice = latent[..., feature_index]

        if PATCH_MODE == "zero":
            target_slice[:] = 0.0
        elif PATCH_MODE == "scale":
            target_slice[:] = target_slice * PATCH_SCALE
        elif PATCH_MODE == "set":
            target_slice[:] = PATCH_VALUE
        else:
            raise ValueError(f"Unknown PATCH_MODE={PATCH_MODE!r}")

        decoded_patched = sae.decode(latent)
        if PATCH_RESIDUAL_UPDATE_MODE == "delta":
            decoded_original = sae.decode(latent_original)
            patched = act_float + (decoded_patched - decoded_original)
        elif PATCH_RESIDUAL_UPDATE_MODE == "replace":
            patched = decoded_patched
        else:
            raise ValueError(
                f"Unknown PATCH_RESIDUAL_UPDATE_MODE={PATCH_RESIDUAL_UPDATE_MODE!r}; "
                "expected 'delta' or 'replace'."
            )
    return patched.to(device=orig_device, dtype=orig_dtype)


def patch_sae_features_batched(activation, sae, feature_indices_for_batch):
    """
    Row-wise feature ablation for a batched set of variants.

    activation: [batch, seq, d_model]
    feature_indices_for_batch: length=batch; row b patches feature_indices[b].

    This lets us run several feature ablations in one forward pass by repeating
    the same prompt across batch rows and assigning a different feature to each
    row. It is much faster on large GPUs than one forward per feature.
    """
    orig_dtype = activation.dtype
    orig_device = activation.device
    act_float = activation.to(dtype=torch.float32)

    if not torch.is_tensor(feature_indices_for_batch):
        feature_indices_for_batch = torch.tensor(
            feature_indices_for_batch,
            device=activation.device,
            dtype=torch.long,
        )
    else:
        feature_indices_for_batch = feature_indices_for_batch.to(
            device=activation.device,
            dtype=torch.long,
        )

    with torch.no_grad():
        latent_original = sae.encode(act_float)
        latent = latent_original.clone()
        batch_size = int(latent.shape[0])
        n_features = int(latent.shape[-1])

        if int(feature_indices_for_batch.shape[0]) != batch_size:
            raise ValueError(
                f"feature_indices_for_batch has length {feature_indices_for_batch.shape[0]}, "
                f"but activation batch has {batch_size} rows."
            )

        for b in range(batch_size):
            f_idx = int(feature_indices_for_batch[b].detach().cpu().item())
            if f_idx < 0 or f_idx >= n_features:
                continue

            if PATCH_POSITION_MODE == "last_token":
                target_slice = latent[b:b + 1, -1:, f_idx]
            else:
                target_slice = latent[b:b + 1, :, f_idx]

            if PATCH_MODE == "zero":
                target_slice[:] = 0.0
            elif PATCH_MODE == "scale":
                target_slice[:] = target_slice * PATCH_SCALE
            elif PATCH_MODE == "set":
                target_slice[:] = PATCH_VALUE
            else:
                raise ValueError(f"Unknown PATCH_MODE={PATCH_MODE!r}")

        decoded_patched = sae.decode(latent)
        if PATCH_RESIDUAL_UPDATE_MODE == "delta":
            decoded_original = sae.decode(latent_original)
            patched = act_float + (decoded_patched - decoded_original)
        elif PATCH_RESIDUAL_UPDATE_MODE == "replace":
            patched = decoded_patched
        else:
            raise ValueError(
                f"Unknown PATCH_RESIDUAL_UPDATE_MODE={PATCH_RESIDUAL_UPDATE_MODE!r}; "
                "expected 'delta' or 'replace'."
            )
    return patched.to(device=orig_device, dtype=orig_dtype)


def safe_to_string_token(model_obj, token_id):
    try:
        return model_obj.to_string([int(token_id)])
    except Exception:
        return str(int(token_id))


def logit_metrics(model_obj, base_logits, patched_logits):
    base_logits = base_logits.float()
    patched_logits = patched_logits.float()
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    patched_logprobs = torch.log_softmax(patched_logits, dim=-1)
    base_probs = torch.softmax(base_logits, dim=-1)
    patched_probs = torch.softmax(patched_logits, dim=-1)

    kl_bp = (base_probs * (base_logprobs - patched_logprobs)).sum(dim=-1)
    kl_pb = (patched_probs * (patched_logprobs - base_logprobs)).sum(dim=-1)
    mix_probs = 0.5 * (base_probs + patched_probs)
    mix_logprobs = torch.log(mix_probs + 1e-30)
    js = 0.5 * (base_probs * (base_logprobs - mix_logprobs)).sum(dim=-1) + (
        0.5 * (patched_probs * (patched_logprobs - mix_logprobs)).sum(dim=-1)
    )

    logit_delta = patched_logits - base_logits
    base_top = base_logits.argmax(dim=-1)
    patched_top = patched_logits.argmax(dim=-1)
    top_changed = base_top != patched_top

    return {
        "kl_base_to_patched_mean": float(kl_bp.mean().detach().cpu().item()),
        "kl_patched_to_base_mean": float(kl_pb.mean().detach().cpu().item()),
        "js_mean": float(js.mean().detach().cpu().item()),
        "logit_l2_mean": float(logit_delta.norm(dim=-1).mean().detach().cpu().item()),
        "logit_max_abs_mean": float(logit_delta.abs().max(dim=-1).values.mean().detach().cpu().item()),
        "top_token_changed_rate": float(top_changed.float().mean().detach().cpu().item()),
        "base_top_token_first": safe_to_string_token(model_obj, int(base_top[0].detach().cpu().item())),
        "patched_top_token_first": safe_to_string_token(model_obj, int(patched_top[0].detach().cpu().item())),
    }


def run_rough_zero_ablation(model_obj, saes, selected):
    prompts = ensure_prompt_list(PATCH_PROMPTS)
    if not prompts:
        raise RuntimeError(
            "RUN_MEDIATION_PATCHING=True, but PATCH_PROMPTS/prompts_target is missing."
        )
    if len(saes) == 0:
        raise RuntimeError("No SAEs loaded; cannot patch.")

    rows = []
    base_logits_batches = []
    token_batches = []

    for start in range(0, len(prompts), PATCH_BATCH_SIZE):
        batch_prompts = prompts[start:start + PATCH_BATCH_SIZE]
        tokens = tokenize_prompts(model_obj, batch_prompts)
        with torch.no_grad():
            base_logits = model_obj(tokens)[:, -1, :].detach()
        token_batches.append((start, tokens))
        base_logits_batches.append(base_logits)

    layer_groups = [
        (int(real_layer), group.copy())
        for real_layer, group in selected.groupby("real_layer", sort=True)
        if int(real_layer) in saes
    ]
    total_chunks = 0
    for _, group in layer_groups:
        total_chunks += int(np.ceil(len(group) / max(1, PATCH_FEATURE_BATCH_SIZE)))

    pbar = tqdm(total=total_chunks, desc="Batched SAE zero-ablation")

    for real_layer, layer_rows in layer_groups:
        if real_layer not in saes:
            continue
        sae = saes[real_layer]
        hook_name = f"blocks.{real_layer}.hook_resid_post"

        layer_rows = layer_rows.reset_index(drop=True)
        for chunk_start in range(0, len(layer_rows), PATCH_FEATURE_BATCH_SIZE):
            feature_chunk = layer_rows.iloc[chunk_start:chunk_start + PATCH_FEATURE_BATCH_SIZE].copy()
            feature_indices = [int(x) for x in feature_chunk["feature_index"].tolist()]
            n_feature_variants = len(feature_indices)

            patched_by_feature = {feature_index: [] for feature_index in feature_indices}
            base_by_feature = {feature_index: [] for feature_index in feature_indices}

            for batch_idx, (_, tokens) in enumerate(token_batches):
                prompt_batch_size = int(tokens.shape[0])
                # Order: prompt0-feature0, prompt0-feature1, ...,
                # prompt1-feature0, prompt1-feature1, ...
                tokens_repeated = tokens.repeat_interleave(n_feature_variants, dim=0)
                feature_indices_for_batch = torch.tensor(
                    feature_indices * prompt_batch_size,
                    device=tokens.device,
                    dtype=torch.long,
                )

                def patching_hook(act, hook):
                    return patch_sae_features_batched(
                        act,
                        sae=sae,
                        feature_indices_for_batch=feature_indices_for_batch,
                    )

                with torch.no_grad():
                    with model_obj.hooks(fwd_hooks=[(hook_name, patching_hook)]):
                        patched_logits = model_obj(tokens_repeated)[:, -1, :].detach()

                # [prompt_batch * n_features, vocab] -> [prompt_batch, n_features, vocab]
                patched_logits = patched_logits.reshape(
                    prompt_batch_size,
                    n_feature_variants,
                    patched_logits.shape[-1],
                )

                base_logits = base_logits_batches[batch_idx]
                for variant_idx, feature_index in enumerate(feature_indices):
                    patched_by_feature[feature_index].append(patched_logits[:, variant_idx, :])
                    base_by_feature[feature_index].append(base_logits)

                del tokens_repeated, patched_logits, feature_indices_for_batch

            feature_chunk_by_id = {
                int(row["feature_index"]): row
                for _, row in feature_chunk.iterrows()
            }

            for feature_index in feature_indices:
                row = feature_chunk_by_id[int(feature_index)]
                base_logits_cat = torch.cat(base_by_feature[feature_index], dim=0)
                patched_logits_cat = torch.cat(patched_by_feature[feature_index], dim=0)
                metrics = logit_metrics(model_obj, base_logits_cat, patched_logits_cat)

                out = {
                    "selected_rank": int(row.get("selected_rank", -1)),
                    "candidate_rank": int(row.get("candidate_rank", -1)),
                    "layer": int(row["layer"]),
                    "real_layer": real_layer,
                    "feature_index": int(feature_index),
                    "x_order_orth_component_delta": float(row.get("x_order_orth_component_delta", np.nan)),
                    "x_content_component_delta": float(row.get("x_content_component_delta", np.nan)),
                    "order_specific_score": float(row.get("order_specific_score", np.nan)),
                    "order_abs_over_content_abs_safe": float(row.get("order_abs_over_content_abs_safe", np.nan)),
                    "interpretation_status": row.get("interpretation_status", ""),
                    "patch_mode": PATCH_MODE,
                    "patch_position_mode": PATCH_POSITION_MODE,
                    "patch_residual_update_mode": PATCH_RESIDUAL_UPDATE_MODE,
                    "patch_feature_batch_size": int(PATCH_FEATURE_BATCH_SIZE),
                    "n_prompts": int(len(prompts)),
                }
                out.update(metrics)
                rows.append(out)

            pbar.update(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    result = pd.DataFrame(rows).sort_values("kl_base_to_patched_mean", ascending=False)
    result.to_csv(SAVE_PATCHING_RESULTS_CSV, index=False)
    print(f"\nRough patching results saved: {SAVE_PATCHING_RESULTS_CSV}")
    print(result.head(20).to_string(index=False))
    return result


# ====================== TOP ACTIVATING CONTEXTS ======================

def normalize_context_token_for_filter(token):
    token = str(token)
    # Common tokenizer display markers. Keep this conservative: it is only for
    # selecting readable focus tokens, not for changing model inputs.
    token = token.replace("▁", " ")
    token = token.replace("Ġ", " ")
    token = token.replace("Ċ", "\n")
    return token


def is_blacklisted_context_token(token):
    raw = str(token)
    normalized = normalize_context_token_for_filter(raw)
    stripped = normalized.strip()
    stripped_lower = stripped.lower()

    if raw in CONTEXT_TOKEN_BLACKLIST_SET:
        return True
    if normalized in CONTEXT_TOKEN_BLACKLIST_SET:
        return True
    if stripped_lower in CONTEXT_TOKEN_BLACKLIST_STRIPPED_SET:
        return True
    if stripped_lower in {"<bos>", "<eos>", "<pad>", "<unk>", "bos", "eos", "pad", "unk"}:
        return True
    if stripped == "":
        return True
    if CONTEXT_TOKEN_PUNCT_RE.match(normalized):
        return True
    return False


def get_feature_top_contexts(model_obj, sae, texts, real_layer, feature_index):
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"
    records = []

    for text_id, text in enumerate(texts):
        tokens = tokenize_prompts(model_obj, [text])
        with torch.no_grad():
            _, cache = model_obj.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name].float()
            latent = sae.encode(act)
            if int(feature_index) >= latent.shape[-1]:
                continue
            scores = latent[..., int(feature_index)][0]
            str_tokens = model_obj.to_str_tokens(tokens[0].detach().cpu())

            if FILTER_CONTEXT_TOKENS:
                keep_mask = torch.ones_like(scores, dtype=torch.bool)
                for pos, token in enumerate(str_tokens[: int(scores.shape[0])]):
                    if is_blacklisted_context_token(token):
                        keep_mask[pos] = False
                filtered_scores = scores.clone()
                filtered_scores[~keep_mask] = -torch.inf
                available = int(keep_mask.sum().detach().cpu().item())
            else:
                filtered_scores = scores
                available = int(scores.shape[0])

            k = min(TOP_N_CONTEXTS, available)
            if k <= 0:
                continue
            values, positions = torch.topk(filtered_scores, k=k)

        for value, pos in zip(values.detach().cpu().tolist(), positions.detach().cpu().tolist()):
            left = max(0, int(pos) - CONTEXT_WINDOW)
            right = min(len(str_tokens), int(pos) + CONTEXT_WINDOW + 1)
            records.append({
                "text_id": int(text_id),
                "real_layer": int(real_layer),
                "layer": int(real_layer) + 1,
                "feature_index": int(feature_index),
                "activation": float(value),
                "token_position": int(pos),
                "token": str_tokens[int(pos)],
                "context": "".join(str_tokens[left:right]),
            })

        del cache, act, latent, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("activation", ascending=False)


def get_layer_features_top_contexts(model_obj, sae, texts, real_layer, feature_rows):
    """
    Efficient grouped context inspection.

    Runs the model once per layer/text batch, encodes the SAE latents once, and
    extracts top contexts for all selected features in that layer. This avoids
    the old feature-by-feature repeated forward passes.
    """
    hook_name = f"blocks.{int(real_layer)}.hook_resid_post"
    records = []
    feature_rows = feature_rows.copy()

    for batch_start in range(0, len(texts), CONTEXT_BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + CONTEXT_BATCH_SIZE]
        tokens = tokenize_prompts(model_obj, batch_texts)

        with torch.no_grad():
            _, cache = model_obj.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name].float()
            latent = sae.encode(act)

        for b in range(latent.shape[0]):
            text_id = batch_start + b
            str_tokens = model_obj.to_str_tokens(tokens[b].detach().cpu())
            seq_len = min(len(str_tokens), int(latent.shape[1]))

            if FILTER_CONTEXT_TOKENS:
                keep_mask = torch.ones(seq_len, dtype=torch.bool, device=latent.device)
                for pos, token in enumerate(str_tokens[:seq_len]):
                    if is_blacklisted_context_token(token):
                        keep_mask[pos] = False
            else:
                keep_mask = torch.ones(seq_len, dtype=torch.bool, device=latent.device)

            available = int(keep_mask.sum().detach().cpu().item())
            if available <= 0:
                continue

            for _, feature_row in feature_rows.iterrows():
                feature_index = int(feature_row["feature_index"])
                if feature_index < 0 or feature_index >= latent.shape[-1]:
                    continue

                scores = latent[b, :seq_len, feature_index]
                filtered_scores = scores.clone()
                filtered_scores[~keep_mask] = -torch.inf
                k = min(TOP_N_CONTEXTS, available)
                if k <= 0:
                    continue

                values, positions = torch.topk(filtered_scores, k=k)
                for value, pos in zip(values.detach().cpu().tolist(), positions.detach().cpu().tolist()):
                    left = max(0, int(pos) - CONTEXT_WINDOW)
                    right = min(len(str_tokens), int(pos) + CONTEXT_WINDOW + 1)
                    records.append({
                        "text_id": int(text_id),
                        "real_layer": int(real_layer),
                        "layer": int(real_layer) + 1,
                        "feature_index": int(feature_index),
                        "activation": float(value),
                        "token_position": int(pos),
                        "token": str_tokens[int(pos)],
                        "context": "".join(str_tokens[left:right]),
                        "selected_rank": int(feature_row.get("selected_rank", -1)),
                        "candidate_rank": int(feature_row.get("candidate_rank", -1)),
                        "interpretation_status": feature_row.get("interpretation_status", ""),
                        "x_order_orth_component_delta": feature_row.get("x_order_orth_component_delta", np.nan),
                        "x_content_component_delta": feature_row.get("x_content_component_delta", np.nan),
                        "order_specific_score": feature_row.get("order_specific_score", np.nan),
                    })

        del cache, act, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["selected_rank", "activation"],
        ascending=[True, False],
    )


def run_top_context_inspection(model_obj, saes, selected):
    texts = ensure_prompt_list(CONTEXT_TEXTS)
    if not texts:
        print("No CONTEXT_TEXTS/PATCH_PROMPTS available; skipping context inspection.")
        return pd.DataFrame()

    frames = []
    rows_to_inspect = selected if N_CONTEXT_FEATURES is None else selected.head(N_CONTEXT_FEATURES)
    for real_layer, layer_rows in rows_to_inspect.groupby("real_layer", sort=True):
        real_layer = int(real_layer)
        if real_layer not in saes:
            continue
        print(f"\nTop contexts for layer={real_layer}, features={len(layer_rows)}")
        contexts = get_layer_features_top_contexts(
            model_obj=model_obj,
            sae=saes[real_layer],
            texts=texts,
            real_layer=real_layer,
            feature_rows=layer_rows,
        )
        if len(contexts) == 0:
            continue
        print(contexts[[
            "selected_rank",
            "feature_index",
            "activation",
            "token",
            "context",
        ]].head(min(20, len(contexts))).to_string(index=False))
        frames.append(contexts)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(SAVE_CONTEXTS_CSV, index=False)
    print(f"\nTop activating contexts saved: {SAVE_CONTEXTS_CSV}")
    return result


# ====================== SUMMARY ======================

def write_summary(manifest, selected, recon_summary, patching_results=None, contexts=None):
    lines = []
    lines.append(f"# Gemma SAE order feature patching v2: {RUN_TAG}")
    lines.append("")
    lines.append("## Tables")
    for _, row in manifest.iterrows():
        status = "loaded" if int(row["loaded"]) else "missing"
        lines.append(f"- {row['filename']}: {status}, rows={row['rows']}, source={row['source']}")
    lines.append("")
    lines.append("## Candidate selection")
    lines.append(
        "Primary source is `sae_order_feature_contrast.csv`; other SAE CSVs are used as support evidence."
    )
    lines.append(f"- tokenization prepend_bos: {PREPEND_BOS}")
    lines.append(
        "Candidates are ranked by `order_specific_score` / absolute `x_order_orth_component_delta`, "
        "so negative order-direction features are preserved."
    )
    lines.append(f"- selected candidates: {len(selected)}")
    if len(selected):
        lines.append("")
        lines.append("| rank | layer | real_layer | feature | x_order_orth | x_content | score | status |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, row in selected.head(20).iterrows():
            lines.append(
                f"| {int(row.get('selected_rank', -1))} "
                f"| {int(row['layer'])} "
                f"| {int(row['real_layer'])} "
                f"| {int(row['feature_index'])} "
                f"| {float(row.get('x_order_orth_component_delta', np.nan)):.6g} "
                f"| {float(row.get('x_content_component_delta', np.nan)):.6g} "
                f"| {float(row.get('order_specific_score', np.nan)):.6g} "
                f"| {row.get('interpretation_status', '')} |"
            )
    lines.append("")
    if len(recon_summary):
        lines.append("## SAE reconstruction quality")
        lines.append(
            f"- mean reconstruction cosine across loaded rows: "
            f"{recon_summary['recon_cosine_mean'].mean():.6f}"
        )
        lines.append(
            f"- mean explained variance proxy across loaded rows: "
            f"{recon_summary['explained_variance_proxy_mean'].mean():.6f}"
        )
    if patching_results is not None and len(patching_results):
        lines.append("")
        lines.append("## Rough zero-ablation signal")
        lines.append(
            "This is an exploratory logit-level perturbation screen, not final causal proof."
        )
        lines.append(f"- patch feature batch size: {PATCH_FEATURE_BATCH_SIZE}")
        lines.append(f"- prompt batch size: {PATCH_BATCH_SIZE}")
        lines.append(f"- residual update mode: {PATCH_RESIDUAL_UPDATE_MODE}")
        lines.append(
            f"- max KL(base||patched): {patching_results['kl_base_to_patched_mean'].max():.6g}"
        )
        lines.append(
            f"- mean KL(base||patched): {patching_results['kl_base_to_patched_mean'].mean():.6g}"
        )
        lines.append(
            f"- mean top-token changed rate: {patching_results['top_token_changed_rate'].mean():.6g}"
        )
    if contexts is not None and len(contexts):
        lines.append("")
        lines.append("## Top activating contexts")
        lines.append(f"- context rows saved: {len(contexts)}")
        lines.append(f"- context batch size: {CONTEXT_BATCH_SIZE}")
        lines.append(f"- context focus token blacklist enabled: {FILTER_CONTEXT_TOKENS}")
        if FILTER_CONTEXT_TOKENS:
            lines.append(
                "- punctuation/whitespace/BOS-like tokens are excluded from top-k focus-token selection only; "
                "the surrounding context text is preserved."
            )
    lines.append("")
    lines.append("## Output files")
    for path in [
        SAVE_TABLE_MANIFEST_CSV,
        SAVE_RECON_SUMMARY_CSV,
        SAVE_RANKED_CANDIDATES_CSV,
        SAVE_SELECTED_CANDIDATES_CSV,
        SAVE_PATCHING_RESULTS_CSV,
        SAVE_CONTEXTS_CSV,
    ]:
        lines.append(f"- `{path}`")

    SAVE_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary saved: {SAVE_SUMMARY_MD}")


# ====================== RUN ======================

tables, manifest_df = load_all_sae_tables()
ranked_candidates, selected_candidates, recon_summary_df = build_candidate_table(tables)

patching_df = pd.DataFrame()
contexts_df = pd.DataFrame()

if not SKIP_MODEL_LOAD and (RUN_MEDIATION_PATCHING or RUN_TOP_CONTEXT_INSPECTION):
    model = load_model()
    validate_selected_layers_against_model(model, selected_candidates)
    needed_saes = load_needed_saes(selected_candidates)

    if RUN_MEDIATION_PATCHING:
        patching_df = run_rough_zero_ablation(model, needed_saes, selected_candidates)
    else:
        print("Rough mediation patching skipped: RUN_MEDIATION_PATCHING=False")

    if RUN_TOP_CONTEXT_INSPECTION:
        contexts_df = run_top_context_inspection(model, needed_saes, selected_candidates)
    else:
        print("Top context inspection skipped: RUN_TOP_CONTEXT_INSPECTION=False")
else:
    print("Model/SAE execution skipped. Candidate tables were still generated.")

write_summary(
    manifest=manifest_df,
    selected=selected_candidates,
    recon_summary=recon_summary_df,
    patching_results=patching_df,
    contexts=contexts_df,
)

print("\nDONE")
print(f"Output directory: {OUTPUT_DIR}")
