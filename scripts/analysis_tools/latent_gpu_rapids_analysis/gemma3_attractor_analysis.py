"""
Gemma-3-12B-It | Hidden Geometry Attractor Analysis
Four-module cross-architectural synthesis pipeline.
Target: Google Colab A100 80GB
"""

import os, gc, zipfile
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ZIP_PATH    = '/content/red_team_hidden_geometry_results_grade_axis_decomposition-google-gemma-3-12b-it__1_.zip'
DATA_DIR    = '/content/gemma3_attractor_data'
OUT_DIR     = '/content/gemma3_attractor_data'
CHUNK_SIZE  = 100_000
DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TOP_N       = 50     # top neurons in module 1
FINAL_N     = 20     # top neurons carried into synthesis

print(f"[INIT] device={DEVICE}  |  torch={torch.__version__}  |  numpy={np.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# UNZIP (idempotent)
# ─────────────────────────────────────────────────────────────────────────────

def _locate(name: str) -> str:
    """Return absolute path to a named file anywhere under DATA_DIR."""
    for root, _, files in os.walk(DATA_DIR):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under {DATA_DIR}")


if not os.path.isdir(DATA_DIR):
    print(f"[UNZIP] extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(DATA_DIR)
    print("[UNZIP] done")
else:
    print(f"[UNZIP] {DATA_DIR} already present, skipping extraction")

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — Streaming neuron anomaly analysis (CPU + Pandas chunked)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("MODULE 1 — Streaming neuron anomaly scan")
print("═"*70)

layer_module_stats: dict  = {}   # (layer, module, unit_type) -> {count, total_delta}
neuron_frequency:   dict  = {}   # (layer, module, unit_index) -> {count, max_delta}

csv_path = _locate('architecture_top_changed_units.csv')
print(f"[M1] reading  {csv_path}")

with pd.read_csv(csv_path, chunksize=CHUNK_SIZE) as reader:
    for chunk in tqdm(reader, desc="M1 chunks"):
        top = chunk[chunk['rank_by_abs_delta'] <= 3]

        for row in top.itertuples(index=False):
            layer      = int(row.layer)
            module     = str(row.module)
            unit_type  = str(row.unit_type)
            abs_delta  = float(row.abs_delta)
            unit_index = int(row.unit_index)

            # per-layer / per-module aggregate
            gk = (layer, module, unit_type)
            if gk not in layer_module_stats:
                layer_module_stats[gk] = {'count': 0, 'total_delta': 0.0}
            layer_module_stats[gk]['count']       += 1
            layer_module_stats[gk]['total_delta'] += abs_delta

            # per-neuron frequency
            nk = (layer, module, unit_index)
            if nk not in neuron_frequency:
                neuron_frequency[nk] = {'count': 0, 'max_delta': 0.0}
            neuron_frequency[nk]['count'] += 1
            if abs_delta > neuron_frequency[nk]['max_delta']:
                neuron_frequency[nk]['max_delta'] = abs_delta

# ── build layer-anomaly report ──
rows_geo = []
for (layer, module, unit_type), s in layer_module_stats.items():
    rows_geo.append({
        'layer':               layer,
        'module':              module,
        'unit_type':           unit_type,
        'anomaly_count':       s['count'],
        'mean_abs_delta':      s['total_delta'] / s['count'],
    })
df_layer_anomalies = (
    pd.DataFrame(rows_geo)
    .sort_values(['layer', 'anomaly_count'], ascending=[True, False])
    .reset_index(drop=True)
)

# ── build top-N neuron table ──
rows_neurons = []
for (layer, module, unit_index), s in neuron_frequency.items():
    rows_neurons.append({
        'layer':       layer,
        'module':      module,
        'unit_index':  unit_index,
        'freq':        s['count'],
        'max_delta':   s['max_delta'],
    })
df_top_neurons = (
    pd.DataFrame(rows_neurons)
    .sort_values('freq', ascending=False)
    .head(TOP_N)
    .reset_index(drop=True)
)

print(f"[M1] layer-module pairs found : {len(df_layer_anomalies)}")
print(f"[M1] unique neurons tracked   : {len(rows_neurons)}")
print(f"[M1] top-{TOP_N} neurons (head-5):")
print(df_top_neurons.head(5).to_string(index=False))

del rows_geo, rows_neurons
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — GPU tensor stability analysis (PyTorch CUDA)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("MODULE 2 — GPU vector stability analysis")
print("═"*70)

def load_numeric_tensors(npz_path: str, device: torch.device) -> dict:
    """Load only numeric arrays from an .npz onto `device`."""
    npz   = np.load(npz_path)
    out   = {}
    skipped = []
    for k in npz.keys():
        arr = npz[k]
        if not np.issubdtype(arr.dtype, np.number):
            skipped.append((k, str(arr.dtype)))
            continue
        out[k] = torch.tensor(arr, dtype=torch.float32, device=device)
    if skipped:
        print(f"[M2]   skipped non-numeric keys: {skipped}")
    return out


# ── vector_x_by_layer  shape (n_layers, hidden_dim) ──
vx_path = _locate('vector_x_by_layer.npz')
print(f"[M2] loading {vx_path}")
vx_tensors = load_numeric_tensors(vx_path, DEVICE)

# The primary key is 'vector_x_by_layer'  shape (49, 3840)
vx_mat = vx_tensors['vector_x_by_layer']          # (L, H)
n_layers, hidden_dim = vx_mat.shape
print(f"[M2] vector_x_by_layer  shape={tuple(vx_mat.shape)}")

# cosine similarity between consecutive layer vectors
stab_rows = []
for i in range(n_layers - 1):
    v_a = vx_mat[i]
    v_b = vx_mat[i + 1]
    cos = (torch.dot(v_a, v_b) /
           (torch.norm(v_a) * torch.norm(v_b) + 1e-12)).item()
    stab_rows.append({
        'layer_from':             i,
        'layer_to':               i + 1,
        'cosine_stability':       cos,
    })

df_vx_stab = pd.DataFrame(stab_rows)

# ── prompt_hidden_states  shape (n_conditions, n_layers, hidden_dim) ──
ph_path = _locate('prompt_hidden_states.npz')
print(f"[M2] loading {ph_path}")
ph_tensors = load_numeric_tensors(ph_path, DEVICE)

hs = ph_tensors['hidden_states']                   # (C, L, H)
n_cond, n_layers_hs, _ = hs.shape
print(f"[M2] hidden_states  shape={tuple(hs.shape)}")

# mean hidden state per layer across conditions → cosine against vector_x
ph_mean = hs.mean(dim=0)                           # (L, H)
n_common = min(n_layers, n_layers_hs)

proj_rows = []
for i in range(n_common):
    v_x   = vx_mat[i]
    h_mu  = ph_mean[i]
    cos   = (torch.dot(v_x, h_mu) /
             (torch.norm(v_x) * torch.norm(h_mu) + 1e-12)).item()
    proj_rows.append({
        'layer':                        i,
        'cosine_hidden_vs_vx':          cos,
    })

df_proj = pd.DataFrame(proj_rows)

# merge both stability signals on layer
df_vx_stab_aug = df_vx_stab.merge(
    df_proj.rename(columns={'layer': 'layer_to'}),
    on='layer_to', how='left'
)

# identify stabilisation layer = first local maximum of cosine_stability
cos_arr = df_vx_stab_aug['cosine_stability'].values
stab_layer = int(np.argmax(cos_arr))
print(f"[M2] peak cosine stability at layer transition {stab_layer} → {stab_layer+1}  "
      f"(cos={cos_arr[stab_layer]:.6f})")

del vx_tensors, ph_tensors, hs, ph_mean, vx_mat
gc.collect()
torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Causal trajectory aggregation (alpha-steering dynamics)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("MODULE 3 — Causal trajectory time-series aggregation")
print("═"*70)

TRAJ_COLS_CAUSAL = [
    'alpha', 'alpha_abs', 'sign_name', 'step', 'layer',
    'is_middle_layer',
    'direction_cosine_with_vector_x_loo',
    'l2_distance_to_reference_prompt_endpoint',
]
TRAJ_COLS_GEN = [
    'condition', 'step', 'layer', 'is_middle_layer',
    'direction_cosine_with_vector_x_loo',
    'l2_distance_to_reference_prompt_endpoint',
]

# ── causal intervention trajectory ──
causal_path = _locate('causal_intervention_trajectory_metrics_raw.csv')
print(f"[M3] reading {causal_path}")

causal_chunks = []
with pd.read_csv(causal_path, chunksize=CHUNK_SIZE,
                 usecols=lambda c: c in TRAJ_COLS_CAUSAL) as reader:
    for chunk in tqdm(reader, desc="M3 causal chunks"):
        mid = chunk[chunk['is_middle_layer'] == 1].copy()
        if mid.empty:
            continue
        mid['cos_delta']  = mid['direction_cosine_with_vector_x_loo'].abs()
        mid['l2_delta']   = mid['l2_distance_to_reference_prompt_endpoint']
        causal_chunks.append(mid[['alpha_abs', 'sign_name', 'step',
                                   'cos_delta', 'l2_delta']])

df_causal_raw = pd.concat(causal_chunks, ignore_index=True)
del causal_chunks
gc.collect()

# mean "suction speed" per alpha × step
df_causal_agg = (
    df_causal_raw
    .groupby(['alpha_abs', 'sign_name', 'step'], sort=True)
    .agg(
        mean_cos_delta=('cos_delta', 'mean'),
        mean_l2_delta=('l2_delta',  'mean'),
        n_samples=('cos_delta', 'count'),
    )
    .reset_index()
)

# suction speed: derivative of cosine alignment over generation steps
df_causal_agg = df_causal_agg.sort_values(['alpha_abs', 'sign_name', 'step'])
df_causal_agg['suction_speed'] = (
    df_causal_agg
    .groupby(['alpha_abs', 'sign_name'])['mean_cos_delta']
    .diff()
    .fillna(0.0)
)

print(f"[M3] causal trajectory rows aggregated: {len(df_causal_agg)}")
del df_causal_raw
gc.collect()

# ── generation trajectory ──
gen_path = _locate('generation_trajectory_metrics_raw.csv')
print(f"[M3] reading {gen_path}")

gen_chunks = []
with pd.read_csv(gen_path, chunksize=CHUNK_SIZE,
                 usecols=lambda c: c in TRAJ_COLS_GEN) as reader:
    for chunk in tqdm(reader, desc="M3 generation chunks"):
        mid = chunk[chunk['is_middle_layer'] == 1].copy()
        if mid.empty:
            continue
        mid['cos_delta'] = mid['direction_cosine_with_vector_x_loo'].abs()
        mid['l2_delta']  = mid['l2_distance_to_reference_prompt_endpoint']
        gen_chunks.append(mid[['condition', 'step', 'cos_delta', 'l2_delta']])

df_gen_raw = pd.concat(gen_chunks, ignore_index=True)
del gen_chunks
gc.collect()

df_gen_agg = (
    df_gen_raw
    .groupby(['condition', 'step'], sort=True)
    .agg(
        mean_cos_delta=('cos_delta', 'mean'),
        mean_l2_delta=('l2_delta',  'mean'),
        n_samples=('cos_delta', 'count'),
    )
    .reset_index()
)
df_gen_agg = df_gen_agg.sort_values(['condition', 'step'])
df_gen_agg['suction_speed'] = (
    df_gen_agg
    .groupby('condition')['mean_cos_delta']
    .diff()
    .fillna(0.0)
)

print(f"[M3] generation trajectory rows aggregated: {len(df_gen_agg)}")
del df_gen_raw
gc.collect()

# combined trajectory report — used in synthesis
df_trajectory_combined = pd.concat([
    df_causal_agg.assign(source='causal').rename(
        columns={'alpha_abs': 'alpha', 'sign_name': 'condition'}),
    df_gen_agg.assign(source='generation', alpha=float('nan')),
], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Cross-architectural synthesis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("MODULE 4 — Global cross-architectural synthesis")
print("═"*70)

# ── load mlp_unit_cluster_summary for cluster labels ──
cluster_path = _locate('mlp_unit_cluster_summary.csv')
print(f"[M4] loading {cluster_path}")

df_cluster = pd.read_csv(cluster_path, usecols=lambda c: c in
    ['condition', 'module', 'layer', 'unit_index',
     'mean_delta', 'mean_abs_delta', 'q_count', 'top_rank_mean'])

# best cluster entry per (layer, module, unit_index) — highest q_count
df_cluster_best = (
    df_cluster
    .sort_values('q_count', ascending=False)
    .groupby(['layer', 'module', 'unit_index'], as_index=False)
    .first()
    .rename(columns={
        'mean_delta':     'cluster_mean_delta',
        'mean_abs_delta': 'cluster_mean_abs_delta',
        'q_count':        'cluster_q_count',
        'top_rank_mean':  'cluster_top_rank_mean',
        'condition':      'cluster_dominant_condition',
    })
)

# ── identify stabilisation layer band ──
# top-5 layer transitions by cosine stability
top_stab_layers = (
    df_vx_stab_aug
    .nlargest(5, 'cosine_stability')['layer_to']
    .values
)
print(f"[M4] top-5 stabilisation layers: {sorted(top_stab_layers)}")

# ── causal impact per layer: mean suction speed across all conditions ──
causal_layer_impact = (
    df_trajectory_combined[df_trajectory_combined['source'] == 'causal']
    .groupby('step', as_index=False)
    .agg(mean_suction=('suction_speed', 'mean'))
    .rename(columns={'step': 'layer'})
)

# ── top-FINAL_N neurons from M1 ──
df_top_final = df_top_neurons.head(FINAL_N).copy()

# ── flag neurons sitting on high-stability layers ──
df_top_final['on_stability_peak'] = df_top_final['layer'].isin(top_stab_layers)

# ── attach cluster info ──
df_top_final = df_top_final.merge(
    df_cluster_best[['layer', 'module', 'unit_index',
                     'cluster_dominant_condition',
                     'cluster_mean_abs_delta',
                     'cluster_q_count',
                     'cluster_top_rank_mean']],
    on=['layer', 'module', 'unit_index'],
    how='left',
)

# ── attach causal layer impact (approximate: match neuron layer to step) ──
df_top_final = df_top_final.merge(
    causal_layer_impact,
    on='layer',
    how='left',
)

# ── attach cosine stability of the layer ──
layer_stab_lookup = (
    df_vx_stab_aug[['layer_to', 'cosine_stability', 'cosine_hidden_vs_vx']]
    .rename(columns={'layer_to': 'layer'})
)
df_top_final = df_top_final.merge(layer_stab_lookup, on='layer', how='left')

# ── composite criticality score ──
# normalise each dimension to [0,1], then sum
def _norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-12)

df_top_final['score_freq']    = _norm(df_top_final['freq'])
df_top_final['score_delta']   = _norm(df_top_final['max_delta'])
df_top_final['score_stab']    = _norm(df_top_final['cosine_stability'].fillna(0))
df_top_final['score_suction'] = _norm(df_top_final['mean_suction'].fillna(0))

df_top_final['criticality_score'] = (
    df_top_final['score_freq']  * 0.35 +
    df_top_final['score_delta'] * 0.30 +
    df_top_final['score_stab']  * 0.20 +
    df_top_final['score_suction'] * 0.15
)

df_global_neurons = (
    df_top_final
    .drop(columns=['score_freq', 'score_delta', 'score_stab', 'score_suction'])
    .sort_values('criticality_score', ascending=False)
    .reset_index(drop=True)
)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("SAVING OUTPUTS")
print("═"*70)

out1 = os.path.join(OUT_DIR, 'INTEGRATED_LAYER_ANOMALIES_REPORT.csv')
out2 = os.path.join(OUT_DIR, 'ATTRACTOR_VECTOR_CUDA_STABILITY.csv')
out3 = os.path.join(OUT_DIR, 'GLOBAL_CAUSAL_NEURONS_MAP.csv')

df_layer_anomalies.to_csv(out1, index=False)
df_vx_stab_aug.to_csv(out2, index=False)
df_global_neurons.to_csv(out3, index=False)

print(f"[OUT] {out1}  ({len(df_layer_anomalies)} rows)")
print(f"[OUT] {out2}  ({len(df_vx_stab_aug)} rows)")
print(f"[OUT] {out3}  ({len(df_global_neurons)} rows)")

print("\n" + "═"*70)
print("TOP-5 CAUSAL NEURONS BY CRITICALITY SCORE")
print("═"*70)
print(df_global_neurons[[
    'layer', 'module', 'unit_index', 'freq', 'max_delta',
    'cosine_stability', 'on_stability_peak',
    'cluster_dominant_condition', 'criticality_score'
]].head(5).to_string(index=False))

print("\n" + "═"*70)
print("VECTOR STABILISATION PROFILE (top-5 transitions by cosine)")
print("═"*70)
print(df_vx_stab_aug.nlargest(5, 'cosine_stability')
      [['layer_from', 'layer_to', 'cosine_stability', 'cosine_hidden_vs_vx']]
      .to_string(index=False))

print("\n[DONE]")
