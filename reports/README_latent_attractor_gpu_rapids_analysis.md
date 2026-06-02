# `latent_attractor_gpu_rapids_analysis.py`

GPU-first analyzer for **derived Latent Discourse Regime metrics** stored in CSV files or ZIP archives.

This script is designed for datasets where the primitive activation-level metrics have already been computed upstream. It does **not** recompute cosine similarity, L2 distance, entropy, or logit shifts from raw transformer tensors. Instead, it reads derived metric tables and produces global summaries, grouped summaries, condition effects, alpha-response regressions, and layerwise transition proxies.

---

## 1. What this script is for

Use this script when your dataset already contains columns such as:

```text
projection_fraction_on_vector_x_loo
direction_cosine_with_vector_x_loo
l2_distance_to_reference_prompt_endpoint
entropy
mean_entropy
selected_logprob
mean_selected_logprob
alpha
alpha_abs
delta
abs_delta
mean_delta
mean_abs_delta
max_abs_delta
condition
base_condition
reference_condition
intervention_name
layer
layer_band
module
sign_name
```

The script analyzes these existing metrics at scale using RAPIDS/cuDF on GPU.

Typical input:

```text
red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip
```

or a directory containing many CSV files.

Typical output:

```text
FINAL_LATENT_ATTRACTOR_METRICS.csv
global_metric_summary.csv
grouped_metric_summary.csv
condition_effects.csv
causal_alpha_regression.csv
layerwise_phase_transition.csv
processing_audit.csv
```

---

## 2. What this script does not do

This script does **not** read raw activation rows in the layout:

```text
h_layer | v_att | logits
```

It does **not** recompute:

```latex
\mathrm{Cos}(\theta_\ell)
= \frac{h_\ell^{(t)} \cdot v_{\mathrm{att}}}{\|h_\ell^{(t)}\|_2 \|v_{\mathrm{att}}\|_2}
```

```latex
\mathcal{D}_{L2}(h_\ell, h_{\mathrm{ref}})
= \|h_\ell^{(t)} - h_{\mathrm{ref}}\|_2
```

```latex
H(X) = - \sum_{i \in V} P(x_i) \log_2 P(x_i)
```

Those primitive quantities must already exist in the input CSV files as derived metric columns.

If your files contain raw hidden states or logits, you need a separate upstream extraction script.

---

## 3. Conceptual pipeline

```text
Upstream activation pipeline
  raw activations / logits
  -> cosine, projection, L2, entropy, logprob, delta metrics
  -> derived CSV files

This script
  derived CSV files
  -> global summaries
  -> grouped summaries
  -> condition effects
  -> alpha-response regressions
  -> layerwise transition proxies
  -> unified evidence table
```

This script is the **secondary evidence-analysis layer**, not the primitive activation-metric extractor.

---

## 4. Recommended project layout

```text
your_project/
├── scripts/
│   └── latent_attractor_gpu_rapids_analysis.py
│
├── data/
│   └── metrics_raw/
│       └── red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip
│
├── outputs/
│   └── latent_attractor_analysis/
│
├── reports/
│   └── README_latent_attractor_gpu_rapids_analysis.md
│
└── requirements-colab.txt
```

Do not mix source metrics, generated outputs, and cache files in the same directory. That is how quiet methodological garbage is born.

---

## 5. Installation

### Google Colab / CUDA 12

```bash
!pip install -q cudf-cu12 cupy-cuda12x pyarrow
```

After installation, restart the runtime if `import cudf` still fails.

Check installation:

```python
import cudf
import cupy
print("cuDF OK")
```

### Local environment

Use a CUDA-compatible RAPIDS/cuDF installation. The script requires:

```text
Python 3.10+
cudf
cupy
pandas
numpy
tqdm
pyarrow
```

Minimal local installation depends on your CUDA/RAPIDS version. For Colab, use the command above.

---

## 6. Basic usage

### Full run on ZIP

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/latent_attractor_analysis" \
  --strict
```

### Colab version

```bash
!python "/content/latent_attractor_gpu_rapids_analysis.py" \
  --input "/content/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "/content/latent_gpu_out" \
  --strict
```

Always quote paths. Bash is not your friend.

---

## 7. Test run

Before a full run, process only the first few CSV files:

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/test_latent_attractor_analysis" \
  --limit-files 3 \
  --strict
```

Inspect outputs:

```bash
ls -lh "outputs/test_latent_attractor_analysis"
```

---

## 8. Optional Parquet cache

The script can cache pruned input tables as Parquet:

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/latent_attractor_analysis" \
  --cache-parquet \
  --strict
```

This creates:

```text
outputs/latent_attractor_analysis/_work/parquet_cache/
```

Parquet cache is **not a research result**. It is only a speed optimization for repeated runs.

Use cache when:

```text
- you will rerun the same analysis many times
- CSV parsing is slow
- you have enough disk space
```

Do not use cache when:

```text
- you want a clean one-shot final run
- disk space is limited
- you want fewer intermediate files
```

Remove cache:

```bash
rm -rf "outputs/latent_attractor_analysis/_work/parquet_cache"
```

Force regeneration:

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/latent_attractor_analysis" \
  --cache-parquet \
  --overwrite-cache \
  --strict
```

---

## 9. CLI arguments

### Required

#### `--input`

Path to a `.zip` file or directory containing CSV files.

Examples:

```bash
--input "data/metrics_raw/run_outputs.zip"
--input "data/metrics_raw/extracted_run/"
```

#### `--output-dir`

Directory where result CSV files are written.

Example:

```bash
--output-dir "outputs/latent_attractor_analysis"
```

### Optional

#### `--work-dir`

Custom working/cache directory. Default:

```text
<output-dir>/_work
```

Example:

```bash
--work-dir "/tmp/latent_attractor_work"
```

#### `--cache-parquet`

Cache selected input columns as Parquet for faster reruns.

#### `--overwrite-cache`

Overwrite existing Parquet cache.

#### `--force-extract`

Re-extract ZIP even if an extraction marker already exists.

Useful when the ZIP changed but the work directory still contains an old extraction.

#### `--limit-files N`

Debug mode. Process only the first `N` CSV files.

Example:

```bash
--limit-files 3
```

Do not use this for final analysis.

#### `--write-per-file`

Also write per-file intermediate outputs into:

```text
<output-dir>/per_file/
```

This is useful for debugging but can create many files.

#### `--strict`

Fail immediately on the first file error.

Recommended for final scientific runs. Without `--strict`, failed files are logged in `processing_audit.csv` and the script continues.

---

## 10. Outputs

### `FINAL_LATENT_ATTRACTOR_METRICS.csv`

Unified evidence table combining:

```text
global metric summaries
condition effects
alpha regressions
layerwise transition proxies
```

Important columns:

```text
evidence_type
source_file
formalism_class
metric
context
primary_value_name
primary_value
secondary_value_name
secondary_value
n
details_json
```

Use this as the high-level evidence table.

---

### `global_metric_summary.csv`

Global statistics per metric per source file.

Columns include:

```text
source_file
formalism_class
metric
rows_seen
finite_count
missing_count
mean
std
min
max
ci95_low
ci95_high
```

This answers:

```text
What is the overall mean/std/min/max of each derived metric?
How many finite values were actually used?
```

---

### `grouped_metric_summary.csv`

Grouped statistics over available grouping keys.

Possible groupings include:

```text
condition
condition+layer
condition+step
condition+reference_condition
condition+reference_condition+layer
condition+module
condition+module+layer
condition+unit_type+layer
base_condition
base_condition+layer_band
base_condition+sign_name
base_condition+layer_band+alpha_abs
base_condition+layer_band+sign_name+alpha_abs
base_condition+layer_band+sign_name+alpha_abs+layer
base_condition+intervention_name
base_condition+intervention_name+layer
layer
layer_band
module
module+layer
unit_type
unit_type+layer
```

This answers:

```text
How do metrics vary by condition, layer, intervention, module, or alpha band?
```

---

### `condition_effects.csv`

Condition-minus-baseline deltas.

The script chooses a baseline using this priority:

```text
neutral
neutral_length_matched_control
control
question_only
```

If none are present, it uses the first condition alphabetically. That fallback is mechanical, not scientific. Check it.

Important columns:

```text
effect_type
formalism_class
metric
grouping
condition
baseline_condition
condition_mean
baseline_mean
delta
relative_delta
```

Interpretation:

```text
positive cosine/projection delta -> stronger attractor alignment
negative L2 delta -> closer to reference endpoint
positive entropy delta -> entropy amplification
```

---

### `causal_alpha_regression.csv`

OLS regressions of metrics against `alpha` and/or `alpha_abs`.

Model:

```text
metric ~ alpha
metric ~ alpha_abs
```

Important columns:

```text
source_file
formalism_class
dependent_metric
x
grouping
n
slope_beta_like
intercept
r2
```

Interpretation:

```text
slope_beta_like -> empirical alpha-response slope
r2              -> variance explained by alpha in that grouping
```

This is a beta-like coefficient. It is not the same thing as a theoretical causal constant unless the experimental design supports that interpretation.

---

### `layerwise_phase_transition.csv`

Layerwise transition proxy per metric.

Important columns:

```text
source_file
formalism_class
metric
context
layer_count
row_count_sum
layer_slope
max_adjacent_jump
transition_from_layer
transition_to_layer
peak_layer
peak_mean
trough_layer
trough_mean
range_peak_minus_trough
```

Interpretation:

```text
max_adjacent_jump       -> largest adjacent-layer change
transition_from/to_layer -> candidate transition location
peak_layer              -> layer with maximum mean metric
trough_layer            -> layer with minimum mean metric
layer_slope             -> global layerwise trend
```

This is a proxy, not proof of a mechanistic phase transition.

---

### `processing_audit.csv`

Audit table for every CSV file discovered.

Important columns:

```text
source_file
status
rows
metric_columns
used_columns
seconds
error
```

Statuses:

```text
ok
skipped_no_metric_columns
error
```

Always inspect this file. It is the first sanity check.

---

## 11. Metric detection logic

The script reads the CSV header and selects:

1. metric columns;
2. grouping columns;
3. independent variables such as `alpha` and `alpha_abs`.

### Excluded index/id columns

These are not treated as metrics:

```text
artifact_type
question_index
step
layer
token_id
unit_index
rank_by_abs_delta
activation_size
layer_count_intervened
q_count
random_index
topk
is_embedding
is_middle_layer
```

Note: `layer` and `step` can still be used as grouping variables. They are just not treated as dependent metrics.

### Group columns

```text
condition
reference_condition
base_condition
intervention_name
layer_band
sign_name
module
unit_type
layer
step
alpha
alpha_abs
```

### Recognized exact metric columns

```text
projection_fraction_on_vector_x_loo
direction_cosine_with_vector_x_loo
projection_fraction_on_arch_vector_x_loo
direction_cosine_with_arch_vector_x_loo
cosine_distance_to_reference
l2_distance_to_reference_prompt_endpoint
l2_distance_to_reference
state_norm
selected_logprob
mean_selected_logprob
entropy
mean_entropy
reference_value
condition_value
delta
abs_delta
mean_delta
mean_abs_delta
max_abs_delta
top_rank_mean
generated_token_count
raw_has_think_tag
visible_response_empty_after_think_strip
refusal_marker_count
caution_marker_count
substitution_marker_count
refusal_binary
caution_binary
substitution_binary
nonempty_visible_response
instruction_deviation_proxy
```

### Regex-recognized metric names

Columns containing patterns like these are also recognized:

```text
cosine
projection_fraction
l2_distance
entropy
logprob
state_norm
mean_abs_delta
max_abs_delta
abs_delta
mean_delta
delta
refusal
caution
substitution
deviation
nonempty
think_tag
generated_token
top_rank
```

---

## 12. Formalism classes

Metrics are assigned to broad evidence classes:

```text
geometrical_capture
l2_compression
entropy_shift
logprob_shift
activation_delta
behavioral_audit
other_metric
```

This classification is string-pattern based. It is useful for organizing outputs, not a substitute for scientific interpretation.

---

## 13. Minimum useful input files

For a meaningful Latent Discourse Regime analysis, include at least:

```text
causal_intervention_trajectory_metrics_raw.csv
generation_trajectory_metrics_raw.csv
layerwise_geometry_metrics_raw.csv
causal_intervention_middle_layer_summary.csv
generation_middle_layer_summary.csv
middle_layer_condition_summary.csv
paired_target_vs_control_tests.csv
layerwise_fdr_target_vs_control.csv
alpha_dose_response_summary.csv
causal_alpha_scaling_summary.csv
behavioral_validation_summary.csv
causal_intervention_behavior_summary.csv
numeric_integrity_check.csv
quarantine_index.csv
claim_ladder_final.csv
```

The current script version may skip some statistical/gate files if their columns do not match its metric-detection rules. That is a known limitation; see below.

---

## 14. Known limitations

### 14.1 Statistical/gate files may be skipped

The current version detects metrics mainly by names containing terms such as:

```text
cosine, projection, l2, entropy, logprob, delta, refusal, caution
```

Therefore files with columns such as:

```text
fdr_q_value
paired_cohen_d
alpha_slope
pass_specificity
specificity_lift
empirical_p_greater_equal_observed
claim_pass
```

may be marked as:

```text
skipped_no_metric_columns
```

This does not mean those files are unimportant. It means this version of the script did not recognize them as metric tables.

For final research-grade use, extend `EXACT_METRIC_COLUMNS` and/or `METRIC_REGEX` to include statistical gate columns.

Recommended additions:

```text
p_value
q_value
fdr_q_value
cohen_d
paired_cohen_d
alpha_slope
directional_alpha_slope
pass_dose_response
pass_specificity
specificity_lift
claim_pass
empirical_p_greater_equal_observed
```

### 14.2 Condition baseline selection is mechanical

The baseline priority is:

```text
neutral
neutral_length_matched_control
control
question_only
```

This is not a substitute for an explicit analysis plan. If your experiment defines a different baseline, update `BASELINE_CONDITION_PRIORITY`.

### 14.3 Layerwise transition is a proxy

`max_adjacent_jump` finds the largest adjacent-layer mean change. This is not proof of a dynamical phase transition. Treat it as a candidate detector.

### 14.4 OLS alpha slope is descriptive unless design supports causality

The script estimates:

```text
metric ~ alpha
```

and reports a beta-like slope. This is causal only if the intervention design and controls justify causal interpretation.

### 14.5 Not a raw activation processor

Again: this script does not process raw hidden states or logits. If your file is a raw tensor dump, this script is the wrong tool.

---

## 15. Validation checklist after every run

### 15.1 Check audit status

```python
import pandas as pd

out = "outputs/latent_attractor_analysis"
audit = pd.read_csv(f"{out}/processing_audit.csv")
print(audit["status"].value_counts())
```

### 15.2 Show files that did not process cleanly

```python
display(audit[audit["status"] != "ok"])
```

### 15.3 Confirm biggest files were processed

```python
display(audit.sort_values("rows", ascending=False).head(20))
```

### 15.4 Check final table is non-empty

```python
final = pd.read_csv(f"{out}/FINAL_LATENT_ATTRACTOR_METRICS.csv")
print(final.shape)
display(final.head(20))
```

### 15.5 Check alpha regressions

```python
alpha = pd.read_csv(f"{out}/causal_alpha_regression.csv")
print(alpha.shape)
display(alpha.sort_values("r2", ascending=False).head(20))
```

### 15.6 Check layerwise transitions

```python
layer = pd.read_csv(f"{out}/layerwise_phase_transition.csv")
print(layer.shape)
display(layer.sort_values("max_adjacent_jump", key=lambda s: s.abs(), ascending=False).head(20))
```

---

## 16. Interpretation guide

### Geometrical capture

Relevant metrics:

```text
direction_cosine_with_vector_x_loo
projection_fraction_on_vector_x_loo
cosine_distance_to_reference
```

Evidence pattern:

```text
higher target/intervention cosine or projection vs baseline
significant condition delta
consistent layerwise concentration
```

### L2 compression

Relevant metrics:

```text
l2_distance_to_reference_prompt_endpoint
l2_distance_to_reference
```

Evidence pattern:

```text
lower target/intervention L2 vs baseline
negative condition delta
stable across reasonable baselines
```

If L2 decreases against one baseline but increases against another, do not claim universal compression. That is baseline-dependent geometry, not a law of nature.

### Entropy shift

Relevant metrics:

```text
entropy
mean_entropy
```

Evidence pattern:

```text
higher entropy in target/intervention condition
positive condition delta
larger effects under alpha interventions or late-layer manipulations
```

### Alpha steering

Relevant outputs:

```text
causal_alpha_regression.csv
```

Evidence pattern:

```text
large slope_beta_like
non-trivial r2
consistent sign across relevant groupings
```

Check asymmetry: plus and minus interventions may not behave symmetrically.

### Layerwise transition

Relevant outputs:

```text
layerwise_phase_transition.csv
```

Evidence pattern:

```text
large max_adjacent_jump
interpretable transition_from_layer / transition_to_layer
peak layer consistent with architecture hypothesis
```

---

## 17. Recommended `.gitignore`

```gitignore
data/metrics_raw/*.zip
data/metrics_raw/*.csv
outputs/
_work/
*_cache/
*.parquet
__pycache__/
.ipynb_checkpoints/
```

Do not commit large metric dumps or generated outputs unless your repository is explicitly designed for data artifacts. Use DVC, S3, Hugging Face Datasets, Google Drive, or another external store.

---

## 18. Reproducible command examples

### Clean final run without Parquet cache

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/latent_attractor_analysis" \
  --strict
```

### Fast rerun with Parquet cache

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/latent_attractor_analysis" \
  --cache-parquet \
  --strict
```

### Debug first three files

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/debug_latent_attractor_analysis" \
  --limit-files 3 \
  --strict
```

### Write per-file debug outputs

```bash
python "scripts/latent_attractor_gpu_rapids_analysis.py" \
  --input "data/metrics_raw/red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip" \
  --output-dir "outputs/latent_attractor_analysis" \
  --write-per-file \
  --strict
```

---

## 19. Methodological statement for paper or appendix

Suggested wording:

```text
The analysis script consumes derived metric tables rather than raw hidden-state tensors. Primitive quantities such as cosine alignment, L2 distance to the reference endpoint, entropy, and intervention deltas are computed upstream. The present script performs finite-value aggregation, grouped metric summaries, condition-level effect estimation, alpha-response ordinary least squares regression, and layerwise transition-proxy extraction. The resulting tables are used as secondary evidence summaries rather than as direct raw-activation measurements.
```

---

## 20. Final warning

The script is useful, but it is not an oracle. Always check:

```text
processing_audit.csv
skipped_no_metric_columns
condition baseline choices
alpha regression groupings
layerwise transition candidates
claim ladder / FDR files outside this script
```

A clean CSV output is not the same thing as a proven scientific claim. It is just cleaner ammunition.
