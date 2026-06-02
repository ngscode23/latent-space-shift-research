# Gemma3-12B-IT Grade 4 SAE Academic Readout

This folder is a compact evidence package for the Gemma3-12B-IT Grade 4 SAE run.

Source zip:

```text
C:\Users\stasv\Downloads\all_hidden_geometry_runs_sae_analyze.zip
```

Primary conclusion file:

```text
academic_conclusion_ru.md
```

Current causal run readout:

```text
causal_xorder_grade4_sae_runbook_ru.md
normctl_causal_conclusion_ru_en.md
context_induced_latent_state_shift_final_conclusion_ru.md
```

Copied metrics:

```text
metrics/
metrics_inventory.csv
```

The folder contains all `.csv`, `.json`, and `.npz` metric-like files from the
source run and metric-lab folders, with one intentional exception:
`FINAL_LATENT_ATTRACTOR_METRICS.csv` was not duplicated because it is the
backward-compatible alias of the already copied
`metric_lab__FINAL_DERIVED_METRIC_EVIDENCE.csv`.

Read first:

```text
metrics/run__grade4_axis_projection_geometry_summary.csv
metrics/run__grade4_axis_component_norm_summary.csv
metrics/run__sae_model_compatibility.csv
metrics/run__sae_reconstruction_quality.csv
metrics/run__sae_order_feature_contrast.csv
metrics/metric_lab__analysis_manifest.json
metrics/metric_lab__anomaly_flags.csv
```

Core result:

```text
Gemma3-12B-IT separates sentence-shuffled content from coherent target order.
x_content reads sentence-shuffle strongly, while x_order_orth reads coherent
target strongly. SAE Lens adds sparse-feature coordinates for this separation.
```

Boundary:

```text
This package now includes a component-causal test. The test supports causal
activity of x_order_orth, but it does not establish x_order_orth as dominant
over x_content in the raw-alpha setup.

A later unit-L2 norm-controlled component-causal run removes the raw-norm
confound, but still does not establish stable bidirectional causal dominance of
x_order_orth. It shows some directional signal, strongest for neutral
+x_order_orth injection, but alpha scaling and target-ablation symmetry are not
stable.

The next required run is norm-controlled natural-scale component causality
before behavioral steering.

The final Russian synthesis for the current evidence state is:
`context_induced_latent_state_shift_final_conclusion_ru.md`.
```
