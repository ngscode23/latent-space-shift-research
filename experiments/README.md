# Experiments Workspace

This folder contains the active experimental code and local run artifacts.

Start from the root-level map:

```text
START_HERE.md
```

## Main Code Families

```text
experiments/gemma3_grade4_sae_academic_readout/
```

Reviewer-facing English/Russian readout package for the main Gemma Grade4 + SAE
conclusions. This is documentation, not source code.

```text
experiments/grade4_axis_decomposition_gemma/
```

Produces hidden-state geometry evidence for Gemma: prompt/generation hidden
states, target/control/shuffle comparisons, Grade4 component axes, SAE readout
tables, and Grade4 ZIP packages.

Key script:

```text
experiments/grade4_axis_decomposition_gemma/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Key docs:

```text
experiments/grade4_axis_decomposition_gemma/RUNBOOK.md
experiments/grade4_axis_decomposition_gemma/metrics/README.md
```

```text
experiments/steering/
```

Consumes Grade4 outputs and runs SAE / dense-axis steering. This includes SAE
feature candidate selection, scale calibration, SAE decoder-direction steering,
teacher-forced KL, and dense `x_order_orth` axis steering.

Current steering workspace:

```text
experiments/steering/sae_gemma_qwen/
```

Key docs:

```text
experiments/steering/sae_gemma_qwen/README_RU_EN.md
experiments/steering/sae_gemma_qwen/RUNBOOK.md
experiments/steering/sae_gemma_qwen/script_docs/README.md
experiments/steering/metrics/README.md
```

## Pipeline

```text
Grade4 hidden geometry
-> post-hoc metric lab analysis
-> SAE candidate evidence
-> SAE scale calibration
-> SAE decoder-direction steering / KL
-> optional dense x_order_orth axis steering
```

The post-hoc metric analyzer lives outside this folder:

```text
scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py
```

## Compatibility Pointers

The old root-level folders:

```text
grade4_axis_decomposition/
steering/
```

are pointer folders only. Do not add new active scripts there. The real active
content is under `experiments/`.

## Do Not Move Again

For readability, prefer adding README/RUNBOOK/index files instead of physically
moving major folders again.
