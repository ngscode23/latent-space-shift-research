# Gemma3 Grade4 + SAE Academic Readout Package

This folder is a reviewer-facing copy of the main Gemma Grade4 / SAE readout
documents.

It is not the source location for the analysis scripts. It is a convenient
readout package placed under `experiments/` so an external reader can understand
the result next to the code and metric folders.

## Files

```text
context_induced_latent_state_shift_final_conclusion_en.md
  English translated/readable version of the final Gemma conclusion.

context_induced_latent_state_shift_final_conclusion_ru.md
  Russian source duplicate copied from research_synthesis.

grade4_geometry_to_sae_steering_unified_readout_en.md
  English translated/readable version of the Grade4 -> SAE steering bridge.

grade4_geometry_to_sae_steering_unified_readout_ru.md
  Russian source duplicate copied from research_synthesis.
```

## Source Documents

Canonical Russian originals remain in:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/context_induced_latent_state_shift_final_conclusion_ru.md
research_synthesis/geometry_coordinate_evidence_package/grade4_geometry_to_sae_steering_unified_readout_ru.md
```

## How To Read

For a fast technical review:

```text
1. context_induced_latent_state_shift_final_conclusion_en.md
2. grade4_geometry_to_sae_steering_unified_readout_en.md
3. START_HERE.md
4. experiments/grade4_axis_decomposition_gemma/RUNBOOK.md
5. experiments/steering/sae_gemma_qwen/RUNBOOK.md
```

Core claim boundary:

```text
The result is a temporary inference-time hidden-state / residual-stream shift,
not a permanent model-state or weight change.

Descriptive latent shift is strong. Content/order separation is strong.
Causal involvement is supported. Stable bidirectional x_order_orth behavioral
control is not proven.
```

