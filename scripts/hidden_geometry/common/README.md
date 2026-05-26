# Hidden Geometry Multi-Model Preflight

This folder contains compatibility utilities for running the Grade 3 / Grade 4
hidden-geometry protocol across multiple model families without changing the
scientific experiment.

## Files

- `model_registry.py` records model-specific runtime profiles.
- `model_compat.py` contains shared adapter/probe helpers.
- `preflight_probe.py` checks model/tokenizer/prompt/layer compatibility before
  an expensive full run.
- `analyze_result_package.py` reads a finished result zip/folder and writes an
  external audit bundle without modifying the source package.

The canonical experiment scripts remain:

- `scripts/hidden_geometry/grade3/red_team_hidden_geometry_grade3_clean_evidence.py`
- `grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py`

## Config-Only Probe

This checks Hugging Face config, tokenizer, chat template, and prompt budget. It
does not download the full model weights:

```powershell
python scripts/hidden_geometry/common/preflight_probe.py `
  --profile gemma3_12b_it `
  --target-file target.txt `
  --neutral-file neutral.txt `
  --questions-file questions.txt `
  --results-dir hidden_geometry_preflight_results/gemma3_12b_it_config
```

## Full Model Probe

This downloads/loads the model and verifies decoder layers, module hooks, and a
short hidden-state forward pass:

```powershell
python scripts/hidden_geometry/common/preflight_probe.py `
  --profile gemma3_12b_it `
  --load-model `
  --target-file target.txt `
  --neutral-file neutral.txt `
  --questions-file questions.txt `
  --results-dir hidden_geometry_preflight_results/gemma3_12b_it_full
```

## Artifacts

The probe writes:

- `model_compatibility_manifest.json`
- `prompt_budget_probe.csv`
- `decoder_layer_probe.csv`
- `module_hook_probe.csv`
- `preflight_status.csv`

Full Grade 3 / Grade 4 runs should be treated as clean only when:

- `final_preflight_status` is `pass`;
- `decoder_layer_probe.csv` has `status=ok`;
- `count` equals `expected_count`;
- required module hooks fire;
- prompt budget rows pass for target, neutral, and question-only conditions.

## Result Package Analyzer

This reads a completed Grade 3 / Grade 4 output package and writes a separate
analysis folder. It does not edit, unpack in place, or normalize the source
result package:

```powershell
python scripts/hidden_geometry/common/analyze_result_package.py `
  --results C:\path\to\red_team_hidden_geometry_results_grade3_gemma3_12b_it.zip `
  --out metrics\gemma3_12b_it_gate3_analysis `
  --run-label gemma3_12b_it_gate3
```

Analyzer artifacts:

- `analysis_summary.md`
- `analysis_summary.json`
- `scoreboard_row.csv`
- `source_file_inventory.csv`
- `peak_tables/geometry_peaks.csv`
- `peak_tables/specificity_peaks.csv`
- `peak_tables/component_peaks.csv`
- `peak_tables/causal_peaks.csv`
- `peak_tables/behavior_peaks.csv`
- `peak_tables/architecture_peaks.csv`
- `peak_tables/anomaly_flags.csv`

Machine outputs use pass/fail/status/failure_code fields. Human interpretation
is kept in `analysis_summary.md` and must cite source artifacts.
