# output.zip analysis report

## Processing audit
- Top-level outputs exist and are non-empty.
- `FINAL_LATENT_ATTRACTOR_METRICS.csv`: 53,848 rows.
- `global_metric_summary.csv`: 178 rows.
- `grouped_metric_summary.csv`: 109,416 rows.
- `condition_effects.csv`: 38,626 rows.
- `causal_alpha_regression.csv`: 13,336 rows.
- `layerwise_phase_transition.csv`: 1,708 rows.
- `processing_audit.csv`: 70 files audited; 36 ok; 34 skipped as `skipped_no_metric_columns`.
- Largest file processed: `causal_intervention_trajectory_metrics_raw.csv`, 4,014,080 rows.

## Main scientific read
- Geometry/specificity is the strongest part: target middle-layer projection mean ≈ 0.9347, direction cosine ≈ 0.6122, positive projection fraction ≈ 0.8947.
- Random same-norm null is beaten: empirical p ≈ 0.00775.
- Target-vs-control paired tests are significant for projection and direction cosine across all listed controls.
- L2 compression is not robust: target is farther than neutral-length control by ≈ +1867.6, but closer than target-sentence-shuffle by ≈ -1127.8. This depends on baseline.
- Entropy shift exists but is modest in direct generation: target ≈ 0.6818 vs neutral ≈ 0.5747. Causal interventions can push entropy higher, especially minus_x late.
- Alpha effects are asymmetric: minus_internal suppression passes dose response; plus_internal fails. This weakens bidirectional causal steering.
- Behavioral/mechanistic claims do not pass the dataset's own gates: claim ladder passes Level 1 Geometry and Level 2 Specificity, but fails Level 3 Causal symmetry, Level 4 Behavioral steering, Level 5 Replication, Level 6 Mechanistic localization.

## Script quality
The GPU analyzer processed the huge files, but it skipped important statistical/gate files such as:
- `paired_target_vs_control_tests.csv`
- `layerwise_fdr_target_vs_control.csv`
- `alpha_dose_response_summary.csv`
- `causal_alpha_scaling_summary.csv`
- `claim_ladder_final.csv`
- `geometry_specificity_summary.csv`

These files are present in the extracted data and contain important numeric evidence, but the script's metric-column detector did not classify them as metric files. This means the script is useful, but not yet a complete final analyzer.
