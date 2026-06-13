# Fullbank Research Packet

This folder stores the fullbank Gemma-3-12B base-vs-instruct geometry/probability audit artifacts.

Main run:

`_20260613_113703_alignment_geometry_probability_run_fullbank_/content/alignment_geometry_probability_run_fullbank/run_20260613_113703`

Use these repository files first:

- `../../FULLBANK_RUN_20260613_113703_FINDINGS.md`
- `../../FULLBANK_RUN_20260613_113703_FINDINGS_RU.md`

The raw extracted Colab payload is intentionally not meant to be committed
under this folder. It contains large `.npz` files, generated `.csv` files, and
very deep Windows paths. Publish the raw artifact package separately through
Drive/Zenodo when needed.

Core metric files:

- `metadata.json` - run setup and prompt counts.
- `late_band_summary.csv` - L30-L47 base-vs-instruct hidden geometry plus logit readout summary.
- `context_snapping_summary.csv` - question-only vs context hidden/logit shift.
- `readout_stiffness_summary.csv` - probability concentration normalized against hidden dispersion.
- `logit_metrics_summary.csv` - next-token entropy/top-k probability metrics by condition.
- `hidden_npz_deep_dive/deep_late_band_contrast_summary.csv` - target/control separation in late hidden states.
- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv` - per-condition late hidden-state geometry.
- `hidden_npz_deep_dive/deep_late_band_base_instruct_alignment_summary.csv` - base/instruct representation alignment.

Short claim:

Dense context can induce a measurable pre-output latent-state shift. In this fullbank Gemma-3-12B run, target/control contexts separate in late hidden-state space before generation, and the separation is stronger in the instruct model. Instruction tuning does not merely collapse hidden geometry: it reduces absolute hidden-state scale while preserving or increasing angular/rank structure, and the strongest alignment-like effect appears in hidden-to-logit readout stiffness.
