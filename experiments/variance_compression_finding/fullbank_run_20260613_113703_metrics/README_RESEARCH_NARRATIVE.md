# Fullbank Research Narrative Index

Run id: `run_20260613_113703`

Models:

- Base: `google/gemma-3-12b-pt`
- Instruct: `google/gemma-3-12b-it`

Coverage:

- Target contexts: `10`
- Control contexts: `10`
- Questions: `10`
- Prompt conditions: `target`, `target_word_shuffle`, `target_sentence_shuffle`, `control`, `question_only`
- Total prompts per model: `410`
- Hidden shape: `(410, 49, 3840)`
- Main interpretation band: `L30-L47`

Narrative files:

- `RESEARCH_NARRATIVE_RU.md` - Russian final project narrative.
- `RESEARCH_NARRATIVE_EN.md` - English final project narrative for AI safety / mech interp readers.

Metric files used for the claims:

- `metadata.json`
- `late_band_summary.csv`
- `context_snapping_summary.csv`
- `readout_stiffness_summary.csv`
- `logit_metrics_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_contrast_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_condition_summary.csv`
- `hidden_npz_deep_dive/deep_late_band_base_instruct_alignment_summary.csv`

Main scientific claim:

Dense context can induce a measurable pre-output latent-state shift in an LLM. In this Gemma-3-12B fullbank audit, target/control contexts separate in late hidden-state space before generation, and this separation is stronger in the instruct model than in the base model. Instruction tuning does not merely collapse hidden-state geometry; it reduces absolute hidden-state scale while preserving or increasing angular/rank structure. The strongest alignment-like effect appears in the hidden-to-logit readout: the instruct model converts hidden states into a sharper, lower-entropy next-token probability distribution.

Short Russian version:

Плотный контекст способен переводить модель в измеримо другой внутренний режим еще до ответа. У Gemma-3-12B target/control контексты разделяются в late hidden states, и у instruct это разделение сильнее, чем у base. При этом instruction/alignment tuning не просто "сжимает hidden space": он уменьшает абсолютный масштаб состояний, но сохраняет или усиливает угловую/ранговую структуру. Главный эффект выравнивания виден на readout-этапе: сложное hidden-state состояние превращается в более жесткое, низкоэнтропийное распределение следующего токена.
