# Qwen3.5-9B Qwen-Scope workspace

This folder is a model-specific workspace for adapting the Grade 4 hidden-geometry and SAE steering pipeline to Qwen3.5-9B / Qwen-Scope.

Important boundaries:
- The Grade 4 script has been adapted to `Qwen/Qwen3.5-9B-Base`.
- The Grade 4 SAE block has been adapted to Qwen-Scope `.pt` dict checkpoints from `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`.
- Qwen-Scope covers transformer layers `0..31`; the Grade 4 hidden-state index uses `layer = block_layer + 1`.
- The steering scripts are still copied from the Gemma workspace and should be treated as templates until their model/SAE loading is adapted.
- Put the model-specific `sae_order_feature_contrast.csv` in `data/` after running the Qwen3.5-9B Grade 4/SAE contrast pipeline.
- Do not reuse Gemma `sae_order_feature_contrast.csv` for Qwen. Each model needs its own coordinate map.

Expected next checks:
- Run a small smoke pass first if runtime/disk is constrained: reduce `SAE_BLOCK_LAYERS` in the Grade 4 script to `[8, 12, 16, 20, 24, 31]`.
- For the full SAE pass, keep `SAE_BLOCK_LAYERS = list(range(32))`.
- Verify `sae_model_compatibility.csv` and `sae_reconstruction_quality.csv` before interpreting `sae_order_feature_contrast.csv`.
- Regenerate Grade 4 axes and `sae_order_feature_contrast.csv` for this model.
- Re-run analyzer and compare against Gemma results.
