# Runbook: полный Grade 4 + SAE causal x_order прогон

## Цель прогона

Этот прогон проверяет не сам факт hidden-state shift. Он уже подтвержден descriptive-метриками.

Цель нового прогона:

```text
Проверить, является ли x_order_orth причинной component axis, которая двигает target-like generation trajectory,
и сравнить ее с x_content.
```

Если `x_order_orth` проходит causal test, claim усиливается:

```text
target text induces a measurable latent-state shift
```

до:

```text
the measured order/response-mode component is causally involved in the shifted generation trajectory.
```

## Текущий правильный профиль скрипта

Файл:

```text
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

Должно быть включено:

```python
RESULTS_DIR = Path("/content/drive/MyDrive/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder")
RUN_LABEL = "grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder"

GENERATION_ENABLED = True
RESEARCH_GRADE_METRICS_ENABLED = True
GRADE4_AXIS_DECOMPOSITION_ENABLED = True

SAE_FEATURE_ANALYSIS_ENABLED = True
SAE_PROMPT_FEATURE_ANALYSIS_ENABLED = True
SAE_GENERATION_FEATURE_ANALYSIS_ENABLED = True
SAE_BLOCK_LAYERS = [12, 18, 24, 30, 36, 41]

GRADE4_COMPONENT_CAUSAL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_AXES = ["x_order_orth", "x_content"]
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late"]
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES = [0.25, 0.50, 0.75]
GRADE4_COMPONENT_CAUSAL_BASE_CONDITIONS = ["neutral", "target"]
GRADE4_COMPONENT_CAUSAL_MAX_QUESTIONS = None
GRADE4_COMPONENT_CAUSAL_MAX_NEW_TOKENS = 128
GRADE4_COMPONENT_CAUSAL_SAVE_STEP_RAW = False
```

Должно остаться выключено для первого causal-component прогона:

```python
CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False
```

## Как запускать в Colab

1. Включить GPU runtime.
2. Смонтировать Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

3. Убедиться, что Hugging Face token доступен, если модель требует access:

```python
import os
os.environ["HF_TOKEN"] = "PASTE_TOKEN_HERE"
```

4. Вставить и запустить весь скрипт:

```text
grade4_axis_decomposition/red_team_hidden_geometry_grade4_axis_decomposition_clean_evidence.py
```

5. В конце скрипт должен напечатать:

```text
Results directory: /content/drive/MyDrive/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder
Grade 4 component geometry summary: .../grade4_axis_projection_geometry_summary.csv
Grade 4 component causal summary: .../grade4_axis_component_causal_projection_summary.csv
SAE order feature contrast: .../sae_order_feature_contrast.csv
```

6. После завершения можно заархивировать папку:

```python
import shutil
run_dir = "/content/drive/MyDrive/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder"
shutil.make_archive(run_dir, "zip", run_dir)
```

## Какие файлы должны появиться

Grade 4 geometry:

```text
grade4_axis_projection_geometry_summary.csv
grade4_axis_component_norm_summary.csv
geometry_decomposition_summary.csv
residual_stream_decomposition.csv
subspace_decomposition_summary.csv
```

Grade 4 causal:

```text
grade4_axis_component_causal_response_audit.csv
grade4_axis_component_causal_projection_raw.csv
grade4_axis_component_causal_projection_summary.csv
grade4_axis_component_causal_symmetry_summary.csv
grade4_axis_component_causal_alpha_scaling_summary.csv
grade4_axis_component_causal_rank_summary.csv
grade4_axis_decomposition_verdict.md
```

SAE:

```text
sae_model_compatibility.csv
sae_reconstruction_quality.csv
sae_prompt_feature_activation_summary.csv
sae_prompt_feature_delta_summary.csv
sae_top_changed_features.csv
sae_grade4_component_feature_summary.csv
sae_generation_feature_summary.csv
sae_generation_top_features.csv
sae_order_feature_contrast.csv
```

Claim ladder:

```text
claim_ladder_final.csv
red_team_hidden_geometry_verdict.md
```

## Как понять, что SAE реально сработал

Первый файл:

```text
sae_model_compatibility.csv
```

Нужно увидеть, что SAE layers загрузились без failure.

Второй файл:

```text
sae_reconstruction_quality.csv
```

Нормальный результат: reconstruction cosine примерно `0.98+`.

Третий файл:

```text
sae_order_feature_contrast.csv
```

Именно он показывает, есть ли sparse features, связанные с `x_order_orth`, и сохраняются ли они в generation.

## Как читать causal result

Главный файл:

```text
grade4_axis_component_causal_symmetry_summary.csv
```

Сильный результат:

```text
x_order_orth имеет стабильный positive plus_minus_projection_gap;
x_order_orth сильнее или чище x_content;
эффект виден в middle и/или late layer bands;
эффект растет с alpha;
neutral + x_order_orth двигается к target-like trajectory;
target - x_order_orth ослабляет target-like trajectory.
```

Второй файл:

```text
grade4_axis_component_causal_alpha_scaling_summary.csv
```

Сильный результат:

```text
signed_alpha_projection_slope положительный и устойчивый для x_order_orth.
```

Третий файл:

```text
grade4_axis_component_causal_rank_summary.csv
```

Сильный результат:

```text
x_order_orth ranks above x_content by causal gap.
```

## Результат фактического causal run

Фактический прогон:

```text
C:\Users\stasv\Downloads\hidden_geometry_runs_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder.zip
```

подтвердил, что component causal block сработал. В run появились
`grade4_axis_component_causal_*` таблицы, `grade4_axis_component_causal_response_audit.csv`
содержит 480 строк, analyzer прошел без ошибок:

```text
files_seen = 82
csv_files_processed = 58
npz_files_processed = 3
plots_written = 19
errors = 0
anomaly_flags = 0
```

Главные числа:

```text
x_order_orth mean causal gap = 1.793915
x_order_orth positive gap rate = 0.861111

x_content mean causal gap = 2.067048
x_content positive gap rate = 0.916667

pairwise x_order_orth beats x_content = 0.361111
mean order_minus_content_gap = -0.273133
```

Вывод: `x_order_orth` причинно активна, но не доминирует над `x_content` в
текущем raw-alpha setup. Критерий:

```text
x_order_orth ranks above x_content by causal gap
```

не прошел. В rank summary при `alpha_abs = 0.75` и `readout_layer_band =
middle` `x_content` занимает rank 1, а `x_order_orth` rank 2 для neutral/middle,
neutral/late, target/middle и target/late.

Главный confound: intervention использует raw component vector:

```text
tensor + alpha * vec
```

При одинаковом `alpha` фактическая сила perturbation отличается, потому что
нормы компонент разные:

```text
middle content_norm = 14422
middle order_orth_norm = 8025

late content_norm = 28844
late order_orth_norm = 14418
```

Поэтому этот run усиливает claim с descriptive separability до causal
involvement, но не до causal dominance.

## Analyzer после Colab

После скачивания zip или папки результата:

```powershell
python scripts/analysis_tools/latent_gpu_rapids_analysis/latent_attractor_gpu_rapids_analysis.py `
  --input "<path_to_causal_run_zip_or_dir>" `
  --output-dir "research_synthesis/gemma3_grade4_sae_academic_readout/causal_xorder_metric_lab" `
  --backend pandas `
  --force-extract
```

Главные analyzer outputs:

```text
FINAL_DERIVED_METRIC_EVIDENCE.csv
grade4_component_causal_matrix.csv
grade4_component_alpha_matrix.csv
grade4_component_rank_matrix.csv
anomaly_flags.csv
plots/grade4_component_causal_gap_heatmap.png
```

## Если Colab падает по памяти

Сначала уменьшать не смысл эксперимента, а объем SAE/generation:

```python
SAE_GENERATION_MAX_STEPS_PER_TRACE = 16
GRADE4_COMPONENT_CAUSAL_MAX_NEW_TOKENS = 96
```

Если все еще падает:

```python
SAE_BLOCK_LAYERS = [24, 30, 36, 41]
```

Не выключать `GRADE4_COMPONENT_CAUSAL_ENABLED`, иначе causal-вопрос не будет закрыт.

## Следующий шаг: norm-controlled component causal run

Следующий прогон не должен сразу включать behavioral steering. Сначала нужно
закрыть confound raw vector norm. Новый causal run должен сравнить `x_order_orth`
и `x_content` при одинаковой intervention energy.

Нужный режим:

```python
RESULTS_DIR = Path("/content/drive/MyDrive/hidden_geometry_runs/grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl")
RUN_LABEL = "grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl"

GRADE4_COMPONENT_CAUSAL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_AXES = ["x_order_orth", "x_content"]
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late"]
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES = [0.25, 0.50, 0.75]
GRADE4_COMPONENT_CAUSAL_BASE_CONDITIONS = ["neutral", "target"]

GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_MODE = "band_l2"
GRADE4_COMPONENT_CAUSAL_READOUT_USES_NORMED_AXIS = True
```

Интервенция должна использовать нормированные component vectors:

```text
x_order_orth_normed = x_order_orth / norm(x_order_orth over intervention band)
x_content_normed = x_content / norm(x_content over intervention band)
```

В новых causal CSV нужно проверить, что для одного и того же `alpha_abs`
колонка `mean_effective_intervention_l2` стала сопоставимой между
`x_order_orth` и `x_content`, а `mean_intervention_axis_band_norm` близка к 1.
Если это не так, norm-control не сработал.

Только после norm-controlled сравнения можно решать, переходить ли к
behavioral steering. Если `x_order_orth` при equal-energy intervention догоняет
или обгоняет `x_content`, тогда следующий прогон:

```python
BEHAVIORAL_CONTROL_AXIS_ENABLED = True
BEHAVIORAL_CONTROL_ALPHA_VALUES = [0.25, 0.50, 0.75]
BEHAVIORAL_CONTROL_LAYER_BANDS = ["middle", "late"]
BEHAVIORAL_CONTROL_PRIMARY_LAYER_BAND = "middle"
BEHAVIORAL_CONTROL_RANDOM_BASELINES = 24
BEHAVIORAL_CONTROL_MAX_TEST_QUESTIONS = None
```

Если после norm-control `x_content` все еще сильнее, честный вывод:

```text
descriptive x_order_orth separation and causal activity are supported, but
dominant causal control is more content-family than order/response-mode.
```

## Результат фактического normctl run

Source zip:

```text
C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41 (1).zip
```

Этот прогон измерял не старый descriptive latent shift. Он измерял более узкий
causal-вопрос:

```text
если x_order_orth и x_content привести к одинаковой L2-норме по intervention
band, какая component direction сильнее двигает generation trajectory при
plus/minus intervention?
```

То есть run проверял `x_order_orth` против `x_content` как равномощные causal
directions, а не как raw vectors разной длины.

Конфиг в zip:

```text
run_label = grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl
model_id = google/gemma-3-12b-it

GRADE4_COMPONENT_CAUSAL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_AXES = ["x_order_orth", "x_content"]
GRADE4_COMPONENT_CAUSAL_LAYER_BANDS = ["middle", "late"]
GRADE4_COMPONENT_CAUSAL_ALPHA_VALUES = [0.25, 0.50, 0.75]
GRADE4_COMPONENT_CAUSAL_BASE_CONDITIONS = ["neutral", "target"]

GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_MODE = "band_l2"
GRADE4_COMPONENT_CAUSAL_READOUT_USES_NORMED_AXIS = True

CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False
```

Sanity check norm-control прошёл:

```text
mean_intervention_axis_band_norm = 1.0
mean_effective_intervention_l2 = alpha_abs
```

Это значит, что старый raw-norm confound был реально закрыт: `x_content` больше
не получал большую intervention energy только потому, что его raw vector norm
больше.

Raw norms до нормализации:

```text
middle x_content raw norm    = 14518.902068
middle x_order_orth raw norm = 8058.432071

late x_content raw norm      = 29315.891582
late x_order_orth raw norm   = 14729.571563
```

Основные causal числа по `plus_minus_projection_gap`:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate   = 0.527778

x_content mean causal gap    = -125.128343
x_content positive rate      = 0.472222
```

Pairwise all readouts:

```text
cells = 36
x_order_orth beats x_content = 0.416667
mean order_minus_content_gap = +59.186823
median order_minus_content   = -128.777290
```

Matching readout only (`middle -> middle`, `late -> late`):

```text
cells = 12
x_order_orth beats x_content = 0.500000
mean order_minus_content_gap = -0.191014
median order_minus_content   = +89.468686
```

Cosine gap did not rescue a dominance claim:

```text
x_order_orth positive cosine-gap rate = 0.583333
x_content positive cosine-gap rate    = 0.416667

all readouts pairwise x_order_orth beats x_content      = 0.416667
matching readouts pairwise x_order_orth beats x_content = 0.500000
```

The cosine gaps themselves are tiny:

```text
x_order_orth mean cosine gap = -0.000101
x_content mean cosine gap    = -0.000054
```

Important asymmetry by base condition:

```text
neutral:
  x_order_orth beats x_content = 0.666667
  mean order_minus_content_gap = +354.870122

target:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -236.496475
```

Interpretation:

```text
x_order_orth has directional signal for neutral injection, but target ablation
does not show stable reverse symmetry. Therefore bidirectional causal symmetry
is not closed.
```

Alpha scaling is weak:

```text
x_order_orth positive slope rate = 0.250000
x_content positive slope rate    = 0.416667
```

Correct verdict:

```text
The normctl run validates that raw-norm advantage was removed, but it does not
show stable causal dominance of x_order_orth over x_content. x_order_orth has
some causal directionality, especially for neutral +x_order_orth injection, but
the effect is not bidirectionally symmetric and does not scale cleanly with
alpha.
```

Important limitation of this normctl design:

```text
The intervention was unit-L2 over the selected band. This is fair for comparing
directions, but it is very small compared with natural component norms in the
actual Grade 4 latent shift. Therefore this run may be underpowered as a test of
natural-scale causal control.
```

Next experiment:

```text
norm-controlled natural-scale causal run
```

Use equal-energy directions, but rescale both axes to a shared natural band
norm:

```text
unit_x_order_orth = x_order_orth / norm(x_order_orth over band)
unit_x_content    = x_content / norm(x_content over band)

shared_band_norm = min(norm(x_order_orth over band), norm(x_content over band))

intervention = alpha * shared_band_norm * unit_axis
```

This keeps the comparison fair while making the perturbation comparable to the
natural latent shift scale.
