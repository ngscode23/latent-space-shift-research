# SAE / Gemma / Qwen Steering Workspace

This folder is the sorted local workspace for SAE-based Gemma steering, Qwen
reference steering, and the bridge from Grade 4 hidden-state geometry to SAE
features.

Русская версия находится ниже после English section.

## English

### Purpose

This folder keeps the SAE / Gemma / Qwen steering materials together without
mixing them with the COAST / conceptor branch. COAST was archived separately in:

```text
archive/coast_posneg_conceptor_branch/
```

The main active line is:

```text
Grade 4 hidden-state geometry -> SAE feature readout -> scale calibration ->
SAE decoder-direction steering / KL metrics -> optional x_order_orth axis steering
```

### Per-Script Documentation

Each Python script has a dedicated English documentation card here:

```text
experiments/steering/sae_gemma_qwen/script_docs/
```

Start with:

```text
experiments/steering/sae_gemma_qwen/script_docs/README.md
```

Those docs explain script status, purpose, inputs, outputs, typical Colab
usage, and whether the script is active, legacy, or Qwen reference.

### Folder Layout

```text
gemma_active/
  Active Gemma SAE and axis scripts.

gemma_active/fast/
  Cached/fast variant of the full SAE steering script.

gemma_legacy/
  Older exploratory Gemma steering scripts. Kept for history and comparison.

gemma_docs/
  Existing script notes and run notes for Gemma SAE steering.

gemma_runs/
  Local analyzed outputs from Gemma SAE candidate discovery runs.

qwen_reference/
  Snapshot of the Qwen steering scripts and Qwen conclusion note. The original
  Qwen workspace remains in model_workspaces/qwen3_5_9b_qwen_scope/.

research_links/
  Links and notes pointing to synthesis documents in research_synthesis/.
```

### Active Gemma Scripts

#### `gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py`

Full Gemma SAE evidence builder and rough patching script.

What it does:

- reads the full Grade 4 Gemma SAE tables, preferably from the raw run ZIP;
- uses `sae_order_feature_contrast.csv` as the primary ranking table;
- adds reconstruction quality, component summary, prompt-delta, generation,
  top-feature, and changed-feature evidence;
- ranks candidate SAE features associated with `x_order_orth` / order-related
  readout;
- optionally runs rough zero-ablation and top activating context inspection.

Use when:

- selecting SAE candidate features for later scale calibration and steering;
- building a richer feature evidence table from the Grade 4 run.

Important inputs:

```text
SAE_TABLE_ZIP_PATH
PATCH_PROMPTS / prompts_target
RUN_MEDIATION_PATCHING
RUN_TOP_CONTEXT_INSPECTION
```

#### `gemma_active/02_scale_calibration.py`

Scale calibration script for SAE decoder-direction steering.

What it does:

- estimates residual-stream norms for selected layers/features;
- compares old tiny scales against native SAE activations;
- proposes scale values as fractions of residual norm;
- optionally runs next-token KL checks.

Use when:

- you have selected SAE features and need reasonable intervention scales before
  running generation steering.

Output role:

- gives `STEERING_SCALES` for `sae_steering_with_kl_full.py`.

#### `gemma_active/sae_steering_with_kl_full.py`

Main full Gemma SAE decoder-direction steering runner.

What it does:

- applies `scale * sae.W_dec[feature]` to residual stream hooks;
- runs generation over configured tasks/features/scales;
- records output metrics and baseline comparisons;
- computes final next-token KL/logit metrics;
- optionally computes teacher-forced per-token KL against baseline
  continuations.

Use when:

- testing whether selected SAE features visibly or distributionally modulate
  model behavior under a target base context.

Expected globals:

```text
model
saes
prompts_target
```

#### `gemma_active/fast/sae_steering_with_kl_full_fast.py`

Cached/fast variant of the full SAE steering runner.

What it changes:

- caches base logits and teacher-forced base forwards;
- writes CSV checkpoints less often;
- avoids storing full base text in every CSV row by default;
- keeps the same conceptual intervention logic as the full script.

Use when:

- running larger SAE steering sweeps and you want less CSV overhead.

Important boundary:

- each generation call is still independent; the model does not remember
  previous questions/scales. The speed bottleneck remains autoregressive
  generation.

#### `gemma_active/x_order_orth_axis_steering_with_kl_full.py`

Grade 4 component-axis steering script.

What it does:

- loads `grade4_axis_component_vectors_by_layer.npz` from a Grade 4 artifact;
- extracts `x_order_orth` or another configured component axis;
- adds that dense residual-stream axis to TransformerLens residual hooks;
- measures generation and KL metrics.

Use when:

- testing the discovered dense Grade 4 axis directly, instead of steering
  through individual SAE decoder directions.

Important distinction:

```text
This is not SAE feature steering. It steers along a dense Grade 4 residual-stream axis.
```

#### `gemma_active/gemma_revision_audit.py`

Gemma revision/audit utility.

What it does:

- helps audit whether the currently loaded/run Gemma setup appears consistent
  with previous assumptions and run metadata.

Use when:

- you suspect model/config/tokenizer/runtime drift and want a lightweight audit
  script.

### Legacy Gemma Scripts

These are preserved because they document how the steering line developed, but
they are not the preferred current pipeline.

#### `gemma_legacy/01_candidate_discovery_and_rough_sae_patching.py`

Original candidate discovery and rough SAE patching script.

Status:

- legacy predecessor of `01b`;
- reads mainly `sae_order_feature_contrast.csv`;
- useful for historical comparison, but `01b` is safer and fuller.

#### `gemma_legacy/sae_feature_steering_light.py`

Small early generation steering test.

Status:

- legacy / quick probe;
- uses a few late Gemma features and small scales;
- not enough for full KL / teacher-forced analysis.

#### `gemma_legacy/sae_feature_steering_v2_no_control.py`

Intermediate SAE steering and diagnostics script.

Status:

- legacy;
- includes generation, next-token KL, unembed projection, positional profile,
  and optional short-prompt ablation;
- replaced for main runs by `sae_steering_with_kl_full.py`.

#### `gemma_legacy/steering_gemma3_V1.py`

Early Gemma steering generation runner.

Status:

- legacy;
- useful for old result interpretation and comparison;
- not the current full metric runner.

### Qwen Reference

`qwen_reference/` contains snapshots copied from:

```text
model_workspaces/qwen3_5_9b_qwen_scope/
```

The original Qwen workspace remains there. The copies here are for nearby
reference only.

Important Qwen role:

```text
Qwen3.5-9B Base replicated the hidden-state / x_order_orth readout in a more
content-heavy form. It supports cross-model existence of the effect, but does
not show x_order_orth as causally dominant over x_content.
```

#### `qwen_reference/scripts_snapshot/01_candidate_discovery_and_rough_sae_patching.py`

Qwen candidate discovery / rough patching script adapted from the Gemma line.
Use it for inspecting Qwen SAE candidate features from Qwen metric tables and
for comparing Qwen candidate logic against Gemma candidate logic. This is a
reference snapshot, not the main Gemma runner.

#### `qwen_reference/scripts_snapshot/qwen35_9b_sae_mediation_top_k.py`

Main Qwen-Scope SAE mediation script.

What it does:

- loads `Qwen/Qwen3.5-9B-Base`;
- loads Qwen-Scope SAE checkpoints from
  `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`;
- uses raw PyTorch `layer*.sae.pt` files, not Gemma-Scope and not the
  `sae_lens` release interface;
- runs targeted SAE mediation / feature patching for top order-related
  candidates;
- can export top activating contexts.

Important boundary:

```text
This is Qwen-specific. Do not run it as a Gemma script.
```

#### `qwen_reference/scripts_snapshot/sae_steering_with_kl_full.py`

Qwen snapshot of the full SAE steering + KL runner. It is useful for comparing
Qwen/Gemma steering logic, but it is not the preferred current Gemma runner.

#### `qwen_reference/scripts_snapshot/sae_steering_with_kl_ful_v2l.py`

Qwen self-contained SAE steering script for Qwen-Scope TopK SAE.

What it does:

- loads Qwen model and tokenizer via `transformers`;
- loads Qwen-Scope SAE feature directions;
- supports chat-template thinking controls;
- applies decoder-direction steering and records generation/KL metrics.

Status:

```text
Qwen-specific experimental steering runner.
```

#### Other Qwen snapshot scripts

```text
02_scale_calibration.py
sae_feature_steering_light.py
sae_feature_steering_v2_no_control.py
steering_gemma3_V1.py
```

These are copied for comparison with the Gemma script family. Treat them as
historical/reference files unless actively working inside the Qwen workspace.

### Related Research Synthesis Documents

Do not move these into `steering/`; they are interpretation/reporting files:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/context_induced_latent_state_shift_final_conclusion_ru.md
research_synthesis/gemma3_grade4_sae_academic_readout/sae_decoder_steering_base_target_control_old_run_readout_ru.md
research_synthesis/geometry_coordinate_evidence_package/grade4_geometry_to_sae_steering_unified_readout_ru.md
research_synthesis/geometry_coordinate_evidence_package/sae_feature_steering_lab_readout_ru.md
model_workspaces/qwen3_5_9b_qwen_scope/context_induced_latent_state_shift_qwen3_5_9b_qwen_scope_final_conclusion_ru.md
```

### Recommended Current Pipeline

1. Candidate evidence:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

2. Scale calibration:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/02_scale_calibration.py
```

3. Full SAE steering:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
```

4. Fast/cached full SAE steering:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

5. Dense Grade 4 axis steering:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

If files are uploaded into Colab root, using `%run -i filename.py` is still
fine. This local folder layout is for project organization and auditability.

---

## Русский

### Назначение

Эта папка собирает в одном месте SAE / Gemma / Qwen steering материалы и не
смешивает их с COAST / conceptor веткой. COAST вынесен отдельно:

```text
archive/coast_posneg_conceptor_branch/
```

Главная активная линия:

```text
Grade 4 hidden-state geometry -> SAE feature readout -> scale calibration ->
SAE decoder-direction steering / KL metrics -> optional x_order_orth axis steering
```

### Документация по каждому скрипту

Для каждого Python-скрипта сделана отдельная английская карточка:

```text
experiments/steering/sae_gemma_qwen/script_docs/
```

Начинать отсюда:

```text
experiments/steering/sae_gemma_qwen/script_docs/README.md
```

Там написано: статус скрипта, зачем он нужен, какие входы ожидает, какие
файлы пишет, как запускать в Colab, и является ли он active, legacy или Qwen
reference.

### Структура папок

```text
gemma_active/
  Активные Gemma SAE и axis скрипты.

gemma_active/fast/
  Ускоренная/кэшированная версия полного SAE steering скрипта.

gemma_legacy/
  Старые экспериментальные Gemma steering скрипты. Сохранены для истории.

gemma_docs/
  Старые описания скриптов и заметки по Gemma SAE steering.

gemma_runs/
  Локальные анализы и output-таблицы от Gemma SAE candidate discovery.

qwen_reference/
  Снимок Qwen steering скриптов и Qwen заключения. Оригинальная Qwen папка
  остается в model_workspaces/qwen3_5_9b_qwen_scope/.

research_links/
  Ссылки на интерпретационные документы в research_synthesis/.
```

### Активные Gemma скрипты

#### `gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py`

Полный скрипт для отбора SAE-кандидатов и грубого patching.

Что делает:

- читает полные Gemma Grade 4 SAE таблицы, лучше из raw run ZIP;
- использует `sae_order_feature_contrast.csv` как главную таблицу ранжирования;
- добавляет reconstruction quality, component summary, prompt delta,
  generation summary, top features, changed features;
- ранжирует SAE-фичи, связанные с `x_order_orth` / order-related readout;
- опционально запускает rough zero-ablation и top activating contexts.

Когда использовать:

- когда выбираем SAE features для scale calibration и steering;
- когда нужно получить полный evidence table по фичам.

#### `gemma_active/02_scale_calibration.py`

Калибровка scale для SAE decoder-direction steering.

Что делает:

- оценивает нормы residual stream на нужных слоях;
- сравнивает старые маленькие scale с натуральными SAE activation;
- предлагает scale как долю нормы residual;
- опционально проверяет next-token KL.

Когда использовать:

- после выбора фич, перед generation steering.

#### `gemma_active/sae_steering_with_kl_full.py`

Главный полный Gemma SAE decoder-direction steering runner.

Что делает:

- добавляет `scale * sae.W_dec[feature]` в residual hook;
- запускает generation по tasks/features/scales;
- считает output metrics, baseline comparison;
- считает final next-token KL/logit metrics;
- опционально считает teacher-forced per-token KL.

Когда использовать:

- для основной проверки, двигают ли выбранные SAE-фичи поведение/распределение.

#### `gemma_active/fast/sae_steering_with_kl_full_fast.py`

Более быстрая/кэшированная версия полного steering runner.

Что меняет:

- кэширует base logits и teacher-forced base forwards;
- реже пишет CSV;
- по умолчанию не дублирует полный base text в каждой CSV строке;
- сохраняет ту же основную логику вмешательства.

Граница:

- каждая генерация остается независимой; модель не помнит предыдущие вопросы
  и scale. Основной bottleneck все равно autoregressive generation.

#### `gemma_active/x_order_orth_axis_steering_with_kl_full.py`

Скрипт для steering по плотной Grade 4 компонентной оси.

Что делает:

- загружает `grade4_axis_component_vectors_by_layer.npz`;
- берет `x_order_orth` или другую axis;
- добавляет dense residual-stream axis в TransformerLens hook;
- считает generation и KL metrics.

Важно:

```text
Это не SAE feature steering. Это steering по плотной Grade 4 residual-stream axis.
```

#### `gemma_active/gemma_revision_audit.py`

Легкий audit скрипт для проверки Gemma setup/runtime drift.

Когда использовать:

- если есть подозрение, что модель/tokenizer/runtime изменились относительно
  старых прогонов.

### Legacy Gemma скрипты

#### `gemma_legacy/01_candidate_discovery_and_rough_sae_patching.py`

Старый predecessor для candidate discovery. Сейчас лучше использовать `01b`.

#### `gemma_legacy/sae_feature_steering_light.py`

Ранний маленький generation steering test. Не полный metric runner.

#### `gemma_legacy/sae_feature_steering_v2_no_control.py`

Средняя версия с generation, KL, unembed projection и positional profile.
Сейчас заменена основным `sae_steering_with_kl_full.py`.

#### `gemma_legacy/steering_gemma3_V1.py`

Первый ранний Gemma steering runner. Сохранен для истории и сравнения.

### Qwen reference

`qwen_reference/` содержит копии из:

```text
model_workspaces/qwen3_5_9b_qwen_scope/
```

Оригинальная Qwen workspace не тронута.

Роль Qwen:

```text
Qwen3.5-9B Base повторил hidden-state / x_order_orth readout, но в более
content-heavy форме. Это поддерживает cross-model существование эффекта, но не
доказывает causal dominance x_order_orth над x_content.
```

#### `qwen_reference/scripts_snapshot/01_candidate_discovery_and_rough_sae_patching.py`

Qwen версия candidate discovery / rough patching. Используется для просмотра
Qwen SAE candidates и сравнения Qwen/Gemma candidate logic. Это reference
snapshot, не основной Gemma runner.

#### `qwen_reference/scripts_snapshot/qwen35_9b_sae_mediation_top_k.py`

Главный Qwen-Scope SAE mediation script.

Что делает:

- загружает `Qwen/Qwen3.5-9B-Base`;
- загружает Qwen-Scope SAE checkpoints из
  `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`;
- использует raw PyTorch `layer*.sae.pt`, а не Gemma-Scope и не обычный
  `sae_lens` release interface;
- запускает SAE mediation / feature patching для top order-related candidates;
- может выгружать top activating contexts.

Важно:

```text
Это Qwen-specific. Не запускать как Gemma скрипт.
```

#### `qwen_reference/scripts_snapshot/sae_steering_with_kl_full.py`

Qwen-снимок полного SAE steering + KL runner. Полезен для сравнения Qwen/Gemma
steering logic, но не является основным текущим Gemma runner.

#### `qwen_reference/scripts_snapshot/sae_steering_with_kl_ful_v2l.py`

Qwen self-contained SAE steering script под Qwen-Scope TopK SAE.

Что делает:

- загружает Qwen model/tokenizer через `transformers`;
- загружает Qwen-Scope SAE feature directions;
- поддерживает chat-template thinking controls;
- делает decoder-direction steering и пишет generation/KL metrics.

Статус:

```text
Qwen-specific experimental steering runner.
```

#### Другие Qwen snapshot scripts

```text
02_scale_calibration.py
sae_feature_steering_light.py
sae_feature_steering_v2_no_control.py
steering_gemma3_V1.py
```

Это копии для сравнения с Gemma script family. Использовать как
historical/reference, если специально не работаешь в Qwen workspace.

### Связанные research_synthesis документы

Их лучше не переносить в `steering/`, потому что это отчеты/интерпретация:

```text
research_synthesis/gemma3_grade4_sae_academic_readout/context_induced_latent_state_shift_final_conclusion_ru.md
research_synthesis/gemma3_grade4_sae_academic_readout/sae_decoder_steering_base_target_control_old_run_readout_ru.md
research_synthesis/geometry_coordinate_evidence_package/grade4_geometry_to_sae_steering_unified_readout_ru.md
research_synthesis/geometry_coordinate_evidence_package/sae_feature_steering_lab_readout_ru.md
model_workspaces/qwen3_5_9b_qwen_scope/context_induced_latent_state_shift_qwen3_5_9b_qwen_scope_final_conclusion_ru.md
```

### Текущий рекомендуемый pipeline

1. Candidate evidence:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/01b_full_sae_evidence_candidate_patching_gemma.py
```

2. Scale calibration:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/02_scale_calibration.py
```

3. Full SAE steering:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/sae_steering_with_kl_full.py
```

4. Fast full SAE steering:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/fast/sae_steering_with_kl_full_fast.py
```

5. Dense axis steering:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/x_order_orth_axis_steering_with_kl_full.py
```

Если в Colab ты загружаешь файлы в корень, можно продолжать запускать просто
`%run -i filename.py`. Эта локальная структура нужна для порядка и audit trail.

