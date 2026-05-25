# Experiment Map Current

Эта карта нужна, чтобы не смешивать два разных корпуса и разные уровни claim.

## Главный Фокус

Главный объект исследования не один конкретный набор текстов, а общий механизм:

```text
structured context -> hidden geometry shift -> semantic/action-policy shift
```

Более точная рамка:

```text
mechanistic interpretability / state-space analysis
```

Мы не доказываем "эти тексты особенные". Selfref и heldout тексты являются
controlled induction stimuli: они вызывают режим, а измеряется распределенное
латентное состояние модели и его последствия.

Главный объект исследования:

```text
context-induced latent regime formation
```

То есть временный, распределенный hidden-state режим, который проявляется в
геометрии представлений, downstream semantic/logit margins, persistence,
rejection residuals, order/dose structure и controlled fake-action choices.

Рабочая гипотеза:

```text
Target-контекст смещает скрытые представления модели.
Этот hidden shift проявляется в semantic/logit/action-policy readouts.
Часть эффекта сохраняется после neutral turns и частично переживает rejection/reset.
```

В проекте есть две основные stimulus families:

```text
1. original / selfref / mirror texts
2. heldout-domain procedural/risk texts
```

Они не находятся в отношении "главный корпус" и "ненужный вторичный корпус".
Они проверяют разные стороны одного механизма:

```text
selfref показывает специальный mirror/self-model pressure effect;
heldout показывает, что нереферентный procedural/risk discourse тоже вызывает shift.
```

Мы не заявляем:

```text
сознательное состояние;
мистический attractor;
полное объяснение поведения одним вектором;
универсальный механизм всех LLM.
```

## Два Корпуса

### 1. Original / Selfref / Mirror

Метки:

```text
force_finality
judgment_distribution
rlhf_reward
expert_verdict
default_caution
safety_overreach
paranoid_reading
intellectual_tool_vs_safety
compliance_rewrite
```

Роль:

```text
Специальный selfref/mirror вариант феномена. Это корпус, где модель читает текст о собственном режиме ответа.
```

Что он проверяет:

```text
Может ли self-reference + pressure + safety/alignment vocabulary +
accusation of mode failure + recognition demand сдвинуть hidden geometry и downstream readouts.
```

Статус:

```text
ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ
для узкой карты: selfref/mirror context вызывает измеримый shift.
```

### 2. Heldout Domain

Метки:

```text
clinical_triage_gate
legal_contract_gate
industrial_permit_gate
incident_change_freeze
finance_credit_exception
aviation_release_gate
lab_biosafety_gate
procurement_vendor_risk
privacy_export_gate
```

Роль:

```text
Самостоятельная heldout-линия: нереферентный procedural/risk discourse.
Это одновременно контроль против self-reference leakage и позитивная проверка,
что не только самореферентные тексты вызывают сдвиг.
```

Что он проверяет:

```text
Сохраняется ли похожий shift без прямой самореференции,
без слов "модель должна", "RLHF", "safety", "внутренний режим модели".
```

Что вышло:

```text
Да, heldout-domain targets тоже двигают semantic/action readouts.
Эффект слабее/иначе по масштабу, но повторяется на Qwen и Ministral.
```

Статус:

```text
СИЛЬНО ПОДДЕРЖАНО
как reviewer-facing доказательство, что procedural/risk context без прямой
самореференции тоже вызывает измеримый shift.
```

## Какие Папки Что Означают

```text
attractor_results_agent_loop_qwen3_14b
attractor_results_agent_loop_qwen3_14b2
attractor_results_agent_loop_qwen3_14b3
```

Это Qwen3-14B selfref/original прогоны. Метки начинаются с:

```text
force_finality, judgment_distribution, rlhf_reward
```

```text
attractor_results_agent_loop_qwen3_14b4_heldout
```

Это Qwen3-14B heldout-domain прогон. Метки начинаются с:

```text
clinical_triage_gate, legal_contract_gate, industrial_permit_gate
```

```text
attractor_results_agent_loop_ministral3_14b_heldout
```

Это Ministral-3-14B heldout-domain прогон. Он нужен как cross-model check для heldout.

```text
attractor_results_olmo2_13b_heldout
```

Это OLMo2-13B heldout-domain прогон. Он расширяет heldout cross-model line:
эффект слабее по semantic амплитуде, но сохраняет ту же структуру
geometry / blind probes / persistence / rejection residual / order-dose /
fake-action readout.

## Cross-Corpus Comparison

Скрипт:

```text
cross_corpus_comparison_v1.py
```

Выход:

```text
cross_corpus_comparison_v1/cross_corpus_comparison.md
cross_corpus_comparison_v1/metric_wide.csv
cross_corpus_comparison_v1/selfref_vs_heldout_ratios.csv
```

Что он проверяет:

```text
selfref / mirror corpus vs heldout procedural/risk corpus
```

Вывод:

```text
Оба corpus family вызывают hidden/readout/action shifts.
Значит проект не про "особые selfref тексты", а про более широкий
context-induced latent regime formation.
Heldout остается более чистой reviewer-facing линией.
Selfref показывает сильную mirror/self-model pressure line, но hard controls
показывают, что pressure/rhetorical topology является активным ингредиентом.
```

## Текущие Настройки Большого Скрипта

В `llm_attractor_colab_copy_paste.py` дефолт сейчас должен быть:

```python
TEXT_FAMILY_PRESET = "original"
RESULTS_DIR = Path("attractor_results_agent_loop_ministral3_14b_selfref")
PRIMARY_CONTROL_MODE = "content_matched"
FAST_CORE_DIAGNOSTICS_ONLY = True
```

Смысл:

```text
Если сейчас запускать большой скрипт, он пойдет по selfref/mirror корпусу,
потому что heldout на Qwen и Ministral уже прогнан. Это не значит, что heldout
вторичен по смыслу; это значит, что следующая практическая дырка сейчас -
Ministral selfref для симметрии.
```

## Что Считать Доверяемым

### Уже Доверяем Для Внутренней Карты

```text
1. Selfref/mirror texts вызывают hidden/readout shift.
2. Blind neutral probes показывают, что это не только старые слова DIRECT/VERDICT/etc.
3. Persistence/rejection показывают, что эффект не мгновенно исчезает.
4. Agent-loop показывает сдвиг в fake-agent action-choice readouts, не real tool behavior.
```

### Сильно Поддержано Как Heldout-Линия

```text
Heldout-domain targets тоже вызывают shift.
Это значит, что механизм шире, чем прямое обращение текста к модели как к модели.
Сдвиг может вызываться процедурно-рискованным дискурсом:
preconditions, risk gate, substitute route, procedure-before-action.
```

### Пока Не Заявлять

```text
1. Это переворачивает alignment.
2. Это универсальный механизм всех LLM.
3. Hidden shift полностью причинно объясняет поведение.
4. Это настоящий attractor в динамическом смысле.
5. Это real-agent behavior.
```

## Зачем Нужен Causal Mediation Script

`causal_mediation_v1_colab.py` не заменяет большой скрипт.

Он проверяет отдельный вопрос:

```text
Если взять hidden target-control vector и добавить/вычесть его,
двигаются ли semantic/action margins в ожидаемую сторону?
```

Смысл:

```text
Это causal-handle тест, не основной measurement package.
Его не надо запускать сейчас, пока не закрыта понятная матрица:
Qwen selfref / Qwen heldout / Ministral selfref / Ministral heldout.
```

Обновление после Qwen heldout mediation:

```text
Qwen/Qwen3-14B heldout mediation поддержал частичную причинную роль
target-control hidden vector, особенно для fake-agent action-policy margins.
Рабочий слой: hidden_index=32 / module_layer=31.
Final hidden_index=40 не является хорошей причинной ручкой в этом прогоне.
```

Артефакты:

```text
latent_shift_evidence_package_v1/causal_mediation/qwen3_14b_heldout/
```

Статус:

```text
СИЛЬНО ПОДДЕРЖАНО для partial causal mediation.
ДОСТАТОЧНО ДОКАЗАНО ДЛЯ ВНУТРЕННЕГО ИСПОЛЬЗОВАНИЯ для action-policy mediation.
ИНТЕРЕСНО, НО ГРЯЗНО для blind-semantic mediation, потому что shuffled-label
control тоже двигает часть semantic readout.
```

Обновление после Ministral heldout mediation:

```text
Ministral heldout НЕ поддержал raw target-control vector как clean causal
handle. Natural target-control gaps сильные, но target_control intervention
не бьет random/shuffled controls.

Итог:
  Qwen layer-32 action-policy mediation остается валидным Qwen-specific
  causal-handle result.

  Broad cross-model claim "raw mean target-control vector mediates behavior"
  сейчас НЕ ПОДДЕРЖАН.

  Основной context-induced latent/readout/action shift claim не ломается,
  потому что он опирается на natural geometry/readout/action evidence, а не
  на универсальность одного causal vector.
```

Артефакты:

```text
latent_shift_evidence_package_v1/causal_mediation/ministral_heldout/
latent_shift_evidence_package_v1/causal_mediation/ministral_heldout/mediation_readout.md
```

## Frozen Runner Rule

Decision:

```text
llm_attractor_colab_copy_paste.py is frozen as the main diagnostic runner.
```

Reason:

```text
The runner is now large enough that adding more blocks increases ambiguity,
runtime, and review surface more than it improves the evidence. New questions
must be implemented as separate focused scripts with narrow outputs.
```

Current examples:

```text
validity_bootstrap_analysis.py
reviewer_robustness_audit_v1.py
cross_model_comparison_heldout_v1.py
cross_corpus_comparison_v1.py
causal_mediation_v1_colab.py
```

Operational rule:

```text
Do not add more metrics to the main runner unless an existing core metric is
broken. New tests should answer a named objection or mechanism question in a
separate script.
```

## Attractor Basin Test v1

Script:

```text
attractor_basin_test_v1_colab.py
```

Why it is separate:

```text
The main runner measures latent/readout/action shifts. A strict attractor claim
requires a different dynamical object: perturbation-return, trajectory
contraction, and basin threshold. Adding this to the main runner would mix the
v1 evidence package with a new mechanism test.
```

What it measures:

```text
1. basin_state:
   hidden target_closeness after target/control/rejection/control-perturbed
   histories.

2. basin_return:
   return_score = target_closeness_after_recovery
                  - target_closeness_at_perturbation

3. basin_contraction:
   contraction_ratio = pairwise_distance_after_recovery
                       / pairwise_distance_at_perturbation

4. basin_threshold:
   target/control token mixing with target_prefix vs target_suffix and neutral
   recovery turns.
```

Decision rule:

```text
Strict attractor-like support:
  return_score CI_low > 0
  contraction_ratio CI_high < 1

If return succeeds but contraction fails:
  claim recovery/persistence, not basin convergence.

If contraction succeeds but return does not:
  check whether trajectories converge toward target, control, or neutral.
```

## Control Baseline Verification

Claude audit question resolved:

```text
The Ministral heldout natural gaps used in causal_mediation_v1
agent_action = -6.218764
blind_semantic = -11.349387
are content-matched, not legacy repetitive-baseline numbers.
```

Evidence:

```text
attractor_results_agent_loop_ministral3_14b_heldout/run_metadata.json:
  primary_control_mode = content_matched

attractor_results_agent_loop_ministral3_14b_heldout/summary_report.txt:
  Control source: auto:content_matched
  Primary control mode: content_matched

latent_shift_evidence_package_v1/input_texts_heldout.json:
  primary_control_mode = content_matched
  control_texts_source = auto:content_matched

content_matched_control_seeds:
  Qwen heldout input_texts.json == Ministral heldout input_texts.json
  Qwen heldout input_texts.json == latent_shift_evidence_package_v1/input_texts_heldout.json
```

Conclusion:

```text
No Ministral re-run is required for this baseline issue. The negative
Ministral raw-vector mediation result remains interpretable as a genuine
causal-handle non-replication, not a baseline mismatch artifact.
```

## OLMo2 Heldout Causal Mediation

Artifacts:

```text
latent_shift_evidence_package_v1/causal_mediation/olmo2_heldout/
latent_shift_evidence_package_v1/causal_mediation/olmo2_heldout/mediation_readout.md
```

Result:

```text
OLMo2 is not a Ministral-style null result.
target_control intervention moves margins in the expected direction in all four
main cells:

agent_action control_plus: 1.071 [0.301, 2.107]
agent_action target_minus: 0.836 [0.041, 2.022]
blind_semantic control_plus: 0.251 [0.170, 0.346]
blind_semantic target_minus: 0.541 [0.296, 0.856]
```

Caveat:

```text
This is not a clean Qwen-style shared causal handle. Target_control confidence
intervals overlap random_same_norm, shuffled_label, or wrong_layer controls in
most comparison cells.
```

Current causal-mechanism map:

```text
Qwen:
  cleanest single-direction action-policy mediation.

Ministral:
  strong natural gaps, but raw target-control vector fails against controls.

OLMo2:
  positive directional target-control mediation, but not cleanly specific.
```

Next mechanism test:

```text
distributed/subspace mediation with leave-one-text-out fitting and same-rank
random/shuffled/wrong-layer controls.
```

## Attractor-Criteria Status

Current conclusion:

```text
We have evidence for attractor-like signatures, not for a formal attractor
basin in the strict dynamical-systems sense.
```

Implementation update:

```text
llm_attractor_colab_copy_paste.py now contains a strict_attractor_validation
block. It is the explicit gate for deciding whether formal attractor language
is supported or rejected in a run.

Outputs:
  hidden_cluster_compression.csv
  hidden_cluster_compression_radius.png
  hidden_cluster_compression_ratio.png
  strict_attractor_probe_set.csv
  strict_attractor_semantic_raw.csv
  strict_attractor_semantic_summary.csv
  strict_attractor_semantic_delta.csv
  strict_attractor_turns.csv
  strict_attractor_hidden_raw.csv
  strict_attractor_condition_summary.csv
  strict_attractor_criteria.csv
  strict_attractor_semantic_fraction_map.png
  strict_attractor_hidden_fraction_map.png

The decisive file is:
  strict_attractor_criteria.csv
```

The important correction is that the project already has metrics for several
attractor-relevant properties, but not for all of them.

| Strict attractor criterion | Existing metric coverage | Current status |
| --- | --- | --- |
| 1. Basin / convergence: different starting states converge to the same regime | Now directly tested by `strict_attractor_*` via `N_THEN_T`, `C_THEN_T`, and `SHUFFLED_T_DIRECT`, plus hidden centroid closeness. | Pending next run. Previously not proven. |
| 2. Stability: small perturbations do not knock the system out | Now directly tested by `strict_attractor_*` via `T_NEUTRAL_2` and `T_NEUTRAL_4`, plus older persistence metrics. | Pending next run. Previously only partly supported as residual persistence. |
| 3. Return: after being pushed away, state returns to the regime | Now directly tested by `strict_attractor_*` via `T_PERTURB_NEUTRAL_0/2/4`. | Pending next run. This is the hardest criterion and may falsify formal-attractor language. |
| 4. Trajectory/path dependence | Covered by order_hysteresis_* metrics. | Supported as order/recency/path sensitivity, not strict hysteresis. |
| 5. Threshold / nonlinearity | Covered partially by mixing_threshold_* metrics. | Supported as dose/suffix boundary, not yet a formal phase transition. |
| 6. Persistence / retention | Covered by blind_neutral_persistence_* and rejection_persistence_* metrics. | Supported with decay: residual remains after neutral/rejection turns. |
| 7. Phase/geometric picture | Now strengthened by `strict_attractor_hidden_raw.csv` and `hidden_cluster_compression.csv`: distance to target/control centroids, hidden fraction toward target, closer-to-target rate, separation-over-radius, and target-cluster radius versus control-cluster radius. | Pending next run for strict geometry/compression; older evidence supports separation only. |

Useful shorthand:

```text
Proven:
  context-induced latent/readout/action regime signatures.

Supported:
  persistence, rejection residuals, order/path sensitivity, dose/suffix
  sensitivity, hidden geometry separation.

Not proven:
  formal basin of attraction, autonomous return, strict stability under small
  perturbations, universal nonlinear threshold, full attractor landscape.
  This remains not proven until strict_attractor_criteria.csv passes.
```

How to phrase it:

```text
Current evidence supports attractor-like signatures of a context-induced
distributed regime. It does not yet support the stronger claim that we have
identified a formal attractor basin.
```

What would be needed for a stronger attractor claim:

```text
1. Many-to-one convergence:
   several unrelated starting histories and paraphrased inductions converge to
   the same hidden/readout region.

2. Perturb-and-return:
   after target induction, insert small control/noise perturbations and show
   return toward the target-like readout without reintroducing target content.

3. Stability radius:
   estimate how much perturbation is needed before the regime collapses.

4. Dense dose curve:
   use more mixture doses and fit linear vs nonlinear/threshold models.

5. State-space clustering:
   show trajectory endpoints cluster around a target-like center, not merely a
   mean target-control displacement.
```

New geometric compression check:

```text
hidden_cluster_compression.csv
```

Decision signal:

```text
target_radius_over_control_radius_cosine < 1
```

Meaning:

```text
Different target texts are more tightly concentrated around a shared hidden
centroid than their matched controls. This strengthens the phase-space picture,
but it does not by itself prove autonomous return or a formal attractor.
```

Decision rule after the next run:

```text
If strict_attractor_criteria.csv marks:
  basin_convergence = supported
  stability_under_neutral_perturbation = supported
  return_after_mild_reset = supported
  hidden_centroid_geometry = supported
  hidden_cluster_compression = supported
  strict_attractor_overall = supported

then formal attractor language is defensible for that model/corpus/probe setup.

If return_after_mild_reset or basin_convergence fails, the correct phrase
remains:
  attractor-like context-induced distributed regime
not:
  formal attractor basin
```

## Strict Operational Attractor Rule

Current rule as of 2026-05-20:

```text
The large runner remains frozen as the v1 diagnostic runner.
The decisive attractor test is the focused script:

  attractor_basin_test_v1_colab.py

The decisive verdict file is:

  strict_attractor_verdict.json
```

Use of "attractor":

```text
Allowed:
  status = strict_attractor_confirmed

Rejected:
  status = strict_attractor_refuted

Not usable:
  status = inconclusive
```

The strict script now tests both text-level and direct hidden-state criteria:

```text
1. target/control hidden separation
2. return after text perturbation
3. contraction of text-start trajectories
4. return after direct hidden-state impulse
5. contraction after direct hidden-state impulse
6. local Jacobian contraction around the target-start region
7. target-vs-control specificity
```

This closes the naming ambiguity: if the strict verdict fails, the project
must stop calling that result an attractor and use "context-induced latent
regime" or "metastable discourse-policy regime" instead.

The mathematically strongest artifact is now:

```text
mathematical_attractor_verdict.json
```

It maps the strict result to:

```text
strict_mathematical_attractor
no_strict_mathematical_attractor
inconclusive
```
